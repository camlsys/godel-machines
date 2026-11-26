"""Check a store's ledger offline: codec identity, checkpoint chain, fold.

Every stored event must decode and re-encode to its stored canonical text, whose sha256 is its id.
Every checkpoint must name the event before it and the running prefix digest over all earlier ids.
The whole ledger is then folded once; the projection digest lets two implementations compare folds.
Cost is one pass over the events and one fold, linear in the event count.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

from gm import ledger
from gm.domain.results import ArtifactRef
from gm.ids import ArtifactOid
from gm.ledger import events as ev


def projection_digest(p: ledger.Fold) -> str:
    """Digest the folded facts a decision reads, in an order-free canonical form."""

    def pairs(mapping):
        return sorted((str(k), str(v)) for k, v in mapping.items())

    facts = {
        "lineage": pairs(p.lineage),
        "pins": pairs(p.active_pins),
        "epochs": pairs(p.epochs),
        "decisions": p.decisions,
        "open": sorted(map(str, p.open_actions)),
        "done": sorted(map(str, p.done)),
        "outcomes": sorted(map(str, p.outcomes)),
        "judgments": [str(j.stored.event_id) for j in p.judgments]
        if p.judgments and hasattr(p.judgments[0], "stored")
        else len(p.judgments),
        "spent": sorted((json.dumps([str(a) for a in k]), pairs(v)) for k, v in p.spent.items()),
        "completed": sorted((str(k), v) for k, v in p.completed.items()),
        "minted": sorted(str(k) for k in p.minted),
    }
    return hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest()


def verify_ledger(db: Path) -> dict[str, object]:
    """Check one ledger file and return what was found; `ok` is true only with no problem."""
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    files: dict[int, dict[str, ArtifactRef]] = {}
    for seq, path, oid, size in connection.execute(
        "select event_seq, path, oid, size from event_artifacts order by event_seq, path"
    ):
        files.setdefault(seq, {})[path] = ArtifactRef(oid=ArtifactOid(oid), size=size)
    started = time.perf_counter()
    problems: list[str] = []
    stored: list[ledger.StoredEvent] = []
    digest = hashlib.sha256(ev._PREFIX_SEED).digest()
    version = ev.SCHEMA
    for seq, event_id, payload in connection.execute("select seq, event_id, payload from events order by seq"):
        try:
            event = ev.event_from_json(text=payload, files=files.get(seq, {}))
            if isinstance(event, ledger.RunConfigured):
                version = event.manifest.version
            again, text = ev.encoded(event, version)
        except Exception as exc:
            problems.append(f"{seq}: decode {type(exc).__name__}: {str(exc)[:200]}")
            continue
        if text != payload:
            problems.append(f"{seq}: re-encoded text differs")
        if again != event_id or hashlib.sha256(payload.encode()).hexdigest() != event_id:
            problems.append(f"{seq}: id differs")
        if isinstance(event, ledger.CheckpointPublished) and (
            str(event.prefix_digest) != digest.hex()
            or str(event.last_event) != (stored[-1].event_id if stored else None)
        ):
            problems.append(f"{seq}: checkpoint digest/last_event differs")
        stored.append(ledger.StoredEvent(seq=seq, event_id=event_id, event=event))
        digest = hashlib.sha256(digest + event_id.encode()).digest()
    connection.close()
    decoded = time.perf_counter()
    projection = None
    try:
        projection = projection_digest(ledger.project(stored))
    except Exception as exc:
        problems.append(f"fold: {type(exc).__name__}: {str(exc)[:300]}")
    folded = time.perf_counter()
    return {
        "events": len(stored),
        "ok": not problems,
        "problems": problems[:5],
        "nproblems": len(problems),
        "projection": projection,
        "codec_s": round(decoded - started, 3),
        "fold_s": round(folded - decoded, 3),
    }


def verify(*, store: Path) -> dict[str, object]:
    """Verify a store directory (holding `ledger.sqlite`) or a bare ledger file."""
    return verify_ledger(store / "ledger.sqlite" if store.is_dir() else store)
