# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""The one durable run store: events, content-addressed artifacts, dataset builds, working state.

The DDL is `sqlite_event_log.schema.sql` (schema 12; a schema-11 store, whose `datasets` rows also carry a
provenance, opens as it is). Artifacts above `INLINE_CEILING_BYTES` keep an empty row and live beside the ledger
under `<ledger>-artifacts/`, zstd-framed (`.zst`) when smaller.
"""

from __future__ import annotations

import fcntl
import json
import os
import sqlite3
import threading
import uuid
from collections.abc import Mapping
from compression import zstd
from dataclasses import dataclass
from importlib.resources import files
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Self

from gm.domain import SampleSetName
from gm.domain.identity import DatasetVersion, SampleKey
from gm.domain.results import Artifact, ArtifactRef, InlineArtifact, artifact_address, resolve
from gm.ids import ActionId, ArtifactOid, CriterionCoordinate, EventId, canonical_json
from gm.ledger.events import (
    EVENT_KEY,
    SCHEMA,
    ActionReserved,
    CheckpointPhase,
    ExpansionRecorded,
    JudgmentRecorded,
    LedgerEvent,
    SoftSkip,
    StoredEvent,
    TaskOutcome,
    encoded,
    event_files,
    event_from_json,
)
from gm.spend import SpendRows

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence
    from pathlib import Path

INLINE_CEILING_BYTES = 2 * 1024 * 1024
INPUT_CHANNEL = "input"
_SPILL_SUFFIX: Final = "-artifacts"
_FRAME_SUFFIX: Final = ".zst"
_EMPTY_OID: Final = ArtifactOid.of_bytes(b"")
_SCHEMA = (files(__package__) / "sqlite_event_log.schema.sql").read_text(encoding="utf-8")
_SETTLING = (TaskOutcome, JudgmentRecorded, ExpansionRecorded, SoftSkip)


class LedgerError(Exception):
    """A ledger contract violation."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Arrival:
    """One completion presenting itself for settlement, once."""

    action: ActionId
    payload: str


def _open_connection(*, path: Path, read_only: bool) -> sqlite3.Connection:
    uri = f"file:{path}?mode=ro" if read_only else str(path)
    conn = sqlite3.connect(uri, uri=read_only, isolation_level=None, check_same_thread=False)
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    if not read_only:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'schema_version'").fetchone() is None:
            conn.executescript(_SCHEMA)
            conn.execute("COMMIT")
    return conn


class SqliteEventLog:
    """The run ledger: one writer connection and one shared read connection."""

    def __init__(self, path: Path, *, read_only: bool = False) -> None:
        """Open one run ledger, taking the writer lock unless read-only."""
        self.path = path
        self._lock = threading.Lock()
        self._selected_sets: dict[
            tuple[str, str, str], Mapping[tuple[DatasetVersion, SampleKey], tuple[ArtifactRef, ...]]
        ] = {}
        self._decoded: dict[tuple[int, str], StoredEvent] = {}
        self._lock_fd = None
        if not read_only:
            self._lock_fd = os.open(f"{path}.lock", os.O_CREAT | os.O_RDWR, 0o644)
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self._conn = _open_connection(path=path, read_only=read_only)
        # One read connection for every thread: jobs run on short-lived threads, so a connection
        # per thread would hold two descriptors per finished job until the run ends.
        self._reader = _open_connection(path=path, read_only=True)
        self._read_lock = threading.Lock()
        first = self._conn.execute("SELECT payload FROM events WHERE seq = 1").fetchone()
        #: The schema this ledger's events are spelled in: its first event's, or this code's for a fresh ledger.
        self.version = SCHEMA if first is None else json.loads(first[0])[EVENT_KEY]

    def _read(self, sql: str, params: tuple = ()) -> list[tuple]:
        with self._read_lock:
            return self._reader.execute(sql, params).fetchall()

    def __enter__(self) -> Self:
        """Return this open ledger for context-manager use."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close the ledger on context exit."""
        self.close()

    def close(self) -> None:
        """Close every connection this ledger opened and release the writer lock."""
        self._reader.close()
        self._conn.close()
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None

    # -- events --------------------------------------------------------------

    def apply(
        self,
        events: Sequence[LedgerEvent],
        *,
        phase: CheckpointPhase | None = None,
        scope: Collection[CriterionCoordinate] | None = None,
        fired: str | None = None,
        arrivals: Sequence[Arrival] = (),
    ) -> tuple[StoredEvent, ...]:
        """Append events, settle their reservations and set the crossing phase in one transaction."""
        with self._lock, self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            for arrival in arrivals:
                self._conn.execute(
                    "INSERT INTO completion_inbox (action, payload) VALUES (?, ?)", (arrival.action, arrival.payload)
                )
            stored = tuple(self._append(event=event) for event in events)
            if phase is not None:
                self._conn.execute(
                    "UPDATE checkpoint_phase SET phase = ?, scope = ?, fired = ? WHERE id = 1 AND phase IN (?, ?, ?, ?)",
                    (
                        phase.value,
                        None if scope is None else canonical_json(sorted(map(str, scope))),
                        fired,
                        *CheckpointPhase,
                    ),
                )
        return stored

    def _append(self, *, event: LedgerEvent) -> StoredEvent:
        eid, payload = encoded(event, self.version)
        seq = self._conn.execute("INSERT INTO events (event_id, payload) VALUES (?, ?)", (eid, payload)).lastrowid
        for path, artifact in sorted(event_files(event=event).items()):
            oid, size = artifact_address(artifact=artifact)
            self._conn.execute(
                "INSERT INTO event_artifacts (event_seq, path, oid, size) VALUES (?, ?, ?, ?)",
                (seq, path, str(oid), size),
            )
        if isinstance(event, ActionReserved):
            self._conn.execute("INSERT INTO reservations (action, reserved_seq) VALUES (?, ?)", (event.action, seq))
        elif isinstance(event, _SETTLING):
            self._conn.execute(
                "UPDATE reservations SET settled_seq = ? WHERE action = ? AND settled_seq IS NULL", (seq, event.action)
            )
            self._conn.execute("DELETE FROM completion_inbox WHERE action = ?", (event.action,))
            self._conn.execute("DELETE FROM retry_state WHERE action = ?", (event.action,))
        return StoredEvent(seq=seq, event_id=eid, event=event)

    def events(self, *, through_seq: int | None = None) -> tuple[StoredEvent, ...]:
        """Return committed events in ledger order, each decoded once per log."""
        where = "" if through_seq is None else " WHERE seq <= ?"
        artifact_where = "" if through_seq is None else " WHERE event_seq <= ?"
        params: tuple[int, ...] = () if through_seq is None else (through_seq,)
        rows = self._read(f"SELECT seq, event_id, payload FROM events{where} ORDER BY seq", params)
        handles: dict[int, dict[str, Artifact]] = {}
        for seq, path, oid, size in self._read(
            f"SELECT event_seq, path, oid, size FROM event_artifacts{artifact_where} ORDER BY event_seq, path", params
        ):
            handles.setdefault(seq, {})[path] = ArtifactRef(oid=ArtifactOid(oid), size=size)
        out = []
        for seq, eid, payload in rows:
            if (hit := self._decoded.get((seq, eid))) is None:
                event = event_from_json(text=payload, files=handles.get(seq, {}))
                hit = self._decoded[seq, eid] = StoredEvent(seq=seq, event_id=EventId(eid), event=event)
            out.append(hit)
        return tuple(out)

    # -- retries -------------------------------------------------------------

    def record_retry(self, *, action: ActionId, attempts: int, error: str, spend: SpendRows = ()) -> None:
        """Record retry state after an open reservation fails transiently."""
        rows = canonical_json([[None if tag is None else str(tag), axis, amount] for tag, axis, amount in spend])
        with self._lock:
            self._conn.execute(
                "INSERT INTO retry_state (action, attempts, last_error, spend_rows) VALUES (?, ?, ?, ?)"
                " ON CONFLICT(action) DO UPDATE SET attempts = excluded.attempts, last_error = excluded.last_error,"
                " spend_rows = excluded.spend_rows WHERE excluded.attempts > retry_state.attempts",
                (action, attempts, error, rows),
            )

    # -- artifacts -----------------------------------------------------------

    def _spilled(self, *, oid: ArtifactOid) -> Path:
        return self.path.with_name(self.path.name + _SPILL_SUFFIX) / str(oid)[:2] / str(oid)

    def put_artifact(self, *, data: bytes) -> ArtifactOid:
        """Store bytes content-addressed, idempotently; safe from worker threads."""
        oid = ArtifactOid.of_bytes(data)
        row = data
        if len(data) > INLINE_CEILING_BYTES:
            spilled = self._spilled(oid=oid)
            framed = spilled.with_name(spilled.name + _FRAME_SUFFIX)
            if not framed.is_file() and not spilled.is_file():
                spilled.parent.mkdir(parents=True, exist_ok=True)
                packed = zstd.compress(data)
                keep, target = (packed, framed) if len(packed) < len(data) else (data, spilled)
                scratch = spilled.with_name(f"{spilled.name}.{uuid.uuid4().hex}.part")
                scratch.write_bytes(keep)
                scratch.replace(target)
            row = b""
        with self._lock:
            self._conn.execute(
                "INSERT INTO artifacts (oid, data) VALUES (?, ?) ON CONFLICT(oid) DO NOTHING", (oid, row)
            )
        return oid

    def get_artifact(self, oid: ArtifactOid) -> bytes:
        """Read stored bytes by content address."""
        ((data,),) = self._read("SELECT data FROM artifacts WHERE oid = ?", (oid,))
        if data or oid == _EMPTY_OID:
            return bytes(data)
        spilled = self._spilled(oid=oid)
        framed = spilled.with_name(spilled.name + _FRAME_SUFFIX)
        return zstd.decompress(framed.read_bytes()) if framed.is_file() else spilled.read_bytes()

    def route_artifact(self, data: bytes) -> ArtifactRef:
        """Store bytes out-of-band and return the reference an event carries."""
        return ArtifactRef(oid=self.put_artifact(data=data), size=len(data))

    def resolve_artifact(self, artifact: Artifact) -> bytes:
        """Resolve an artifact handle through the ledger's content store."""
        return resolve(artifact, self.get_artifact)

    # -- datasets ------------------------------------------------------------

    def put_dataset(
        self,
        *,
        version: DatasetVersion,
        sets: Mapping[str, Sequence[str]],
        files: Mapping[tuple[str, str], Sequence[bytes | Artifact]],
        images: Mapping[str, str] | Mapping[SampleKey, str],
    ) -> None:
        """Persist one immutable dataset build in one transaction, once."""
        if self.has_dataset(version):
            return
        rows, carried = [], []
        for (sample, channel), row in files.items():
            for ordinal, entry in enumerate(row):
                if isinstance(entry, bytes):
                    entry = InlineArtifact(data=entry)
                oid, _ = artifact_address(artifact=entry)
                rows.append((version, str(sample), str(channel), ordinal, oid))
                if isinstance(entry, InlineArtifact):
                    carried.append((oid, entry.data))
        with self._lock, self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            self._conn.execute("INSERT INTO datasets VALUES (?)", (version,))
            self._conn.executemany("INSERT INTO artifacts VALUES (?, ?) ON CONFLICT(oid) DO NOTHING", carried)
            self._conn.executemany(
                "INSERT INTO dataset_sets VALUES (?, ?, ?, ?)",
                [(version, name, index, sample) for name, keys in sets.items() for index, sample in enumerate(keys)]
                + [(version, name, 0, "") for name, keys in sets.items() if not keys],
            )
            self._conn.executemany("INSERT INTO dataset_files VALUES (?, ?, ?, ?, ?)", rows)
            self._conn.executemany(
                "INSERT INTO dataset_images VALUES (?, ?, ?)", [(version, s, image) for s, image in images.items()]
            )

    def has_dataset(self, version: str) -> bool:
        """Return whether a dataset build is present."""
        return bool(self._read("SELECT 1 FROM datasets WHERE version = ?", (version,)))

    def selected_inputs(
        self,
        *,
        versions: Collection[DatasetVersion],
        sample_sets: Collection[SampleSetName],
        channel: str = INPUT_CHANNEL,
    ) -> Mapping[tuple[DatasetVersion, SampleKey], tuple[ArtifactRef, ...]]:
        """Return artifacts selected from named sets across dataset builds, on one channel."""
        pieces = [
            self._selected_set(version=version, name=name, channel=channel)
            for version in sorted({str(version) for version in versions})
            for name in sorted({str(name) for name in sample_sets})
        ]
        merged: dict[tuple[DatasetVersion, SampleKey], tuple[ArtifactRef, ...]] = {}
        for key in dict.fromkeys(chain.from_iterable(pieces)):
            merged[key] = tuple(chain.from_iterable(piece[key] for piece in pieces if key in piece))
        return MappingProxyType(merged)

    def _selected_set(
        self, *, version: str, name: str, channel: str
    ) -> Mapping[tuple[DatasetVersion, SampleKey], tuple[ArtifactRef, ...]]:
        if (held := self._selected_sets.get((version, name, channel))) is not None:
            return held
        rows = self._read(
            "SELECT s.sample, f.oid, LENGTH(a.data)"
            " FROM dataset_sets s"
            " LEFT JOIN dataset_files f"
            " ON f.version = s.version AND f.sample = s.sample AND f.channel = ?"
            " LEFT JOIN artifacts a ON a.oid = f.oid"
            " WHERE s.version = ? AND s.name = ?"
            " ORDER BY s.ordinal, f.ordinal",
            (channel, version, name),
        )
        grouped: dict[tuple[DatasetVersion, SampleKey], list[ArtifactRef]] = {}
        for sample, oid, size in rows:
            if not sample:
                continue
            handles = grouped.setdefault((DatasetVersion(version), SampleKey(sample)), [])
            if oid is not None:
                handles.append(ArtifactRef(oid=ArtifactOid(oid), size=size))
        selected = {key: tuple(handles) for key, handles in grouped.items()}
        if self.has_dataset(version):
            self._selected_sets[version, name, channel] = selected
        return selected
