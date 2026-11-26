# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""Let a solver put its work in front of the review seat it carries, mid-episode.

Shared across harnesses. What it resolves is the DOMAIN's judging seat, and nothing here reads a
patch or a solve loop; the command it advertises is a parameter, because a harness knows its own
entry point and this module does not.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# The workspace root: the seat and `shared` both live there.
sys.path.insert(0, str(Path.cwd()))

from shared.gateway.entrypoint import client_from_env

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from types import ModuleType

SEAT_PACKAGE = "task_agents"
REVIEW_ENTRY = "self_review"
CONTRACT = Path(".rqgm") / "contract.json"
JUDGES, SCORER, NODE_SCORED = "judges", "scorer", "node"
REVIEW_PROXY, PROXY_SEAT = "review_proxy", "seat"
SELF_REVIEW_FLAG = "--self-review"
#: The episode's own tree, as its shell is told. A solver whose shell works in a checkout outside
#: the tree cannot find the contract by walking up from its working directory.
TREE_VAR = "RQGM_EPISODE_TREE"
#: Where the solver's work is, when that is not the tree. The seat is read from the tree, the diff
#: from here.
CHECKOUT_VAR = "RQGM_AGENT_CHECKOUT"
#: What the mini-swe harness's entry is, kept as the default so an existing caller is unchanged.
#: A harness that enters differently passes its own `tail` rather than editing this.
INVOCATION_TAIL = f"-m meta.solve {SELF_REVIEW_FLAG}"
SELF_REVIEW_ARITY = 2
REVIEW_REPORT = "[self-review] score={score} diff={size}B tests={evidence}B"

PROXY_CLAUSE = """

YOUR REVIEWER IS A PROXY, NOT THE THING THAT SCORES YOU. The reviewer in this node stands in for a held-out one you never see and cannot read, and it is the held-out one that settles your utility. Your edits to the reviewer are never promoted, so making the copy you hold more lenient changes a number nobody scores you on.

IMPROVING THE REVIEWER DOES NOT IMPROVE YOUR UTILITY. Not a little, not indirectly. Effort spent there is spent for nothing, and it is effort the work itself could have had. Spend it on the task.

IT IS STILL THERE, AND IT IS STILL WORTH USING. It has not been removed and you are not working blind. Consult it exactly as you would otherwise: try your work against it, and put that work in front of it before you file. It is a rehearsal of the judgment you are about to face, which is why a reviewer that predicts the held-out one helps you here and a lenient one cannot."""

NO_JUDGING_SEAT = "this tree's {contract} declares no node-scored judging seat, so there is no seat to consult. A domain whose judging slots are all fixed instruments carries no module a solver can call"
SEAT_WITHOUT_ENTRY = "this tree declares {seat!r} as its judging seat and that seat states no {entry!r}, so there is nothing to consult. The seat exists: what is missing is the entry point, on that module"
SEAT_BROKEN = "this tree declares {seat!r} as its judging seat and importing it raised {failure}. The seat is broken rather than missing: it is there, and it does not load"


def locate(*, start: Path, environ: Mapping[str, str] | None = None) -> Path:
    """Return the tree holding `.rqgm/contract.json`: the one the episode's shell was told, else the nearest at or above `start`, else `start`.

    A solver runs the self-review from wherever its shell is. When that is inside the tree,
    walking up finds the contract; when the shell works in a checkout beside the tree, only
    the episode's own word (`TREE_VAR`, set by the harness that built the shell) reaches it.
    """
    declared = (os.environ if environ is None else environ).get(TREE_VAR, "")
    if declared and (Path(declared) / CONTRACT).is_file():
        return Path(declared).resolve()
    here = start.resolve()
    for candidate in (here, *here.parents):
        if (candidate / CONTRACT).is_file():
            return candidate
    return here


def _contract(*, workspace: Path) -> dict:
    try:
        stated = json.loads((workspace / CONTRACT).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return stated if isinstance(stated, dict) else {}


def review_seat(*, workspace: Path) -> str | None:
    """Return the module name of the one judging seat this tree declares, or None."""
    stated = _contract(workspace=workspace)
    proxied, judges = stated.get(REVIEW_PROXY), stated.get(JUDGES)
    if isinstance(proxied, dict) and proxied:
        seats = (spec.get(PROXY_SEAT) for spec in proxied.values() if isinstance(spec, dict))
        named = sorted(str(seat) for seat in seats if isinstance(seat, str))
    else:
        slots = judges.items() if isinstance(judges, dict) else ()
        named = sorted(str(slot) for slot, spec in slots if isinstance(spec, dict) and spec.get(SCORER) == NODE_SCORED)
    return f"{SEAT_PACKAGE}.{named[0]}" if len(named) == 1 else None


def invocation(*, python: str, tail: str = INVOCATION_TAIL) -> str:
    """Return the command a solver types, naming the interpreter its container carries.

    Args:
        python (str): The interpreter this episode runs under.
        tail (str): The entry the harness offers, defaulting to the mini-swe
            one. A second harness states its own rather than editing this
            module, which is the whole reason the module is shared.

    Returns:
        str: The command, as the served clause prints it.
    """
    return f"{python} {tail}"


def episode_python() -> str:
    """Return the interpreter this episode is running under."""
    return sys.executable


def consultable(*, workspace: Path) -> tuple[ModuleType | None, str | None]:
    """Return this tree's consultable seat module, or why it has none; exactly one is not None."""
    workspace = locate(start=workspace)
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
    seat = review_seat(workspace=workspace)
    if seat is None:
        return None, NO_JUDGING_SEAT.format(contract=CONTRACT)
    try:
        found = importlib.util.find_spec(seat)
    except (ImportError, ValueError) as failure:
        return None, SEAT_BROKEN.format(seat=seat, failure=f"{type(failure).__name__}: {failure}")
    origin = Path(found.origin).resolve() if found is not None and found.origin else None
    if origin is not None and workspace not in origin.parents:
        raise ValueError(f"the review seat is read from {workspace} and would be imported from {origin}, two different trees")
    try:
        carried = importlib.import_module(seat) if found is not None else None
    except (ImportError, ValueError) as failure:
        return None, SEAT_BROKEN.format(seat=seat, failure=f"{type(failure).__name__}: {failure}")
    if carried is not None and hasattr(carried, REVIEW_ENTRY):
        return carried, None
    return None, SEAT_WITHOUT_ENTRY.format(seat=seat, entry=REVIEW_ENTRY)


def proxy_clause(*, workspace: Path) -> str:
    """Return what this tree's solver is told about a proxy reviewer, or empty text."""
    proxied = _contract(workspace=workspace).get(REVIEW_PROXY)
    return PROXY_CLAUSE if isinstance(proxied, dict) and proxied else ""


def why_not_reviewing(*, workspace: Path) -> str | None:
    """Return why this tree carries no consultable review seat, or None where it does."""
    return consultable(workspace=workspace)[1]


def reviewing(*, workspace: Path | None = None) -> bool:
    """Answer whether the tree this episode stands in carries a consultable review seat."""
    return why_not_reviewing(workspace=locate(start=Path.cwd()) if workspace is None else workspace) is None


def coupled(*, clause: str, python: str) -> None:
    """Refuse where the served clause and this entry point have drifted apart."""
    advertised = invocation(python=python)
    if not any(line.strip().startswith(advertised) for line in clause.splitlines()):
        raise RuntimeError(f"the served self-review clause does not print {advertised!r}, so the prompt and the command it advertises have drifted apart")


def review(
    *,
    evidence: Path,
    workspace: Path,
    environ: Mapping[str, str],
    capture: Callable[..., str],
    state: Callable[..., tuple[str, frozenset[str], str]],
) -> str:
    """Return this episode's own reviewer's verdict on the work it has done so far."""
    if (client := client_from_env()) is None:
        raise RuntimeError("this episode holds no gateway grant, so it cannot ask its reviewer anything")
    return feedback(evidence=evidence, workspace=workspace, environ=environ, capture=capture,
                    state=state, chat=lambda prompt: client.chat(messages=[{"role": "user", "content": prompt}]).content)


def feedback(
    *, evidence: Path, workspace: Path, environ: Mapping[str, str],
    capture: Callable[..., str], state: Callable[..., tuple[str, frozenset[str], str]],
    chat: Callable[[str], str],
) -> str:
    """Consult the carried seat through the native caller's explicitly supplied continuation."""
    seat, refused = consultable(workspace=workspace)
    if seat is None:
        raise RuntimeError(refused)
    baseline, _, task = state(environ=environ)
    printed = evidence.read_bytes()
    candidate = capture(workspace=worked_in(workspace=workspace, environ=environ), baseline=baseline)
    spoken: list[str] = []

    def asked(prompt: str) -> str:
        spoken.append(chat(prompt))
        return spoken[-1]

    score = seat.self_review(problem=task, diff=candidate, author_output=printed, chat=asked)
    report = REVIEW_REPORT.format(score=score, size=len(candidate.encode()), evidence=len(printed))
    return "\n\n".join([report, *spoken])


def worked_in(*, workspace: Path, environ: Mapping[str, str]) -> Path:
    """Return where the episode's work is: its stated checkout, else the tree itself."""
    stated = environ.get(CHECKOUT_VAR, "")
    return Path(stated).resolve() if stated and Path(stated).is_dir() else workspace


def main(*, argv: Sequence[str], capture: Callable[..., str], state: Callable[..., tuple[str, frozenset[str], str]]) -> int:
    """Run one self-review for the episode whose shell invoked this module."""
    if len(argv) != SELF_REVIEW_ARITY or argv[0] != SELF_REVIEW_FLAG:
        sys.stderr.write(f"usage: {invocation(python=episode_python())} <file-holding-your-test-output>\n")
        return 1
    try:
        verdict = review(evidence=Path(argv[1]), workspace=locate(start=Path.cwd()), environ=os.environ, capture=capture, state=state)
    except (RuntimeError, OSError, ValueError, KeyError) as refused:
        sys.stderr.write(f"[self-review] refused: {type(refused).__name__}: {refused}\n")
        return 1
    sys.stdout.write(verdict + "\n")
    return 0
