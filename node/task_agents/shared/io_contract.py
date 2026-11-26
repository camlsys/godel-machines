# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""Refuse a seat's inputs before it runs and its answer when it returns, per declared arm."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

CONTRACT = Path(".rqgm/contract.json")
IO_KEY = "io"
DEPTH_KEY = "depth"
EXPERIENCE = "experience"
DATA = "data"
GENERATIONS = "generations"
GENERATE = "generate"
JUDGE = "judge"
HOST_INSTRUMENT = "host-instrument"
ARM_KEYS = ("reads", "positional", "writes", "evidence")


class IoRefusalError(RuntimeError):
    """A seat's inputs or its answer do not match what the form declared."""


def _stated(*, contract: Path) -> Mapping[str, Any]:
    if not contract.is_file():
        raise IoRefusalError(
            f"{contract} is absent, so nothing states what this seat reads and writes or which"
            " generation it is; refusing rather than running unchecked",
        )
    stated = json.loads(contract.read_text(encoding="utf-8"))
    if not isinstance(stated, dict):
        raise IoRefusalError(f"{contract} is not an object: {stated!r}")
    return stated


def declared_depth(*, contract: Path = CONTRACT) -> int:
    """Return this node's generation as the host wrote it; genesis carries none, so absence refuses."""
    depth = _stated(contract=contract).get(DEPTH_KEY)
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
        raise IoRefusalError(
            f"{contract} states {DEPTH_KEY}={depth!r}, not a whole number at or above zero: the"
            " host failed to stage the key, and a stand-in files this episode's work under"
            " another node's generation",
        )
    return depth


def own_experience_path(*, workspace: Path = Path(), contract: Path | None = None) -> Path:
    """Return the one folder this node's own work is written to, and no other node's."""
    stated = workspace / CONTRACT if contract is None else contract
    return workspace / EXPERIENCE / GENERATIONS / str(declared_depth(contract=stated))


def own_attempt_cell(
    *,
    role: str,
    task: str,
    workspace: Path = Path(),
    contract: Path | None = None,
) -> Path:
    """Return the folder this node's attempts for one seat and task are numbered under."""
    return own_experience_path(workspace=workspace, contract=contract) / role / task


def shared_data_path(*, workspace: Path = Path()) -> Path:
    """Return the chain's shared material root, which carries no depth."""
    return workspace / EXPERIENCE / DATA


def declared_arm(*, role: str, mode: str, contract: Path = CONTRACT) -> dict[str, Any]:
    """Return one seat's declared vocabulary on one arm, refusing an undeclared seat or arm."""
    block = _stated(contract=contract).get(IO_KEY)
    if not isinstance(block, dict):
        raise IoRefusalError(f"{contract} states no {IO_KEY!r} object, so no seat has a vocabulary")
    arms = block.get(role)
    if not isinstance(arms, dict) or not arms:
        raise IoRefusalError(f"{contract} declares seats {sorted(map(str, block))}, not {role!r}")
    if list(arms) == [HOST_INSTRUMENT]:
        raise IoRefusalError(
            f"{contract} declares {role!r} as a host-owned instrument, which runs kernel code"
            " rather than a seat this tree carries, so there is nothing here to run",
        )
    arm = arms.get(mode)
    if not isinstance(arm, dict) or any(key not in arm for key in ARM_KEYS):
        raise IoRefusalError(
            f"{contract} declares seat {role!r} on {sorted(map(str, arms))}; the arm {mode!r} is"
            f" absent or states less than {list(ARM_KEYS)}: {arm!r}",
        )
    return dict(arm)


def check_inputs(*, role: str, mode: str, files: Mapping[str, Any], arm: Mapping[str, Any]) -> None:
    """Refuse a staged input set that is not the one the form declared for this arm."""
    required = {str(name) for name in arm["reads"]}
    present = {str(name) for name in files}
    if missing := sorted(required - present):
        raise IoRefusalError(
            f"seat {role!r} on arm {mode!r} reads {sorted(required)} and was staged"
            f" {sorted(present)}: {missing} never arrived",
        )
    surplus = sorted(present - required)
    if not arm["positional"]:
        if surplus:
            raise IoRefusalError(
                f"seat {role!r} on arm {mode!r} reads exactly {sorted(required)} and was staged"
                f" {surplus} beside them, material the form never said it may see",
            )
        return
    if unnumbered := [name for name in surplus if not name.isdigit()]:
        raise IoRefusalError(
            f"seat {role!r} on arm {mode!r} reads the blind channel by position and was staged"
            f" {unnumbered}, which are not positions",
        )
    positions = sorted(int(name) for name in surplus)
    if positions != list(range(len(positions))):
        raise IoRefusalError(
            f"seat {role!r} on arm {mode!r} was staged blind positions {positions}, not a"
            " gapless run from zero",
        )


def check_output(
    *,
    role: str,
    mode: str,
    answer: Mapping[str, Any],
    arm: Mapping[str, Any],
) -> None:
    """Refuse an answer that is not the one the form declared this seat produces."""
    files = answer.get("files")
    if not isinstance(files, dict):
        raise IoRefusalError(
            f"seat {role!r} on arm {mode!r} answered {sorted(map(str, answer))} with no 'files'"
            " mapping; a seat answers in files on either arm",
        )
    declared = {str(name) for name in arm["writes"]}
    produced = {str(name) for name in files}
    if missing := sorted(declared - produced):
        raise IoRefusalError(
            f"seat {role!r} on arm {mode!r} produces {sorted(declared)} and produced"
            f" {sorted(produced)}, omitting {missing}",
        )
    if undeclared := sorted(produced - declared):
        raise IoRefusalError(
            f"seat {role!r} on arm {mode!r} produced {undeclared}, which the form declares no"
            " channel for, so the work would be lost",
        )
    evidence = answer.get("evidence") or {}
    if not isinstance(evidence, dict):
        raise IoRefusalError(
            f"seat {role!r} on arm {mode!r} stated evidence that is not a mapping: {evidence!r}",
        )
    allowed = {str(name) for name in arm["evidence"]}
    if stray := sorted({str(name) for name in evidence} - allowed):
        raise IoRefusalError(
            f"seat {role!r} on arm {mode!r} states evidence {stray} the form declares no home"
            " for; unwritten evidence is legal, undeclared evidence is not",
        )
