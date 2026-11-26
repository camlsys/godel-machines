# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""Propose this node's successor genomes, then run the meta agent over them."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

GRID = 100

GIT_TIMEOUT_SECONDS = 60.0


def record() -> bytes:
    """Return the canonical bytes of what this lineage is, read from git.

    Raises:
        RuntimeError: The chain could not be read, for any reason.
    """
    argv = ("git", "rev-list", "--reverse", "HEAD")
    try:
        done = subprocess.run(
            argv,
            capture_output=True,
            check=False,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"{' '.join(argv)}: could not run git: {exc}") from exc
    if done.returncode != 0:
        detail = done.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"{' '.join(argv)}: exited {done.returncode}: {detail}")
    if not done.stdout:
        raise RuntimeError(f"{' '.join(argv)}: no commit, so no lineage to read")
    return done.stdout


def _step(*, carried: str, evidence: bytes) -> int:
    """Return the never-zero step one parent takes under one staged record."""
    drawn = hashlib.sha256(evidence + b"|" + carried.encode()).digest()
    return 1 + int.from_bytes(drawn[:8]) % GRID


def proposal(*, carried: str, evidence: bytes) -> str:
    """Return the successor threshold for one parent under one staged record."""
    stepped = round(float(carried) * GRID) + _step(carried=carried, evidence=evidence)
    return f"{stepped % (GRID + 1) / GRID:.2f}"


def answer(*, carried: str, evidence: bytes) -> str:
    """Return the successor answer, spelled as the whole number the solver emits."""
    return f"{(int(carried) + _step(carried=carried, evidence=evidence)) % (GRID + 1)}"


LADDERS = {"strictness.txt": proposal, "answer.txt": answer}


def main() -> int:
    """Write this node's proposals, then hand the episode to the meta agent.

    Returns:
        int: The meta agent's exit status, or 1 when the episode named no genome or named one this file.
    """
    named = sys.argv[1:]
    if not named:
        sys.stderr.write(f"usage: python -B meta/expand.py <{'|'.join(LADDERS)}>...\n")
        return 1
    action = json.loads(Path(".rqgm/job.json").read_text(encoding="utf-8"))["action"].encode()
    for path in named:
        genome = Path(path)
        ladder = LADDERS.get(genome.name)
        if ladder is None:
            sys.stderr.write(f"no ladder for the genome {path!r}; this form walks {[*LADDERS]}\n")
            return 1
        genome.write_text(
            ladder(carried=genome.read_text(encoding="utf-8"), evidence=action),
            encoding="utf-8",
        )
    import entry

    return entry.main()


if __name__ == "__main__":
    sys.exit(main())
