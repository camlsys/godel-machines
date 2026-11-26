# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""Own the names the harness reserves inside a candidate's checkout."""

from __future__ import annotations

RESERVED = ".rqgm"

RESERVED_SIBLING = f"{RESERVED}-"

EXPERIENCE = "experience"
GENERATIONS = "generations"

SCRATCH = "evidence"

_SCRATCH_DEPTH = 5


def reserved(*, name: str) -> bool:
    """Say whether a path is the harness's own bookkeeping rather than the candidate's work."""
    if name.startswith((f"{RESERVED}/", RESERVED_SIBLING)):
        return True
    parts = name.split("/")
    return (
        len(parts) >= _SCRATCH_DEPTH
        and parts[0] == EXPERIENCE
        and parts[1] == GENERATIONS
        and parts[3] == SCRATCH
    )
