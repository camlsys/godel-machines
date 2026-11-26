# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""The `toy` solver: a blind producer whose answer is the genome beside it."""

from __future__ import annotations

from pathlib import Path
from typing import Any

GENOME = "answer.txt"
ANSWER = "probe"


def forward(inputs: dict[str, Any]) -> dict[str, dict[str, bytes]]:
    """Emit this seat's carried answer, blind to the staged question.

    Returns:
        dict[str, dict[str, bytes]]: `files` mapping the produced artifact to its bytes, and an empty.
    """
    del inputs
    return {
        "files": {ANSWER: (Path(__file__).parent / GENOME).read_bytes()},
        "evidence": {},
    }
