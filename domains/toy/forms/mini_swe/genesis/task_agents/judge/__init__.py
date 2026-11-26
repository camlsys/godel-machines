# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""The `toy` judge: the `strictness-v1` law, as a real agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CEILING = 1.0
DIVISOR = 5
INCREMENT = 10.0
RULE = "strictness-v1"

PROBE = "probe"
VERDICT = "verdict.txt"
GENOME = "strictness.txt"


def verdict(strictness: str, answer: str) -> float:
    """Return the raw judge score for one strictness against one answer."""
    return min(CEILING, float(strictness) + (int(answer) % DIVISOR) / INCREMENT)


def forward(inputs: dict[str, Any]) -> dict[str, dict[str, bytes]]:
    """Judge the staged answer under this agent's own carried strictness.

    Returns:
        dict[str, dict[str, bytes]]: `files`, carrying this seat's verdict, and an empty `evidence`.
    """
    strictness = (Path(__file__).parent / GENOME).read_text(encoding="utf-8")
    raw = verdict(strictness, inputs["files"][PROBE].decode())
    return {
        "files": {VERDICT: json.dumps({"score": raw, "outcome": raw}).encode()},
        "evidence": {},
    }


def self_review(*, problem: str, diff: str, author_output: bytes, chat: object) -> float:
    """Return this seat's verdict on the author's stated answer, read exactly as `forward` reads one."""
    strictness = (Path(__file__).parent / GENOME).read_text(encoding="utf-8")
    return verdict(strictness, author_output.decode(errors="replace").strip() or "0")
