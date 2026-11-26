# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""The grading seat: one call from a marking text to an integer mark on the IMO scale."""

from __future__ import annotations

import json
from collections.abc import Callable

from shared import reply as replies

MARKING = "marking"
GUIDANCE = "guidance.txt"
GRADE = "grade.txt"
RESPONSE_HEADING = "### Response"
FIELD = "points"
LOW, HIGH = 0, 7
MALFORMED = -1
DRAFTED = "### Draft proof under review"

INSTRUCTION = """You are grading the work above as an IMO coordinator would, against the official marking scheme where one is given. Read the whole argument before you decide anything, then check every step the author leans on: a claim asserted without proof, a case left out, a lemma quoted wrongly, an inequality applied the wrong way round, an object used before it is shown to exist. A confident tone and a correct final answer earn nothing on their own; the argument earns the mark.

Award an integer from 0 to 7. 7 is a complete and rigorous proof. 5 or 6 is a proof with a small gap or slip the author could clearly have closed. 1 to 4 is genuine partial progress, a lemma proved, a case settled, a reduction made, in proportion to how much of the problem it covers. 0 is no meaningful progress, however long the text. Where the marking scheme says what a partial step is worth, follow the scheme.

Write your reasoning first, in plain prose with LaTeX wherever you need it; nothing in the reasoning is parsed. Say what is proved, what is missing, and what the mark rests on. Then end your reply with the mark in one fenced JSON block and write nothing after it:

```json
{"points": 0}
```

Give exactly one such block, holding one integer from 0 to 7. A reply with no block, with two different marks, or with anything else in the block is recorded as no mark at all."""


def judged(*, guidance: bytes, attempt: bytes) -> str:
    """Return the marking text: the guidance's three sections, then the attempt as the response."""
    return f"{guidance.decode(errors='replace')}\n\n{RESPONSE_HEADING}\n\n{attempt.decode(errors='replace')}"


def prompt(*, work: str) -> str:
    """Return the grading prompt, work first because the instruction says "the work above"."""
    return f"{work}\n\n{INSTRUCTION}"


def mark(*, reply: str) -> int:
    """Return the one mark the reply states, or `MALFORMED` for none, several, or one off the scale."""
    stated = {value for value in replies.field(reply=reply, name=FIELD) if type(value) is int and LOW <= value <= HIGH}
    return stated.pop() if len(stated) == 1 else MALFORMED


def forward(inputs: dict) -> dict:
    """Mark the staged work and write the grade as `{"score": mark}`."""
    chat = inputs.get("chat")
    if chat is None:
        raise RuntimeError("the imo grader has no gateway grant, so it cannot grade")
    files = inputs["files"]
    if MARKING not in files:
        raise KeyError(f"the imo grader was staged {sorted(files)}, which carries no {MARKING!r} to grade")
    work = judged(guidance=files[GUIDANCE], attempt=files[MARKING]) if GUIDANCE in files else files[MARKING].decode(errors="replace")
    reply = chat(prompt(work=work))
    score = mark(reply=reply)
    return {
        "files": {GRADE: json.dumps({"score": score}, sort_keys=True).encode()},
        "evidence": {
            "reply": reply.encode(),
            "parsed": json.dumps({"field": FIELD, "score": score, "malformed": score == MALFORMED}, indent=2, sort_keys=True).encode(),
        },
    }


def self_review(*, problem: str, diff: str, author_output: bytes, chat: Callable[[str], object]) -> float:
    """Return this seat's mark for a proof its author is still writing, read exactly as `forward` reads one."""
    said = author_output.decode(errors="replace").strip()
    work = "\n\n".join(part for part in (problem.strip(), DRAFTED, diff, said) if part)
    answered = forward({"files": {MARKING: work.encode()}, "chat": chat})
    return float(json.loads(answered["files"][GRADE])["score"])


def _traced(*, chat: Callable[[str], object]) -> tuple[object, list[dict[str, str]]]:
    """Return a chat that records every exchange, and the list it records into."""
    trace: list[dict[str, str]] = []

    def asked(prompt: str) -> str:
        reply = str(chat(prompt))
        trace.append({"prompt": prompt, "reply": reply})
        return reply

    return asked, trace


def review(*, problem: str, work: str, chat: Callable[[str], object]) -> tuple[float, str, list[dict[str, str]]]:
    """Return this seat's mark for a draft proof, the reasoning behind it, and every exchange it made, for the writer to revise on."""
    asked, trace = _traced(chat=chat)
    answered = forward({"files": {MARKING: f"{problem.strip()}\n\n{DRAFTED}\n\n{work}".encode()}, "chat": asked})
    return float(json.loads(answered["files"][GRADE])["score"]), answered["evidence"]["reply"].decode(errors="replace"), trace
