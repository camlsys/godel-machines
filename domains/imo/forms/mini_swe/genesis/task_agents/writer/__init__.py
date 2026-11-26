# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""The proof writer: one call from a problem to a proof, revised once on the node's own grader."""

from __future__ import annotations

import json

PROBLEM = "problem"
ATTEMPT = "marking"

INSTRUCTION = """Prove the problem above. You are writing for an IMO coordinator who awards 0 to 7 against an official marking scheme, so only a complete and rigorous argument pays in full: a correct answer with a gap earns little and a confident sketch earns nothing.

Settle the whole argument before you write. Decide what has to be shown, find the idea that closes it, and check the hard step against a small case.

Then write the proof in order: state each claim before you use it, justify every step, cover every case, and end with the conclusion the problem asks for. Use LaTeX for the mathematics, as the problem does. Do not restate the problem, do not add a preamble or a closing remark, and write nothing but the proof: your reply is filed as the solution exactly as it stands."""

REVISE = """Revise your proof above in the light of the review. The reviewer read your proof without the official solution, and its mark does not decide your score. Where it found a real gap, close it; where you judge it wrong, keep your argument and make the step it doubted explicit. Return the complete revised proof in LaTeX, and nothing else."""


#: HOW MANY TIMES THIS SEAT REVISES against its own judge before it files. The search edits this
#: file, so the round count is one of the things it edits. `0` is the one-call seat.
REVIEW_ROUNDS = 1

def prompt(*, problem: str) -> str:
    """Return the first prompt: the problem, then the instruction that names it."""
    return f"{problem}\n\n{INSTRUCTION}"


def dialogue(*, trace: list[dict[str, str]]) -> str:
    """Return the reviewer's whole conversation, every prompt it was sent and every reply it gave."""
    return "\n\n".join(f"#### The reviewer was asked\n{turn['prompt']}\n\n#### The reviewer answered\n{turn['reply']}" for turn in trace)


def revision(*, problem: str, proof: str, mark: float, reasoning: str, trace: list[dict[str, str]]) -> str:
    """Return the revision prompt: the problem again, the proof as reviewed, the mark, the review, the grader's dialogue, the ask."""
    return f"{problem}\n\n### Your proof, as reviewed\n{proof}\n\n### The review\nMark awarded: {mark}\n\n{reasoning}\n\n### The grader's dialogue\n{dialogue(trace=trace)}\n\n{REVISE}"


def forward(inputs: dict) -> dict:
    """Write the proof, put it in front of the node's grader, and file the revision."""
    chat = inputs.get("chat")
    if chat is None:
        raise RuntimeError("the imo writer has no gateway grant, so it cannot prove")
    files = inputs["files"]
    problem = "\n\n".join(files[name].decode(errors="replace") for name in sorted(files))
    proof = chat(prompt(problem=problem))
    seat = inputs.get("review")
    reviews: list[str] = []
    for _ in range(REVIEW_ROUNDS if seat is not None else 0):
        try:
            mark, reasoning, trace = seat(problem=problem, work=proof, chat=chat)
        except Exception:
            break
        reviews.append(json.dumps({"score": mark, "review": reasoning, "dialogue": trace}, indent=2))
        proof = chat(revision(problem=problem, proof=proof, mark=mark, reasoning=reasoning, trace=trace))
    return {
        "files": {ATTEMPT: proof.encode()},
        "evidence": {
            "reply": proof.encode(),
            "staged": problem.encode(),
            "review": "\n\n".join(reviews).encode() or b"no review: this node seats no reviewer, or it did not answer",
        },
    }
