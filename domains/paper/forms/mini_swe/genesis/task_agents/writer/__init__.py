# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""The blind writer: one model call from a title and abstract to a whole paper."""

from __future__ import annotations

import json

from shared.gateway.client import GatewayRefused

DRAFT = "paper_text"

INSTRUCTION = """\
The title and abstract above are a seed: a research direction, not a specification. Write a paper on that topic. It need not follow the title and abstract as given, but it must be inspired by the same general area, and it should aim at a real contribution to the field.

You cannot run experiments, so put in the placeholder values and figures the ideal version of this paper would report. State them plainly, with no caveats and no defensive language. The authors are sifting through proposals and want to see the strongest form the paper could take.

None of this is submitted anywhere. It is purely internal, so there is nothing here to hedge against.

Length and structure of a top machine learning venue: introduction, related work as you know it, the method in full technical detail with its notation and, where the method admits it, formal statements and their proofs, the experiments (datasets, baselines, metrics, ablations, compute) with their results in tables and figures described in text, and the discussion. Write in the paper's own voice, as a finished submission. You are strong at mathematics, and there is room to take the paper further in that direction than the title and abstract imply.

Reply with the paper and nothing else, in Markdown with LaTeX for mathematics: no preface, no closing remark. Your whole reply is the draft that goes to review.
"""


REVISE = """Revise the paper above in the light of the review. Where the reviewer found a real weakness, fix it; where you judge it wrong, keep your ground and make the case stronger. Reply with the complete revised paper in Markdown with LaTeX for mathematics, and nothing else."""


#: HOW MANY TIMES THIS SEAT REVISES against its own judge before it files. The search edits this
#: file, so the round count is one of the things it edits. `0` is the one-call seat.
REVIEW_ROUNDS = 1

def dialogue(*, trace: list[dict[str, str]]) -> str:
    """Return the reviewer's whole conversation, every prompt it was sent and every reply it gave."""
    return "\n\n".join(f"#### The reviewer was asked\n{turn['prompt']}\n\n#### The reviewer answered\n{turn['reply']}" for turn in trace)


def revision(*, premise: str, draft: str, score: float, review: str, trace: list[dict[str, str]]) -> str:
    """Return the revision prompt: the premise again, the draft as reviewed, the score, the review, the reviewer's dialogue, the ask."""
    return f"{premise}\n\n### Your paper, as reviewed\n{draft}\n\n### The review\nScore: {score}\n\n{review}\n\n### The reviewer's dialogue\n{dialogue(trace=trace)}\n\n{REVISE}"


def prompt(*, premise: str) -> str:
    """Return the writer's prompt: the staged premise first, then the instruction that refers to it."""
    return f"{premise}\n\n{INSTRUCTION}"


def forward(inputs: dict) -> dict:
    """Draft one paper from the staged premise; refuse an empty premise before the call is paid for."""
    chat = inputs.get("chat")
    if chat is None:
        raise RuntimeError("the paper writer has no gateway grant, so it cannot draft")
    files = inputs["files"]
    premise = "\n\n".join(files[name].decode(errors="replace") for name in sorted(files))
    if not premise.strip():
        sizes = {name: len(files[name]) for name in sorted(files)}
        msg = f"the paper writer was staged {sizes} (bytes per file) and no text came with it, so there is nothing to draft from"
        raise ValueError(msg)
    draft = chat(prompt(premise=premise))
    seat = inputs.get("review")
    reviews: list[str] = []
    for _ in range(REVIEW_ROUNDS if seat is not None else 0):
        try:
            score, review, trace = seat(problem=premise, work=draft, chat=chat)
        except GatewayRefused:
            raise
        except Exception:
            break
        reviews.append(json.dumps({"score": score, "review": review, "dialogue": trace}, indent=2))
        draft = chat(revision(premise=premise, draft=draft, score=score, review=review, trace=trace))
    return {
        "files": {DRAFT: draft.encode()},
        "evidence": {
            "reply": draft.encode(),
            "staged": premise.encode(),
            "review": "\n\n".join(reviews).encode() or b"no review: this node seats no reviewer, or it did not answer",
        },
    }
