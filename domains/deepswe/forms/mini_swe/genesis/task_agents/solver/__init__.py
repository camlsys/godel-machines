# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""The deepswe solver: one corpus cell in, one patch out, through the shared mini-swe harness."""

from __future__ import annotations

import json
from pathlib import Path

from shared.gateway.client import GatewayRefused
from task_agents.shared import io_contract

ASSESSED, PATCH = "assessed", "model.patch"
REVISE_MARK = "\n\n### REVIEW OF YOUR PATCH SO FAR\n"
REVISE = "\n\nThe reviewer above rejected the patch as it stands. Revise it in the same working directory: address every case it found missing, keep what it accepted, and leave the tree holding the complete patch."
PROBLEM_MARK, DIFF_MARK = "### PROBLEM\n", "\n\n### DIFF\n"

#: The corpus field holding the benchmark's own task statement. A name, not a prompt.
INSTRUCTION = "instruction"

#: WHAT THIS SEAT IS TOLD BEFORE THE TASK, and the search edits it like any other line here.
#: Deliberately minimal: the statement below it is the benchmark's, and every word added here is
#: a word this domain chose rather than one DeepSWE wrote.
PREAMBLE = "You are a coding agent. Solve the following task."

CONTEXT = (("repository", "### REPOSITORY\n"), ("language", "### LANGUAGE\n"), ("path", "### WORKING DIRECTORY\n"))

TRAIL = (("observations", Path("observations") / "observations.jsonl"), ("scratch", Path("evidence") / "scratch.md"))


#: HOW MANY TIMES THIS SEAT REVISES against its own judge before it files. The search edits this
#: file, so the round count is one of the things it edits. `0` is the one-call seat.
REVIEW_ROUNDS = 1

def dialogue(*, trace: list[dict[str, str]]) -> str:
    """Return the reviewer's whole conversation, every prompt it was sent and every reply it gave."""
    return "\n\n".join(f"#### The reviewer was asked\n{turn['prompt']}\n\n#### The reviewer answered\n{turn['reply']}" for turn in trace)


def _row(*, cell: str) -> dict[str, str]:
    """Return the staged cell as this domain's row, refusing anything else by name."""
    try:
        row = json.loads(cell)
    except ValueError as unreadable:
        raise TypeError(f"the staged task opens {cell[:80]!r} and does not parse as JSON, so it is not this domain's cell") from unreadable
    if not isinstance(row, dict):
        raise TypeError(f"the staged task parses as {type(row).__name__}, not the object the corpus builder writes")
    for field in (INSTRUCTION, *(name for name, _ in CONTEXT)):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError(f"the staged cell states no {field!r}; it carries {sorted(row)}")
    return row


def _asked(*, row: dict[str, str]) -> str:
    """Return the task as the episode is asked it: the preamble, the statement, then its context under headings."""
    return "\n\n".join([PREAMBLE.strip(), row[INSTRUCTION].strip(), *(mark + row[field].strip() for field, mark in CONTEXT)])


def _trail() -> dict[str, bytes]:
    """Return the harness's own records from this generation's folder, where it wrote them."""
    own = io_contract.own_experience_path()
    return {name: (own / leaf).read_bytes() for name, leaf in TRAIL if (own / leaf).is_file()}


def forward(inputs: dict) -> dict:  # seat ABI, host calls it positionally
    """Solve the staged task and publish the patch beside the envelope the reviewer reads."""
    if inputs.get("chat") is None:
        raise RuntimeError("the solver has no gateway grant, so it cannot run an episode")
    if (solve := inputs.get("solve")) is None:
        raise RuntimeError("the solver has no solve capability, so it cannot run an episode")
    files = inputs["files"]
    if all(payload == b"" for payload in files.values()):
        raise KeyError(f"the deepswe solver was staged {sorted(files)}, which carries no bytes, so it has no problem to solve")
    row = _row(cell="\n\n".join(files[name].decode(errors="replace") for name in sorted(files)))
    asked = _asked(row=row)
    episode = solve(asked)
    seat = inputs.get("review")
    reviews: list[str] = []
    for _ in range(REVIEW_ROUNDS if seat is not None else 0):
        try:
            score, review, trace = seat(problem=row[INSTRUCTION], work=episode["patch"], chat=inputs["chat"])
        except GatewayRefused:
            raise
        except Exception:  # a broken reviewer costs the revision, never the patch
            break
        reviews.append(json.dumps({"score": score, "review": review, "dialogue": trace}, indent=2))
        if score == 1.0:
            break
        episode = solve(asked + REVISE_MARK + f"Score: {score}\n\n{review}\n\n### The reviewer's dialogue\n" + dialogue(trace=trace) + REVISE)
    return {
        "files": {
            ASSESSED: (PROBLEM_MARK + row[INSTRUCTION] + DIFF_MARK + episode["patch"]).encode(),
            PATCH: episode["patch"].encode(),
        },
        "evidence": {
            "staged": asked.encode(),
            "messages": json.dumps(episode["messages"], indent=2, default=str).encode(),
            "stop": json.dumps({key: episode[key] for key in ("exit_status", "turns", "tokens")}, indent=2).encode(),
            "review": "\n\n".join(reviews).encode() or b"no review: this node seats no reviewer, or it did not answer",
            **_trail(),
        },
    }
