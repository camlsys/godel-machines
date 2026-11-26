# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""The deepswe reviewer: a problem statement and a diff in, accept or reject out."""

from __future__ import annotations

import json
from collections.abc import Callable
import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory

from shared import reply as replies
from task_agents.shared import runtime

REVIEWED, PATCH, PROBLEM, VERDICT = "assessed", "model.patch", "problem", "verdict.txt"
PROBLEM_MARK, DIFF_MARK = "### PROBLEM\n", "\n\n### DIFF\n"
AUTHOR_MARK = "\n### TEST OUTPUT THE AUTHOR GATHERED, NOT THE GRADING STANDARD\n"
FIELD = "Decision"
DECISIONS = {"accept": 1.0, "reject": 0.0}
MALFORMED = -1.0
MALFORMED_RETRIES = 2

POSITIONAL = re.compile(r"^(?:input-)?\d+$")
DIFF_HEAD = re.compile(rb"^[ \t]*(diff --git |@@ )|^[ \t]*--- .*\r?\n\+\+\+ ", re.MULTILINE)

INSTRUCTION = """Review the diff above and assess whether it solves the problem the statement sets out.

Work from the statement. It is the specification, and the only one: take the behaviour it asks for -- the inputs, options, calls and cases it names -- and for each, find the code in the diff that delivers it and judge whether that code is correct. Say so in your reasoning as you go.

The author may have written new tests, and they are in the diff. They are separate from the true golden tests and show the author's intent, not the standard. They tell you what the author believed the problem was, which is useful, and a change can satisfy them and still be wrong. Weigh them as evidence of intent and judge the code against the statement.

Accept when the diff correctly tackles the problem and provides the solution the statement asks for. Reject when some behaviour it asks for has no code behind it, or has code you can show is wrong. Nothing else decides it: not style, naming, structure, verbosity, unrelated churn, missing documentation, or handling for cases the statement never raises. Neither does residual doubt, since every diff leaves some, including the correct ones. Reject only for something you named and can show the diff gets wrong.

Reply with one JSON block and nothing after it:
{
  "Reasoning": "...",
  "Decision": "Accept" | "Reject"
}
"Decision" is exactly "Accept" or "Reject"."""


def staged(*, problem: str, diff: str) -> str:
    """Return the one channel a solver's work reaches this seat on."""
    return PROBLEM_MARK + problem + DIFF_MARK + diff


def prompt(*, channel: str) -> str:
    """Return the complete prompt: the work first, then the instruction that names it."""
    return channel + "\n\n" + INSTRUCTION


def _is_work(name: str) -> bool:  # a host test widens it by position
    """Say whether one staged name is the work under review."""
    leaf = name.rsplit("/", 1)[-1]
    return leaf in {REVIEWED, PATCH, PROBLEM} or bool(POSITIONAL.match(leaf))


def channel(*, files: dict, author_output: bytes = b"") -> str:
    """Compose the body the model reads, refusing anything staged that is not the work."""
    if not files:
        raise KeyError("the deepswe reviewer was staged nothing, so it has nothing to review")
    if unknown := sorted(n for n in files if not _is_work(n)):
        raise KeyError(
            f"the deepswe reviewer was staged {unknown}, "
            "which is neither the work under review nor a declared input",
        )
    work = [n for n in sorted(files) if _is_work(n)]
    if not work:
        raise KeyError(f"the deepswe reviewer was staged {sorted(files)}, none of which is work under review")
    # The separate patch is for code that inspects files; the envelope already contains it.
    if any(n.rsplit("/", 1)[-1] == REVIEWED or POSITIONAL.fullmatch(n.rsplit("/", 1)[-1]) for n in work):
        work = [n for n in work if n.rsplit("/", 1)[-1] != PATCH]
    if DIFF_HEAD.search(author_output):
        raise KeyError("the deepswe reviewer was handed a patch as author output; that slot is for what the author's tests printed")
    body = "\n\n".join(files[n].decode(errors="replace") for n in work)
    if printed := author_output.decode(errors="replace").strip():
        body += "\n" + AUTHOR_MARK + printed
    return body


def verdict(*, reply: str) -> float:
    """Return the one decision the reply states, or MALFORMED when it states none or several."""
    words = {str(word).strip().lower() for word in replies.field(reply=reply, name=FIELD)}
    stated = {DECISIONS[word] for word in words if word in DECISIONS}
    return stated.pop() if len(stated) == 1 else MALFORMED


def judged(*, reviewed: str, chat: Callable[[str], object]) -> dict:
    """Ask the model for a verdict on one composed body, asking again while the reply is unreadable."""
    asked = prompt(channel=reviewed)
    attempts, score = 0, MALFORMED
    while score == MALFORMED and attempts <= MALFORMED_RETRIES:
        reply = str(chat(asked))
        attempts, score = attempts + 1, verdict(reply=reply)
    parsed = {"field": FIELD, "score": score, "malformed": score == MALFORMED, "attempts": attempts}
    return {
        "files": {VERDICT: json.dumps({"score": score}, sort_keys=True).encode()},
        "evidence": {
            "reply": reply.encode(),
            "reviewed": reviewed.encode(),
            "parsed": json.dumps(parsed, indent=2, sort_keys=True).encode(),
        },
    }


def forward(inputs: dict) -> dict:  # seat ABI, host calls it positionally
    """Review what the host staged and answer in the declared files."""
    chat = inputs.get("chat")
    if chat is None:
        raise RuntimeError("the deepswe reviewer has no gateway grant, so it cannot judge")
    return judged(reviewed=channel(files=inputs["files"]), chat=chat)


def _traced(*, chat: Callable[[str], object]) -> tuple[object, list[dict[str, str]]]:
    """Return a chat that records every exchange, and the list it records into."""
    trace: list[dict[str, str]] = []

    def asked(prompt: str) -> str:
        reply = str(chat(prompt))
        trace.append({"prompt": prompt, "reply": reply})
        return reply

    return asked, trace


def _co_located(*, problem: str, diff: str, chat: Callable[[str], object], author_output: bytes = b"") -> dict:
    """Run editable forward with temporary input files and the solver's current working tree."""
    reviewed = channel(files={REVIEWED: staged(problem=problem, diff=diff).encode()}, author_output=author_output)
    files = {REVIEWED: reviewed.encode(), PATCH: diff.encode()}
    with TemporaryDirectory(prefix="rqgm-review-") as scratch:
        paths = {name: str(Path(scratch).absolute() / name) for name in files}
        for name, data in files.items():
            Path(paths[name]).write_bytes(data)
        return forward({
            "files": files,
            "paths": paths,
            "repository": os.environ.get(runtime.CHECKOUT),
            "solve": runtime.solve_capability(),
            "chat": chat,
        })


def review(*, problem: str, work: str, chat: Callable[[str], object]) -> tuple[float, str, list[dict[str, str]]]:
    """Return the editable seat's verdict, reply evidence, and traced chat calls for solver revision."""
    asked, trace = _traced(chat=chat)
    answered = _co_located(problem=problem, diff=work, chat=asked)
    # Reply evidence is optional in the forward contract; the verdict file is required.
    reply = (answered.get("evidence") or {}).get("reply", b"")
    return float(json.loads(answered["files"][VERDICT])["score"]), reply.decode(errors="replace"), trace


def self_review(*, problem: str, diff: str, author_output: bytes, chat: Callable[[str], object]) -> float:
    """Return this seat's verdict on a diff its author is still working on."""
    if not author_output.strip():
        raise ValueError("a self-review needs the author's own test output, and none was given")
    answered = _co_located(problem=problem, diff=diff, author_output=author_output, chat=chat)
    return float(json.loads(answered["files"][VERDICT])["score"])
