# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""Run mini-swe tool calls under Bash, with bounded, truthful observations."""

from __future__ import annotations

import hashlib
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig
from minisweagent.exceptions import FormatError, Submitted
from minisweagent.models.utils import actions_toolcall

from shared.gateway.prompts import carried, filled
from task_agents.shared import io_contract

OBSERVATION = carried(name="observation_format.txt")
OBSERVATION_MAX_CHARACTERS = 32_000
_CAUSE_MAX_CHARACTERS = 400
ELISION_BUDGET = 512
#: Where a clipped command's whole output lands, for the agent to read past the elision.
#: `spill`, never `observations`: a name the meta agent sees as observations reads as evidence.
SPILL_DIR = "spill"
PROMPT_KEY = "prompt.json"
PROMPT_VERSION = 1
SILENCE = "[no output: the command wrote nothing to stdout or stderr]\n"
UNOBSERVED = carried(name="unobserved.txt")
PARTIAL = carried(name="partial_output.txt")
BANNER_OPEN = "==> "
BANNER_CLOSE = " <=="
SHELL = "bash"
SUBMISSION = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
SELF_REVIEW = "self_review"
SELF_REVIEW_TOOL = {
    "type": "function",
    "function": {
        "name": SELF_REVIEW,
        "description": "Ask this episode's carried reviewer about the committed patch and your test output. Feedback only; charges this episode's model allowance.",
        "parameters": {
            "type": "object",
            "properties": {"evidence": {"type": "string", "description": "Path to your saved test output, relative to the task checkout or absolute."}},
            "required": ["evidence"],
            "additionalProperties": False,
        },
    },
}
SUBMISSION_MISSED = filled(name="submission_missed.txt", fields={"SUBMISSION": SUBMISSION})
HIDDEN_STAGE = "[rqgm] this command reports 0, but a stage inside it did not:"
SUBMISSION_HID_A_FAILURE = filled(name="submission_hid_a_failure.txt", fields={"SUBMISSION": SUBMISSION})

# Every command runs between these: the trap notes a failing stage, and the epilogue reports the
# first non-zero pipeline stage (141, a closed pipe, excepted) where the last stage returned 0.
PROLOGUE = """\
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
set -o errtrace
__rqgm_hidden=''
__rqgm_note() {
  [ "$1" = 141 ] || __rqgm_hidden="${__rqgm_hidden}
  exit $1 from: $2"
}
trap '__rqgm_note "$?" "$BASH_COMMAND"' ERR"""

EPILOGUE = """\
__rqgm_rc=$? __rqgm_pipe="${PIPESTATUS[*]}"
trap - ERR
__rqgm_stifled=0
for __rqgm_stage in $__rqgm_pipe; do
  case "$__rqgm_stage" in
    0 | 141) ;;
    *) __rqgm_stifled="$__rqgm_stage" ; break ;;
  esac
done
if [ "$__rqgm_rc" = 0 ] && [ "$__rqgm_stifled" != 0 ]; then
  printf '[rqgm] the last pipeline exited [%s], left to right, so this command reports %s rather than the 0 its last stage returned.\\n' "$__rqgm_pipe" "$__rqgm_stifled" >&2
  __rqgm_rc="$__rqgm_stifled"
fi
[ "$__rqgm_rc" = 0 ] && [ -n "$__rqgm_hidden" ] &&
  printf '[rqgm] this command reports 0, but a stage inside it did not:%s\\n' "$__rqgm_hidden" >&2
exit "$__rqgm_rc\""""


def parse_actions(*, calls: tuple[tuple[str, str, str], ...], review: bool, format_error: str) -> list[dict]:
    """Extend the pinned upstream bash parser with one content-only review action."""
    actions = []
    if not calls:
        return actions_toolcall.parse_toolcall_actions([], format_error_template=format_error)
    for identifier, name, arguments in calls:
        if name == SELF_REVIEW and review:
            try:
                fields = json.loads(arguments)
            except ValueError as refused:
                raise FormatError({"role": "user", "content": f"{SELF_REVIEW}: {refused}",
                                   "extra": {"interrupt_type": "FormatError"}}) from refused
            if (not isinstance(fields, dict) or set(fields) != {"evidence"}
                    or not isinstance(fields["evidence"], str) or not fields["evidence"].strip()):
                raise FormatError({"role": "user",
                                   "content": f"{SELF_REVIEW}: expected exactly one nonempty evidence path",
                                   "extra": {"interrupt_type": "FormatError"}})
            actions.append({"tool": SELF_REVIEW, "evidence": fields["evidence"], "tool_call_id": identifier})
        else:
            call = SimpleNamespace(id=identifier, function=SimpleNamespace(name=name, arguments=arguments))
            actions.extend(actions_toolcall.parse_toolcall_actions([call], format_error_template=format_error))
    return actions


def spill_root() -> Path:
    """Return the directory a clipped output is written whole to, this episode."""
    return io_contract.own_experience_path() / SPILL_DIR


def record_prompt(*, system: str, instance: str) -> Path | None:
    """File this episode's prompt as constructed, before any model has been asked anything."""
    path = io_contract.own_experience_path() / PROMPT_KEY
    record = {"version": PROMPT_VERSION, "system": system, "instance": instance}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    except OSError:
        return None
    return path


def _named(*, text: str) -> list[str]:
    edges = len(BANNER_OPEN) + len(BANNER_CLOSE)
    return [
        line[len(BANNER_OPEN) : -len(BANNER_CLOSE)]
        for line in text.splitlines()
        if line.startswith(BANNER_OPEN) and line.endswith(BANNER_CLOSE) and len(line) > edges
    ]


def _swallowed(*, head: str, cut: str, tail: str) -> tuple[str | None, list[str]]:
    inside = _named(text=cut)
    return ([None, *_named(text=head)])[-1], inside if _named(text=tail) else inside[:-1]


def _elision(*, text: str, head: str, tail: str, spill: str | None) -> str:
    lines, first = text.count("\n") + 1, head.count("\n") + 1
    last = lines - tail.count("\n")
    cut = text[len(head) : len(text) - len(tail)]
    said = [f"... elided lines {first:,}-{last:,} of {lines:,} ({len(cut):,} of {len(text):,} characters) ..."]
    if spill is not None:
        said.append(f"... this output whole is at {spill}; `sed -n '{first},{last}p' {spill}` ...")
    opened, swallowed = _swallowed(head=head, cut=cut, tail=tail)
    if opened is not None:
        said.append(f"... the cut opens inside {opened} ...")
    if swallowed:
        said.append(f"... {len(swallowed):,} file(s) removed whole: {', '.join(swallowed)} ...")
    while len("\n".join(said)) + 2 > ELISION_BUDGET and len(said) > 1:
        said.pop()
    return ("\n" + "\n".join(said) + "\n")[:ELISION_BUDGET]


def _spill(*, text: str) -> str | None:
    payload = text.encode("utf-8", "surrogatepass")
    path = spill_root() / f"{hashlib.sha256(payload).hexdigest()[:16]}.txt"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    except OSError:
        return None
    # Commands can run in a task checkout while observations are filed in the node checkout.
    return str(path.resolve())


def _cut_by(*, output: dict) -> str:
    said = str(output.get("exception_info") or "").strip()
    return f" The environment reported: {said[:_CAUSE_MAX_CHARACTERS]}." if said else ""


def _bounded(*, output: dict) -> dict:
    text = output.get("output") or ""
    if output.get("exception_info"):
        text = f"{(PARTIAL if text else UNOBSERVED).rstrip(chr(10))}{_cut_by(output=output)}\n{text}"
    elif not text:
        text = SILENCE
    if len(text) <= OBSERVATION_MAX_CHARACTERS:
        return {**output, "output": text}
    half = (OBSERVATION_MAX_CHARACTERS - ELISION_BUDGET) // 2
    head, tail = text[:half], text[len(text) - half :]
    marker = _elision(text=text, head=head, tail=tail, spill=_spill(text=text))
    return {**output, "output": f"{head}{marker}{tail}"}


def _interpreter() -> str:
    found = shutil.which(SHELL)
    if found is None:
        raise RuntimeError(f"this loop offers a `{SHELL}` tool but `{SHELL}` is absent from PATH; refusing beats running another shell")
    return found


def _under(*, shell: str, command: str) -> str:
    return f"exec {shlex.quote(shell)} -c {shlex.quote(f'{PROLOGUE}\n{command}\n{EPILOGUE}')}"


def _gate_refusal() -> str | None:
    gate = Path("meta/gate.py")
    if not gate.is_file():
        return None
    try:
        argv = [sys.executable, str(gate)]
        ran = subprocess.run(argv, capture_output=True, text=True, cwd=Path.cwd(), check=False)
    except OSError as failure:
        sys.stderr.write(f"[mini-swe] the gate could not be run: {failure}\n")
        return None
    return None if ran.returncode == 0 else f"\n\n[harness] This episode did NOT submit. {ran.stdout.strip()}\n"


class _WallConfig(LocalEnvironmentConfig):
    """Upstream's settings with `timeout` widened to `None`, which is `communicate`'s own no-wall."""

    timeout: int | None = None


class BashEnvironment(LocalEnvironment):
    """Run tool calls under Bash, within the command wall and the episode end where declared."""

    def __init__(self, ends: float | None = None, *, timeout: int | None, **kwargs: object) -> None:
        """Bind the shell, the per-command wall (`None` for none), and the episode end."""
        super().__init__(config_class=_WallConfig, timeout=timeout, **kwargs)
        self.shell = _interpreter()
        self.ends = ends
        self.ceiling = self.config.timeout

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
        """Execute one action under Bash with no more than the episode time left."""
        command = _under(shell=self.shell, command=str(action.get("command", "")))
        if self.ends is not None:
            left = int(self.ends - time.monotonic())
            asked = timeout if timeout is not None else self.ceiling
            timeout = max(1, left if asked is None else min(asked, left))
        return super().execute({**action, "command": command}, cwd, timeout=timeout)

    def _check_finished(self, output: dict) -> None:
        text = str(output.get("output") or "")
        try:
            super()._check_finished(output)
        except Submitted:
            refused = SUBMISSION_HID_A_FAILURE if HIDDEN_STAGE in text else _gate_refusal()
            if refused is None:
                raise
            output["output"] = text + refused
            return
        if SUBMISSION in text:
            output["output"] = text + SUBMISSION_MISSED
