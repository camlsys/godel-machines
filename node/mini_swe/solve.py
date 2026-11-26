# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""Adapt the shared gateway to upstream mini-swe-agent, and run one solving episode."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path.cwd()))

from execution import (
    OBSERVATION,
    OBSERVATION_MAX_CHARACTERS,
    SELF_REVIEW,
    SELF_REVIEW_TOOL,
    SHELL,
    SUBMISSION,
    BashEnvironment,
    _bounded,
    parse_actions,
    record_prompt,
    spill_root,
)
from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import FormatError, InterruptAgentFlow, Submitted
from minisweagent.models.utils import actions_toolcall
from shared.self_review import CHECKOUT_VAR, TREE_VAR, feedback, proxy_clause, reviewing
from shared.self_review import main as _self_review_main

from shared import reserved_paths
from shared.gateway.budget import Budget
from shared.gateway.client import GatewayRefused
from shared.gateway.compaction import ContextOverflowError, Summarizer, pin
from shared.gateway.entrypoint import client_from_env, declared_budget
from shared.gateway.estimate import Estimator
from shared.gateway.prompts import carried, filled
from shared.gateway.refusal import Ladder, Rung, guardrail_refusal
from shared.gateway.window import Window
from task_agents.shared import io_contract

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from shared.gateway.client import Client, Reply

TOOLS: Sequence[Mapping[str, object]] = (actions_toolcall.BASH_TOOL,)
FORMAT_ERROR = carried(name="format_error.txt")
STEP_LIMIT = 250
REASONING_BLOCK = "<reasoning>\n{text}\n</reasoning>\n\n"
POSTED_FIELDS = ("tool_calls", "tool_call_id")
_PINNED = 2
CONTEXT_EXIT = "ContextExceeded"
GUARDRAIL_EXIT = "GuardrailTripped"
LADDER_SAID = {
    Rung.COMPACT: "compacting eagerly and asking the same turn again",
    Rung.CLEAR: "restarting from the starting prompt against the tree as it now stands; every edit this episode made stays on disk",
    Rung.PUBLISH: "ending the episode so the work already on disk publishes",
}
WORKSPACE_NAME = "."
BASELINE_VAR = "RQGM_EPISODE_BASELINE"
CARRIED_VAR = "RQGM_EPISODE_CARRIED"
TASK_VAR = "RQGM_EPISODE_TASK"
# This serialization stays in native review state rather than each command's environment.
CARRIED_SEPARATOR = "\n"
LOOP = carried(name="act_loop.txt")
REVIEW_LOOP = (
    f"HOW YOU ACT. Call `bash` with a `command` to read, edit and test, or `{SELF_REVIEW}`"
    " with an `evidence` path to consult your carried reviewer. Review runs in the native"
    " task process. You receive each tool's result before acting again."
)
SOLVE_PROMPT = filled(name="solve_prompt.txt", fields={"LOOP": LOOP, "SUBMISSION": SUBMISSION})
SOLVE_INSTANCE = filled(name="solve_instance.txt", fields={"SUBMISSION": SUBMISSION})
EXPANSION_INSTANCE = filled(name="expansion_instance.txt", fields={"SUBMISSION": SUBMISSION})
REVIEW_MARK = "{REVIEW}"
REVIEW_CLAUSE = """

ASK YOUR REVIEWER BEFORE YOU SUBMIT, AND BY DEFAULT. This node carries a review seat. Run the tests you care about, save what they PRINTED to a file, commit the patch you want reviewed, then call the {TOOL} tool with:

    {{"evidence": "<the-file-holding-your-test-output>"}}

This is a native tool call. Shell commands have no model access. The evidence path is relative to your task checkout, or absolute. The tool reads the committed diff from the episode's original baseline to HEAD, just as the harness captures your submission; uncommitted edits are excluded. You name only your test output: the diff, baseline, task and review seat come from the episode. Its observation contains the score, diff byte count and reviewer's reply. Read the reasoning; the number alone tells you little.

1.0 is accept, 0.0 is reject, -1.0 means the seat could not read its own answer. It judges on the problem statement and your diff, and it holds no graded tests and no reference solution -- it knows what you know and nothing more. It is a heuristic, not ground truth: read a reject as a reason to look at your diff again, and do not rewrite working code to chase the number.

It costs a model call, so ask about a candidate worth checking rather than about every edit. It is not the reviewer that grades you, and nothing it prints reaches your score."""
DIFF_ARGV = ("git", "diff", "--no-color", "--binary")
REVISION_ARGV = ("git", "rev-parse", "--verify", "HEAD")
UNTRACKED_ARGV = ("git", "ls-files", "--others", "--exclude-standard", "-z")
ANSWER_PATHS = ("--", ".", f":(exclude){reserved_paths.RESERVED}", f":(exclude){reserved_paths.RESERVED_SIBLING}*")
GIT_TIMEOUT = 300
SCRATCH_DIR = "evidence"
SCRATCH_NAME = "scratch.md"
SALVAGE_BUDGET = 12_000
SALVAGE_BANNER = "## Salvaged by the harness, not written by the agent"


def _posted(*, message: Mapping[str, object]) -> dict[str, object]:
    posted: dict[str, object] = {"role": message.get("role"), "content": message.get("content")}
    posted.update({key: message[key] for key in POSTED_FIELDS if key in message})
    return posted


def _calls(*, calls: tuple[tuple[str, str, str], ...]) -> list[dict[str, object]]:
    return [{"id": i, "type": "function", "function": {"name": n, "arguments": a}} for i, n, a in calls]


def note(*, entry: str, body: str) -> None:
    """Write one categorized episode-log entry."""
    sys.stderr.write(f"\n### {entry}\n{body}\n")


class GatewayModel:
    """Fill upstream's model protocol from the shared gateway client."""

    config = None

    def __init__(self, client: Client, *, output_tokens: int | None, tools: Sequence[Mapping[str, object]] = TOOLS) -> None:
        """Bind one gateway client and its declared generation ceiling."""
        self._client, self._output_tokens = client, output_tokens
        self.tools = tools
        self.tokens, self.estimator = 0, Estimator()
        self.served: float | None = None

    def query(self, messages: list[dict[str, str]], **kwargs: object) -> dict:
        """Return one upstream assistant message with parsed tool actions."""
        del kwargs
        reply = self._chat(messages=[_posted(message=m) for m in messages], tools=self.tools)
        review = SELF_REVIEW_TOOL in self.tools
        correction = (
            "{{error}} Call bash with a command or self_review with an evidence path. No action ran."
            if review else FORMAT_ERROR
        )
        try:
            actions = parse_actions(calls=reply.tool_calls, review=review, format_error=correction)
        except FormatError as refused:
            refused.messages[0].setdefault("extra", {})["cost"] = float(reply.total_tokens)
            raise
        return self.format_message(
            role="assistant",
            content=(REASONING_BLOCK.format(text=reply.reasoning) if reply.reasoning else "") + reply.content,
            tool_calls=_calls(calls=reply.tool_calls),
            extra={"actions": actions, "cost": float(reply.total_tokens)},
        )

    def _chat(self, *, messages: Sequence[Mapping[str, object]], tools: Sequence[Mapping[str, object]] | None) -> Reply:
        reply = self._client.chat(messages=messages, tools=tools, output_tokens=self._output_tokens)
        if reply.reasoning:
            note(entry="model reasoning", body=reply.reasoning)
        self.estimator.observe(sent=reply.prompt_chars, counted=reply.prompt_tokens)
        self.tokens += reply.total_tokens
        self.served = reply.budget_remaining
        return reply

    def review(self, prompt: str) -> str:
        """Meter a fresh review conversation against this solver's own allowance."""
        reply = self._chat(messages=[{"role": "user", "content": prompt}], tools=None)
        # Keep even an unreadable reply if a later retry exhausts the grant.
        note(entry="self-review reply", body=reply.content)
        return reply.content

    @staticmethod
    def format_message(
        *, role: str, content: str, extra: dict | None = None, tool_calls: list[dict[str, object]] | None = None,
    ) -> dict:
        """Return one message in upstream's stored shape."""
        message = {"role": role, "content": content, "extra": dict(extra or {})}
        if tool_calls:
            message["tool_calls"] = list(tool_calls)
        return message

    @staticmethod
    def format_observation_messages(
        message: dict, outputs: list[dict], template_vars: dict | None = None,
    ) -> list[dict]:
        """Return bounded tool-result messages paired to their calls."""
        return actions_toolcall.format_toolcall_observation_messages(
            actions=message.get("extra", {}).get("actions", []),
            outputs=[_bounded(output=output) for output in outputs],
            observation_template=OBSERVATION,
            template_vars=template_vars,
        )

    @staticmethod
    def get_template_vars(**kwargs: object) -> dict:
        """Return no model-specific template variables."""
        del kwargs
        return {}

    def serialize(self) -> dict:
        """Return the token statistic stored with the trajectory."""
        return {"info": {"model_stats": {"tokens": self.tokens}}}


class ContextExceeded(InterruptAgentFlow):
    """Declared context can't hold even maximally compacted history."""


class GuardrailTripped(InterruptAgentFlow):
    """A safety guardrail refused this episode past every recovery."""


def context_exit(*, overflow: ContextOverflowError) -> dict:
    """Return an upstream exit message for context exhaustion."""
    extra = {"exit_status": CONTEXT_EXIT, "submission": "", "exception_str": str(overflow)}
    return {"role": "exit", "content": str(overflow), "extra": extra}


def _review_evidence(*, workspace: Path, tree: Path, name: str) -> Path:
    evidence = (workspace / name).resolve()
    if not evidence.is_relative_to(workspace) and not evidence.is_relative_to(tree):
        raise ValueError("test output must belong to this episode's tree or task checkout")
    return evidence


class WindowedAgent(DefaultAgent):
    """Upstream's loop with context fitting, host budget exposure and the guardrail ladder."""

    def __init__(self, *args: Any, window: Window, budget: Budget,
                 episode_state: Mapping[str, str] | None, **kwargs: Any) -> None:
        """Bind upstream's agent to one window, one budget, and one guardrail ladder."""
        super().__init__(*args, **kwargs)
        self._window, self._budget = window, budget
        self._episode_state = None if episode_state is None else dict(episode_state)
        self._warned: set[str] = set()
        self._ladder = Ladder()

    @property
    def charged(self) -> float:
        """What this episode's grant has been charged, children included."""
        remaining = self._budget.remaining(served=self.model.served, spent=self.model.tokens)
        return self._budget.used(remaining=remaining, spent=self.model.tokens)

    def execute_actions(self, message: dict) -> list[dict]:
        """Dispatch review inside the native process, retaining earlier observations on interruption."""
        actions = message.get("extra", {}).get("actions", [])
        outputs = []
        spent = self.model.tokens
        try:
            for action in actions:
                if action.get("tool") == SELF_REVIEW:
                    outputs.append(self._review(action=action))
                else:
                    outputs.append(self.env.execute(action))
        except Submitted as ended:
            outputs.append({"output": SUBMISSION + "\n" + ended.messages[0]["content"],
                            "returncode": 0, "exception_info": ""})
            raise
        except GatewayRefused as refused:
            outputs.append({"output": f"[self-review] refused: {refused}", "returncode": 1,
                            "exception_info": ""})
            raise
        finally:
            self.cost += self.model.tokens - spent
            observed = self.model.format_observation_messages(message, outputs, self.get_template_vars())
            self.add_messages(*observed)
        return observed

    def _review(self, *, action: dict) -> dict:
        state = self._episode_state
        if state is None:
            raise RuntimeError("native review requires the episode's capture state")
        try:
            tree = Path(state[TREE_VAR])
            workspace = Path(self.env.config.cwd).resolve()
            evidence = _review_evidence(workspace=workspace, tree=tree, name=action["evidence"])
            output = feedback(
                evidence=evidence, workspace=tree, environ={**state, CHECKOUT_VAR: str(workspace)},
                capture=diff, state=stated_state, chat=self.model.review,
            )
        except GatewayRefused:
            # A host/provider failure remains an episode failure, never a review score.
            raise
        except (RuntimeError, OSError, ValueError, KeyError) as refused:
            return {"output": f"[self-review] refused: {type(refused).__name__}: {refused}",
                    "returncode": 1, "exception_info": ""}
        return {"output": output, "returncode": 0, "exception_info": ""}

    def query(self) -> dict:
        """Fit stored history before asking for the next assistant turn."""
        eager = False
        while True:
            self.messages = self._fitted(messages=self._told(messages=self.messages), eager=eager)
            try:
                return super().query()
            except Exception as refused:
                if not guardrail_refusal(error=refused):
                    raise
                eager = self._recovered(refused=refused)

    def _recovered(self, *, refused: BaseException) -> bool:
        rung = self._ladder.next()
        note(entry="guardrail", body=f"{' -> '.join(r.value for r in self._ladder.taken)}: {LADDER_SAID[rung]}\n{refused}")
        if rung is Rung.PUBLISH:
            extra = {"exit_status": GUARDRAIL_EXIT, "submission": "", "exception_str": str(refused)}
            raise GuardrailTripped({"role": "exit", "content": str(refused), "extra": extra}) from refused
        if rung is Rung.CLEAR:
            self.messages = self.messages[:_PINNED]
        return rung is Rung.COMPACT

    def get_template_vars(self, **kwargs: object) -> dict:
        """Return upstream's variables plus the currently binding limit."""
        variables = super().get_template_vars(**kwargs)
        remaining = self._budget.remaining(served=self.model.served, spent=self.model.tokens)
        binding = self._budget.binding(
            taken=int(variables["n_model_calls"]), step_limit=int(variables["step_limit"]), remaining_tokens=remaining,
        )
        return {**variables, "binding": binding}

    def _told(self, *, messages: list[dict]) -> list[dict]:
        remaining = self._budget.remaining(served=self.model.served, spent=self.model.tokens)
        fragments = self._budget.exposure(remaining_tokens=remaining)
        said = [f"[host] {'; '.join(fragments)}."] if fragments else []
        said += self._budget.nudge(used_tokens=self._budget.used(remaining=remaining, spent=self.model.tokens), warned=self._warned)
        return messages + [self.model.format_message(role="user", content=text) for text in said]

    def _fitted(self, *, messages: list[dict], eager: bool) -> list[dict]:
        held = [pin(message=message) for message in messages[:_PINNED]]
        try:
            overhead = self.model.estimator.payload(value=list(self.model.tools))
            fitted, report = self._window.fitted(messages=held + messages[_PINNED:], overhead=overhead, eager=eager)
        except ContextOverflowError as overflow:
            note(entry="host", body=str(overflow))
            raise ContextExceeded(context_exit(overflow=overflow)) from overflow
        if report is not None and report.fired:
            note(entry="compaction", body=str(report))
        return fitted


def _bounds(*, wall: int | None) -> str:
    killed = (
        "No wall-clock limit stops a command or ends the episode."
        if wall is None
        else f"A command is killed after {wall} seconds, and nothing sooner."
    )
    return (
        "THE BOUNDS THIS LOOP ENFORCES, on top of anything said above."
        f" You get {STEP_LIMIT} steps -- one command each -- declared as"
        " `STEP_LIMIT` in `meta/solve.py`; edits affect the episode after yours."
        " A token allotment may end the episode sooner. While a limit binds,"
        " every observation carries a `<budget>` element naming whichever one"
        " it is and the estimated turns it leaves, so the count is kept for"
        " you and you need not keep it yourself."
        f" {killed}"
        f" The `bash` tool is `{SHELL}`; brace expansion, arrays, and `[[` work."
        f" Command output is clipped to {OBSERVATION_MAX_CHARACTERS:,} characters"
        " with the middle removed. The marker names the transient whole-output"
        f" file under `{spill_root()}/` and a `sed -n` command for the missing lines."
        " Declared token and machine limits are in `.rqgm/environment.json`."
    )


def _standing() -> str:
    return (
        f"WHERE YOU ARE. Your shell starts in `{WORKSPACE_NAME}`, and that directory is the"
        " tree you may edit. Nothing above it is yours."
    )


def episode_state(*, baseline: str, carried: frozenset[str], task: str, tree: Path) -> dict[str, str]:
    """Return native review state, whose task and carried paths can exceed process limits."""
    return {BASELINE_VAR: baseline, CARRIED_VAR: CARRIED_SEPARATOR.join(sorted(carried)), TASK_VAR: task, TREE_VAR: str(tree.resolve())}


def shell_state(*, state: Mapping[str, str] | None) -> dict[str, str]:
    """Export only scalar locations; native review retains the full task and carried paths."""
    return {} if state is None else {
        key: value for key, value in state.items() if key not in {CARRIED_VAR, TASK_VAR}
    }


def stated_state(*, environ: Mapping[str, str]) -> tuple[str, frozenset[str], str]:
    """Return the capture state the episode stated, refusing an unstated one."""
    baseline, task = environ.get(BASELINE_VAR, ""), environ.get(TASK_VAR, "")
    if not baseline:
        raise RuntimeError(f"this episode stated no {BASELINE_VAR}, so there is no commit to read the work against")
    if not task:
        raise RuntimeError(f"this episode stated no {TASK_VAR}, so there is no problem statement to review the work against")
    return baseline, frozenset(name for name in environ.get(CARRIED_VAR, "").split(CARRIED_SEPARATOR) if name), task


def build_agent(
    *,
    client: Client,
    budget: Budget,
    system_template: str,
    workspace: Path,
    state: Mapping[str, str] | None = None,
    instance: str | None = None,
) -> DefaultAgent:
    """Build upstream's loop with the shared gateway, workspace, and declared limits."""
    output_tokens = declared_budget(raw=os.environ["RQGM_AGENT_MAX_OUTPUT_TOKENS"])
    tree = Path(state[TREE_VAR]) if state is not None and TREE_VAR in state else None
    enabled = tree is not None and reviewing(workspace=tree)
    if enabled:
        stated_state(environ=state)
    model = GatewayModel(client, output_tokens=output_tokens, tools=(*TOOLS, SELF_REVIEW_TOOL) if enabled else TOOLS)
    served = review_clause(workspace=tree) + proxy_clause(workspace=tree) if enabled else ""
    # No host states a command wall; `None` is what the prompt says and what the shell enforces.
    wall = None
    system_template = system_template.replace(REVIEW_MARK, served)
    if enabled:
        system_template = system_template.replace(LOOP, REVIEW_LOOP)
    composed = f"{system_template}\n\n{_standing()}\n\n{_bounds(wall=wall)}"
    instance_used = SOLVE_INSTANCE if instance is None else instance
    if enabled:
        instance_used = instance_used.replace(
            "2. Provide one or more bash tool calls to execute",
            f"2. Call bash with a command or {SELF_REVIEW} with an evidence path",
        )
    record_prompt(system=composed, instance=instance_used)
    return WindowedAgent(
        model,
        BashEnvironment(cwd=str(workspace), timeout=wall, ends=None, env=shell_state(state=state)),
        system_template=composed,
        instance_template=instance_used,
        step_limit=STEP_LIMIT,
        cost_limit=float(budget.tokens or 0),
        wall_time_limit_seconds=0,
        window=Window(
            sequence=declared_budget(raw=os.environ["RQGM_AGENT_MAX_SEQUENCE_TOKENS"]),
            output=output_tokens,
            estimator=model.estimator,
            summarizer=Summarizer(chat=client.chat, note=note),
        ),
        budget=budget,
        episode_state=state,
    )


def review_clause(*, workspace: Path) -> str:
    """Return the clause as served to an episode standing in `workspace`."""
    del workspace
    return REVIEW_CLAUSE.format(TOOL=SELF_REVIEW)


def _git(*, argv: tuple[str, ...], workspace: Path) -> str:
    finished = subprocess.run(argv, cwd=str(workspace), capture_output=True, text=True, check=False, timeout=GIT_TIMEOUT)
    return finished.stdout if finished.returncode == 0 else ""


def revision(*, workspace: Path) -> str:
    """Return the commit a tree's HEAD names right now, or empty text outside a checkout."""
    return _git(argv=REVISION_ARGV, workspace=workspace).strip()


#: The commit each workspace answers against, taken once and kept for the whole episode.
#:
#: `solve` can be called more than once per episode: `task_agents/solver` runs a revise round
#: against its own review seat in the same working directory, and a candidate that commits its
#: work moves `HEAD` between the rounds. Reading `git rev-parse HEAD` on every call would base the
#: second round on the first round's commit, so the diff that leaves would be the increment and
#: not the work, and an increment does not apply to the task's base commit.
_BASELINES: dict[str, str] = {}


def episode_baseline(*, workspace: Path) -> str:
    """Return the commit this EPISODE answers against, the same one for every round it runs."""
    key = str(workspace.resolve())
    if not _BASELINES.get(key):
        _BASELINES[key] = revision(workspace=workspace)
    return _BASELINES[key]


def untracked(*, workspace: Path) -> frozenset[str]:
    """Return the non-ignored untracked paths, excluding harness bookkeeping."""
    names = _git(argv=UNTRACKED_ARGV, workspace=workspace).split("\0")
    return frozenset(name for name in names if name and not reserved_paths.reserved(name=name))


def diff(*, workspace: Path, baseline: str = "") -> str:
    """Return the complete committed patch consumed by both review and executable grading."""
    if not baseline:
        return ""
    # Replacing a hunk in model.patch makes executable grading fail to apply it.
    return _git(argv=(*DIFF_ARGV, baseline, "HEAD", *ANSWER_PATHS), workspace=workspace)


def salvage(*, notes: Path, messages: Sequence[Mapping[str, object]], died: BaseException) -> str | None:
    """Append a dying episode's own assistant turns to its scratch file, and say where."""
    said = [text for turn in messages if turn.get("role") == "assistant" if (text := str(turn.get("content") or "").strip())]
    kept: list[str] = []
    total = 0
    for text in reversed(said):
        if kept and total + len(text) > SALVAGE_BUDGET:
            break
        kept.append(text[:SALVAGE_BUDGET])
        total += len(text)
    kept.reverse()
    header = (
        f"\n\n{SALVAGE_BANNER}\n\nThis episode did not finish. It ended at turn {len(said)} on"
        f" `{type(died).__name__}: {died}`, before it could write up what it had learned. What follows is its own"
        " assistant turns, verbatim and unreviewed -- findings here are as reliable as the turn that stated them and"
        f" no more, and a plan stated in the last turn was never carried out. {len(said) - len(kept)} earlier turn(s)"
        " omitted for length.\n\n"
    )
    path = io_contract.own_experience_path(workspace=notes) / SCRATCH_DIR / SCRATCH_NAME
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as opened:
            opened.write(header + "\n\n---\n\n".join(kept) + "\n")
    except OSError:
        return None
    return str(path)


def _absent(*, task: str | None) -> str | None:
    if task is None:
        return "a null problem statement"
    if not task:
        return "an empty problem statement"
    if not task.strip():
        return f"a problem statement of {len(task)} whitespace characters and nothing else"
    return None


def solve(*, task: str, workspace: Path, notes: Path) -> dict:
    """Run one bounded solver episode and return its patch, dialogue, and spend."""
    if (shape := _absent(task=task)) is not None:
        raise ValueError(
            f"this episode was handed {shape}, so there is nothing to solve; refusing rather than asking a model an empty"
            " `<pr_description>` and scoring the patch on an answer to nothing",
        )
    baseline = episode_baseline(workspace=workspace)
    if (client := client_from_env()) is None:
        raise RuntimeError("the solver has no gateway grant, so it cannot run an episode")
    state = episode_state(baseline=baseline, carried=untracked(workspace=workspace), task=task, tree=notes)
    agent = build_agent(client=client, budget=Budget.declared(), system_template=SOLVE_PROMPT, workspace=workspace, state=state)
    try:
        outcome = agent.run(task)
    except BaseException as died:
        salvage(notes=notes, messages=agent.messages, died=died)
        raise
    return {
        "patch": diff(workspace=workspace, baseline=baseline),
        "messages": agent.messages,
        "exit_status": str(outcome.get("exit_status", "unknown")),
        "turns": agent.n_calls,
        "tokens": agent.charged,
    }


def main(*, argv: Sequence[str]) -> int:
    """Run one self-review for the episode whose shell invoked this module."""
    return _self_review_main(argv=argv, capture=diff, state=stated_state)


if __name__ == "__main__":
    sys.exit(main(argv=sys.argv[1:]))
