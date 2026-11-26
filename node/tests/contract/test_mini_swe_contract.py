# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""The mini-swe harness, driven where it runs: the shell, the observations, the loop, the capture."""

import importlib.util
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from shared.gateway.budget import Budget
from shared.gateway.client import Reply


@pytest.fixture
def harness(root, node, monkeypatch):
    """The carried meta harness, imported the way an episode imports it, standing in `node`."""
    if not (root / "meta" / "solve.py").is_file():
        pytest.skip("this form carries no mini-swe harness under meta/")
    monkeypatch.syspath_prepend(str(root / "meta"))
    for name in ("execution", "self_review", "solve"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    import minisweagent
    import solve

    assert Path(minisweagent.__file__).resolve().is_relative_to(root / "meta"), "exercise the carried upstream"
    return solve


class _Client:
    """A gateway that answers each chat with the next scripted reply and remembers what it was sent."""

    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    def chat(self, *, messages, **kwargs):
        self.seen.append((list(messages), kwargs))
        return self.replies.pop(0)


def _git(where, *argv):
    return subprocess.run(["git", *argv], cwd=str(where), check=True, capture_output=True, text=True).stdout


def test_a_command_runs_under_bash_and_a_hidden_failure_is_reported(harness):
    """Bash is the shell; a pipeline whose early stage failed reports that stage, not the 0."""
    environment = harness.BashEnvironment(cwd=".", timeout=None)
    out = environment.execute({"command": "echo ${BASH_VERSION:+bash} && printf '%s\\n' a{1,2}"})
    assert out["returncode"] == 0 and out["output"].split() == ["bash", "a1", "a2"]
    failed = environment.execute({"command": "false | true"})
    assert failed["returncode"] == 1 and "the last pipeline exited [1 0]" in failed["output"]
    hidden = environment.execute({"command": "false; true"})
    assert hidden["returncode"] == 0 and sys.modules["execution"].HIDDEN_STAGE in hidden["output"]


@pytest.mark.parametrize("separate_task_checkout", [False, True])
def test_an_observation_is_bounded_and_the_whole_output_is_spilled_under_this_generation(harness, node, separate_task_checkout):
    """Read the advertised full output through the same shell that produced it, from either cwd."""
    workspace = node / "task-checkout" if separate_task_checkout else node
    workspace.mkdir(exist_ok=True)
    environment = harness.BashEnvironment(cwd=str(workspace), timeout=10)
    model = harness.GatewayModel(_Client(), output_tokens=None)
    long = "".join(f"line {n}: café\n" for n in range(20_000))
    command = "for ((n=0; n<20000; n++)); do printf 'line %s: café\\n' \"$n\"; done"
    message = {"extra": {"actions": [{"command": command, "tool_call_id": "c1"}]}}
    ran = environment.execute({"command": command})
    assert ran["returncode"] == 0 and ran["output"] == long
    [seen] = model.format_observation_messages(message, [ran])
    assert len(seen["content"]) < len(long) and "elided lines" in seen["content"]
    spilled = list((node / "experience" / "generations" / "1" / "spill").glob("*.txt"))
    assert len(spilled) == 1 and spilled[0].read_bytes() == long.encode()
    advertised = re.search(r"this output whole is at (.*?); `", seen["content"])
    assert advertised is not None
    readback = environment.execute({"command": f"cat -- {shlex.quote(advertised[1])}"})
    assert readback["returncode"] == 0 and readback["output"].encode() == long.encode()
    spilled[0].unlink()
    missing = environment.execute({"command": f"cat -- {shlex.quote(advertised[1])}"})
    assert missing["returncode"] != 0 and "No such file" in missing["output"]
    [quiet] = model.format_observation_messages(message, [{**ran, "output": ""}])
    assert "no output" in quiet["content"]
    [killed] = model.format_observation_messages(message, [{"output": "", "returncode": 1, "exception_info": "timed out"}])
    assert "timed out" in killed["content"]


def test_an_unwritable_spill_does_not_advertise_a_file(harness, node):
    """A failed spill still clips the observation without promising unavailable bytes."""
    (node / "experience").write_text("not a directory")
    model = harness.GatewayModel(_Client(), output_tokens=None)
    message = {"extra": {"actions": [{"command": "x", "tool_call_id": "c1"}]}}
    [seen] = model.format_observation_messages(message, [{"returncode": 0, "exception_info": "", "output": "x" * 100_000}])
    assert "elided lines" in seen["content"] and len(seen["content"]) < 100_000
    assert "this output whole is at" not in seen["content"]


def test_the_model_seam_turns_a_reply_into_an_upstream_turn_with_actions(harness):
    """query() posts only wire fields, meters tokens, and hands upstream the parsed bash action."""
    client = _Client(Reply("run it", (("c1", "bash", '{"command": "ls"}'),), 30, 970.0, prompt_tokens=20, prompt_chars=60, reasoning="because"))
    model = harness.GatewayModel(client, output_tokens=64)
    turn = model.query([{"role": "user", "content": "go", "extra": {"secret": 1}}])
    assert turn["role"] == "assistant" and turn["content"].endswith("run it") and "because" in turn["content"]
    assert [a["command"] for a in turn["extra"]["actions"]] == ["ls"] and turn["extra"]["cost"] == 30.0
    assert (model.tokens, model.served, model.serialize()) == (30, 970.0, {"info": {"model_stats": {"tokens": 30}}})
    posted, kwargs = client.seen[0]
    assert posted == [{"role": "user", "content": "go"}] and kwargs["output_tokens"] == 64


def test_build_agent_composes_the_prompt_records_it_and_binds_the_declared_limits(harness, node, grant, monkeypatch):
    """The system prompt gains the standing and bounds clauses, is filed before any call, and the loop is bounded."""
    monkeypatch.setenv("RQGM_AGENT_MAX_OUTPUT_TOKENS", "128")
    monkeypatch.setenv("RQGM_AGENT_MAX_SEQUENCE_TOKENS", "20000")
    agent = harness.build_agent(
        client=_Client(),
        budget=Budget(tokens=500, exposed=frozenset({"tokens"}), nudge_at=None),
        system_template="BASE {REVIEW}",
        workspace=node,
        state=harness.episode_state(baseline="abc", carried=frozenset({"n.txt"}), task="fix it", tree=node),
    )
    system = agent.config.system_template
    assert system.startswith("BASE") and harness._standing() in system and harness._bounds(wall=None) in system
    assert agent.config.step_limit == harness.STEP_LIMIT and agent.config.cost_limit == 500.0
    assert agent.config.wall_time_limit_seconds == 0 and agent.env.config.timeout is None
    assert "RQGM_EPISODE_TASK" not in agent.env.config.env
    assert "RQGM_EPISODE_CARRIED" not in agent.env.config.env
    assert agent._episode_state["RQGM_EPISODE_TASK"] == "fix it"
    recorded = json.loads((node / "experience" / "generations" / "1" / "prompt.json").read_text())
    assert recorded["system"] == system and recorded["instance"] == harness.SOLVE_INSTANCE
    assert agent.get_template_vars()["binding"].startswith(f"{harness.STEP_LIMIT} turn(s) left")
    assert agent.env.config.env["RQGM_EPISODE_BASELINE"] == "abc"
    assert agent._episode_state["RQGM_EPISODE_CARRIED"] == "n.txt"


def test_the_capture_reads_committed_work_against_the_baseline_and_skips_harness_paths(harness, tmp_path):
    """revision, untracked and diff read the checkout as the submission capture does."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "base")
    baseline = harness.revision(workspace=repo)
    assert baseline == _git(repo, "rev-parse", "HEAD").strip()
    (repo / "work.txt").write_text("done\n")
    (repo / ".rqgm").mkdir()
    (repo / ".rqgm" / "note").write_text("harness\n")
    assert harness.untracked(workspace=repo) == frozenset({"work.txt"})
    assert harness.diff(workspace=repo, baseline=baseline) == ""
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "work")
    patch = harness.diff(workspace=repo, baseline=baseline)
    assert "+done" in patch and ".rqgm/note" not in patch
    assert harness.revision(workspace=tmp_path) == "" and harness.diff(workspace=repo) == ""


def test_a_dying_episode_salvages_its_own_words_and_an_empty_task_never_starts(harness, node):
    """salvage() appends the assistant turns to this generation's scratch; solve() refuses no task."""
    where = harness.salvage(notes=node, messages=[{"role": "assistant", "content": "I learned X"}, {"role": "user", "content": "no"}], died=RuntimeError("boom"))
    text = (node / "experience" / "generations" / "1" / "evidence" / "scratch.md").read_text()
    assert where.endswith("scratch.md") and harness.SALVAGE_BANNER in text and "I learned X" in text and "boom" in text
    for task in ("", "   ", None):
        with pytest.raises(ValueError, match="nothing to solve"):
            harness.solve(task=task, workspace=node, notes=node)


def test_a_self_review_needs_a_seat_a_grant_and_the_episode_state(harness, node, capsys):
    """The advertised command resolves here; without a consultable seat it refuses and says why."""
    review = harness._self_review_main.__module__
    module = sys.modules[review]
    assert module.review_seat(workspace=node) == "task_agents.probe"
    assert not module.reviewing(workspace=node) and "task_agents.probe" in module.why_not_reviewing(workspace=node)
    assert module.proxy_clause(workspace=node) == ""
    state = harness.episode_state(baseline="abc", carried=frozenset({"n.txt"}), task="fix it", tree=node)
    assert state["RQGM_EPISODE_TREE"] == str(node.resolve())
    assert harness.stated_state(environ=state) == ("abc", frozenset({"n.txt"}), "fix it")
    with pytest.raises(RuntimeError, match="stated no RQGM_EPISODE_BASELINE"):
        harness.stated_state(environ={})
    clause = harness.review_clause(workspace=node)
    assert harness.SELF_REVIEW in clause and '"evidence"' in clause
    assert module.SELF_REVIEW_FLAG not in clause
    assert harness.main(argv=[]) == 1 and "usage:" in capsys.readouterr().err
    assert harness.main(argv=[module.SELF_REVIEW_FLAG, "evidence.txt"]) == 1
    assert "[self-review] refused" in capsys.readouterr().err


@pytest.mark.parametrize("arguments", ['{}', '[]', '{"evidence": ""}', '{"evidence": 1}',
                                        '{"evidence": "out", "model": "other"}', 'not-json'])
def test_native_review_refuses_invalid_arguments_before_any_action_and_keeps_the_charge(harness, arguments):
    client = _Client(Reply("", (("r1", harness.SELF_REVIEW, arguments),), 13, 87.0))
    model = harness.GatewayModel(client, output_tokens=64, tools=(*harness.TOOLS, harness.SELF_REVIEW_TOOL))
    with pytest.raises(harness.FormatError) as refused:
        model.query([{"role": "user", "content": "review"}])
    assert model.tokens == 13 and model.served == 87.0
    assert refused.value.messages[0]["extra"]["cost"] == 13.0
    assert len(client.seen) == 1


def test_an_unoffered_review_is_refused_by_the_parser(harness):
    model = harness.GatewayModel(
        _Client(Reply("", (("r1", harness.SELF_REVIEW, '{"evidence": "out"}'),), 7)), output_tokens=64,
    )
    with pytest.raises(harness.FormatError):
        model.query([{"role": "user", "content": "review"}])
    assert model.tokens == 7


@pytest.mark.parametrize("large_state", [False, True], ids=["small", "over-linux-argument-limit"])
def test_native_review_keeps_mixed_tool_ids_feedback_and_retry_usage_through_submission(harness, node, monkeypatch, grant, large_state):
    task = "fix it" + (" large revision context" * 12000 if large_state else "")
    carried_paths = frozenset(f"scratch/{n:05d}-" + "x" * 40 for n in range(5000)) if large_state else frozenset()
    seat_path = node / "task_agents" / "contract_review" / "__init__.py"
    seat_path.parent.mkdir(parents=True)
    seat_path.write_text('''def self_review(*, problem, diff, author_output, chat):
    assert problem == EXPECTED_TASK and "+done" in diff
    assert author_output == b"3 passed\\n"
    assert chat("review first") == "unreadable"
    assert chat("review again") == "accepted"
    return 1.0
''')
    spec = importlib.util.spec_from_file_location("task_agents.contract_review", seat_path)
    seat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(seat)
    seat.EXPECTED_TASK = task
    monkeypatch.setitem(sys.modules, spec.name, seat)
    contract = node / ".rqgm" / "contract.json"
    stated = json.loads(contract.read_text())
    stated["judges"] = {"contract_review": {"scorer": "node"}}
    contract.write_text(json.dumps(stated))
    checkout = node / "checkout"
    checkout.mkdir()
    _git(checkout, "init", "-q")
    _git(checkout, "config", "user.name", "fixture")
    _git(checkout, "config", "user.email", "fixture@example.invalid")
    _git(checkout, "commit", "-qm", "base", "--allow-empty")
    baseline = harness.revision(workspace=checkout)
    command = "test -z \"${RQGM_EPISODE_TASK+x}\" && test -z \"${RQGM_EPISODE_CARRIED+x}\" && printf 'done\\n' > work.txt && git add work.txt && git commit -qm work && printf '3 passed\\n' > tests.out"
    calls = (("edit", "bash", json.dumps({"command": command})),
             ("review", harness.SELF_REVIEW, '{"evidence": "tests.out"}'),
             ("finish", "bash", json.dumps({"command": "echo " + harness.SUBMISSION})))
    client = _Client(Reply("", calls, 10, 90.0), Reply("unreadable", (), 11, 79.0), Reply("accepted", (), 12, 67.0))
    agent = harness.build_agent(
        client=client, budget=Budget(tokens=100, exposed=frozenset({"tokens"}), nudge_at=None),
        system_template="{REVIEW}", workspace=checkout,
        state=harness.episode_state(baseline=baseline, carried=carried_paths, task=task, tree=node),
    )
    result = agent.run(task)
    assert result["exit_status"] == "Submitted" and agent.n_calls == 1
    assert harness.stated_state(environ=agent._episode_state) == (baseline, carried_paths, task)
    assert agent.cost == agent.model.tokens == agent.charged == 33
    observed = [turn for turn in agent.messages if turn["role"] == "tool"]
    assert [turn["tool_call_id"] for turn in observed] == ["edit", "review", "finish"]
    assert "[self-review] score=1.0" in observed[1]["content"]
    assert "unreadable" in observed[1]["content"] and "accepted" in observed[1]["content"]
    assert "action was not executed" not in observed[2]["content"]
    assert client.seen[1][0] == [{"role": "user", "content": "review first"}]
    assert all(fields["tools"] is None for _, fields in client.seen[1:])
    assert all(fields["output_tokens"] is None for _, fields in client.seen)


def test_native_review_does_not_advertise_without_episode_state(harness, node, grant):
    agent = harness.build_agent(
        client=_Client(), budget=Budget(tokens=100, exposed=frozenset(), nudge_at=None),
        system_template="{REVIEW}", workspace=node,
    )
    assert harness.SELF_REVIEW_TOOL not in agent.model.tools
    assert "ASK YOUR REVIEWER" not in agent.config.system_template
