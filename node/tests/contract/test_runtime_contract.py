# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""The task-agent runtime: one dispatch in, the declared files and the host's answer document out."""

import hashlib
import json
import sys
import types

import pytest

from task_agents.shared import runtime


def _seat(monkeypatch, forward):
    """Stand a seat named `probe` in the carried package, answering with `forward`."""
    module = types.ModuleType("task_agents.probe")
    module.forward = forward
    monkeypatch.setitem(sys.modules, "task_agents.probe", module)
    return module


def _dispatch(node, job):
    (node / ".rqgm" / "dispatch.json").write_text(json.dumps(job), encoding="utf-8")


def test_no_dispatch_means_no_job_and_says_so(node, capsys):
    """A seat is never run by hand: without the host's dispatch the runtime exits 1 and explains."""
    assert runtime.run() == 1
    said = capsys.readouterr().err
    assert ".rqgm/dispatch.json is absent" in said and "rqgm-call submit" in said


def test_a_generate_dispatch_writes_each_members_files_and_the_generation_document(node, monkeypatch):
    """Generate: every member's declared outputs and evidence land where the job says, once."""
    seen = []
    _seat(monkeypatch, lambda inputs: seen.append(inputs) or {
        "files": {"answer.txt": b"A" + str(inputs["seed"]).encode()},
        "evidence": {"log.txt": b"ran"},
    })
    staged = node / "in0"
    staged.write_bytes(b"row zero")
    members = [
        {"member": m, "seed": n, "inputs": [str(staged)], "outputs": {"answer.txt": str(node / f"{m}.out")}, "evidence": {"log.txt": str(node / f"{m}.log")}}
        for n, m in enumerate(("m1", "m2"))
    ]
    _dispatch(node, {"role": "probe", "mode": "generate", "seed": 0, "members": members})
    assert runtime.run() == 0
    assert not (node / ".rqgm" / "dispatch.json").exists(), "the dispatch is consumed, never re-run"
    assert (node / "m1.out").read_bytes() == b"A0" and (node / "m2.out").read_bytes() == b"A1"
    assert (node / "m2.log").read_bytes() == b"ran"
    assert json.loads((node / ".rqgm" / "generation.json").read_text()) == [
        {"evidence": ["log.txt"], "member": "m1", "outputs": ["answer.txt"]},
        {"evidence": ["log.txt"], "member": "m2", "outputs": ["answer.txt"]},
    ]
    assert [s["files"] for s in seen] == [{"0": b"row zero"}] * 2 and [s["seed"] for s in seen] == [0, 1]


def test_a_judge_dispatch_scores_every_member_from_the_verdict_file(node, monkeypatch):
    """Judge: one call per member, the score read from the declared verdict channel."""
    _seat(monkeypatch, lambda inputs: {"files": {"verdict.json": b'{"score": 0.25}'}})
    paper = node / "paper.txt"
    paper.write_bytes(b"the paper")
    _dispatch(node, {
        "role": "probe", "mode": "judge", "seed": 0, "members": ["a", "b"],
        "inputs": {"paper": str(paper)},
        "outputs": {"verdict.json": str(node / "verdict.json")},
        "evidence": {},
    })
    assert runtime.run() == 0
    assert json.loads((node / ".rqgm" / "judgment.json").read_text()) == [
        {"member": "a", "score": 0.25},
        {"member": "b", "score": 0.25},
    ]
    assert (node / "verdict.json").read_bytes() == b'{"score": 0.25}'


def test_a_verdict_that_is_not_a_scored_object_refuses_by_name(node, monkeypatch):
    """A verdict without a numeric score is a refusal, never a score of nothing."""
    paper = node / "paper.txt"
    paper.write_bytes(b"p")
    job = {"role": "probe", "mode": "judge", "seed": 0, "members": ["a"], "inputs": {"paper": str(paper)}, "outputs": {"verdict.json": str(node / "v")}}
    for verdict, said in ((b"not json", "cannot parse"), (b'{"grade": 1}', "names no 'score'"), (b'{"score": true}', "not a JSON number")):
        _seat(monkeypatch, lambda _, v=verdict: {"files": {"verdict.json": v}})
        _dispatch(node, job)
        with pytest.raises(ValueError, match=said):
            runtime.run()


def test_an_unserved_mode_and_a_malformed_membership_refuse(node, monkeypatch):
    """The runtime serves generate and judge, and a member owes its four fields."""
    _seat(monkeypatch, lambda _: {"files": {}})
    _dispatch(node, {"role": "probe", "mode": "train", "seed": 0})
    with pytest.raises(TypeError, match="does not serve"):
        runtime.run()
    _dispatch(node, {"role": "probe", "mode": "generate", "seed": 0, "members": [{"member": "m"}]})
    with pytest.raises(ValueError, match="states no"):
        runtime.run()
    _dispatch(node, {"role": "probe", "mode": "judge", "seed": 0, "members": [1]})
    with pytest.raises(ValueError, match="names no members"):
        runtime.run()


def test_harvesting_records_which_rows_a_call_received(node, monkeypatch):
    """Provenance: a staged file whose digest matches the host's selection is a received sample."""
    paper = node / "paper.txt"
    paper.write_bytes(b"row one")
    digest = hashlib.sha256(b"row one").hexdigest()
    seat = types.SimpleNamespace(forward=lambda _: {"files": {"verdict.json": b'{"score": 0}'}})
    job = {
        "role": "probe", "mode": "judge", "seed": 0, "inputs": {"paper": str(paper)},
        "provenance": {"call": "c", "dataset": "d", "task": "t", "basis": "b", "samples": {"s1": {"paper": digest}, "s2": {"paper": "0" * 64}}},
    }
    with runtime.harvesting() as rows:
        runtime._forward(seat, job)
        with runtime.harvesting() as inner:
            runtime._forward(seat, job)
    assert len(rows) == 1 and len(inner) == 1
    assert rows[0]["samples"] == ["s1"] and rows[0]["missing"] == ["s2"] and rows[0]["carried"] == 1
    assert rows[0]["received"] == {"paper": 7} and rows[0]["role"] == "probe"


def test_the_capabilities_a_seat_is_handed_are_none_without_a_grant_or_a_judge(node, root, monkeypatch):
    """chat needs a gateway grant, review a node-scored judge over the role, solve a meta/solve.py."""
    for key in ("OPENAI_BASE_URL", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    assert runtime.chat_capability() is None
    assert runtime.review_capability(role="nobody") is None
    assert runtime.review_capability(role="probe") is None, "the judging seat is not importable here"
    _seat(monkeypatch, lambda _: None).review = "the entry"
    assert runtime.review_capability(role="probe") == "the entry"
    assert callable(runtime.solve_capability()) == (root / "meta" / "solve.py").is_file()


def test_a_grant_makes_chat_a_text_call_through_the_client(node, grant, monkeypatch):
    """With the episode environment stated, chat is one user message through the gateway client."""
    from shared.gateway import client as wire

    asked = []
    monkeypatch.setattr(wire.Client, "chat", lambda self, *, messages, **_: asked.append(messages) or wire.Reply("pong", (), 1))
    chat = runtime.chat_capability()
    assert chat("ping") == "pong"
    assert asked == [[{"role": "user", "content": "ping"}]]
