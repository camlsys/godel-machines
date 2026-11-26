# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""The io contract: what a seat may read, must write, and may leave as evidence."""

import json
import types
from pathlib import Path

import pytest

from task_agents.shared import io_contract

FOREIGN = "a-channel-no-form-declares"


def _arms(block):
    """Return every `(role, mode)` a descriptor declares a carried seat arm for."""
    return sorted(
        (role, mode)
        for role, arms in (block or {}).items()
        for mode in (arms or {})
        if mode != io_contract.HOST_INSTRUMENT
    )


def _staged(arm):
    return {str(name): b"STAGED BYTES" for name in arm["reads"]}


def _answered(arm):
    return {"files": {str(name): b"ANSWER BYTES" for name in arm["writes"]}, "evidence": {}}


def test_a_seat_reads_its_own_generation_folder_from_the_depth_the_host_wrote(node):
    """Every path a seat writes hangs off experience/generations/<depth>; genesis has no depth."""
    assert io_contract.declared_depth() == 1
    assert io_contract.own_experience_path(workspace=node) == node / "experience" / "generations" / "1"
    assert io_contract.own_attempt_cell(role="probe", task="t") == Path("experience/generations/1/probe/t")
    assert io_contract.shared_data_path() == Path("experience/data")
    (node / ".rqgm" / "contract.json").write_text(json.dumps({"io": {}}), encoding="utf-8")
    with pytest.raises(io_contract.IoRefusalError, match="failed to stage"):
        io_contract.declared_depth()


def test_an_absent_or_malformed_contract_refuses_by_name(tmp_path):
    """No contract, no run: a seat measured against nothing has a contract nobody can break."""
    with pytest.raises(io_contract.IoRefusalError, match="absent"):
        io_contract.declared_arm(role="probe", mode="judge", contract=tmp_path / "none.json")
    listed = tmp_path / "list.json"
    listed.write_text("[]", encoding="utf-8")
    with pytest.raises(io_contract.IoRefusalError, match="not an object"):
        io_contract.declared_depth(contract=listed)


def test_only_a_declared_seat_on_a_declared_arm_gets_a_vocabulary(node):
    """An undeclared seat, an undeclared arm, and a host instrument all refuse and say which."""
    arm = io_contract.declared_arm(role="probe", mode="judge")
    assert set(arm) == set(io_contract.ARM_KEYS)
    with pytest.raises(io_contract.IoRefusalError, match="not 'nobody'"):
        io_contract.declared_arm(role="nobody", mode="judge")
    with pytest.raises(io_contract.IoRefusalError) as refusal:
        io_contract.declared_arm(role="probe", mode=FOREIGN)
    assert FOREIGN in str(refusal.value) and "probe" in str(refusal.value)
    with pytest.raises(io_contract.IoRefusalError, match="host-owned instrument"):
        io_contract.declared_arm(role="fixed", mode=None)


def test_inputs_are_exactly_the_declared_reads_or_a_gapless_blind_run(node):
    """A missing read, a stray file, a named file among positions, and a gap all refuse."""
    judge = io_contract.declared_arm(role="probe", mode="judge")
    io_contract.check_inputs(role="probe", mode="judge", files={"paper": b""}, arm=judge)
    with pytest.raises(io_contract.IoRefusalError, match="never arrived"):
        io_contract.check_inputs(role="probe", mode="judge", files={}, arm=judge)
    with pytest.raises(io_contract.IoRefusalError, match=FOREIGN):
        io_contract.check_inputs(role="probe", mode="judge", files={"paper": b"", FOREIGN: b""}, arm=judge)
    generate = io_contract.declared_arm(role="probe", mode="generate")
    io_contract.check_inputs(role="probe", mode="generate", files={"0": b"", "1": b""}, arm=generate)
    with pytest.raises(io_contract.IoRefusalError, match="not positions"):
        io_contract.check_inputs(role="probe", mode="generate", files={"0": b"", "x": b""}, arm=generate)
    with pytest.raises(io_contract.IoRefusalError, match="gapless"):
        io_contract.check_inputs(role="probe", mode="generate", files={"0": b"", "2": b""}, arm=generate)


def test_an_answer_is_exactly_the_declared_writes_and_at_most_the_declared_evidence(node):
    """No files mapping, an omitted write, an undeclared write, and stray evidence all refuse."""
    arm = io_contract.declared_arm(role="probe", mode="judge")
    good = {"files": {"verdict.json": b"{}"}, "evidence": {"log.txt": b""}}
    io_contract.check_output(role="probe", mode="judge", answer=good, arm=arm)
    io_contract.check_output(role="probe", mode="judge", answer={"files": {"verdict.json": b""}}, arm=arm)
    with pytest.raises(io_contract.IoRefusalError, match="no 'files'"):
        io_contract.check_output(role="probe", mode="judge", answer={"evidence": {}}, arm=arm)
    with pytest.raises(io_contract.IoRefusalError, match="omitting"):
        io_contract.check_output(role="probe", mode="judge", answer={"files": {}}, arm=arm)
    with pytest.raises(io_contract.IoRefusalError, match=FOREIGN):
        io_contract.check_output(role="probe", mode="judge", answer={"files": {"verdict.json": b"", FOREIGN: b""}}, arm=arm)
    with pytest.raises(io_contract.IoRefusalError, match=FOREIGN):
        io_contract.check_output(role="probe", mode="judge", answer={**good, "evidence": {FOREIGN: b""}}, arm=arm)
    with pytest.raises(io_contract.IoRefusalError, match="not a mapping"):
        io_contract.check_output(role="probe", mode="judge", answer={**good, "evidence": ["log.txt"]}, arm=arm)


def test_this_nodes_own_declaration_drives_every_carried_arm(contract):
    """The host's io block names arms this module reads whole, and each accepts its own shape."""
    arms = _arms(contract.get("io"))
    assert arms, "this node's contract carries no io block, so every seat refuses at the boundary"
    for role, mode in arms:
        arm = io_contract.declared_arm(role=role, mode=mode)
        assert set(io_contract.ARM_KEYS) <= set(arm)
        if not arm["positional"]:
            io_contract.check_inputs(role=role, mode=mode, files=_staged(arm), arm=arm)
        io_contract.check_output(role=role, mode=mode, answer=_answered(arm), arm=arm)
        with pytest.raises(io_contract.IoRefusalError):
            io_contract.check_output(role=role, mode=mode, answer={"evidence": {}}, arm=arm)
    for role, arms_of in sorted((contract.get("io") or {}).items()):
        if list(arms_of or {}) == [io_contract.HOST_INSTRUMENT]:
            with pytest.raises(io_contract.IoRefusalError, match="host-owned instrument"):
                io_contract.declared_arm(role=role, mode=None)


def test_the_runtime_checks_both_sides_of_the_seat_it_forwards_to(node):
    """`_forward` refuses an undeclared input before the seat runs and an undeclared output after."""
    from task_agents.shared import runtime

    entered = []

    def conforming(subject):
        entered.append(subject)
        return {"files": {"verdict.json": b'{"score": 1}'}, "evidence": {}}

    staged = node / "paper.txt"
    staged.write_bytes(b"STAGED")
    job = {"role": "probe", "mode": "judge", "inputs": {"paper": str(staged)}, "seed": 3}
    seat = types.SimpleNamespace(forward=conforming)
    with pytest.raises(io_contract.IoRefusalError):
        runtime._forward(seat, {**job, "inputs": {FOREIGN: str(staged)}})
    assert not entered, "the seat ran on an input the form never declared"
    with pytest.raises(io_contract.IoRefusalError):
        runtime._forward(types.SimpleNamespace(forward=lambda _: {"files": {FOREIGN: b""}}), job)
    with pytest.raises(io_contract.IoRefusalError, match="no `mode`"):
        runtime._forward(seat, {**job, "mode": ""})
    assert runtime._forward(seat, job)["files"] == {"verdict.json": b'{"score": 1}'}
    assert entered[0]["files"] == {"paper": b"STAGED"} and entered[0]["seed"] == 3
    assert set(entered[0]) == {"files", "seed", "chat", "solve", "review", "evidence"}
