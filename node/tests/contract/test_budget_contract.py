# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""The spend policy: what the agent is told about its allotment, and when it is warned."""

import pytest

from shared.gateway.budget import Budget


def test_the_policy_is_read_from_the_episode_environment(grant, monkeypatch):
    """Budget.declared reads the allotment, the exposed axes and the nudge the host stated."""
    declared = Budget.declared()
    assert (declared.tokens, declared.exposed, declared.nudge_at) == (1000, frozenset({"tokens"}), 0.5)
    monkeypatch.setenv("RQGM_AGENT_TOKEN_ALLOTMENT", "unbounded")
    monkeypatch.setenv("RQGM_AGENT_BUDGET_EXPOSE", "")
    monkeypatch.setenv("RQGM_AGENT_BUDGET_NUDGE", "off")
    unbounded = Budget.declared()
    assert (unbounded.tokens, unbounded.exposed, unbounded.nudge_at) == (None, frozenset(), None)
    with pytest.raises(ValueError, match="unknown exposed budget axes"):
        Budget(tokens=1, exposed=frozenset({"seconds"}), nudge_at=None)
    with pytest.raises(ValueError, match="nudge_at"):
        Budget(tokens=1, exposed=frozenset(), nudge_at=1.5)


def test_the_meters_reading_wins_and_the_remainder_is_told_in_tokens_and_turns():
    """remaining prefers the gateway's figure; exposure states tokens, then turns once a step is known."""
    policy = Budget(tokens=1000, exposed=frozenset({"tokens"}), nudge_at=None)
    assert policy.remaining(served=None, spent=100) == 900
    assert policy.remaining(served=850.0, spent=100) == 850.0
    assert policy.used(remaining=850.0, spent=100) == 150 and policy.used(remaining=None, spent=100) == 100
    assert policy.exposure(remaining_tokens=900) == ["~900 budget tokens remain"]
    assert policy.turns_left(remaining_tokens=900) is None, "no step is known before a turn is metered"
    assert policy.nudge(used_tokens=100, warned=set()) == [], "nudges are off, and the step is still recorded"
    assert policy.turns_left(remaining_tokens=800) == 8
    assert len(policy.exposure(remaining_tokens=800)) == 2, "tokens and turns, with nudges off"
    hidden = Budget(tokens=1000, exposed=frozenset(), nudge_at=None)
    assert hidden.exposure(remaining_tokens=900) == []
    assert Budget(tokens=None, exposed=frozenset({"tokens"}), nudge_at=None).exposure(remaining_tokens=None) == []


def test_the_binding_limit_is_named_as_turns_or_steps():
    """Whichever ends the episode first is the sentence: the allotment in turns, or the step count."""
    policy = Budget(tokens=1000, exposed=frozenset(), nudge_at=0.9)
    assert policy.binding(taken=2, step_limit=10, remaining_tokens=900).startswith("8 turn(s) left: step 2 of 10")
    policy.nudge(used_tokens=100, warned=set())
    assert policy.binding(taken=2, step_limit=10, remaining_tokens=700).startswith("~4 turn(s) left. The token allotment")
    assert policy.binding(taken=2, step_limit=0, remaining_tokens=None) == ""


def test_the_warning_comes_once_on_the_last_turn_before_the_threshold():
    """The nudge fires when the next turn's cost would cross nudge_at, and never twice."""
    policy = Budget(tokens=1000, exposed=frozenset(), nudge_at=0.5)
    warned = set()
    assert policy.nudge(used_tokens=100, warned=warned) == [], "the first reading has no step yet"
    assert policy.nudge(used_tokens=200, warned=warned) == []
    assert policy.nudge(used_tokens=300, warned=warned) == [], "300 + a step of 100 stays under 500"
    said = policy.nudge(used_tokens=420, warned=warned)
    assert len(said) == 1 and "tokens" in said[0] and "4 more full turn" in said[0] and warned == {"tokens"}
    assert policy.nudge(used_tokens=900, warned=warned) == []
    assert Budget(tokens=None, exposed=frozenset(), nudge_at=0.5).nudge(used_tokens=1, warned=set()) == []
