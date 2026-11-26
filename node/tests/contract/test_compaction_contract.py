# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""Context fitting: a history that no longer fits is compacted, the pinned tier never touched."""

import pytest

from shared.gateway import compaction, window
from shared.gateway.estimate import Estimator

PINNED = [
    compaction.pin(message={"role": "system", "content": "the instructions"}),
    compaction.pin(message={"role": "user", "content": "the task"}),
]


def _turn(n, size):
    """One tool-calling assistant turn and its long tool result."""
    return [
        {"role": "assistant", "content": f"step {n}", "tool_calls": [{"id": f"c{n}", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": f"c{n}", "content": "x" * size},
    ]


def _history(turns, size=2000):
    return PINNED + [message for n in range(turns) for message in _turn(n, size)]


def test_units_pair_a_call_with_its_replies_and_pairing_is_checked():
    """A tool-calling assistant and its tool replies move as one unit."""
    history = _history(2)
    assert compaction.units(messages=history) == [[0], [1], [2, 3], [4, 5]]
    assert compaction.pairing_valid(messages=history)
    assert not compaction.pairing_valid(messages=history[:3])
    assert not compaction.pairing_valid(messages=[{"role": "tool", "content": "orphan"}])


def test_a_fitting_history_is_left_alone_and_an_overflowing_one_is_compacted_under_the_target():
    """compact() is idle under the limit and lands under the limit over it, with the tiers it promises kept."""
    estimator = Estimator()
    config = compaction.CompactionConfig(limit_tokens=4000, estimator=estimator)
    small = _history(1, size=100)
    kept, report = compaction.compact(messages=small, config=config)
    assert kept == small and not report.fired and report.before == report.after
    big = _history(12)
    fitted, report = compaction.compact(messages=big, config=config)
    assert report.fired and report.after <= config.limit_tokens < report.before
    assert fitted[:2] == big[:2], "the pinned tier is never clipped, cleared or dropped"
    assert fitted[-2:] == big[-2:], "the verbatim tail survives"
    assert estimator.messages(messages=fitted) == report.after
    assert compaction.pairing_valid(messages=[{k: v for k, v in m.items() if k != compaction.PIN} for m in fitted])
    assert report.clips + report.clears + report.dropped > 0


def test_clipping_keeps_a_head_a_marker_and_a_tail():
    """clip() shortens to the token ceiling with the middle elided and says how much went."""
    estimator = Estimator()
    text = "".join(f"line {n}\n" for n in range(500))
    short = compaction.clip(text=text, max_tokens=50, estimator=estimator)
    assert len(short) <= int(50 * estimator.ratio) and "chars elided" in short
    assert short.startswith("line 0") and short.endswith("line 499\n")
    assert compaction.clip(text="tiny", max_tokens=50, estimator=estimator) == "tiny"


def test_a_history_no_stage_can_fit_raises_context_overflow():
    """Even maximal clipping cannot shrink a pinned tier past the limit: the loop is told, not fed."""
    estimator = Estimator()
    config = compaction.CompactionConfig(limit_tokens=200, estimator=estimator)
    huge = [compaction.pin(message={"role": "system", "content": "p" * 5000}), *_history(3)[2:]]
    with pytest.raises(compaction.ContextOverflowError, match="context overflow") as overflow:
        compaction.compact(messages=huge, config=config)
    assert overflow.value.limit == 200 and overflow.value.pinned > 200


def test_a_summarizer_stands_in_for_a_dropped_span_and_falls_back_when_it_fails():
    """A dropped span becomes the model's summary once; a failing summarizer drops the span plain."""
    notes = []
    summarizer = compaction.Summarizer(chat=lambda **_: type("R", (), {"content": "what happened"})(), note=lambda **kw: notes.append(kw))
    config = compaction.CompactionConfig(limit_tokens=3000, estimator=Estimator(), summarizer=summarizer)
    fitted, report = compaction.compact(messages=_history(12), config=config)
    assert report.dropped > 0 and report.summarized and summarizer.calls == 1, "one model call per compaction"
    assert any("what happened" in str(m.get("content")) for m in fitted)
    broken = compaction.Summarizer(chat=lambda **_: (_ for _ in ()).throw(RuntimeError("down")), note=lambda **kw: notes.append(kw))
    assert broken(history="span") == "" and broken.failed and "down" in notes[-1]["body"]


def test_the_window_applies_the_declared_ceilings_or_none():
    """Window fits to sequence minus output minus overhead; an unbounded binding never compacts."""
    estimator = Estimator()
    open_window = window.Window(sequence=None, output=None, estimator=estimator)
    assert not open_window.declared and open_window.config(overhead=0) is None
    assert open_window.fitted(messages=_history(12)) == (_history(12), None)
    bounded = window.Window(sequence=5000, output=1000, estimator=estimator)
    built = bounded.config(overhead=100)
    assert built.limit_tokens == compaction.limit_tokens(sequence=5000, output=1000, overhead=100) == 3900
    assert compaction.limit_tokens(sequence=4000, output=None, overhead=0) == 3000
    fitted, report = bounded.fitted(messages=_history(12), eager=True)
    assert report.fired and report.after <= built.limit_tokens
    assert window.Window(sequence=5000, output=None, estimator=estimator).fitted(messages=_history(1, 10))[1].fired is False
