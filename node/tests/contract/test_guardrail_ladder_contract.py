# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""The guardrail recovery ladder, driven from inside the node that carries it."""

import hashlib
import pathlib
import sys

import pytest

ROOT = next(
    parent
    for parent in pathlib.Path(__file__).absolute().parents
    if (parent / "task_agents").is_dir()
)

SHARED = ROOT / "shared"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OBSERVED = (
    "ContentPolicyViolationError: Invalid prompt: your prompt was flagged as"
    " potentially violating our usage policy"
)

WRAPPED = f"gateway refused (400) after 1 attempt: {OBSERVED!r}"

TRANSIENT = "gateway refused (429) after 8 attempts, waited 14.0s: 'rate limit exceeded'"

FILES = {
    "solution.py": b"def solve():\n    return '\\u2713'\n",
    "notes/measured.txt": "yield 92 %, \u0394H = -57 kJ/mol\n\"quoted\"\n".encode(),
    "notes/deep/nested.bin": bytes(range(256)),
}

def _module():
    """Return the carried `refusal` module, skipping where this form ships no gateway."""
    if not (SHARED / "gateway" / "refusal.py").is_file():
        pytest.skip("this form carries no shared/gateway/refusal.py")
    from shared.gateway import refusal

    return refusal

def _refused():
    """Return the carried `GatewayRefused`, skipping where this form ships no gateway."""
    if not (SHARED / "gateway" / "client.py").is_file():
        pytest.skip("this form carries no shared/gateway/client.py")
    from shared.gateway.client import GatewayRefused

    return GatewayRefused

def _writes(tree, files=FILES):
    """Fill a worktree with the bytes an episode is supposed to keep."""
    for name, payload in files.items():
        path = tree / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

def _digest(tree):
    """Return every file under a tree, keyed by relative path, hashed by content."""
    return {
        str(path.relative_to(tree).as_posix()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(tree.rglob("*"))
        if path.is_file()
    }

@pytest.fixture
def tree(tmp_path, monkeypatch):
    """Return a throwaway worktree, entered, with the cwd restored afterwards."""
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    monkeypatch.chdir(worktree)
    return worktree

def test_the_ladder_hands_out_compact_then_clear_then_publish_in_that_order():
    """The order is the cheap recovery first: compaction, then a clear, then publication."""
    refusal = _module()
    ladder = refusal.Ladder()

    handed = [ladder.next(), ladder.next(), ladder.next()]

    assert handed == [refusal.Rung.COMPACT, refusal.Rung.CLEAR, refusal.Rung.PUBLISH], (
        "the ladder no longer escalates compact -> clear -> publish; it handed out"
        f" {[rung.name for rung in handed]}. A skipped rung either discards a conversation"
        " compaction could have saved, or ends an episode that had a cheaper way out."
    )
    assert [rung.value for rung in handed] == ["compact", "clear", "publish"], (
        "a rung's published value changed; the wiring in solve.py and entry.py branches on"
        f" these strings, so the branch goes silently unreached: {[r.value for r in handed]}"
    )

def test_a_refusal_after_a_clear_publishes_forever_and_never_returns_to_compaction():
    """Once cleared, every further guardrail verdict publishes. The ladder never resets."""
    refusal = _module()
    ladder = refusal.Ladder()
    ladder.next()
    ladder.next()
    assert ladder.cleared, "the ladder does not report a clear it just handed out"

    later = [ladder.next() for _ in range(12)]

    assert set(later) == {refusal.Rung.PUBLISH}, (
        "the ladder walked back down after clearing; it handed out"
        f" {sorted({rung.name for rung in later})}. An episode whose STARTING PROMPT trips"
        " the guardrail would then clear itself forever and finish with nothing published."
    )
    assert ladder.cleared, (
        "`cleared` went back to False after further refusals; escalation is per-episode and"
        " monotonic, so a caller reading it would re-run a clear it already paid for"
    )
    assert ladder.taken.count(refusal.Rung.CLEAR) == 1, (
        f"the episode cleared more than once: {[rung.name for rung in ladder.taken]}"
    )
    assert ladder.taken.count(refusal.Rung.COMPACT) == 1, (
        f"the episode compacted more than once: {[rung.name for rung in ladder.taken]}"
    )

def test_each_episode_starts_its_own_ladder_at_the_foot():
    """Escalation is PER-EPISODE: a fresh ladder has taken nothing and begins at compaction."""
    refusal = _module()
    spent = refusal.Ladder()
    for _ in range(3):
        spent.next()

    fresh = refusal.Ladder()

    assert fresh.taken == (), (
        f"a new episode's ladder already reports rungs taken: {[r.name for r in fresh.taken]}"
    )
    assert not fresh.cleared, "a new episode's ladder already reports itself cleared"
    assert fresh.next() is refusal.Rung.COMPACT, (
        "a new episode does not start at compaction; ladder state is leaking between"
        " episodes, so a later episode publishes on a verdict compaction could have cleared"
    )

def test_taken_reports_the_rungs_in_the_order_they_were_handed_out():
    """`taken` is the audit trail: same rungs, same order, growing one entry per refusal."""
    refusal = _module()
    ladder = refusal.Ladder()
    expected = []

    for rung in (refusal.Rung.COMPACT, refusal.Rung.CLEAR, refusal.Rung.PUBLISH, refusal.Rung.PUBLISH):
        handed = ladder.next()
        expected.append(rung)
        assert handed is rung, f"expected {rung.name} at call {len(expected)}, got {handed.name}"
        assert ladder.taken == tuple(expected), (
            "`taken` disagrees with the rungs actually handed out:"
            f" {[r.name for r in ladder.taken]} against {[r.name for r in expected]}."
            " A reader cannot tell from the log why this episode published."
        )

    assert isinstance(ladder.taken, tuple), (
        f"`taken` is a {type(ladder.taken).__name__}, not the tuple the contract names;"
        " a mutable list would let a caller edit an episode's own history"
    )

def test_the_real_observed_guardrail_verdict_is_a_guardrail_refusal():
    """A provider's real refusal string, verbatim, must enter the ladder."""
    refusal = _module()
    GatewayRefused = _refused()

    bare = GatewayRefused(OBSERVED, status=400)
    wrapped = GatewayRefused(WRAPPED, status=400)

    assert refusal.guardrail_refusal(error=bare), (
        f"the observed content-policy verdict is not recognised: {OBSERVED!r}."
        " The episode that produced it would end and discard its finished worktree."
    )
    assert refusal.guardrail_refusal(error=wrapped), (
        "the verdict is not recognised once the client has wrapped it, which is the ONLY"
        f" shape the ladder ever sees at runtime: {WRAPPED!r}"
    )

def test_transient_weather_and_ordinary_bugs_stay_out_of_the_ladder():
    """SCOPE: only a content-policy verdict escalates. Weather and defects must not."""
    refusal = _module()
    GatewayRefused = _refused()

    rate_limited = GatewayRefused(TRANSIENT, status=429)
    exhausted = GatewayRefused("gateway refused: retries exhausted without a verdict", status=0)
    unavailable = GatewayRefused("gateway refused (503): 'service unavailable'", status=503)
    bug = ValueError("solve() got an unexpected keyword argument")
    disguised = ValueError(OBSERVED)

    for error in (rate_limited, exhausted, unavailable, bug, disguised):
        assert not refusal.guardrail_refusal(error=error), (
            f"{type(error).__name__}({str(error)!r}) was routed into the guardrail ladder."
            " A transient would clear a conversation over a capacity blip; a bug would be"
            " retried instead of surfaced. These leave as weather or as failures, not here."
        )

def test_a_clear_keeps_every_byte_the_episode_already_wrote(tree):
    """A clear drops HISTORY and keeps the WORKTREE. This is the claim the feature exists for."""
    refusal = _module()
    _writes(tree)
    before = _digest(tree)
    assert before, "the fixture staged no files, so this test would pass over an empty tree"

    pinned = [{"role": "system", "content": "instructions"}, {"role": "user", "content": "the task"}]
    history = pinned + [{"role": "assistant", "content": f"turn {turn}"} for turn in range(40)]

    ladder = refusal.Ladder()
    seen = []
    while True:
        rung = ladder.next()
        seen.append(rung)
        if rung is refusal.Rung.PUBLISH:
            break
        if rung is refusal.Rung.COMPACT:
            history = pinned + history[-4:]
        else:
            history = list(pinned)
        assert _digest(tree) == before, (
            f"the worktree changed while taking the {rung.value} rung. The work an episode"
            " already did must survive escalation, because publication is what keeps it."
        )

    assert seen == [refusal.Rung.COMPACT, refusal.Rung.CLEAR, refusal.Rung.PUBLISH], (
        f"the episode did not escalate through every rung: {[rung.name for rung in seen]}"
    )
    assert history == pinned, (
        "a clear left history beyond the starting prompt; the accumulated bytes that tripped"
        f" the guardrail would still be sent: {history}"
    )
    assert _digest(tree) == before, (
        "the worktree is not byte-identical after a full escalation to publication. The node"
        " would publish work it had already destroyed."
    )
    for name, payload in FILES.items():
        assert (tree / name).read_bytes() == payload, (
            f"{name} did not read back byte-identical after the clear; the episode's own"
            " output was rewritten by a step that is only allowed to drop conversation"
        )
