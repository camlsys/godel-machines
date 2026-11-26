# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""The settlement wire: the host runs a form's `settle.py` over JSON members and reads one verdict per member."""

from __future__ import annotations

import json
import subprocess
from base64 import b64decode, b64encode
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field

from gm.domain.identity import SampleKey, ScoringRuleId
from gm.domain.results import EvaluatorJudgment, InlineArtifact, TaskSuccess
from gm.ids import ArtifactOid, canonical_json, require_safe_relpath
from gm.pointer import materialized

_STDERR_KEPT = 2048
#: The exit code a settlement uses for host weather: it failed to stage, the candidate is not at fault.
WEATHER_EXIT = 75
JUDGMENT_KEYS = frozenset({"score", "outcome", "rule", "evidence", "utilities", "units", "labels"})
WIRE_KEYS = frozenset({"score"})
GENERATION_KEYS = frozenset({"outputs", "evidence"})
MEMBER_KEY = "member"
MEMBERS_KEY = "members"


class ScorerRefusalError(Exception):
    """A settlement process exited non-zero on a membership."""


def _covered(offered: Sequence[str], required: AbstractSet[str], called: str, noun: str) -> None:
    counted = Counter(offered)
    if repeated := sorted(name for name, times in counted.items() if times > 1):
        raise ValueError(f"{called} returned {noun} more than once: {repeated}")
    if (keys := frozenset(counted)) != frozenset(required):
        missing, extra = sorted(frozenset(required) - keys), sorted(keys - frozenset(required))
        raise ValueError(f"{called} returned {noun} missing={missing} unexpected={extra}")


def batch_answers(
    *, returned: object, requested: AbstractSet[str], declared: AbstractSet[str], called: str
) -> dict[str, dict[str, object]]:
    """Return one batch reply's per-member answers, refusing a missing, repeated or extra member or key."""
    answers: list[tuple[str, dict[str, object]]] = []
    for offered in returned:
        member = offered.get(MEMBER_KEY)
        stated = {key: value for key, value in offered.items() if key != MEMBER_KEY}
        _covered([str(key) if isinstance(key, str) else repr(key) for key in stated], declared,
                 f"{called} answering for {member!r}", "keys")
        answers.append((member, stated))
    _covered([member for member, _ in answers], requested, called, "members")
    return dict(answers)


def _judgment(answer: Mapping[str, object]) -> EvaluatorJudgment:
    files = {name: InlineArtifact(data=b64decode(payload, validate=True)) for name, payload in answer["evidence"].items()}
    for name in files:
        require_safe_relpath(path=name)
    return EvaluatorJudgment(
        files=files,
        spend=(),
        score=answer["score"],
        outcome=float(answer["outcome"]),
        scoring_rule=ScoringRuleId(answer["rule"]),
        utilities={name: float(value) for name, value in answer["utilities"].items()},
        units={name: float(value) for name, value in answer["units"].items()},
        labels=dict(answer["labels"]),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ScoredMember:
    """One member of a scoring membership, and everything scoring it needs."""

    member: str
    sample: SampleKey | None
    produced: TaskSuccess
    channels: Mapping[str, bytes] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class ProcessInstrument:
    """A form's settlement process, `python settle.py <slot>` run in the form's folder."""

    argv: tuple[str, ...]
    home: str
    deadline_seconds: float

    def score(
        self, members: Sequence[ScoredMember], *, fetch: Callable[[ArtifactOid], bytes]
    ) -> dict[str, EvaluatorJudgment]:
        """Return the settlement's verdict for every member: a fixed judge scoring produced files."""
        return self._answers([
            {
                MEMBER_KEY: str(one.member),
                "sample": None if one.sample is None else str(one.sample),
                "files": {
                    name: b64encode(materialized(data=artifact.data, fetch=fetch)).decode("ascii")
                    for name, artifact in one.produced.files.items()
                },
                "channels": {name: value.decode() for name, value in one.channels.items()},
            }
            for one in members
        ])

    def finish(
        self, sample: SampleKey, judgment: EvaluatorJudgment, *, channels: Mapping[str, bytes], member: str
    ) -> EvaluatorJudgment:
        """Return the settlement's verdict over a node judge's raw score."""
        finished = self._answers([{
            MEMBER_KEY: member,
            "sample": str(sample),
            "score": judgment.score,
            "channels": {name: value.decode() for name, value in channels.items()},
        }])[member]
        return EvaluatorJudgment(
            files={**dict(judgment.files), **dict(finished.files)},
            spend=judgment.spend,
            score=finished.score,
            outcome=finished.outcome,
            scoring_rule=finished.scoring_rule,
            utilities=finished.utilities,
            units=finished.units,
            labels=finished.labels,
        )

    def _answers(self, members: Sequence[Mapping[str, object]]) -> dict[str, EvaluatorJudgment]:
        called = " ".join(self.argv)
        named = [str(body[MEMBER_KEY]) for body in members]
        if repeated := sorted(name for name, times in Counter(named).items() if times > 1):
            raise ValueError(f"the request to {called} names a member more than once: {repeated}")
        done = subprocess.run(
            self.argv, input=canonical_json(list(members)).encode(), capture_output=True, check=False,
            cwd=self.home, timeout=self.deadline_seconds,
        )
        if done.returncode != 0:
            detail = done.stderr.decode(errors="replace").strip()[-_STDERR_KEPT:]
            raise ScorerRefusalError(f"{called} exited {done.returncode}: {detail}")
        answers = batch_answers(
            returned=json.loads(done.stdout), requested=frozenset(named), declared=JUDGMENT_KEYS, called=called,
        )
        return {member: _judgment(answer) for member, answer in answers.items()}
