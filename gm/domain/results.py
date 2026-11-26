# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""Define the closed set of task and evaluator result values."""

from __future__ import annotations

from dataclasses import field
from enum import StrEnum
from typing import TYPE_CHECKING, Self, assert_never

from pydantic import ConfigDict, JsonValue, NonNegativeInt, model_validator
from pydantic.dataclasses import dataclass

from gm.domain.identity import ScoringRuleId
from gm.frozen import FrozenMapping, frozen_default
from gm.ids import ArtifactOid, Finite, NonBlank, UnitInterval
from gm.spend import Amount, AxisName, SpendRows

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

CONFIG = ConfigDict(extra="forbid", strict=True, ser_json_bytes="base64", val_json_bytes="base64")


@dataclass(frozen=True, slots=True, config=CONFIG, kw_only=True)
class InlineArtifact:
    """Carry a payload by value, up to the ceiling past which the ledger uses a ref."""

    data: bytes


@dataclass(frozen=True, slots=True, config=CONFIG, kw_only=True)
class ArtifactRef:
    """Identify an out-of-line payload by its content address."""

    oid: ArtifactOid
    size: NonNegativeInt


type Artifact = InlineArtifact | ArtifactRef


def artifact_address(*, artifact: Artifact) -> tuple[ArtifactOid, int]:
    """Return an artifact's content address and byte size."""
    match artifact:
        case InlineArtifact(data=data):
            return ArtifactOid.of_bytes(data), len(data)
        case ArtifactRef(oid=oid, size=size):
            return oid, size
        case _:
            assert_never(artifact)


def resolve(artifact: Artifact, fetch: Callable[[ArtifactOid], bytes]) -> bytes:
    """Return an artifact handle's bytes, reaching ``fetch`` only for a reference."""
    match artifact:
        case InlineArtifact(data=data):
            return data
        case ArtifactRef(oid=oid):
            return fetch(oid)
        case _:
            assert_never(artifact)


class FailureReason(StrEnum):
    """Define the closed classification vocabulary for candidate failures."""

    AGENT_ERROR = "agent-error"
    MALFORMED_OUTPUT = "malformed-output"
    TOKEN_CAP = "token-cap"
    TIMEOUT = "timeout"
    RETRY_EXHAUSTED = "retry-exhausted"
    UNKNOWN = "unknown"


class ExpansionOutcome(StrEnum):
    """Define the closed vocabulary for why an expansion published a node, or didn't."""

    PUBLISHED = "published"
    DECLINED = "declined"
    DUPLICATE = "duplicate"
    FAILED = "failed"


class RowKind(StrEnum):
    """Define the closed vocabulary a results row is classified under."""

    MEASURED = "MEASURED"
    FAILED = "FAILED"
    REFUSED = "REFUSED"
    UNFAIR = "UNFAIR"


class RefusalCode(StrEnum):
    """Say why the host refused before the seat ran, in the only seven words there are."""

    NO_OUTPUT = "no_output"
    TIMEOUT = "timeout"
    GRADER_ERROR = "grader_error"
    IMAGE_REFUSED = "image_refused"
    STAGING_REFUSED = "staging_refused"
    GATEWAY_REFUSED = "gateway_refused"
    KILLED = "killed"


def row_kind(*, reason: FailureReason) -> tuple[RowKind, RefusalCode | None]:
    """Return the classification one candidate-failure reason earns on the record."""
    match reason:
        case FailureReason.AGENT_ERROR | FailureReason.MALFORMED_OUTPUT | FailureReason.TOKEN_CAP:
            return RowKind.FAILED, None
        case FailureReason.TIMEOUT:
            return RowKind.REFUSED, RefusalCode.TIMEOUT
        case FailureReason.RETRY_EXHAUSTED | FailureReason.UNKNOWN:
            return RowKind.REFUSED, RefusalCode.NO_OUTPUT
        case _:
            assert_never(reason)


@dataclass(frozen=True, slots=True, config=CONFIG, kw_only=True)
class TaskSuccess:
    """Record one task execution that produced output."""

    files: FrozenMapping[NonBlank, Artifact]
    spend: SpendRows
    metadata: FrozenMapping[NonBlank, Artifact] = field(default_factory=frozen_default)


@dataclass(frozen=True, slots=True, config=CONFIG, kw_only=True)
class TaskFailure:
    """Record one failed task execution without a score."""

    files: FrozenMapping[NonBlank, Artifact]
    error: NonBlank
    reason: FailureReason
    spend: SpendRows


@dataclass(frozen=True, slots=True, kw_only=True, config=CONFIG)
class EvaluatorJudgment:
    """Record one successful evaluator judgment over an outcome."""

    files: FrozenMapping[NonBlank, Artifact]
    spend: SpendRows
    score: int | Finite
    outcome: UnitInterval
    scoring_rule: ScoringRuleId
    utilities: FrozenMapping[NonBlank, UnitInterval] = field(default_factory=frozen_default)
    units: FrozenMapping[AxisName, Amount] = field(default_factory=frozen_default)
    labels: FrozenMapping[NonBlank, JsonValue] = field(default_factory=frozen_default)
    declared_outputs: FrozenMapping[NonBlank, NonBlank] | None = None
    kind: RowKind = RowKind.MEASURED
    refusal_code: RefusalCode | None = None

    @model_validator(mode="after")
    def _classified(self) -> Self:
        return self

    def utility(self, *, interpretation: str) -> float:
        """Return what one declared interpretation made of this raw score."""
        return stated(self.utilities, "utility", interpretation, self.scoring_rule)

    def label(self, name: str) -> JsonValue:
        """Return the domain's own eligibility answer under one declared label."""
        return stated(self.labels, "label", name, self.scoring_rule)


@dataclass(frozen=True, slots=True, config=CONFIG, kw_only=True)
class InvalidJudgment:
    """Record a completed evaluator attempt that produced no verdict."""

    files: FrozenMapping[NonBlank, Artifact]
    spend: SpendRows
    error: NonBlank


FORFEIT = 0.0
FORFEIT_NOTE = "forfeit.log"


def forfeit(
    *, failure: TaskFailure, scoring_rule: ScoringRuleId, interpretations: frozenset[str] = frozenset()
) -> EvaluatorJudgment:
    """Return the host-minted verdict on a task agent that raised."""
    kind, refusal_code = row_kind(reason=failure.reason)
    return EvaluatorJudgment(
        files=dict(failure.files),
        score=FORFEIT,
        outcome=0.0,
        scoring_rule=scoring_rule,
        utilities=dict.fromkeys(interpretations, 0.0),
        units={},
        labels={},
        spend=failure.spend,
        kind=kind,
        refusal_code=refusal_code,
    )


def stated[T](carried: Mapping[str, T], kind: str, name: str, subject: ScoringRuleId) -> T:
    """Select one thing a domain stated at settlement, and refuse what it didn't."""
    return carried[name]


type JudgmentResult = EvaluatorJudgment | InvalidJudgment


type TaskResult = TaskSuccess | TaskFailure
