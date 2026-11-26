# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""Typed work-queue port between the controller and execution."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Literal, Protocol

from pydantic import PositiveInt
from pydantic.dataclasses import dataclass

from gm.domain import Artifact, EpisodeBinding, TrainSpec
from gm.domain.identity import EvaluatorSlot, RoleId, SampleKey, TaskId
from gm.domain.results import CONFIG, EvaluatorJudgment, InvalidJudgment, TaskFailure, TaskSuccess
from gm.frozen import FrozenMapping
from gm.ids import (
    ActionId,
    BindingTag,
    CriterionCoordinate,
    NodeId,
    NonBlank,
    Sha256Hex,
)
from gm.ledger import JudgedSubject
from gm.ledger.events import AttemptDialogue
from gm.spend import SpendRows

# The rule each criterion pins, keyed by coordinate. Frozen: callers hash an export spec.
type Pins = FrozenMapping[CriterionCoordinate, str]


@dataclass(frozen=True, slots=True, kw_only=True, config=CONFIG)
class ExportSpec:
    """The criterion pins an expansion was issued under, and the digest of the question it answers."""

    pins: Pins
    expected_hash: Sha256Hex


@dataclass(frozen=True, slots=True, kw_only=True, config=CONFIG)
class ExpansionJob:
    """Run the meta episode over the parent's experience view."""

    action: ActionId
    parent: NodeId
    export: ExportSpec
    binding: EpisodeBinding
    allowed_tags: frozenset[BindingTag] = frozenset()
    train: TrainSpec | None = None
    attempt_no: PositiveInt = 1


@dataclass(frozen=True, slots=True, config=CONFIG, kw_only=True)
class ExpansionCompletion:
    """Return a published child and its closed episode evidence."""

    child: NodeId | None
    result: TaskSuccess | TaskFailure


@dataclass(frozen=True, slots=True, kw_only=True, config=CONFIG)
class GenerationJob:
    """Run one role from the complete pin-selected node."""

    action: ActionId
    role: RoleId
    # Keyed by the name the seat reads bytes at, never by position; blind `input` keeps positions.
    inputs: Mapping[str, Artifact]
    seed: int
    node: NodeId
    task: TaskId
    pins: Pins


@dataclass(frozen=True, slots=True, kw_only=True, config=CONFIG)
class JudgmentJob:
    """Score one subject with one declared scorer slot."""

    action: ActionId
    node: NodeId

    slot: EvaluatorSlot
    subject: JudgedSubject
    sample: SampleKey
    # A failed production still owes its cover a host-minted `0.0`; it never reaches an evaluator episode.
    produced: TaskSuccess | TaskFailure
    role: RoleId | None
    pins: Pins
    # A minted member keeps its provenance key as `sample`; staging reads the original corpus row.
    source_sample: SampleKey | None = None
    # The host-materialized inputs a selected train sample's producer received, for its judge only.
    source_inputs: tuple[Artifact, ...] | None = None
    source_channels: Mapping[str, Artifact] | None = None


type Job = ExpansionJob | GenerationJob | JudgmentJob


@dataclass(frozen=True, slots=True, config=CONFIG, kw_only=True)
class Skipped:
    """Record infrastructure retry exhaustion as a soft skip."""

    attempts: int
    error: NonBlank
    spend: SpendRows = ()
    # What every exhausted attempt said, empty when none reached a model.
    dialogues: tuple[AttemptDialogue, ...] = ()


type Completion = (
    ExpansionCompletion | TaskSuccess | TaskFailure | EvaluatorJudgment | InvalidJudgment | Skipped
)


#: The issuing scope of the run's own search, and of a node-triggered train evaluation.
SEARCH_SCOPE, TRAIN_SCOPE = "search", "train"

#: The capacity classes a running job is charged against, highest priority first.
TRAIN_CLASS: Final = "train"
EXPANSION_CLASS: Final = "expansion"
VALIDATION_CLASS: Final = "validation"
CLASS_ORDER: Final = {EXPANSION_CLASS: 0, TRAIN_CLASS: 1, VALIDATION_CLASS: 2}


def job_class(*, scope: str, expansion: bool) -> str:
    """Return the capacity class one job is charged against."""
    if expansion:
        return EXPANSION_CLASS
    return TRAIN_CLASS if scope == TRAIN_SCOPE else VALIDATION_CLASS


@dataclass(frozen=True, slots=True, kw_only=True, config=CONFIG)
class Ticket:
    """The capacity class a job dispatches under, and the expansion a train seat belongs to."""

    kind: str
    owner: str | None = None
    key: str = ""
    batch: str | None = None


@dataclass(frozen=True, slots=True, config=CONFIG, kw_only=True)
class CompletionEnvelope:
    """Carry one completion with its originating job."""

    job: Job
    outcome: Completion


EXPANSION_QUEUED: Final = "queued"
EXPANSION_STARTED: Final = "started"
EXPANSION_FINISHED: Final = "finished"
type ExpansionState = Literal["queued", "started", "finished"]


@dataclass(frozen=True, slots=True, config=CONFIG, kw_only=True)
class ExpansionExecution:
    """Each open expansion's execution state, and whether another expansion could start now.

    A finished expansion's completion has arrived but is not yet applied to the ledger.
    """

    states: FrozenMapping[ActionId, ExpansionState]
    available: bool


class WorkQueue(Protocol):
    """Submit jobs; receive completions in arrival order.

    Another executor plugs in by implementing these methods.
    """

    def submit(self, job: Job, *, ticket: Ticket | None = None) -> None:
        """Submit one closed job under its capacity ticket."""
        ...

    def next_completion(self, timeout: float | None) -> CompletionEnvelope | None:
        """Return the next arrival, blocking up to `timeout` seconds (None: until one arrives)."""
        ...

    def expansion_execution(self, *, actions: frozenset[ActionId]) -> ExpansionExecution:
        """Return the execution state of exactly these open expansions, and the expansion room."""
        ...
