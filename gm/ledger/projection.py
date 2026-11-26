# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""The fold: one object, updated once per ledger event, holding what decisions, settlement and verification read."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, assert_never

from gm.domain.results import InvalidJudgment, TaskSuccess
from gm.ids import CriterionCoordinate
from gm.ledger.events import (
    ActionKind,
    ActionReserved,
    Cell,
    CheckpointPublished,
    CriterionTransition,
    ExpansionRecorded,
    JudgmentRecorded,
    LabeledRef,
    NodeState,
    NodeStateRecorded,
    OutcomeRef,
    RunConfigured,
    SoftSkip,
    StoredEvent,
    TaskOutcome,
    settled_verdict,
)
from gm.ledger.judged_index import JudgedIndex
from gm.ledger.membership_index import MembershipIndex
from gm.ledger.schedule_index import ScheduleIndex
from gm.ledger.settings import Settings, settings
from gm.ledger.trial_index import TrialIndex

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Hashable, Iterable, Mapping, Sequence

    from gm.domain import JudgedWork, MintedSet
    from gm.domain.identity import EvaluatorSlot, RoleId, TaskId
    from gm.ids import ActionId, BindingTag, EventId, NodeId
    from gm.ledger.events import JudgedSubject, LedgerEvent, RunManifest
    from gm.spend import AxisName, SpendRows


class _Seat(NamedTuple):
    node: NodeId
    role: RoleId
    task: TaskId | None
    scope: str
    sample: str


@dataclass(frozen=True, slots=True, kw_only=True)
class _Issuing:
    seat: _Seat
    cover: ActionId


@dataclass(slots=True, kw_only=True)
class CoverIndex:
    """Which cover each open judgment belongs to, and how much of it is still open.

    Consecutive judgment reservations on one seat form one cover; any other event ends it.
    """

    cover: dict[ActionId, ActionId]
    outstanding: dict[ActionId, int]
    issuing: _Issuing | None

    @classmethod
    def empty(cls) -> CoverIndex:
        """Return the index a run begins with, holding no cover at all."""
        return cls(cover={}, outstanding={}, issuing=None)

    def witness(self, event: LedgerEvent) -> None:
        """Fold one event's effect on the fan-out now under issue."""
        if not isinstance(event, ActionReserved) or event.kind is not ActionKind.JUDGMENT:
            self.issuing = None
            return
        seat = _Seat(event.node, event.role, event.task, str(event.scope), str(event.sample))
        if self.issuing is not None and self.issuing.seat == seat:
            cover = self.issuing.cover
        else:
            cover = event.action
            self.outstanding[cover] = 0
        self.issuing = _Issuing(seat=seat, cover=cover)
        self.cover[event.action] = cover
        self.outstanding[cover] += 1

    def close(self, action: ActionId) -> None:
        """Drop one settled judgment from its cover, and the cover once it's empty."""
        if (cover := self.cover.pop(action, None)) is None:
            return
        if (left := self.outstanding[cover] - 1) == 0:
            del self.outstanding[cover]
        else:
            self.outstanding[cover] = left

    def remaining(self, action: ActionId) -> int:
        """Return how many seats of this action's cover are still open, itself included."""
        if (cover := self.cover.get(action)) is None:
            msg = f"no open judgment cover for action: {action}"
            raise ValueError(msg)
        return self.outstanding[cover]


@dataclass(slots=True)
class _Expansion:
    """One admitted expansion: its scope, and what has settled about it so far."""

    scope: str
    terminal: bool = False
    settled: bool = False
    final: bool = False


def _scope_kind(reservation: ActionReserved) -> tuple[str, ActionKind]:
    return (str(reservation.scope), reservation.kind)


def _authored(pins: Mapping[CriterionCoordinate, str], interpretation: CriterionCoordinate | None) -> tuple:
    """The pins that authored a raw verdict, without the interpretation, as a sorted key."""
    return tuple(sorted((str(axis), value) for axis, value in pins.items() if axis != interpretation))


class Fold:
    """A run's state folded from its ledger one event at a time; only `apply` writes it.

    `open_actions` and the open counts are replaced rather than changed in place, because train
    batches reserve from other threads while the decision loop iterates them.
    """

    def __init__(self) -> None:
        """Start from the empty ledger."""
        self.manifest: RunManifest | None = None
        #: The decision settings `manifest` fixes (`gm.ledger.settings`).
        self.settings: Settings | None = None
        #: Replaced, never changed in place, so a reservation can keep the vector it was issued under.
        self.active_pins: Mapping[CriterionCoordinate, str] = {}
        self.epochs: dict[CriterionCoordinate, int] = {}
        #: Reservations so far; the next action id is one past it.
        self.decisions = 0
        self.last: EventId | None = None
        self.last_seq = 0
        #: Reservations not yet settled, in reservation order, and the pins each was issued under.
        self.open_actions: dict[ActionId, ActionReserved] = {}
        self.issued_pins: dict[ActionId, Mapping[CriterionCoordinate, str]] = {}
        self.scopes: dict[ActionId, str] = {}
        self.outcomes: dict[EventId, TaskOutcome] = {}
        self.outcome_events: list[StoredEvent] = []
        #: Each cell's outcomes in append order.
        self.cell_outcomes: dict[Cell, list[tuple[EventId, TaskOutcome]]] = {}
        #: Valid verdicts on recorded outcomes, in append order, and the stored events carrying them.
        self.judgments: list[JudgmentRecorded] = []
        self.judgment_events: list[StoredEvent] = []
        #: Valid verdicts on labeled subjects, per judging slot.
        self.labeled: dict[EvaluatorSlot, list[tuple[EventId, JudgmentRecorded]]] = {}
        self.completed: Counter[tuple[Cell, str]] = Counter()
        self.spent: dict[tuple[str, BindingTag | None], dict[AxisName, int | float]] = {}
        self.lineage: dict[NodeId, NodeId] = {}
        #: Each node's publication ordinal (root 0), which names it in a run's draws.
        self.ordinals: dict[str, int] = {}
        #: Nodes whose final turn said they are done.
        self.done: set[NodeId] = set()
        self.expansion_attempts: dict[NodeId, list[StoredEvent]] = {}
        self.minted: dict[str, MintedSet] = {}
        self.retained: dict[str, tuple[MintedSet, ...]] = {}
        self.minted_by: dict[str, CriterionCoordinate] = {}
        self.covers = CoverIndex.empty()
        self.trials = TrialIndex()
        self.schedules = ScheduleIndex()
        self.members = MembershipIndex()
        self.judged_index = JudgedIndex()
        self._expansions: dict[ActionId, _Expansion] = {}
        self._verdicts: set[tuple[EvaluatorSlot, JudgedSubject, tuple]] = set()
        self._open_counts: dict[Callable[[ActionReserved], Hashable], dict[Hashable, int]] = {}
        self.track(_scope_kind)

    def apply(self, stored: StoredEvent) -> None:
        """Fold one event."""
        event = stored.event
        match event:
            case RunConfigured(manifest=configured):
                self.ordinals[str(configured.source)] = 0
                self.manifest = configured
                self.settings = settings(configured)
                self.active_pins = dict(configured.pins)
                self.epochs = dict.fromkeys(self.active_pins, 0)
            case ActionReserved() as reservation:
                self._reserve(reservation)
            case TaskOutcome() as outcome:
                self._outcome(outcome, stored)
            case JudgmentRecorded() as record:
                self._judgment(record, stored)
            case ExpansionRecorded() as grown:
                self._expansion(grown, stored)
            case NodeStateRecorded() as moved:
                if moved.state is NodeState.FINAL_TURN:
                    (self.done.add if moved.done else self.done.discard)(moved.node)
                    if (attempt := self._expansions.get(moved.action)) is not None:
                        attempt.final = True
            case CriterionTransition() as transition:
                self._transition(transition)
            case SoftSkip() as skipped:
                self._skip(skipped)
            case CheckpointPublished():
                pass
            case _:
                assert_never(event)
        self.covers.witness(event)
        self.last, self.last_seq = stored.event_id, stored.seq

    def _reserve(self, reservation: ActionReserved) -> None:
        action = reservation.action
        self.open_actions = {**self.open_actions, action: reservation}
        self._count(reservation, 1)
        self.schedules.reserve(action, self.active_pins.get(self.manifest.coordinates.dataset))
        self.issued_pins[action] = self.active_pins
        self.scopes[action] = str(reservation.scope)
        self.decisions += 1
        if reservation.kind is ActionKind.EXPANSION:
            self._expansions[action] = _Expansion(str(reservation.scope))

    def _close(self, action: ActionId) -> str | None:
        """Settle one reservation; return the dataset pin it was issued under."""
        held = dict(self.open_actions)
        self._count(held.pop(action), -1)
        self.open_actions = held
        self.issued_pins.pop(action)
        self.covers.close(action)
        return self.schedules.close(action)

    def _transition(self, transition: CriterionTransition) -> None:
        coordinate = transition.coordinate
        self.active_pins = {**self.active_pins, coordinate: transition.replacement}
        self.epochs[coordinate] += 1
        for one in transition.minted:
            self.minted[str(one.name)] = one
            self.minted_by[str(one.name)] = coordinate
            family = self.retained.get(str(one.name), ())
            if one not in family:
                self.retained[str(one.name)] = (*family, one)

    def _outcome(self, outcome: TaskOutcome, stored: StoredEvent) -> None:
        reservation = self.open_actions[outcome.action]
        cell = Cell(node=outcome.node, role=outcome.role, task=outcome.task)
        self._close(outcome.action)
        self.outcomes[stored.event_id] = outcome
        self.outcome_events.append(stored)
        self.members.log.append((stored.event_id, cell, outcome))
        self.cell_outcomes.setdefault(cell, []).append((stored.event_id, outcome))
        self.completed[cell, str(reservation.scope)] += 1
        if not isinstance(outcome.result, TaskSuccess):
            self.trials.settle_outcome(stored=stored, reservation=reservation, cell=cell, outcome=outcome)
        axes = self.manifest.coordinates
        self.schedules.settle_outcome(
            reservation=reservation,
            cell=cell,
            outcome=outcome,
            sample=outcome.pins.get(axes.sample),
            dataset=outcome.pins.get(axes.dataset),
        )
        self._charge(reservation, outcome.result.spend)

    def _judgment(self, record: JudgmentRecorded, stored: StoredEvent) -> None:
        reservation = self.open_actions[record.action]
        cell = (
            None
            if reservation.task is None
            else Cell(node=reservation.node, role=reservation.role, task=reservation.task)
        )
        self._close(record.action)
        self.members.log.append((stored.event_id, cell, record))
        axes = self.manifest.coordinates
        subject = record.subject
        if cell is not None:
            self.completed[cell, str(reservation.scope)] += 1
            if isinstance(subject, LabeledRef):
                self.schedules.settle_judgment(
                    reservation=reservation, cell=cell, record=record, dataset=record.pins.get(axes.dataset)
                )
        target = self.outcomes[subject.outcome] if isinstance(subject, OutcomeRef) else None
        if not isinstance(record.judgment, InvalidJudgment):
            self._verdicts.add((record.slot, subject, _authored(record.pins, axes.interpretation)))
            if isinstance(subject, OutcomeRef):
                self.judgments.append(record)
                self.judgment_events.append(stored)
            else:
                self.labeled.setdefault(record.slot, []).append((stored.event_id, record))
            self.judged_index.settle(
                scope=str(reservation.scope),
                record=record,
                event_id=stored.event_id,
                outcome=target,
                sample_axis=axes.sample,
            )
        self.trials.settle_judgment(stored=stored, reservation=reservation, record=record, cell=cell, target=target)
        self._charge(reservation, record.judgment.spend)

    def _expansion(self, grown: ExpansionRecorded, stored: StoredEvent) -> None:
        reservation = self.open_actions[grown.action]
        self._close(grown.action)
        self.expansion_attempts.setdefault(grown.parent, []).append(stored)
        attempt = self._expansions[grown.action]
        attempt.settled = True
        attempt.terminal |= grown.child is None
        if grown.child is not None:
            self.lineage[grown.child] = grown.parent
            self.ordinals.setdefault(str(grown.child), len(self.ordinals))
        self._charge(reservation, grown.result.spend)

    def _skip(self, skipped: SoftSkip) -> None:
        reservation = self.open_actions[skipped.action]
        if reservation.kind is ActionKind.EXPANSION:
            self._expansions[skipped.action].terminal = True
        self.schedules.settle_skip(
            reservation=reservation,
            cell=None
            if reservation.task is None
            else Cell(node=reservation.node, role=reservation.role, task=reservation.task),
            dataset=self._close(skipped.action),
        )
        self._charge(reservation, skipped.spend)

    def _charge(self, reservation: ActionReserved, spend: SpendRows) -> None:
        scope = str(reservation.scope)
        for tag, axis, amount in spend:
            bucket = self.spent.setdefault((scope, tag), {})
            bucket[axis] = bucket.get(axis, 0) + amount

    def _count(self, reservation: ActionReserved, step: int) -> None:
        for key, counts in self._open_counts.items():
            value = key(reservation)
            moved = {**counts, value: counts.get(value, 0) + step}
            if not moved[value]:
                del moved[value]
            self._open_counts[key] = moved

    def track(self, key: Callable[[ActionReserved], Hashable]) -> None:
        """Count open reservations per `key` value from now on; register before other threads reserve."""
        self._open_counts[key] = dict(Counter(map(key, self.open_actions.values())))

    def open_count(self, key: Callable[[ActionReserved], Hashable]) -> Mapping[Hashable, int]:
        """Return how many open reservations each value of a tracked `key` holds."""
        return self._open_counts[key]

    @property
    def open_kinds(self) -> Mapping[tuple[str, ActionKind], int]:
        """How many reservations are open per scope and kind."""
        return self._open_counts[_scope_kind]

    def total(self, axis: AxisName, scopes: Collection[str] | None = None) -> int | float:
        """Return one declared resource axis summed across selected scopes."""
        return sum(
            values.get(axis, 0) for (scope, _), values in self.spent.items() if scopes is None or scope in scopes
        )

    def counted_in(self, record: JudgmentRecorded | TaskOutcome) -> bool:
        """Return whether a stored pinned outcome or judgment currently counts."""
        return self.counted_for(record, None)

    def counted_for(self, record: JudgmentRecorded | TaskOutcome, identity: CriterionCoordinate | None) -> bool:
        """Return whether a record counts as evidence about one named candidate."""
        currency = self.manifest.coordinates.currency
        return all(
            coord == identity or not currency.get(coord, True) or self.active_pins.get(coord) == value
            for coord, value in record.pins.items()
        )

    def labeled_counted(self, cell: Cell, record: JudgmentRecorded | TaskOutcome) -> bool:
        """Return whether a labeled trial counts for its candidate cell."""
        return self.counted_for(record, CriterionCoordinate(str(cell.role)))

    def holds_verdict(
        self,
        *,
        slot: EvaluatorSlot,
        subject: JudgedSubject,
        pins: Mapping[CriterionCoordinate, str],
        interpretation: CriterionCoordinate | None,
    ) -> bool:
        """Return whether this slot already returned a valid raw verdict on this subject under these pins."""
        return (slot, subject, _authored(pins, interpretation)) in self._verdicts

    def expansion_completions(self, *, scopes: Collection[str]) -> tuple[bool, ...]:
        """Return terminal status in expansion admission order, including childless attempts.

        A published child needs a final turn as well as its settlement before it counts as complete.
        """
        return tuple(
            one.terminal or (one.settled and one.final) for one in self._expansions.values() if one.scope in scopes
        )

    def judged_streams(
        self, *, scopes: Collection[str]
    ) -> tuple[Sequence[JudgedWork], Mapping[EvaluatorSlot, Sequence[JudgedWork]]]:
        """Return the settled judgment facts a transition rule reads."""
        return self.judged_index.streams(projection=self, scopes=scopes)

    def meaning(self, record: JudgmentRecorded, pins: Mapping[CriterionCoordinate, str]) -> float:
        """Return what one verdict means under ``pins``; the single selection."""
        axes = self.manifest.coordinates
        verdict = settled_verdict(record=record)
        if (value := self.minted_interpretation(record.pins.get(axes.sample))) is not None:
            return verdict.utility(interpretation=value)
        if (seat := self.manifest.seat_interpretations.get(record.slot)) is not None:
            return verdict.utility(interpretation=seat)
        interpretation = axes.interpretation
        if interpretation is None or (active := pins.get(interpretation)) is None:
            return verdict.outcome
        return verdict.utility(interpretation=active)

    def effective(
        self, record: JudgmentRecorded, *, active_pins: Mapping[CriterionCoordinate, str] | None = None
    ) -> float:
        """Return the verdict's counted meaning; the fold assigns no other."""
        axes = self.manifest.coordinates
        active_vector = self.active_pins if active_pins is None else active_pins
        interpretation = axes.interpretation
        if (
            interpretation is not None
            and record.pins.get(interpretation) == active_vector.get(interpretation)
            and self.minted_interpretation(record.pins.get(axes.sample)) is None
        ):
            return settled_verdict(record=record).outcome
        return self.meaning(record, active_vector)

    def minted_interpretation(self, sample: str | None) -> str | None:
        """Return the minted-set interpretation owning this sample, if any."""
        for minted in self.minted.values():
            if minted.interpretation is not None and sample in minted.members:
                return minted.interpretation
        return None

    def state(self) -> tuple[object, ...]:
        """The folded facts any two folds of the same events agree on."""
        return (
            self.manifest, self.active_pins, self.epochs, self.decisions, self.open_actions, self.open_kinds,
            self.scopes, self.outcomes, self.cell_outcomes, self.judgments, self.labeled, self.completed,
            self.spent, self.lineage, self.done, self.minted, self.retained, self.minted_by, self.covers,
        )



def project(events: Iterable[StoredEvent]) -> Fold:
    """Return a fresh fold of an event prefix."""
    fold = Fold()
    for stored in events:
        fold.apply(stored)
    return fold


class ClaimKey(NamedTuple):
    """The cell and sample one open judgment reservation authors a verdict for."""

    node: NodeId
    role: RoleId
    task: TaskId | None
    sample: str


def claimed(projection: Fold, *, scopes: Collection[str]) -> frozenset[ClaimKey]:
    """Return the seats a judgment is already reserved for and not yet settled."""
    return frozenset(
        ClaimKey(r.node, r.role, r.task, str(r.sample))
        for r in projection.open_actions.values()
        if r.kind is ActionKind.JUDGMENT and r.sample is not None and str(r.scope) in scopes
    )
