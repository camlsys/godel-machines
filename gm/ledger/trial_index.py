# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""Maintain the cell aggregate every decision reads, one event at a time."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from gm.domain.identity import SampleKey
from gm.domain.results import InvalidJudgment, TaskSuccess
from gm.ids import CriterionCoordinate
from gm.ledger.events import OutcomeRef, TaskOutcome

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence

    from gm.ids import EventId
    from gm.ledger.events import ActionReserved, Cell, JudgmentRecorded, StoredEvent
    from gm.ledger.projection import Fold


class Trial(NamedTuple):
    """One counted contribution and the evidence mass it carries."""

    value: float | None
    mass: float


_JUDGED = (0,)
_FAILED = (2,)


class CellEvidence(NamedTuple):
    """Each cell's aggregated evidence, and the events that measured it."""

    evidence: Mapping[Cell, object]
    measured: Mapping[Cell, set[EventId]]


class _Trial(NamedTuple):
    scope: str
    cell: Cell
    record: JudgmentRecorded | TaskOutcome
    identity: CriterionCoordinate | None
    gate: TaskOutcome | None
    measured: EventId | None
    rank: tuple[int, ...] | None


class TrialIndex:
    """The ``Cell`` to trials aggregate, maintained one event at a time."""

    def __init__(self) -> None:
        """Initialize an empty aggregate over an empty log."""
        self._log: list[_Trial] = []
        self._slots: dict[object, int] = {}
        self._key: object = None
        self._cursor = 0
        self._buckets: dict[Cell, list[tuple[tuple[int, ...], int, Trial]]] = {}
        self._measured: dict[Cell, set[EventId]] = {}
        self._heads: dict[Cell, tuple[tuple[int, ...], int]] = {}
        self._evidence: dict[Cell, object] = {}

    def __repr__(self) -> str:
        """Return a constant, because this index isn't an observation."""
        return f"{type(self).__name__}()"

    def settle_outcome(
        self, *, stored: StoredEvent, reservation: ActionReserved, cell: Cell, outcome: TaskOutcome
    ) -> None:
        """Record what one settled task outcome standingly contributes."""
        if not isinstance(outcome.result, TaskSuccess):
            self._log.append(
                _Trial(
                    scope=str(reservation.scope),
                    cell=cell,
                    record=outcome,
                    identity=None,
                    gate=outcome,
                    measured=stored.event_id,
                    rank=_FAILED,
                )
            )

    def settle_judgment(
        self,
        *,
        stored: StoredEvent,
        reservation: ActionReserved,
        record: JudgmentRecorded,
        cell: Cell | None,
        target: TaskOutcome | None,
    ) -> None:
        """Record what one settled judgment standingly contributes."""
        counted = not isinstance(record.judgment, InvalidJudgment)
        if isinstance(record.subject, OutcomeRef) and target is not None and cell is not None:
            self._log.append(
                _Trial(
                    scope=str(reservation.scope),
                    cell=cell,
                    record=record,
                    identity=None,
                    gate=target,
                    measured=record.subject.outcome,
                    rank=_JUDGED if counted else None,
                )
            )
        elif cell is not None:
            rank = self._slots.setdefault(record.slot, len(self._slots)) if counted else None
            self._log.append(
                _Trial(
                    scope=str(reservation.scope),
                    cell=cell,
                    record=record,
                    identity=CriterionCoordinate(str(cell.role)),
                    gate=None,
                    measured=stored.event_id,
                    rank=None if rank is None else (1, rank),
                )
            )

    def evidence(
        self,
        projection: Fold,
        *,
        scopes: Collection[str],
        samples: Collection[SampleKey],
        aggregation: Callable[[Sequence[Trial]], object],
    ) -> CellEvidence:
        """Return each cell's aggregated evidence and the events it measured."""
        key = (
            scopes,
            samples,
            aggregation,
            sum(sorted(projection.epochs.values())),
        )
        if key != self._key:
            self._key, self._cursor = key, 0
            self._buckets, self._measured, self._heads, self._evidence = {}, {}, {}, {}
        axis = projection.manifest.coordinates.sample
        moved: set[Cell] = set()
        for index in range(self._cursor, len(self._log)):
            trial = self._log[index]
            gate = trial.gate
            sample = None if gate is None else SampleKey(gate.pins[axis])
            if (
                trial.scope not in scopes
                or (sample is not None and sample not in samples)
                or not projection.counted_for(trial.record, trial.identity)
            ):
                continue
            if trial.measured is not None:
                self._measured.setdefault(trial.cell, set()).add(trial.measured)
            if (rank := trial.rank) is not None:
                record = trial.record
                counted = None if isinstance(record, TaskOutcome) else projection.effective(record)
                self._buckets.setdefault(trial.cell, []).append(
                    (rank, index, Trial(counted, 1.0))
                )
                moved.add(trial.cell)
        self._cursor = len(self._log)
        resort = False
        for cell in moved:
            (bucket := self._buckets[cell]).sort()
            self._evidence[cell] = aggregation([counted for _, _, counted in bucket])
            if self._heads.get(cell) != (head := bucket[0][:2]):
                self._heads[cell] = head
                resort = True
        if resort:
            self._evidence = {cell: self._evidence[cell] for cell in sorted(self._heads, key=self._heads.__getitem__)}
        return CellEvidence(self._evidence, self._measured)
