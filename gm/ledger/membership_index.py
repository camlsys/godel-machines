# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""Maintain each cell's completed and awaiting samples under the current pins, one event at a time.

A produced success completes its sample once every required judge has a current finished verdict;
until then it awaits one, unless a required judge already failed it. Pins move only on a criterion
transition, which re-derives the whole index once.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from gm.domain.identity import SampleKey
from gm.domain.results import EvaluatorJudgment, RowKind, TaskSuccess
from gm.ids import CriterionCoordinate
from gm.ledger.events import OutcomeRef, TaskOutcome

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from gm.domain.identity import EvaluatorSlot, RoleId
    from gm.ids import EventId
    from gm.ledger.events import Cell, JudgmentRecorded
    from gm.ledger.projection import Fold


def required(*, cover: Mapping[RoleId, frozenset[EvaluatorSlot]], role: RoleId) -> frozenset[EvaluatorSlot]:
    """Return the judges one sampling role requires."""
    if role not in cover:
        msg = f"sampling role {role!r} declares no judging coverage"
        raise ValueError(msg)
    return cover[role]


def _current(*, projection: Fold, record: JudgmentRecorded, candidate: CriterionCoordinate | None) -> bool:
    axes = projection.manifest.coordinates
    # Reinterpreting a finished verdict needs no new execution. A labeled verdict's candidate
    # pin names the node being tested, not the incumbent judging that node.
    return all(
        axis in {candidate, axes.interpretation}
        or not axes.currency.get(axis, True) or projection.active_pins.get(axis) == value
        for axis, value in record.pins.items()
    )


def _finished(*, record: JudgmentRecorded) -> bool:
    return isinstance(record.judgment, EvaluatorJudgment) and record.judgment.kind in {
        RowKind.MEASURED, RowKind.FAILED,
    }


class _Produced:
    """One counted success and the verdicts its required judges returned so far."""

    def __init__(self, *, cell: Cell, sample: SampleKey, slots: frozenset[EvaluatorSlot]) -> None:
        self.cell, self.sample, self.slots = cell, sample, slots
        self.finished: set[EvaluatorSlot] = set()
        self.failed: set[EvaluatorSlot] = set()
        self.table: dict[Cell, Counter[SampleKey]] | None = None


class MembershipIndex:
    """Each cell's completed and awaiting sample counts, caught up lazily from the fold's log."""

    def __init__(self) -> None:
        """Start empty over an empty log."""
        self.log: list[tuple[EventId, Cell | None, TaskOutcome | JudgmentRecorded]] = []
        self._key: tuple[object, ...] | None = None
        self._cursor = 0
        self._tick = 0
        self._produced: dict[EventId, _Produced] = {}
        self.completed: dict[Cell, Counter[SampleKey]] = {}
        self.awaiting: dict[Cell, Counter[SampleKey]] = {}
        #: The tick of each cell's last change: equal ticks mean equal membership.
        self.version: dict[Cell, int] = {}
        #: What a reader last derived from one cell, keyed by everything it read; cleared with the index.
        self.derived: dict[Cell, tuple[object, object]] = {}

    def __repr__(self) -> str:
        """Return a constant, because this index isn't an observation."""
        return f"{type(self).__name__}()"

    def current(
        self,
        projection: Fold,
        *,
        cover: Mapping[RoleId, frozenset[EvaluatorSlot]],
        scopes: Collection[str],
    ) -> MembershipIndex:
        """Return this index caught up to the projection's log, re-derived if pins, cover or scopes moved."""
        key = (frozenset(scopes), cover, sum(sorted(projection.epochs.values())))
        if key != self._key:
            self._key, self._cursor = key, 0
            self._produced, self.completed, self.awaiting, self.version, self.derived = {}, {}, {}, {}, {}
        axes = projection.manifest.coordinates
        dataset = projection.active_pins[axes.dataset]
        for position in range(self._cursor, len(self.log)):
            ref, cell, record = self.log[position]
            if projection.scopes[record.action] not in scopes:
                continue
            if isinstance(record, TaskOutcome):
                if (
                    isinstance(record.result, TaskSuccess)
                    and projection.counted_in(record)
                    and record.pins[axes.dataset] == dataset
                ):
                    one = _Produced(
                        cell=cell, sample=SampleKey(record.pins[axes.sample]),
                        slots=required(cover=cover, role=record.role),
                    )
                    self._produced[ref] = one
                    self._place(one)
            elif isinstance(record.subject, OutcomeRef):
                one = self._produced.get(record.subject.outcome)
                if one is not None and _current(projection=projection, record=record, candidate=None):
                    (one.finished if _finished(record=record) else one.failed).add(record.slot)
                    self._place(one)
            elif (
                cell is not None and _finished(record=record)
                and _current(projection=projection, record=record, candidate=CriterionCoordinate(str(cell.role)))
            ):
                self._count(self.completed, cell, SampleKey(record.pins[axes.sample]), 1)
        self._cursor = len(self.log)
        return self

    def awaiting_of(self, cell: Cell) -> frozenset[SampleKey]:
        """Return the samples one cell holds a reusable output for, awaiting a verdict."""
        return frozenset(self.awaiting.get(cell, ()))

    def _place(self, one: _Produced) -> None:
        if one.slots <= one.finished:
            table = self.completed
        elif not (one.failed - one.finished) & one.slots:
            table = self.awaiting
        else:
            table = None
        if table is one.table:
            return
        if one.table is not None:
            self._count(one.table, one.cell, one.sample, -1)
        if table is not None:
            self._count(table, one.cell, one.sample, 1)
        one.table = table

    def _count(self, table: dict[Cell, Counter[SampleKey]], cell: Cell, sample: SampleKey, step: int) -> None:
        counts = table.setdefault(cell, Counter())
        if counts[sample] + step:
            counts[sample] += step
        else:
            del counts[sample]
        self._tick += 1
        self.version[cell] = self._tick
