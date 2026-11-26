# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""Maintain each cell's schedule position, one event at a time."""

from __future__ import annotations

from collections import Counter
from functools import partial
from typing import TYPE_CHECKING, NamedTuple

from gm.ledger.events import ActionKind, Cell, CellSlot

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping

    from gm.ids import ActionId
    from gm.ledger.events import ActionReserved, JudgmentRecorded, TaskOutcome
    from gm.ledger.projection import Fold


class _Source(NamedTuple):
    kind: ActionKind
    scope: str
    cell: Cell
    sample: str | None
    dataset: str | None
    record: JudgmentRecorded | TaskOutcome | None


class _Group(NamedTuple):
    at: int
    by_sample: dict[str | None, int]

    @property
    def total(self) -> int:
        """Every position this group holds, declared or not."""
        return sum(sorted(self.by_sample.values()))


class _Tally(NamedTuple):
    total: int
    at: int


class ScheduleIndex:
    """Each cell's settled source count, broken down by what a caller filters on."""

    def __init__(self) -> None:
        """Initialize an empty breakdown over an empty log."""
        self.log: list[_Source] = []
        self.epoch = -1
        self.cursor = 0
        self.generated: dict[CellSlot, dict[tuple[Cell, str, bool], _Group]] = {}
        self.judged: dict[CellSlot, dict[tuple[Cell, str, bool], _Tally]] = {}
        self.reserved_dataset: dict[ActionId, str | None] = {}

    def reserve(self, action: ActionId, dataset: str | None) -> None:
        """Record the dataset pin one reservation carries."""
        self.reserved_dataset[action] = dataset

    def close(self, action: ActionId) -> str | None:
        """Forget one reservation's issue-time pin, returning it."""
        return self.reserved_dataset.pop(action, None)

    def __repr__(self) -> str:
        """Return a constant, because this index isn't an observation."""
        return f"{type(self).__name__}()"

    def settle_outcome(
        self, *, reservation: ActionReserved, cell: Cell, outcome: TaskOutcome, sample: str | None, dataset: str | None
    ) -> None:
        """Record one settled generation as a candidate schedule position."""
        self.log.append(_Source(ActionKind.GENERATION, str(reservation.scope), cell, sample, dataset, outcome))

    def settle_judgment(
        self, *, reservation: ActionReserved, cell: Cell, record: JudgmentRecorded, dataset: str | None
    ) -> None:
        """Record one settled labeled attempt as a candidate schedule position."""
        self.log.append(_Source(ActionKind.JUDGMENT, str(reservation.scope), cell, None, dataset, record))

    def settle_skip(self, *, reservation: ActionReserved, cell: Cell | None, dataset: str | None) -> None:
        """Record one reservation that closed without an outcome as a schedule position."""
        if cell is None or reservation.sample is None:
            return
        self.log.append(_Source(reservation.kind, str(reservation.scope), cell, str(reservation.sample), dataset, None))


def _current(*, projection: Fold, source: _Source, active: str | None) -> bool:
    if source.record is None:
        if source.kind is ActionKind.JUDGMENT:
            pool = projection.minted.get(str(source.cell.task))
            if pool is not None and source.sample not in pool.members:
                return False
        return source.dataset == active
    if source.kind is ActionKind.GENERATION:
        return source.dataset == active and projection.counted_in(source.record)
    return projection.labeled_counted(source.cell, source.record)


def _folded(projection: Fold) -> ScheduleIndex:
    if (epoch := sum(projection.epochs.values())) != (index := projection.schedules).epoch:
        index.epoch, index.cursor = epoch, 0
        index.generated, index.judged = {}, {}
    axes = projection.manifest.coordinates
    active = projection.active_pins.get(axes.dataset)
    for position in range(index.cursor, len(index.log)):
        source = index.log[position]
        if not _current(projection=projection, source=source, active=active):
            continue
        held = source.record is not None
        if source.kind is ActionKind.GENERATION:
            into = index.generated.setdefault(source.cell.slot, {})
            keyed = (source.cell, source.scope, held)
            group = into.get(keyed) or _Group(position, {})
            group.by_sample[source.sample] = group.by_sample.get(source.sample, 0) + 1
            into[keyed] = group
        else:
            grouped = index.judged.setdefault(source.cell.slot, {})
            judged = (source.cell, source.scope, held)
            tally = grouped.get(judged)
            grouped[judged] = _Tally(tally.total + 1, tally.at) if tally else _Tally(1, position)
    index.cursor = len(index.log)
    return index


def _counted(group: _Group, declared: Collection[str]) -> int:
    outside = group.by_sample.keys() - declared
    if not outside:
        return group.total
    return group.total - sum(group.by_sample[sample] for sample in outside)


def _ordered(
    generated: Mapping[Cell, int],
    judged: Mapping[Cell, int],
    *,
    gen_first: Mapping[Cell, int],
    jud_first: Mapping[Cell, int],
) -> dict[Cell, int]:
    counts: dict[Cell, int] = {cell: generated[cell] for cell in sorted(generated, key=gen_first.__getitem__)}
    for cell in sorted(judged, key=jud_first.__getitem__):
        counts[cell] = counts.get(cell, 0) + judged[cell]
    return counts


def _walk(
    projection: Fold,
    *,
    scopes: Collection[str],
    owners: Mapping[CellSlot, ActionKind],
    declared: Collection[str],
    counting: Callable[[_Group, Collection[str]], int],
) -> tuple[dict[Cell, int], dict[Cell, int]]:
    index = _folded(projection)
    standing: dict[Cell, int] = {}
    carried: dict[Cell, int] = {}
    labeled: dict[Cell, int] = {}
    answered: dict[Cell, int] = {}
    gen_first: dict[Cell, int] = {}
    jud_first: dict[Cell, int] = {}
    for slot, kind in owners.items():
        if kind is ActionKind.GENERATION:
            for (cell, scope, held), group in index.generated.get(slot, {}).items():
                gen_first[cell] = min(gen_first.get(cell, group.at), group.at)
                if scope not in scopes or not (total := counting(group, declared)):
                    continue
                standing[cell] = standing.get(cell, 0) + total
                if held:
                    carried[cell] = carried.get(cell, 0) + total
        elif kind is ActionKind.JUDGMENT:
            for (cell, scope, held), tally in index.judged.get(slot, {}).items():
                jud_first[cell] = min(jud_first.get(cell, tally.at), tally.at)
                if scope not in scopes:
                    continue
                labeled[cell] = labeled.get(cell, 0) + tally.total
                if held:
                    answered[cell] = answered.get(cell, 0) + tally.total
    return (
        _ordered(standing, labeled, gen_first=gen_first, jud_first=jud_first),
        _ordered(carried, answered, gen_first=gen_first, jud_first=jud_first),
    )


_counts = partial(_walk, counting=_counted)


def open_positions(
    *,
    projection: Fold,
    scopes: Collection[str],
    owners: Mapping[CellSlot, ActionKind],
    declared: Collection[str],
) -> dict[Cell, int]:
    """Return each cell's in-flight schedule positions."""
    active = projection.active_pins.get(projection.manifest.coordinates.dataset)
    pinned = projection.schedules.reserved_dataset
    return dict(
        Counter(
            Cell(node=r.node, role=r.role, task=r.task)
            for action, r in projection.open_actions.items()
            if r.task is not None
            and owners.get(CellSlot.of(role=r.role, task=r.task)) is r.kind
            and str(r.scope) in scopes
            and str(r.sample) in declared
            and pinned.get(action) == active
        )
    )


def position_counts(
    *,
    projection: Fold,
    scopes: Collection[str],
    owners: Mapping[CellSlot, ActionKind],
    population: Collection[object],
) -> tuple[dict[Cell, int], dict[Cell, int], dict[Cell, int]]:
    """Return each cell's settled positions, its in-flight ones, and its evidenced ones."""
    declared = frozenset(map(str, population))
    completed, evidenced = _counts(projection, scopes=scopes, owners=owners, declared=declared)
    reserved = open_positions(projection=projection, scopes=scopes, owners=owners, declared=declared)
    return completed, reserved, evidenced
