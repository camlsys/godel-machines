# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""What the kernel owns about samples: the populations a run's cells draw from, which sample an evaluation
reads next, the per-cell in-flight caps, which cells are exhausted, and the decision snapshot built from them.

A policy never picks a sample. Each cell walks its population in declared order (a schema-8 run's recorded
`epochs` add seeded permutation passes), skipping samples held by an open action or awaiting a verdict.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import TYPE_CHECKING

from gm import ids, ledger, policy
from gm.domain import CompositeObjective, RoleId, SampleKey, TaskId
from gm.ledger import ActionKind, Cell, CellSlot, Fold
from gm.ledger.membership_index import required
from gm.ledger.schedule_index import position_counts

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from gm.domain import DatasetSpec, DomainManifest, EvaluatorSlot, SurfaceSide
    from gm.work_queue import ExpansionExecution

type JudgingCover = Mapping[RoleId, frozenset[EvaluatorSlot]]
#: One in-flight bucket: the kind of work and the sample set it is about (None: no declared set).
type CapCell = tuple[ActionKind, str | None]

#: The scopes a decision reads: the run's own search.
SEARCH = frozenset({ledger.SEARCH})
#: The bucket an owed judgment occupies: its producer forced it, so it shares no cell with chosen work.
OWED = "owed"
#: Open actions allowed per (kind, sample set) cell a run names no cap for: one, so every draw sees settled state.
IN_FLIGHT = 1


def judging_cover(domain: DomainManifest) -> JudgingCover:
    """Return each declared role's required judges."""
    cover: dict[RoleId, set[EvaluatorSlot]] = defaultdict(set)
    for slot, roles in domain.jurisdiction.items():
        for role in roles:
            cover[role].add(slot)
    return {role: frozenset(slots) for role, slots in cover.items()}


def labeled_roles(projection: Fold) -> dict[RoleId, tuple[TaskId, ...]]:
    """Return the judged-evaluator cells over minted pools, which sit beside the generator cells."""
    cells: dict[RoleId, set[TaskId]] = {}
    for name, one in projection.minted.items():
        if one.judge is not None and one.members:
            cells.setdefault(RoleId(str(one.judge)), set()).add(TaskId(name))
    return {role: tuple(sorted(tasks)) for role, tasks in cells.items()}


def role_cells(
    domain: DomainManifest, projection: Fold,
) -> tuple[dict[RoleId, tuple[TaskId, ...]], dict[CellSlot, ActionKind]]:
    """Return every role's cells and the kind each cell reserves (generation, or judgment over a minted pool)."""
    pools, labeled = domain.generator.task_pools, labeled_roles(projection)
    if collisions := {(str(r), str(t)) for r, tasks in labeled.items() for t in tasks if t in pools.get(r, ())}:
        raise ValueError(f"labeled cells collide with declared task pools: {sorted(collisions)}")
    roles = {role: (*pools.get(role, ()), *labeled.get(role, ())) for role in (*pools, *labeled)}
    owners = {
        CellSlot(role=role, task=task): ActionKind.GENERATION if task in pools.get(role, ()) else ActionKind.JUDGMENT
        for role, tasks in roles.items() for task in tasks
    }
    return roles, owners


def objective_sides(domain: DomainManifest) -> dict[CellSlot, SurfaceSide]:
    """Return which side of a seat objective each evaluation cell answers on."""
    pools, sets = domain.generator.task_pools, domain.generator.task_sets
    sides: dict[CellSlot, SurfaceSide] = {}
    for coordinate, rule in domain.transition.coordinates.items():
        if not isinstance(rule, CompositeObjective):
            continue
        role = RoleId(str(coordinate))
        for surface, side in rule.surfaces.items():
            declared = [task for task in pools.get(role, ()) if sets[task] == surface.samples]
            for task in declared or [TaskId(str(surface.samples))]:
                sides[CellSlot.of(role=role, task=task)] = side
    return sides


def pinned_build(*, projection: Fold, domain: DomainManifest) -> DatasetSpec:
    """Return the dataset build this run's active dataset pin names."""
    axis = projection.manifest.coordinates.dataset
    if (version := projection.active_pins.get(axis)) is None:
        raise ValueError(f"the run pins no value at its dataset coordinate {str(axis)!r}")
    return domain.dataset.build(version=version)


def run_population(*, projection: Fold, domain: DomainManifest) -> tuple[SampleKey, ...]:
    """Return every sample any declared cell of this run may stand at."""
    build = pinned_build(projection=projection, domain=domain)
    seen: dict[SampleKey, None] = {}
    for name in sorted({str(samples) for samples in domain.generator.task_sets.values()}):
        seen.update(dict.fromkeys(build.splits[name]))
    for name in sorted(projection.minted):
        seen.update(dict.fromkeys(projection.minted[name].members))
    return tuple(seen)


def cell_samples(*, projection: Fold, domain: DomainManifest, cell: Cell) -> tuple[SampleKey, ...]:
    """Return the population one evaluation cell draws from: its task's set, or the minted pool it names."""
    if cell.task in domain.generator.task_pools.get(cell.role, ()):
        return pinned_build(projection=projection, domain=domain).splits[domain.generator.task_sets[cell.task]]
    return projection.minted[str(cell.task)].members


def reserved_members(*, projection: Fold, scopes: Collection[str]) -> dict[Cell, frozenset[SampleKey]]:
    """Return the samples open actions hold per cell, under the active dataset build."""
    active = projection.active_pins[projection.manifest.coordinates.dataset]
    held: dict[Cell, set[SampleKey]] = defaultdict(set)
    for action, one in projection.open_actions.items():
        if (
            one.kind is ActionKind.EXPANSION or one.task is None or one.sample is None
            or str(one.scope) not in scopes or projection.schedules.reserved_dataset[action] != active
        ):
            continue
        held[Cell(node=one.node, role=one.role, task=one.task)].add(one.sample)
    return {cell: frozenset(samples) for cell, samples in held.items()}


def _round(key: bytes, index: int, block: int) -> int:
    digest = hashlib.sha256(key + index.to_bytes(4, "big") + block.to_bytes(8, "big")).digest()
    return int.from_bytes(digest[:8], "big")


def _permuted(*, seed: int, label: str, size: int, index: int) -> int:
    """A four-round Feistel permutation of range(size), cycle-walked (schema 8's `epoch_permutation_v2`)."""
    if size < 2:
        return 0
    key = hashlib.sha256(f"rqgm/sample_order/{seed}/{label}".encode()).digest()
    half = ((size - 1).bit_length() + 1) // 2
    mask = (1 << half) - 1
    value = index
    while True:
        left, right = value >> half, value & mask
        for round_ in range(4):
            left, right = right, left ^ (_round(key, round_, right) & mask)
        value = (left << half) | right
        if value < size:
            return value


def _member(seed: int, size: int, epoch: int, index: int) -> int:
    """The member at `index` of pass `epoch`: pass 0 is the declared order; later passes rotate an anchor first."""
    if epoch == 0:
        return index
    turn = epoch % size
    anchor = 0 if turn == 0 else 1 + _permuted(seed=seed, label="anchor", size=size - 1, index=turn - 1)
    if index == 0:
        return anchor
    shifted = _permuted(seed=seed, label=f"epoch/{epoch}", size=size - 1, index=index - 1)
    return shifted if shifted < anchor else shifted + 1


def _walk(
    samples: tuple[SampleKey, ...], *, completed: Mapping[SampleKey, int], held: frozenset[SampleKey], epochs: int, seed: int,
) -> SampleKey | None:
    """Return the first sample of the current pass that is neither evaluated in this pass nor held."""
    if not samples:
        return None
    epoch = min(completed.get(one, 0) for one in samples)
    if epoch > epochs:
        return None
    order = (samples[_member(seed, len(samples), epoch, index)] for index in range(len(samples)))
    return next((one for one in order if completed.get(one, 0) <= epoch and one not in held), None)


class Sampler:
    """The kernel's sample choice and per-cell caps for one run, read from its fold."""

    def __init__(self, *, domain: DomainManifest, projection: Fold) -> None:
        """Resolve the run's cap buckets against its corpus and start counting open actions per bucket."""
        self.domain, self.projection = domain, projection
        self.cover = judging_cover(domain)
        settings = projection.settings
        self.epochs, self._caps = settings.epochs, settings.caps
        index: dict[SampleKey, str] = {}
        shared: set[SampleKey] = set()
        for name, keys in domain.dataset.rows().items():
            for sample in keys:
                if index.setdefault(sample, str(name)) != str(name):
                    shared.add(sample)
        #: The set each sample falls under; a sample two sets name is resolved per role by `_by_role`.
        self._splits = {sample: name for sample, name in index.items() if sample not in shared}
        bound = dict(domain.generator.task_sets)
        rows = domain.dataset.rows()
        self._by_role = {
            (str(role), sample): str(one)
            for role, tasks in domain.generator.task_pools.items()
            for one in {bound[task] for task in tasks if task in bound}
            for sample in rows.get(one, ())
        }
        projection.track(self.bucket)

    def split(self, *, role: object, sample: SampleKey | None) -> str | None:
        """Return the sample set one seat's sample occupies, as the caps key it."""
        if sample is not None and (named := self._by_role.get((str(role), sample))) is not None:
            return named
        return None if sample is None else self._splits.get(sample)

    def bucket(self, reservation: ledger.ActionReserved) -> CapCell:
        """Return the cap cell one open action occupies."""
        if reservation.owed:
            return (reservation.kind, OWED)
        return (reservation.kind, self.split(role=reservation.role, sample=reservation.sample))

    def room(self, *, kind: ActionKind, split: str | None) -> int:
        """Return how many more actions one cap cell admits open."""
        return self._caps.get((kind, split), IN_FLIGHT) - self.projection.open_count(self.bucket).get((kind, split), 0)

    def next_sample(self, cell: Cell, *, held: frozenset[SampleKey] | None = None) -> SampleKey | None:
        """Return the sample this cell's next evaluation reads, or None when none is free."""
        p = self.projection
        if cell.task in self.domain.generator.task_pools.get(cell.role, ()):
            required(cover=self.cover, role=cell.role)
        index = p.members.current(p, cover=self.cover, scopes=SEARCH)
        if held is None:
            held = reserved_members(projection=p, scopes=SEARCH).get(cell, frozenset())
        return _walk(
            cell_samples(projection=p, domain=self.domain, cell=cell),
            completed=index.completed.get(cell, {}), held=held | index.awaiting_of(cell),
            epochs=self.epochs, seed=p.manifest.seed,
        )

    def exhausted(self, roles: Mapping[RoleId, tuple[TaskId, ...]], nodes: Collection[ids.NodeId]) -> frozenset[Cell]:
        """Return the cells with no free sample and no output still awaiting a verdict."""
        p = self.projection
        held = reserved_members(projection=p, scopes=SEARCH)
        index = p.members.current(p, cover=self.cover, scopes=SEARCH)
        out: set[Cell] = set()
        for node in nodes:
            for role, tasks in roles.items():
                for task in tasks:
                    cell = Cell(node=node, role=role, task=task)
                    samples = cell_samples(projection=p, domain=self.domain, cell=cell)
                    # A cell's answer moves only with its own membership, holds or population.
                    read = (index.version.get(cell, 0), held.get(cell, frozenset()), samples)
                    known = index.derived.get(cell)
                    if known is None or known[0] != read:
                        free = _walk(
                            samples, completed=index.completed.get(cell, {}),
                            held=read[1] | index.awaiting_of(cell), epochs=self.epochs, seed=p.manifest.seed,
                        )
                        awaiting = (index.awaiting_of(cell) - read[1]).intersection(samples)
                        known = index.derived[cell] = (read, free is None and not awaiting)
                    if known[1]:
                        out.add(cell)
        return frozenset(out)

    def snapshot(self, *, execution: ExpansionExecution | None, commitments: int) -> policy.Snapshot:
        """Return the decision snapshot, built from the fold."""
        p, domain = self.projection, self.domain
        roles, owners = role_cells(domain, p)
        build = pinned_build(projection=p, domain=domain)
        snapshot_samples = frozenset(sample for name in p.manifest.snapshot_sets for sample in build.splits[name])
        population = run_population(projection=p, domain=domain)
        completed, reserved, evidenced = position_counts(projection=p, scopes=SEARCH, owners=owners, population=population)
        cells = p.trials.evidence(p, scopes=SEARCH, samples=snapshot_samples, aggregation=policy.paper_aggregation)
        # Only currently measured artifacts and current skips count as completed positions.
        current = {
            cell: len(cells.measured.get(cell, ())) + completed.get(cell, 0) - evidenced.get(cell, 0)
            for cell in dict.fromkeys((*completed, *cells.measured))
        }
        pending = self._pending(population=frozenset(population), owners=owners)
        # A failed producer or partly settled cover can be measured while a judge is still open; that judge
        # holds its source position rather than opening another.
        held = {
            cell: min(count, max(0, completed.get(cell, 0) + reserved.get(cell, 0) - current.get(cell, 0)))
            for cell, count in pending.items()
        }
        parents = policy.searchable(root=p.manifest.source, lineage=p.lineage, done=frozenset(p.done))
        return policy.Snapshot(
            parents=parents,
            roles=roles,
            sides=objective_sides(domain),
            completed=current,
            reserved=held,
            evidence=cells.evidence,
            clock_completed=float(p.total(ledger.CLOCK, SEARCH)),
            budget=p.manifest.budget,
            action_index=p.decisions,
            seed=p.manifest.seed,
            ordinals=p.ordinals,
            policy=p.settings.policy,
            expansion_execution=execution,
            expansion_completed=p.expansion_completions(scopes=SEARCH),
            evaluation_commitments=commitments,
            closed=self.exhausted(roles, parents),
        )

    def _pending(self, *, population: frozenset[SampleKey], owners: Mapping[CellSlot, ActionKind]) -> dict[Cell, int]:
        """Count open covers per cell whose issue-time pins are still current."""
        p = self.projection
        open_ = {
            action: one for action, one in p.open_actions.items()
            if one.task is not None and str(one.scope) in SEARCH and one.sample in population
            and CellSlot.of(role=one.role, task=one.task) in owners and one.kind is not ActionKind.EXPANSION
        }
        counted: dict[Cell, set[ids.ActionId]] = {}
        for action, one in open_.items():
            if not self._current(one, owners=owners, pins=p.issued_pins[action]):
                continue
            cell = Cell(node=one.node, role=one.role, task=one.task)
            cover = p.covers.cover[action] if one.kind is ActionKind.JUDGMENT else action
            counted.setdefault(cell, set()).add(cover)
        return {cell: len(covers) for cell, covers in counted.items()}

    def _current(self, one: ledger.ActionReserved, *, owners: Mapping[CellSlot, ActionKind], pins: Mapping) -> bool:
        p = self.projection
        axes = p.manifest.coordinates
        coordinates = {axes.dataset}
        role = one.role
        if one.kind is ActionKind.JUDGMENT:
            if one.slot is None:
                raise ValueError(f"judgment reservation names no judging slot: {one.action}")
            role = RoleId(str(one.slot))
            coordinates.add(ids.CriterionCoordinate(str(one.slot)))
            if axes.interpretation is not None:
                coordinates.add(axes.interpretation)
            if owners[CellSlot.of(role=one.role, task=one.task)] is ActionKind.JUDGMENT:
                coordinates.discard(ids.CriterionCoordinate(str(one.role)))
                coordinates.add(p.minted_by[str(one.task)])
        if (provenance := p.settings.provenance.get(str(role))) is not None:
            coordinates.add(ids.CriterionCoordinate(provenance))
        return not any(
            axes.currency.get(axis, True) and pins.get(axis) != p.active_pins.get(axis) for axis in coordinates
        )

