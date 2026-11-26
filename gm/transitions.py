# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""Criterion transitions: the declared rules (curriculum, pool mint, objective promotion), the amortized checkpoint
that runs them, and the search cells over minted pools that feed them."""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from gm import ids, ledger, policy, sampling, settlement
from gm import work_queue as wq
from gm.domain import (
    Artifact,
    CompositeObjective,
    Curriculum,
    EvaluatorSlot,
    MintedSet,
    PoolMint,
    RoleId,
    SampleKey,
    TaskId,
    TaskSuccess,
    TransitionFacts,
)
from gm.ledger.trial_index import Trial
from gm.policy import Evidence

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from gm.controller import Controller
    from gm.domain import SurfaceSide, UtilitySurface
    from gm.ledger.events import LabeledRef, OutcomeRef

def rescore(
    host: Controller,
    node: ids.NodeId,
    role: RoleId,
    task: TaskId,
) -> bool:
    """Return whether this call reserved any judgment, reusing the cell's immutable artifacts."""
    stale = False
    claims = ledger.claimed_judgments(
        host.projection,
        scopes={ledger.SEARCH},
    )
    for ref, outcome in host.projection.cell_outcomes.get(ledger.Cell(node=node, role=role, task=task), ()):
        if host.projection.scopes[outcome.action] != ledger.SEARCH:
            continue  # measurement outcomes never re-enter search
        if not host.projection.counted_in(outcome):
            continue  # source pins changed; generate under the current source
        axes = host.projection.manifest.coordinates
        if outcome.pins.get(axes.dataset) != host.projection.active_pins.get(axes.dataset):
            continue
        sample = SampleKey(outcome.pins[axes.sample])
        if ledger.ClaimKey(node, role, task, str(sample)) in claims:
            continue  # a verdict on this cell and sample is already asked for
        split = host.sampler.split(role=role, sample=sample)
        if host.sampler.room(kind=ledger.ActionKind.JUDGMENT, split=split) <= 0:
            continue
        stale |= host.judge(ref, outcome)
    return stale



def labeled_cell(*, host: Controller, node: ids.NodeId, role: RoleId, task: TaskId) -> bool:
    """Return whether this dispatched a search-scheduled cell trial over a minted pool."""
    name = str(task)
    coordinate = ids.CriterionCoordinate(str(role))
    minted = host.projection.minted[name]
    if (judge := minted.judge) is None:
        msg = f"labeled cell names a judgeless minted set: {name!r}"
        raise ValueError(msg)
    if not minted.members:
        return False
    staged, extra = minted_staged(host=host, name=name, one=minted)
    sample = host.sampler.next_sample(ledger.Cell(node=node, role=role, task=task))
    if sample is None:
        return False
    source = host.projection.outcomes[minted.sources[sample]]
    source_sample = SampleKey(source.pins[host.projection.manifest.coordinates.sample])
    _trial(
        host,
        node,
        role,
        task,
        judge=judge,
        sample=sample,
        staged=staged,
        source_sample=source_sample,
        agent=role,
        pins=settlement.judgment_pins(host.projection, host.domain, sample, coordinate)
        | {coordinate: str(node)}
        | extra,
    )
    return True


def _trial(
    host: Controller,
    node: ids.NodeId,
    role: RoleId,
    task: TaskId | None,
    *,
    judge: EvaluatorSlot,
    sample: SampleKey,
    staged: Mapping[SampleKey, Mapping[str, Artifact]],
    source_sample: SampleKey | None = None,
    agent: RoleId,
    pins: Mapping[ids.CriterionCoordinate, str],
) -> None:
    subject = ledger.LabeledRef(node, sample)
    pins = MappingProxyType(dict(pins) | settlement.provenance_pin(host.projection, judge))
    (action,) = host.reserve_all(
        [
            (
                ledger.ActionKind.JUDGMENT,
                node,
                role,
                task,
                ledger.SEARCH,
                sample,
                False,
                judge,
            ),
        ],
    )
    host.queue.submit(
        wq.JudgmentJob(
            action=action,
            node=node,
            slot=judge,
            subject=subject,
            sample=sample,
            produced=TaskSuccess(files=staged.get(sample, {}), spend=()),
            role=agent,
            pins=pins,
            source_sample=source_sample,
        ),
    )


RECOMPARISON = "recomparison"
AMORTIZED = "amortized"
#: A crossing costs work proportional to the evidence count n. Crossing again at ceil(n_last x B / (B - 1)) spends at
#: most B units of comparison work per unit of evidence over the whole run.
SCANNING_BUDGET = 9


def amortized_threshold(crossings: int, budget: int = SCANNING_BUDGET) -> int:
    """Return the evaluation count at which the crossing after `crossings` is due: 1, 2, ..., 9, 11, 13, 15, 17, 20, ...

    Each threshold is ceil(previous x B / (B - 1)), at least previous + 1, so the comparison work summed over a run
    stays within B times the evidence accumulated, whatever the run's length.
    """
    at = 1
    for _ in range(crossings):
        at = max(at + 1, -(-at * budget // (budget - 1)))
    return at


class Triggers:
    """When a checkpoint is due: the amortized schedule, and comparisons a crossing's own transitions invalidated."""

    def __init__(self, coordinates: Collection[ids.CriterionCoordinate]) -> None:
        """Start at the empty ledger."""
        self._coordinates = frozenset(coordinates)
        self._crossings = 0
        self._since_publish = 0
        self._moved: set[ids.CriterionCoordinate] = set()
        self._owed: frozenset[ids.CriterionCoordinate] = frozenset()

    def advance(self, stored: ledger.StoredEvent) -> None:
        """Fold one event."""
        event = stored.event
        published = isinstance(event, ledger.CheckpointPublished)
        self._since_publish = 0 if published else self._since_publish + 1
        if isinstance(event, ledger.CriterionTransition):
            self._moved.add(event.coordinate)
        if published:
            compared = self._coordinates if event.scope is None else frozenset(event.scope) & self._coordinates
            self._owed = (self._owed - compared) | frozenset(c for c in compared if self._moved - {c})
            self._moved.clear()
            self._crossings += event.trigger == AMORTIZED

    def owed(self) -> frozenset[ids.CriterionCoordinate]:
        """Return the comparisons a crossing's own transitions invalidated."""
        return self._owed

    def due(self, projection: ledger.Fold) -> str | None:
        """Return the trigger due now, if any; a due amortized checkpoint compares every coordinate."""
        if self._since_publish and projection.total(ledger.CLOCK) >= amortized_threshold(self._crossings):
            return AMORTIZED
        return None


def checkpoint(
    host: Controller,
    scope: frozenset[ids.CriterionCoordinate] | None = None,
    trigger: str | None = None,
) -> None:
    """Compare settled evidence, commit transitions, and publish their ledger prefix."""
    moved = False
    host.apply_events(events=[], phase=ledger.CheckpointPhase.DRAIN, scope=scope, fired=trigger)
    host.apply_events(
        events=[],
        phase=ledger.CheckpointPhase.COMPARE,
        scope=scope,
        fired=trigger,
    )
    transitions = _compare(host=host, scope=scope)  # compare settled evidence
    if transitions:
        host.apply_events(
            events=transitions,
            phase=ledger.CheckpointPhase.TRANSITION,
            scope=scope,
            fired=trigger,
        )  # commit transitions
        moved = True
    with host.ledger_held():
        head, digest = host.events[-1].event_id, ledger.prefix_digest(events=host.events)  # publish
        published = ledger.CheckpointPublished(
            head,
            digest,
            trigger,
            None if scope is None else tuple(sorted(scope)),
        )
        host.apply_events(events=[published], phase=ledger.CheckpointPhase.IDLE)
    if moved and ledger.project(host.events).state() != host.projection.state():
        msg = "incremental fold diverged from the from-scratch refold"
        raise RuntimeError(msg)



def _compare(*, host: Controller, scope: frozenset[ids.CriterionCoordinate] | None) -> list[ledger.LedgerEvent]:
    p = host.projection
    pins = MappingProxyType(dict(p.active_pins))
    epochs = MappingProxyType(dict(p.epochs))
    minted_before = MappingProxyType(dict(p.minted))
    batch_minted = dict(p.minted)
    events: list[ledger.LedgerEvent] = []
    for coordinate, rule in host.domain.transition.coordinates.items():
        if scope is not None and coordinate not in scope:
            continue
        incumbent = pins[coordinate]
        scopes = frozenset(host.domain.transition.scopes[coordinate])
        replacement, minted = _apply_rule(
            host, rule, coordinate=coordinate, incumbent=incumbent, scopes=scopes, pins=pins, epochs=epochs,
            minted=minted_before,
        )
        if replacement is None or replacement == incumbent:
            continue
        for one in minted:  # validate against the batch; expose only to the validator
            one.require_declared(splits=host.domain.dataset.rows(), live=batch_minted)
            batch_minted[one.name] = one
        events.append(ledger.CriterionTransition(coordinate, incumbent, replacement, epochs[coordinate] + 1, minted))
    return events


def _apply_rule(
    host: Controller,
    rule: CompositeObjective | Curriculum | PoolMint,
    *,
    coordinate: ids.CriterionCoordinate,
    incumbent: str,
    scopes: frozenset[str],
    pins: Mapping[ids.CriterionCoordinate, str],
    epochs: Mapping[ids.CriterionCoordinate, int],
    minted: Mapping[str, MintedSet],
) -> tuple[str | None, tuple[MintedSet, ...]]:
    """Run one declared rule: return the replacement incumbent (None: none) and any sets it mints."""
    p, settings = host.projection, host.projection.settings
    match rule:
        case Curriculum():
            return rule.crossing(p.total(ledger.CLOCK, scopes), incumbent), ()
        case PoolMint():
            signer, cut = _displaced(host, rule, epochs=epochs, pins=pins) or (None, None)
            # Pool comparison may run after the reviewer changed: freeze the population before that displacement.
            judged, labeled = ((), {}) if cut is None else ledger.project(host.events[: cut - 1]).judged_streams(scopes=scopes)
            facts = TransitionFacts(
                clock=p.total(ledger.CLOCK, scopes), epoch=epochs[coordinate], pins=pins, judged=tuple(judged),
                labeled={slot: tuple(work) for slot, work in labeled.items()}, minted=minted,
                retained=MappingProxyType(dict(p.retained)), ordinals=p.ordinals if settings.policy is None else None,
            )
            digest, one = rule.crossing(facts, incumbent, displaced=signer)
            if str(coordinate) in settings.withheld_pools:
                return None, ()
            return digest, () if one is None else (one,)
        case CompositeObjective():
            winner = _promoted(host, rule, coordinate=coordinate, incumbent=incumbent, scopes=scopes, minted=minted)
            return (None if settings.fixed_evaluator else winner), ()


def _displaced(
    host: Controller, rule: PoolMint, *, epochs: Mapping[ids.CriterionCoordinate, int], pins: Mapping[ids.CriterionCoordinate, str],
) -> tuple[str, int] | None:
    """Return the signer the latest committed transition displaced, and the seq of its first appointment."""
    changes = [
        (stored.seq, stored.event) for stored in host.events
        if isinstance(stored.event, ledger.CriterionTransition) and stored.event.coordinate == rule.signer
    ]
    if len(changes) != epochs[rule.signer]:
        raise ValueError("pool cutoff requires complete signer transition history")
    if not changes:
        return None
    if changes[-1][1].replacement != pins[rule.signer]:
        raise ValueError("pool facts disagree with the committed signer transition")
    signer = changes[-1][1].incumbent
    return signer, next(seq for seq, event in changes if event.incumbent == signer)


def _promoted(
    host: Controller,
    objective: CompositeObjective,
    *,
    coordinate: ids.CriterionCoordinate,
    incumbent: str,
    scopes: frozenset[str],
    minted: Mapping[str, MintedSet],
) -> str | None:
    """Return the challenger whose BB_eps over the objective's evidence beats the incumbent's, or None."""
    p = host.projection
    splits = sampling.pinned_build(projection=p, domain=host.domain).splits
    live = set(host.domain.dataset.splits) | {name for name, one in minted.items() if one.members}
    # A stationary empty pool cannot supply a trial, so it contributes no surface.
    sides = {surface: side for surface, side in objective.surfaces.items() if str(surface.samples) in live}
    if not sides:
        return None
    rows = _labeled_rows(p, coordinate, sides, splits=splits, scopes=scopes, minted_mass=p.settings.pool_mass)
    # A pool holding no rows right now is exempt from the coverage requirement, and from nothing else.
    unpopulated = {s for s in sides if not (splits.get(str(s.samples)) or getattr(minted.get(str(s.samples)), "members", ()))}
    beliefs = {}
    for arm, row in rows.items():
        weights = _drawn(sides, row)
        if weights is None or any(0 < row[k].trials < objective.min_trials for k in weights if k in row):
            continue
        pooled = _pooled(weights, row)
        covered = all(key in row and row[key].trials for key in sides if key not in unpopulated)
        if pooled.trials and pooled.score_sum / pooled.trials > 0 and covered:
            beliefs[arm] = policy.best_belief(pooled.score_sum, pooled.shortfall)
    row = rows.get(incumbent, {})
    weights = _drawn(sides, row)
    pooled = Evidence() if weights is None else _pooled(weights, row)
    held = policy.best_belief(pooled.score_sum, pooled.shortfall)
    beliefs = {arm: b for arm, b in beliefs.items() if b > held} or {incumbent: held}
    winner = policy.argmax(beliefs, ordinals=p.ordinals, by_name=p.settings.policy is not None)
    return None if winner == incumbent else str(winner)


def _drawn(sides: Mapping[UtilitySurface, SurfaceSide], row: Mapping[UtilitySurface, Evidence]) -> dict | None:
    """The trial mass a row drew on each declared surface, or None when it drew none."""
    drawn = {key: row[key].trials if key in row else 0.0 for key in sides}
    return drawn if math.fsum(drawn.values()) else None


def _pooled(surfaces: Collection[UtilitySurface], row: Mapping[UtilitySurface, Evidence]) -> Evidence:
    """Pool already weighted per-surface evidence once."""
    measured = [row[key] for key in surfaces if key in row and row[key].trials]
    return Evidence(math.fsum(one.score_sum for one in measured), math.fsum(one.trials for one in measured))


def _labeled_rows(
    p: ledger.Fold,
    coordinate: ids.CriterionCoordinate,
    surfaces: Collection[UtilitySurface],
    *,
    splits: Mapping[str, tuple[SampleKey, ...]],
    scopes: frozenset[str],
    minted_mass: float,
) -> dict[str, dict[UtilitySurface, Evidence]]:
    """Return each candidate seat's per-surface evidence from the judgments its slot produced or received."""
    axis = p.manifest.coordinates.sample
    produced = RoleId(str(coordinate))
    admitted = p.judged_index.slot_rows(projection=p, scopes=scopes)
    rows: dict[str, dict[UtilitySurface, list[Trial]]] = {}
    for surface in surfaces:
        name = str(surface.samples)
        members = set(splits.get(name) or p.minted[name].members)
        labeled, judged = admitted.get(surface.judge, ((), ()))
        for records, from_labeled in ((labeled, True), (judged, False)):
            for _, record in records:
                if not p.counted_for(record, coordinate):
                    continue
                if from_labeled:
                    value, sample = record.pins.get(coordinate), cast("LabeledRef", record.subject).sample
                else:
                    outcome = p.outcomes[cast("OutcomeRef", record.subject).outcome]
                    sample = SampleKey(outcome.pins[axis])
                    value = str(outcome.node) if outcome.role == produced else record.pins.get(coordinate)
                if value is None or sample not in members:
                    continue
                trial = Trial(p.effective(record), 1.0 if name in splits else minted_mass)
                rows.setdefault(value, {}).setdefault(surface, []).append(trial)
    return {arm: {s: policy.paper_aggregation(t) for s, t in row.items()} for arm, row in rows.items()}


def minted_staged(
    *,
    host: Controller,
    name: str,
    one: MintedSet,
) -> tuple[dict[SampleKey, Mapping[str, Artifact]], dict[ids.CriterionCoordinate, str]]:
    """Return a minted set's staged member bundles and its pool-provenance pin."""
    staged = {
        member: _staged(host=host, name=name, source=source)
        for member, source in one.sources.items()
    }
    pool = host.projection.minted_by[name]
    return staged, {pool: host.projection.active_pins[pool]}


def _staged(*, host: Controller, name: str, source: ids.EventId) -> Mapping[str, Artifact]:
    produced = host.projection.outcomes[source].result
    if not isinstance(produced, TaskSuccess):
        msg = f"minted set {name!r} stages from a failed outcome: {source}"
        raise TypeError(msg)
    return produced.files
