# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""Derive the settle side without touching the writer."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from gm import ids, ledger
from gm import work_queue as wq
from gm.domain.identity import EvaluatorSlot, RoleId, SampleKey
from gm.domain.results import (
    EvaluatorJudgment,
    ExpansionOutcome,
    InvalidJudgment,
    TaskFailure,
    TaskResult,
    TaskSuccess,
)
from gm.frozen import freeze_fields
from gm.spend import AxisName, canonical, merged

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from gm.domain.manifest import DomainManifest
    from gm.domain import CoordinateDeclarations


def _is_trial(*, reservation: ledger.ActionReserved, event: ledger.LedgerEvent) -> bool:
    match event:
        case ledger.TaskOutcome(result=TaskFailure()):
            return True
        case ledger.JudgmentRecorded():
            return reservation.task is not None
        case _:
            return False


def _judges_a_failure(*, event: ledger.LedgerEvent, projection: ledger.Fold) -> bool:
    if not isinstance(event, ledger.JudgmentRecorded) or not isinstance(event.subject, ledger.OutcomeRef):
        return False
    produced = projection.outcomes.get(event.subject.outcome)
    return produced is not None and isinstance(produced.result, TaskFailure)


def clock_units(
    reservation: ledger.ActionReserved,
    event: ledger.LedgerEvent,
    projection: ledger.Fold,
) -> Mapping[AxisName, int | float]:
    """Return one settlement's metered units: one clock unit per counted trial, none otherwise.

    Not counted: a settlement outside the clock's scopes, a judgment its cover does not close,
    a judgment of a failed generation, and anything that is not a trial.
    """
    counted = (
        str(reservation.scope) == ledger.SEARCH
        and not (isinstance(event, ledger.JudgmentRecorded) and projection.covers.remaining(reservation.action) != 1)
        and not _judges_a_failure(event=event, projection=projection)
        and _is_trial(reservation=reservation, event=event)
    )
    return {ledger.CLOCK: 1} if counted else {}


def _authored(
    record: ledger.JudgmentRecorded,
    projection: ledger.Fold,
) -> ledger.JudgmentRecorded:
    if not isinstance(record.judgment, EvaluatorJudgment):
        return record
    seat = projection.manifest.seat_interpretations.get(record.slot)
    if seat is not None:
        pinned = seat
    else:
        interpretation = projection.manifest.coordinates.interpretation
        if interpretation is None or (found := record.pins.get(interpretation)) is None:
            return record
        pinned = found
    meant = record.judgment.utility(interpretation=pinned)
    if meant == record.judgment.outcome:
        return record
    return dataclasses.replace(record, judgment=dataclasses.replace(record.judgment, outcome=meant))


def _publication(
    *,
    child: ids.NodeId | None,
    result: TaskResult,
    held: Mapping[ids.NodeId, ids.NodeId],
) -> ExpansionOutcome:
    if isinstance(result, TaskFailure):
        return ExpansionOutcome.FAILED
    if child is None:
        return ExpansionOutcome.DECLINED
    if child in held:
        return ExpansionOutcome.DUPLICATE
    return ExpansionOutcome.PUBLISHED


def settle(
    job: wq.Job,
    outcome: wq.Completion,
    projection: ledger.Fold,
) -> ledger.LedgerEvent:
    """Return the one typed settlement event for a completion."""
    match job, outcome:
        case (
            wq.ExpansionJob() | wq.GenerationJob(),
            wq.Skipped(attempts=attempts, error=error, spend=spend),
        ):
            event: ledger.LedgerEvent = ledger.SoftSkip(
                job.action,
                attempts,
                error,
                spend,
                outcome.dialogues,
            )
        case (wq.ExpansionJob() as e, wq.ExpansionCompletion(child=child, result=result)):
            hashed = _publication(child=child, result=result, held=projection.lineage)
            child = None if child in projection.lineage else child
            event = ledger.ExpansionRecorded(
                e.action,
                e.parent,
                child,
                result,
                e.export.expected_hash,
                e.attempt_no,
                e.binding,
                hashed,
            )
        case (wq.GenerationJob() as g, TaskSuccess() | TaskFailure() as result):
            event = ledger.TaskOutcome(g.action, g.node, g.role, g.task, g.pins, result)
        case (wq.JudgmentJob() as j, EvaluatorJudgment() | InvalidJudgment() as result):
            event = _authored(
                ledger.JudgmentRecorded(j.action, j.slot, j.subject, j.pins, result),
                projection,
            )
    reservation = projection.open_actions[job.action]
    units = clock_units(reservation, event, projection)
    rows = canonical((None, axis, amount) for axis, amount in units.items())
    if rows:
        match event:
            case ledger.TaskOutcome(result=result):
                event = dataclasses.replace(
                    event,
                    result=dataclasses.replace(result, spend=merged(result.spend, rows)),
                )
            case ledger.JudgmentRecorded(judgment=judgment):
                event = dataclasses.replace(
                    event,
                    judgment=dataclasses.replace(judgment, spend=merged(judgment.spend, rows)),
                )
            case ledger.ExpansionRecorded(result=result):
                event = dataclasses.replace(
                    event,
                    result=dataclasses.replace(result, spend=merged(result.spend, rows)),
                )
            case ledger.SoftSkip(spend=spend):
                event = dataclasses.replace(event, spend=merged(spend, rows))
    return event


def generation_pins(
    *,
    projection: ledger.Fold,
    domain: DomainManifest,
    sample: SampleKey,
    role: RoleId | None = None,
) -> dict[ids.CriterionCoordinate, str]:
    """Return a generated artifact's identity-dependency pins: the dataset build, the sample and the role's provenance."""
    axes = projection.manifest.coordinates
    return {axes.dataset: projection.active_pins[axes.dataset], axes.sample: str(sample)} | provenance_pin(projection, role)


def provenance_pin(projection: ledger.Fold, role: object) -> dict[ids.CriterionCoordinate, str]:
    """Return the model-provenance pin a schema-8 run recorded on what `role` produced or judged."""
    coordinate = projection.settings.provenance.get(str(role))
    return {} if coordinate is None else {ids.CriterionCoordinate(coordinate): projection.active_pins[coordinate]}


def judging_pins(
    projection: ledger.Fold,
    *coordinates: ids.CriterionCoordinate,
) -> dict[ids.CriterionCoordinate, str]:
    """Return the pins a judgment takes its *meaning* from, at the active vector."""
    pins: dict[ids.CriterionCoordinate, str] = {}
    interpretation = projection.manifest.coordinates.interpretation
    for coordinate in (*coordinates, *((interpretation,) if interpretation else ())):
        if (value := projection.active_pins.get(coordinate)) is not None:
            pins[coordinate] = value
    return pins


def subject_pins(
    *,
    axes: CoordinateDeclarations,
    outcome: ledger.TaskOutcome,
) -> dict[ids.CriterionCoordinate, str]:
    """Return the structural identity a judgment inherits from its subject."""
    return {
        axes.dataset: outcome.pins[axes.dataset],
        axes.sample: outcome.pins[axes.sample],
    }


def judgment_pins(
    projection: ledger.Fold,
    domain: DomainManifest,
    sample: SampleKey,
    *coordinates: ids.CriterionCoordinate,
) -> dict[ids.CriterionCoordinate, str]:
    """Return the pins a judgment's generation and meaning depend on."""
    return generation_pins(projection=projection, domain=domain, sample=sample) | judging_pins(
        projection,
        *coordinates,
    )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class OwedJudgment:
    """One judgment a produced success still owes, resolved and ready to reserve."""

    slot: EvaluatorSlot
    subject: ledger.OutcomeRef
    sample: SampleKey
    node: ids.NodeId
    agent: RoleId | None
    pins: Mapping[ids.CriterionCoordinate, str]

    def __post_init__(self) -> None:
        """Detach the dependency vector recorded against this owed judgment."""
        freeze_fields(self, "pins")


def owed_judgments(
    *,
    projection: ledger.Fold,
    domain: DomainManifest,
    ref: ids.EventId,
    outcome: ledger.TaskOutcome,
    seated_at: Callable[[EvaluatorSlot], ids.NodeId] | None = None,
) -> tuple[OwedJudgment, ...]:
    """Return which judgments a produced success still owes."""
    axes = projection.manifest.coordinates
    sample = SampleKey(outcome.pins[axes.sample])
    structure = subject_pins(axes=axes, outcome=outcome)
    subject = ledger.OutcomeRef(ref)
    interpretation = axes.interpretation
    owed: list[OwedJudgment] = []
    for slot, roles in domain.jurisdiction.items():
        if outcome.role not in roles:
            continue
        coordinate = ids.CriterionCoordinate(str(slot))
        agent = RoleId(str(slot)) if slot in domain.agent_slots else None
        node = (
            (str(seated_at(slot)) if seated_at is not None else projection.active_pins[coordinate])
            if agent
            else outcome.node
        )
        pins = (
            structure
            | judging_pins(projection, coordinate)
            | provenance_pin(projection, slot)
            | ({coordinate: node} if agent else {})
        )
        if projection.holds_verdict(slot=slot, subject=subject, pins=pins, interpretation=interpretation):
            continue  # Raw judgment is current; interpretation folds it again.
        owed.append(
            OwedJudgment(
                slot=slot,
                subject=subject,
                sample=sample,
                node=ids.NodeId(node),
                agent=agent,
                pins=pins,
            ),
        )
    return tuple(owed)
