# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""Accumulate the settled judgment streams a crossing reads, one event at a time."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from gm.domain.identity import SampleKey
from gm.domain import JudgedWork
from gm.ledger.events import LabeledRef, OutcomeRef, settled_verdict

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from gm.domain.identity import EvaluatorSlot
    from gm.ids import CriterionCoordinate, EventId, NodeId
    from gm.ledger.events import JudgmentRecorded, TaskOutcome
    from gm.ledger.projection import Fold


type _Row = tuple[EventId, JudgmentRecorded]


def _outcome_subject(*, record: JudgmentRecorded) -> OutcomeRef:
    subject = record.subject
    return subject


def _labeled_subject(*, record: JudgmentRecorded) -> LabeledRef:
    subject = record.subject
    return subject


@dataclasses.dataclass(slots=True, kw_only=True)
class _Streams:
    judged: list[JudgedWork] = dataclasses.field(default_factory=list)
    judged_at: int = 0
    labeled: dict[EvaluatorSlot, list[JudgedWork]] = dataclasses.field(default_factory=dict)
    labeled_at: dict[EvaluatorSlot, int] = dataclasses.field(default_factory=dict)
    rows: dict[EvaluatorSlot, tuple[list[_Row], list[_Row]]] = dataclasses.field(default_factory=dict)

    def bucket_for(self, *, slot: EvaluatorSlot) -> tuple[list[_Row], list[_Row]]:
        """Return one slot's labeled and judged row lists, creating them once."""
        return self.rows.setdefault(slot, ([], []))


class JudgedIndex:
    """Maintain the settled judgment streams and the slot grouping over them."""

    __slots__ = ("_streams",)

    def __init__(self) -> None:
        """Initialize an index that has read nothing."""
        self._streams: dict[frozenset[str], _Streams] = {}

    def _bucket(self, *, projection: Fold, scopes: Collection[str]) -> _Streams:
        key = frozenset(scopes)
        bucket = self._streams.get(key)
        if bucket is None:
            bucket = self._streams[key] = _Streams()
        axis = projection.manifest.coordinates.sample
        records = projection.judgments
        for index in range(bucket.judged_at, len(records)):
            record = records[index]
            if projection.scopes[record.action] not in key:
                continue
            ref = _outcome_subject(record=record).outcome
            outcome = projection.outcomes[ref]
            bucket.judged.append(
                _work(record=record, node=outcome.node, sample=SampleKey(outcome.pins[axis]), source=ref)
            )
            bucket.bucket_for(slot=record.slot)[1].append((ref, record))
        bucket.judged_at = len(records)
        for slot, labeled in projection.labeled.items():
            held = bucket.labeled.setdefault(slot, [])
            for index in range(bucket.labeled_at.get(slot, 0), len(labeled)):
                event_id, record = labeled[index]
                if projection.scopes[record.action] in key:
                    subject = _labeled_subject(record=record)
                    held.append(_work(record=record, node=subject.node, sample=subject.sample, source=None))
                    bucket.bucket_for(slot=slot)[0].append((event_id, record))
            bucket.labeled_at[slot] = len(labeled)
        return bucket

    def settle(
        self,
        *,
        scope: str,
        record: JudgmentRecorded,
        event_id: EventId,
        outcome: TaskOutcome | None,
        sample_axis: CriterionCoordinate,
    ) -> None:
        """Fold one settled judgment into every stream already asked for."""
        if outcome is not None:
            source = _outcome_subject(record=record).outcome
            work = _work(record=record, node=outcome.node, sample=SampleKey(outcome.pins[sample_axis]), source=source)
            for key, bucket in self._streams.items():
                bucket.judged_at += 1
                if scope in key:
                    bucket.judged.append(work)
                    bucket.bucket_for(slot=record.slot)[1].append((source, record))
            return
        subject = _labeled_subject(record=record)
        work = _work(record=record, node=subject.node, sample=subject.sample, source=None)
        for key, bucket in self._streams.items():
            bucket.labeled_at[record.slot] = bucket.labeled_at.get(record.slot, 0) + 1
            into = bucket.labeled.setdefault(record.slot, [])
            if scope in key:
                into.append(work)
                bucket.bucket_for(slot=record.slot)[0].append((event_id, record))

    def streams(
        self, *, projection: Fold, scopes: Collection[str]
    ) -> tuple[Sequence[JudgedWork], Mapping[EvaluatorSlot, Sequence[JudgedWork]]]:
        """Return the two settled judgment streams one scope set sees."""
        bucket = self._bucket(projection=projection, scopes=scopes)
        return bucket.judged, bucket.labeled

    def slot_rows(
        self, *, projection: Fold, scopes: Collection[str]
    ) -> Mapping[EvaluatorSlot, tuple[Sequence[_Row], Sequence[_Row]]]:
        """Return one scope set's admitted records, grouped by judging slot."""
        return self._bucket(projection=projection, scopes=scopes).rows


def _work(*, record: JudgmentRecorded, node: NodeId, sample: SampleKey, source: EventId | None) -> JudgedWork:
    verdict = settled_verdict(record=record)
    return JudgedWork(
        slot=record.slot,
        pins=record.pins,
        score=verdict.score,
        outcome=verdict.outcome,
        node=node,
        sample=sample,
        source=source,
        labels=verdict.labels,
    )
