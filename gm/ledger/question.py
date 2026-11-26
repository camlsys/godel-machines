# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""The expansion-question digest: what a parent's chain of experience says, at the active pins, as one sha256.

The digest spells one canonical JSON payload (chain, pins, axes, economy, the chain's experience rows, the
events and artifacts they select). Each node's share of it (its rows, selected events and artifacts, already
canonical) is kept per fold and rebuilt only when an outcome or judgment on that node lands or the pins move,
so a question costs one join over the chain's cached pieces, not a rescan of its evidence.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from itertools import chain as chained
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

from gm.domain.identity import DatasetVersion, EvaluatorSlot, RoleId, SampleKey, TaskId
from gm.domain.results import Artifact, EvaluatorJudgment, TaskResult, TaskSuccess, artifact_address
from gm.frozen import freeze_fields
from gm.ids import BindingTag, CriterionCoordinate, EventId, NodeId, canonical_json
from gm.ledger.events import DIALOGUE_ARTIFACT, FILE_EFFECTS_ARTIFACT, JudgmentRecorded, StoredEvent, canonical_fact

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from gm.ledger.projection import Fold
    from gm.spend import AxisName


@dataclass(frozen=True, slots=True, kw_only=True)
class _ExperienceRow:
    production_seq: int
    production_event: EventId
    generation: int
    node: NodeId
    dataset: DatasetVersion | None
    sample: SampleKey
    role: RoleId
    binding_tag: BindingTag | None
    binding_endpoint: str | None
    binding_effort: str | None
    resources: Mapping[AxisName, int | float]
    task: TaskId
    result: TaskResult
    judgment_seq: int | None
    judgment_event: EventId | None
    slot: EvaluatorSlot | None
    judgment: EvaluatorJudgment | None
    effective_outcome: float | None

    def __post_init__(self) -> None:
        """Detach the exported spend row from the caller that accumulated it."""
        freeze_fields(self, "resources")

    def artifacts(self) -> tuple[Artifact, ...]:
        """Return every handle this row stands behind."""
        result = self.result
        return (
            *result.files.values(),
            *(result.metadata.values() if isinstance(result, TaskSuccess) else ()),
            *(self.judgment.files.values() if self.judgment else ()),
        )


@dataclass(frozen=True, slots=True)
class _Piece:
    """One node's canonical share of a question, valid while its key holds."""

    key: tuple
    rows: str
    selected: list[tuple[int, str]]
    artifacts: set[tuple[str, int]]


@dataclass(slots=True, kw_only=True)
class _Model:
    """What the fold's appends have added since the last question, indexed by node."""

    last_seq: int = 0
    outcomes_seen: int = 0
    judgments_seen: int = 0
    by_node: dict[NodeId, list[StoredEvent]] = field(default_factory=dict)
    grouped: dict[EventId, dict[EvaluatorSlot, list[StoredEvent]]] = field(default_factory=dict)
    #: Bumped on every outcome or judgment about a node: a node's piece is current while its version is.
    versions: dict[NodeId, int] = field(default_factory=dict)
    pins: tuple = ()
    pieces: dict[NodeId, _Piece] = field(default_factory=dict)


_MODELS: WeakKeyDictionary[Fold, _Model] = WeakKeyDictionary()


def _model(fold: Fold) -> _Model:
    model = _MODELS.get(fold)
    if model is None or fold.last_seq < model.last_seq:
        model = _MODELS[fold] = _Model()
    for stored in fold.outcome_events[model.outcomes_seen:]:
        node = stored.event.node
        model.by_node.setdefault(node, []).append(stored)
        model.versions[node] = model.versions.get(node, 0) + 1
    model.outcomes_seen = len(fold.outcome_events)
    for stored in fold.judgment_events[model.judgments_seen:]:
        record = stored.event
        model.grouped.setdefault(record.subject.outcome, {}).setdefault(record.slot, []).append(stored)
        node = fold.outcomes[record.subject.outcome].node
        model.versions[node] = model.versions.get(node, 0) + 1
    model.judgments_seen = len(fold.judgment_events)
    model.last_seq = fold.last_seq
    return model


def _settled(fold: Fold, pins: Mapping[CriterionCoordinate, str], group: list[StoredEvent]) -> StoredEvent:
    """The judgment a slot's group stands on: the latest whose pins are current, else the latest."""
    currency = fold.manifest.coordinates.currency
    exact = [one for one in group if all(
        not currency.get(coordinate, True) or pins.get(coordinate) == value for coordinate, value in one.event.pins.items()
    )]
    return max(exact or group, key=lambda one: one.seq)


def _piece(
    fold: Fold, model: _Model, node: NodeId, generation: int, published: StoredEvent | None,
    pins: Mapping[CriterionCoordinate, str], axes: tuple[AxisName, ...],
) -> _Piece:
    key = (model.versions.get(node, 0), generation, published is not None)
    if (held := model.pieces.get(node)) is not None and held.key == key:
        return held
    manifest = fold.manifest
    coordinates = manifest.coordinates
    rows: list[_ExperienceRow] = []
    for stored in model.by_node.get(node, ()):
        outcome = stored.event
        if coordinates.dataset not in outcome.pins or coordinates.sample not in outcome.pins:
            continue
        tag = manifest.tags.get(outcome.role)
        binding = None if tag is None else manifest.bindings[tag]
        resources = {axis: sum(amount for _, current, amount in outcome.result.spend if current == axis) for axis in axes}
        for group in model.grouped.get(stored.event_id, {}).values():
            selected = _settled(fold, pins, group)
            record: JudgmentRecorded = selected.event
            rows.append(_ExperienceRow(
                production_seq=stored.seq, production_event=stored.event_id, generation=generation, node=node,
                dataset=DatasetVersion(outcome.pins[coordinates.dataset]), sample=SampleKey(outcome.pins[coordinates.sample]),
                role=outcome.role, binding_tag=tag,
                binding_endpoint=None if binding is None else str(binding.endpoint),
                binding_effort=None if binding is None else binding.effort,
                resources=resources, task=outcome.task, result=outcome.result,
                judgment_seq=selected.seq, judgment_event=selected.event_id, slot=record.slot,
                judgment=record.judgment, effective_outcome=fold.effective(record, active_pins=pins),
            ))
    rows.sort(key=lambda row: (row.production_seq, row.judgment_seq))
    selected = {(row.production_seq, row.production_event) for row in rows}
    selected |= {(row.judgment_seq, row.judgment_event) for row in rows}
    artifacts = {artifact_address(artifact=artifact) for row in rows for artifact in row.artifacts()}
    # The expansion that published this node from a parent on the chain: its dialogue is selected evidence, its
    # dialogue and file effects are selected artifacts.
    if published is not None:
        stored, result = published, published.event.result
        holder = result.metadata if isinstance(result, TaskSuccess) else result.files
        if (dialogue := holder.get(DIALOGUE_ARTIFACT)) is not None:
            selected.add((stored.seq, stored.event_id))
            artifacts.add(artifact_address(artifact=dialogue))
        if (effects := holder.get(FILE_EFFECTS_ARTIFACT)) is not None:
            artifacts.add(artifact_address(artifact=effects))
    piece = _Piece(
        key=key,
        rows=",".join(canonical_json(canonical_fact(value=row)) for row in rows),
        selected=sorted((seq, canonical_json([seq, str(event)])) for seq, event in selected),
        artifacts={(str(oid), size) for oid, size in artifacts},
    )
    model.pieces[node] = piece
    return piece


def question_digest(
    fold: Fold, chain: Sequence[NodeId], pins: Mapping[CriterionCoordinate, str], axes: tuple[AxisName, ...],
) -> str:
    """Return the digest of the question an expansion of `chain[-1]` asks, for a root-first ancestry chain."""
    model = _model(fold)
    frozen_pins = tuple(sorted(pins.items(), key=lambda item: str(item[0])))
    # What a row means moves with the pins and with what the run has minted (both change only at a transition).
    pinned = (frozen_pins, tuple(sorted(fold.epochs.items())), axes)
    if model.pins != pinned:
        model.pins, model.pieces = pinned, {}
    on_chain = frozenset(chain)
    published = {
        stored.event.child: stored
        for parent in chain for stored in fold.expansion_attempts.get(parent, ()) if stored.event.child in on_chain
    }
    pieces = [_piece(fold, model, node, generation, published.get(node), pins, axes) for generation, node in enumerate(chain)]
    economy = sorted(
        (str(tag), axis, float(factor)) for tag, factors in fold.manifest.weights.items() for axis, factor in factors.items()
    )
    selected = sorted(set(chained.from_iterable(piece.selected for piece in pieces)))
    artifacts = sorted(set().union(*(piece.artifacts for piece in pieces)))
    text = (
        '{"axes":' + canonical_json(list(axes))
        + ',"chain":' + canonical_json([str(node) for node in chain])
        + ',"economy":' + canonical_json([list(one) for one in economy])
        + ',"pins":' + canonical_json([[str(coordinate), value] for coordinate, value in frozen_pins])
        + ',"rows":[' + ",".join(piece.rows for piece in pieces if piece.rows)
        + '],"selected":[' + ",".join(one for _, one in selected)
        + '],"selected_artifacts":[' + ",".join(canonical_json([oid, size]) for oid, size in artifacts) + "]}"
    )
    return hashlib.sha256(text.encode()).hexdigest()
