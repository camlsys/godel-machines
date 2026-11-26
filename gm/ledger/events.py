# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""Closed typed events and their content identities."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from enum import StrEnum
from typing import Annotated, Final, Literal, cast

from pydantic import ConfigDict, Field, JsonValue, NonNegativeInt, PositiveInt, TypeAdapter
from pydantic.dataclasses import dataclass as validated

from gm.domain import BindingConfig, CoordinateDeclarations, EpisodeBinding, MintedSet, TrainSpec
from gm.domain.identity import EvaluatorSlot, RoleId, SampleKey, TaskId
from gm.domain.results import CONFIG as _CONFIG
from gm.domain.results import (
    Artifact,
    ArtifactRef,
    EvaluatorJudgment,
    ExpansionOutcome,
    InlineArtifact,
    JudgmentResult,
    TaskResult,
    TaskSuccess,
    artifact_address,
)
from gm.frozen import FrozenMapping, frozen_default
from gm.ids import ActionId, BindingTag, CriterionCoordinate, EventId, NodeId, NonBlank, NonEmpty, RunId, Sha256Hex, SortedSet, canonical_json
from gm.spend import Amount, AxisName, SpendRows, Weight

SCHEMA = 10
LEGACY_SCHEMA = 8
"""The schema every recorded ledger before this one was written in: it decodes, re-encodes and decides by its own rules."""
# What a schema-8 manifest and expansion arm state that this schema no longer names, kept verbatim on decode.
_LEGACY_MANIFEST = (
    "behavior_id", "ceilings", "clock", "episode_envelopes", "fence_admissions", "rate_revision", "resolved_config",
    "resolved_images", "resources", "sample_sets", "scopes", "surface", "tags", "train", "transition_digest", "watchdogs",
)
_LEGACY_ARM = ("call_seconds", "expected_wall_seconds", "gpus", "repository", "wall_seconds")
EVENT_KEY = "rqgm-event"
_PREFIX_SEED = b"rqgm-prefix"
DIALOGUE_ARTIFACT = "dialogue.log"
SEAL_MANIFEST_ARTIFACT = "pending.manifest.json"
FILE_EFFECTS_ARTIFACT = "file-effects.json"
SEAL_OMITTED: Final = "omitted"


class ActionKind(StrEnum):
    """Closed reservation kinds."""

    EXPANSION = "expansion"
    GENERATION = "generation"
    JUDGMENT = "judgment"


class CheckpointPhase(StrEnum):
    """Name each durable phase of a criterion-transition crossing."""

    IDLE = "idle"
    DRAIN = "drain"
    COMPARE = "compare"
    TRANSITION = "transition"


SEARCH, TRAIN = "search", "train"
"""The two fact scopes: search-time evaluations, and the train submissions a node makes from inside its expansion."""
CLOCK = "evaluations"
"""The budget axis: one unit per counted evaluation trial."""
CLOCK_RESERVE = 1.0
"""Units an open evaluation holds against the budget until it settles."""


@validated(frozen=True, slots=True, config=ConfigDict(extra="forbid", strict=True), kw_only=True)
class RunManifest:
    """What every decision and verification reads about a run, recorded as its first event."""

    run: RunId
    form: NonBlank
    seed: NonNegativeInt
    workers: PositiveInt
    budget: PositiveInt
    source: NodeId
    source_commit: NodeId
    pins: NonEmpty[FrozenMapping[CriterionCoordinate, NonBlank]]
    coordinates: CoordinateDeclarations
    snapshot_sets: SortedSet[NonBlank]
    seat_interpretations: FrozenMapping[EvaluatorSlot, NonBlank] = field(default_factory=frozen_default)
    bindings: FrozenMapping[BindingTag, BindingConfig] = field(default_factory=frozen_default)
    tags: FrozenMapping[RoleId, BindingTag] = field(default_factory=frozen_default)
    weights: FrozenMapping[BindingTag, FrozenMapping[AxisName, Weight]] = field(default_factory=frozen_default)
    images: FrozenMapping[NonBlank, NonBlank] = field(default_factory=frozen_default)
    train: TrainSpec | None = None
    #: The event schema this run's ledger is written in.
    version: Annotated[int, Field(exclude=True)] = SCHEMA
    #: A schema-8 manifest's fields this schema does not name, verbatim; empty for a run this schema started.
    recorded: Annotated[FrozenMapping[str, JsonValue], Field(exclude=True)] = field(default_factory=frozen_default)

    @property
    def ceilings(self) -> dict[str, dict[AxisName, Amount]]:
        """The budget per scope: the search scope is capped on the clock, training is not."""
        return {SEARCH: {CLOCK: self.budget}, TRAIN: {}}


@validated(frozen=True, slots=True, config=_CONFIG)
class RunConfigured:
    """Anchor a run with its immutable self-description."""

    manifest: RunManifest
    variant: Literal["run-configured"] = "run-configured"


@validated(frozen=True, slots=True, config=_CONFIG)
class ActionReserved:
    """One durable reservation issued before dispatch."""

    action: ActionId
    kind: ActionKind
    node: NodeId
    role: RoleId
    task: TaskId | None
    scope: NonBlank
    sample: SampleKey | None
    owed: bool = False
    slot: EvaluatorSlot | None = None
    variant: Literal["action-reserved"] = "action-reserved"


@validated(frozen=True, slots=True, config=_CONFIG)
class TaskOutcome:
    """One task execution settling a generation reservation."""

    action: ActionId
    node: NodeId
    role: RoleId
    task: TaskId
    pins: NonEmpty[FrozenMapping[CriterionCoordinate, NonBlank]]
    result: TaskResult
    variant: Literal["task-outcome"] = "task-outcome"


@validated(frozen=True, slots=True, config=_CONFIG)
class OutcomeRef:
    """Refer to the produced outcome scored by a validation judgment."""

    outcome: EventId


@validated(frozen=True, slots=True, config=_CONFIG)
class LabeledRef:
    """A sample-labeled judgment's subject: a candidate node probed on one sample."""

    node: NodeId
    sample: SampleKey


type JudgedSubject = OutcomeRef | LabeledRef


@validated(frozen=True, slots=True, config=_CONFIG)
class JudgmentRecorded:
    """One evaluator judgment settling a judgment reservation."""

    action: ActionId
    slot: EvaluatorSlot
    subject: JudgedSubject
    pins: NonEmpty[FrozenMapping[CriterionCoordinate, NonBlank]]
    judgment: JudgmentResult
    variant: Literal["judgment-recorded"] = "judgment-recorded"


def settled_verdict(*, record: JudgmentRecorded) -> EvaluatorJudgment:
    """Return the verdict a counted judgment record carries."""
    return record.judgment


@validated(frozen=True, slots=True, config=_CONFIG)
class ExpansionRecorded:
    """One expansion settlement: a published child or childless evidence."""

    action: ActionId
    parent: NodeId
    child: NodeId | None
    result: TaskResult
    experience_v4: Sha256Hex
    attempt_no: PositiveInt
    arm: EpisodeBinding
    outcome: ExpansionOutcome | None = None
    variant: Literal["expansion-recorded"] = "expansion-recorded"


@validated(frozen=True, slots=True, config=_CONFIG)
class CriterionTransition:
    """One committed pin replacement."""

    coordinate: CriterionCoordinate
    incumbent: NonBlank
    replacement: NonBlank
    epoch: PositiveInt
    minted: tuple[MintedSet, ...] = ()
    variant: Literal["criterion-transition"] = "criterion-transition"


@validated(frozen=True, slots=True, config=_CONFIG)
class AttemptDialogue:
    """Say which attempt of a retried action one recorded conversation belongs to."""

    attempt_no: PositiveInt
    spoken: Artifact


def attempt_dialogue_name(*, attempt_no: int) -> str:
    """Return the evidence path one attempt's conversation is filed under."""
    return f"attempt-{attempt_no}.{DIALOGUE_ARTIFACT}"


CASCADE_SKIP: Final = "the train seat it was reserved to judge ({producer}) authored no outcome"


def cascaded_skip(*, producer: object) -> str:
    """Return the reason recorded on a seat whose producer authored no outcome."""
    return CASCADE_SKIP.format(producer=producer)


@validated(frozen=True, slots=True, config=_CONFIG)
class SoftSkip:
    """Settle retry exhaustion without authoring a trial or candidate failure."""

    action: ActionId
    attempts: PositiveInt
    error: NonBlank
    spend: SpendRows
    dialogues: tuple[AttemptDialogue, ...] = ()
    variant: Literal["soft-skip"] = "soft-skip"


class NodeState(StrEnum):
    """Name the two moments in a node's life the store records, and no others."""

    ADMITTED = "admitted"
    FINAL_TURN = "final_turn"


@validated(frozen=True, slots=True, config=_CONFIG)
class NodeStateRecorded:
    """One moment in a node's life: admitted at its first turn, or its final turn's answer."""

    action: ActionId
    node: NodeId
    state: NodeState
    reason: NonBlank | None = None
    done: bool | None = None
    grounds: NonBlank | None = None
    variant: Literal["node-state-recorded"] = "node-state-recorded"


@validated(frozen=True, slots=True, config=_CONFIG)
class CheckpointPublished:
    """One atomic restart point."""

    last_event: EventId
    prefix_digest: Sha256Hex
    trigger: NonBlank | None
    scope: tuple[CriterionCoordinate, ...] | None
    variant: Literal["checkpoint-published"] = "checkpoint-published"


type LedgerEvent = (
    RunConfigured
    | ActionReserved
    | TaskOutcome
    | JudgmentRecorded
    | ExpansionRecorded
    | NodeStateRecorded
    | CriterionTransition
    | SoftSkip
    | CheckpointPublished
)


@dataclass(frozen=True, slots=True, kw_only=True)
class StoredEvent:
    """Pair one applied event with its durable ledger position and identity."""

    seq: int
    event_id: EventId
    event: LedgerEvent


@dataclass(frozen=True, slots=True, kw_only=True)
class CellSlot:
    """Identify one dispatch cell across every node, by role and task."""

    role: RoleId
    task: TaskId

    @classmethod
    def of(cls, *, role: RoleId, task: TaskId) -> CellSlot:
        """Return *the* slot for this role and task, built once and shared."""
        key = (role, task)
        got = _slots.get(key)
        if got is None:
            got = _slots[key] = cls(role=role, task=task)
        return got


_slots: dict[tuple[RoleId, TaskId], CellSlot] = {}


@dataclass(frozen=True, slots=True, kw_only=True)
class Cell:
    """Identify one dispatch cell by its node, role, and task."""

    node: NodeId
    role: RoleId
    task: TaskId

    @property
    def slot(self) -> CellSlot:
        """This cell's node-independent owner-map key."""
        return CellSlot.of(role=self.role, task=self.task)


def canonical_fact(*, value: object) -> object:
    """Return a validated fact in its canonical identity representation."""
    if isinstance(value, InlineArtifact | ArtifactRef):
        oid, size = artifact_address(artifact=value)
        return {"sha256": str(oid), "size": size}
    if is_dataclass(value):
        return {f.name: canonical_fact(value=getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {
            str(key): canonical_fact(value=item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return [canonical_fact(value=item) for item in value] if isinstance(value, tuple) else value


def prefix_digest_ids(*, events: Sequence[str]) -> str:
    """Return the running prefix digest over event ids alone."""
    digest = hashlib.sha256(_PREFIX_SEED).digest()
    for one in events:
        digest = hashlib.sha256(digest + one.encode()).digest()
    return digest.hex()


def prefix_digest(*, events: Sequence[StoredEvent]) -> str:
    """Return the running prefix digest with the applied event ids folded in."""
    return prefix_digest_ids(events=[stored.event_id for stored in events])


_ROW_CODEC: TypeAdapter[LedgerEvent] = TypeAdapter(Annotated[LedgerEvent, Field(discriminator="variant")])


def event_files(*, event: LedgerEvent) -> Mapping[str, Artifact]:
    """Return artifact handles carried outside one event's canonical payload."""
    match event:
        case TaskOutcome(result=TaskSuccess() as result) | ExpansionRecorded(result=TaskSuccess() as result):
            channels = (("files", result.files), ("metadata", result.metadata))
        case TaskOutcome(result=result) | ExpansionRecorded(result=result):
            channels = (("files", result.files),)
        case JudgmentRecorded(judgment=judgment):
            channels = (("files", judgment.files),)
        case SoftSkip(dialogues=spoken) if spoken:
            channels = (("files", {attempt_dialogue_name(attempt_no=one.attempt_no): one.spoken for one in spoken}),)
        case _:
            channels = ()
    return {
        canonical_json([channel, path]): artifact
        for channel, artifacts in channels
        for path, artifact in artifacts.items()
    }


def _claim_attempts(*, payload: dict[str, object], event: SoftSkip) -> None:
    payload.pop("dialogues", None)
    if event.dialogues:
        payload["dialogues"] = [{"attempt_no": one.attempt_no} for one in event.dialogues]


def event_payload(*, event: LedgerEvent, version: int = SCHEMA) -> dict[str, object]:
    """Return a payload tagged with the ledger's schema, with artifacts represented by a manifest."""
    payload: dict[str, object] = {EVENT_KEY: version, **_ROW_CODEC.dump_python(event, mode="json")}
    files = event_files(event=event)
    holder = payload.get("result") or payload.get("judgment")
    manifest = []
    for key, artifact in sorted(files.items()):
        channel, path = json.loads(key)
        oid, size = artifact_address(artifact=artifact)
        manifest.append([channel, path, str(oid), size])
    if isinstance(holder, dict):
        typed_holder = cast("dict[str, object]", holder)
        typed_holder.pop("files", None)
        typed_holder.pop("metadata", None)
    if isinstance(event, SoftSkip):
        _claim_attempts(payload=payload, event=event)
    _drop_unstated(payload=payload, event=event)
    if version == LEGACY_SCHEMA:
        _as_legacy(payload=payload, event=event)
    if manifest:
        payload["artifacts"] = manifest
    return payload


def _drop_unstated(*, payload: dict[str, object], event: LedgerEvent) -> None:
    if isinstance(event, ExpansionRecorded) and event.outcome is None:
        payload.pop("outcome", None)
    if isinstance(event, ExpansionRecorded) and event.arm.idle_seconds is None:
        arm = payload.get("arm")
        if isinstance(arm, dict):
            cast("dict[str, object]", arm).pop("idle_seconds", None)
    if isinstance(event, NodeStateRecorded):
        for absent in ("reason", "done", "grounds"):
            if getattr(event, absent) is None:
                payload.pop(absent, None)
    if isinstance(event, ActionReserved) and event.slot is None:
        payload.pop("slot", None)


def _as_legacy(*, payload: dict, event: LedgerEvent) -> None:
    """Spell a payload the way schema 8 recorded it: its own manifest and arm fields back in place of derived ones."""
    match event:
        case RunConfigured(manifest=manifest):
            for derived in ("budget", "snapshot_sets", "images"):
                del payload["manifest"][derived]
            payload["manifest"].update(manifest.recorded)
        case ExpansionRecorded(arm=arm):
            del payload["arm"]["checkout"]
            payload["arm"].update(arm.recorded)


def _from_legacy(raw: dict) -> None:
    """Read a schema-8 payload into this schema's shape, keeping what this schema does not name as recorded data."""
    if raw["variant"] == "run-configured":
        anchor = raw["manifest"]
        old = {key: anchor.pop(key) for key in _LEGACY_MANIFEST}
        anchor.update(
            version=LEGACY_SCHEMA, recorded=old, budget=old["ceilings"][SEARCH][CLOCK],
            snapshot_sets=old["sample_sets"]["snapshot"], images=old["resolved_images"],
            tags={role: one["tag"] for role, one in old["tags"].items()},
            train=old["train"] and {"task_sets": old["train"]["task_sets"]},
        )
    elif raw["variant"] == "expansion-recorded":
        raw["arm"]["recorded"] = {key: raw["arm"].pop(key) for key in _LEGACY_ARM}


def encoded(event: LedgerEvent, version: int = SCHEMA) -> tuple[EventId, str]:
    """Return the event's identifier and canonical bytes in the ledger's schema, encoding it once."""
    text = canonical_json(event_payload(event=event, version=version))
    return EventId(hashlib.sha256(text.encode()).hexdigest()), text


def event_from_json(*, text: str, files: Mapping[str, Artifact]) -> LedgerEvent:
    """Return the event decoded from a stored payload and the artifact handles its manifest names."""
    raw = json.loads(text)
    if raw.pop(EVENT_KEY) == LEGACY_SCHEMA:
        _from_legacy(raw)
    restored: dict[str, dict[str, Artifact]] = {"files": {}, "metadata": {}}
    for channel, path, _, _ in raw.pop("artifacts", []):
        if (artifact := files.get(canonical_json([channel, path]))) is not None:
            restored[channel][path] = artifact
    holder = raw.get("result") or raw.get("judgment")
    if isinstance(holder, dict):
        holder["files"] = restored["files"]
        if raw.get("variant") in {"task-outcome", "expansion-recorded"} and "error" not in holder:
            holder["metadata"] = restored["metadata"]
    elif raw.get("variant") == "soft-skip":
        raw["dialogues"] = [
            {
                "attempt_no": one["attempt_no"],
                "spoken": restored["files"][attempt_dialogue_name(attempt_no=one["attempt_no"])],
            }
            for one in raw.get("dialogues") or ()
        ]
    return _ROW_CODEC.validate_python(raw, strict=False)
