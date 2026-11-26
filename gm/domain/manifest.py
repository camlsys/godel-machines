# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""A form, read once from `form.json`, its `genesis/` tree and its domain's data into one frozen manifest.

The manifest also writes the contract every episode is staged at `.rqgm/contract.json`.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from gm.domain.dataset import DatasetSpec, SampleSetName, TrainSpec, read_dataset
from gm.domain.identity import DomainId, EvaluatorSlot, RoleId, ScoringRuleId, TaskId
from gm.domain.seams import ProcessInstrument
from gm.domain.standard import (
    CompositeObjective,
    Curriculum,
    PoolCriterion,
    PoolMint,
    SurfaceSide,
    TransitionPolicy,
    TransitionSpec,
    UtilitySurface,
)
from gm.ids import BindingTag, Ceiling, CriterionCoordinate, canonical_json

REPO = Path(__file__).resolve().parents[2]
#: The node layers every form's root tree is composed from, below its meta folder and its own `genesis/`.
NODE_LAYERS = ("shared", "task_agents", "tests", "pytest.ini")
OUTPUT_TOKENS_ENV = "RQGM_AGENT_MAX_OUTPUT_TOKENS"
SEQUENCE_TOKENS_ENV = "RQGM_AGENT_MAX_SEQUENCE_TOKENS"
SEARCH = "search"


@dataclass(frozen=True, slots=True, kw_only=True)
class CoordinateDeclarations:
    """The structural axes a run pins, and whether each coordinate's value must be current for evidence to count."""

    currency: Mapping[CriterionCoordinate, bool]
    sample: CriterionCoordinate
    dataset: CriterionCoordinate
    interpretation: CriterionCoordinate | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class BindingConfig:
    """One model endpoint, its provider settings, and its token ceilings."""

    endpoint: str
    effort: str | None = None
    kwargs: Mapping[str, object] = field(default_factory=dict)
    output_tokens: Ceiling
    sequence_tokens: Ceiling

    def agent_env(self) -> dict[str, str]:
        """Return this binding's ceilings, for the agent that runs under it."""
        return {OUTPUT_TOKENS_ENV: str(self.output_tokens), SEQUENCE_TOKENS_ENV: str(self.sequence_tokens)}


@dataclass(frozen=True, slots=True, kw_only=True)
class EpisodeBinding:
    """One episode: its command, its image digest, its model binding, and the files it writes."""

    argv: tuple[str, ...]
    image: str
    tag: BindingTag | None = None
    outputs: Mapping[str, str] = field(default_factory=dict)
    evidence: Mapping[str, str] = field(default_factory=dict)
    token_allotment: Ceiling | None = None
    idle_seconds: float | None = None
    #: Where the sample's own repository sits in its image; set, the episode runs in the sample's image.
    checkout: str | None = None
    #: A schema-8 arm's fields this schema does not name, verbatim; empty for an episode this schema declared.
    recorded: Annotated[Mapping[str, object], Field(exclude=True)] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class StageSpec:
    """What a judgment receives: the produced files it reads, channels staged as files, and the channels it settles."""

    identity: bool
    reads: frozenset[str]
    prepare: Mapping[str, str] = field(default_factory=dict)
    finish: ProcessInstrument | None = None
    settles: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True, kw_only=True)
class FixedInstrument:
    """A judge only the host's settlement runs: it never evolves."""

    implementation: ProcessInstrument


@dataclass(frozen=True, slots=True, kw_only=True)
class ScorerSpec:
    """One judge slot: its rule, who runs it (a node episode or a fixed instrument), and its staging."""

    rule: ScoringRuleId
    scorer: EpisodeBinding | FixedInstrument
    stage: StageSpec
    interpretation: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class GeneratorSpec:
    """The producing roles: their tasks, episodes, the channel each reads, and the set each task searches."""

    task_pools: Mapping[RoleId, tuple[TaskId, ...]]
    episodes: Mapping[RoleId, EpisodeBinding]
    reads: Mapping[RoleId, str | None]
    task_sets: Mapping[TaskId, SampleSetName]


@dataclass(frozen=True, slots=True, kw_only=True)
class DomainManifest:
    """Everything a run reads about its form."""

    name: str
    domain: DomainId
    dataset: DatasetSpec
    generator: GeneratorSpec
    scorers: Mapping[EvaluatorSlot, ScorerSpec]
    jurisdiction: Mapping[EvaluatorSlot, frozenset[RoleId]]
    transition: TransitionSpec
    coordinates: CoordinateDeclarations
    interpretations: frozenset[str]
    pins: Mapping[CriterionCoordinate, str]
    train: TrainSpec | None
    meta: EpisodeBinding
    workspace: Mapping[str, bytes]
    contract: bytes
    agent_slots: frozenset[EvaluatorSlot]
    tags: Mapping[RoleId, BindingTag]
    bindings: Mapping[BindingTag, BindingConfig]
    weights: Mapping[BindingTag, Mapping[str, float]]
    axes: tuple[str, ...]
    cpus: float
    memory_bytes: int | Literal["unbounded"]
    images: Mapping[str, str] = field(default_factory=dict)


def _episode(declared: Mapping) -> EpisodeBinding:
    return EpisodeBinding(
        argv=tuple(declared["argv"]),
        image=declared["image"],
        tag=None if declared.get("tag") is None else BindingTag(declared["tag"]),
        outputs=dict(declared.get("outputs", {})),
        evidence=dict(declared.get("evidence", {})),
        token_allotment=declared.get("token_allotment"),
        idle_seconds=declared.get("idle_seconds"),
        checkout=declared.get("checkout"),
    )


def _policy(declared: Mapping) -> TransitionPolicy:
    match declared["kind"]:
        case "curriculum":
            return Curriculum(at=declared["at"], source=declared["source"], target=declared["target"])
        case "objective_transition":
            return CompositeObjective(
                surfaces={
                    UtilitySurface(judge=EvaluatorSlot(one["judge"]), samples=one["samples"]): SurfaceSide(one["side"])
                    for one in declared["surfaces"]
                },
                min_trials=declared["min_trials"],
            )
        case "pool":
            criterion = declared["criterion"]
            return PoolMint(
                name=declared["name"],
                criterion=PoolCriterion(
                    slot=EvaluatorSlot(criterion["slot"]), low=criterion["low"], high=criterion["high"],
                ),
                signer=CriterionCoordinate(declared["signer"]),
                judge=EvaluatorSlot(declared["judge"]) if declared.get("judge") else None,
                interpretation=declared.get("interpretation"),
            )
    raise ValueError(f"unknown transition kind {declared['kind']!r}")


def compose(*, genesis: Path, meta: str) -> dict[str, bytes]:
    """Return a form's root tree: the node layers, then `node/<meta>/` under `meta/`, then its `genesis/`."""
    node = REPO / "node"
    layers = [(node / entry, "" if entry == "pytest.ini" else entry) for entry in NODE_LAYERS]
    layers += [(node / meta, "meta"), (genesis, "")]
    files: dict[str, bytes] = {}
    for root, prefix in layers:
        if root.is_file():
            files[f"{prefix}/{root.name}".lstrip("/")] = root.read_bytes()
            continue
        for directory, folders, names in os.walk(root):
            folders[:] = sorted(one for one in folders if one != "__pycache__")
            for name in sorted(names):
                path = Path(directory) / name
                files[f"{prefix}/{path.relative_to(root).as_posix()}".lstrip("/")] = path.read_bytes()
    return _with_harnesses(files)


_HARNESS_CACHE = Path.home() / ".cache" / "rqgm" / "harness"


def _selected(path: str, keys: list[str]) -> bool:
    """Whether a path is one of `keys`, or lies under one ending in `/`."""
    return any(path.startswith(k) if k.endswith("/") else path == k for k in keys)


def _archive(*, url: str, sha256: str) -> bytes:
    cached = _HARNESS_CACHE / f"{sha256}.tar.gz"
    if cached.is_file() and hashlib.sha256(data := cached.read_bytes()).hexdigest() == sha256:
        return data
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    if (digest := hashlib.sha256(data).hexdigest()) != sha256:
        raise ValueError(f"{url} hashes to sha256 {digest}, and its harness.json pins {sha256}")
    _HARNESS_CACHE.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(data)
    return data


def _with_harnesses(files: dict[str, bytes]) -> dict[str, bytes]:
    """Replace each `harness.json` with the pinned upstream files it names, its local overrides, and `upstream.json`."""
    grown = dict(files)
    for name in sorted(one for one in files if Path(one).name == "harness.json"):
        here = name[: -len("harness.json")]
        declared = json.loads(files[name])
        archive, rule = declared["archive"], declared["fetched"]
        with tarfile.open(fileobj=io.BytesIO(_archive(url=archive["url"], sha256=archive["sha256"])), mode="r:gz") as tar:
            for member in tar.getmembers():
                inside = "/".join(member.name.split("/")[archive["strip_components"]:])
                if member.isfile() and _selected(inside, rule["take"]) and not _selected(inside, rule["skip"]):
                    grown[f"{here}{inside.removeprefix(rule['prefix'])}"] = tar.extractfile(member).read()
        del grown[name]
        for source, target in declared["local"].items():
            grown[f"{here}{target}"] = grown.pop(f"{here}{source}")
        record = {key: value for key, value in declared.items() if key not in ("fetched", "local")}
        grown[f"{here}upstream.json"] = json.dumps(record, indent=2).encode()
    return grown


def _contract(
    *, generator: GeneratorSpec, scorers: Mapping[EvaluatorSlot, ScorerSpec],
    jurisdiction: Mapping[EvaluatorSlot, frozenset[RoleId]], train: TrainSpec | None,
    dataset: DatasetSpec, workspace: Mapping[str, bytes],
) -> bytes:
    """The facts a node cannot read off itself: roles, judges, channels, set sizes, and each seat's I/O."""
    rows = dataset.rows()

    def half(name: str | None) -> dict | None:
        return None if name is None else {"set": name, "rows": len(rows[name]) if name in rows else None}

    def arm(*, reads: list[str], positional: bool, episode: EpisodeBinding | None) -> dict:
        return {
            "reads": reads, "positional": positional,
            "writes": sorted(episode.outputs) if episode else [], "evidence": sorted(episode.evidence) if episode else [],
        }

    io_: dict[str, dict[str, dict]] = {}
    for role, episode in generator.episodes.items():
        channel = generator.reads[role]
        named = [] if channel is None or channel == "input" else [channel]
        io_.setdefault(str(role), {})["generate"] = arm(reads=named, positional=channel == "input", episode=episode)
    for slot, spec in scorers.items():
        reads = sorted({*spec.stage.reads, *spec.stage.prepare.values()})
        node = isinstance(spec.scorer, EpisodeBinding)
        io_.setdefault(str(slot), {})["judge" if node else "host-instrument"] = arm(
            reads=reads, positional=False, episode=spec.scorer if node else None,
        )
    agents = {
        path.split("/")[1] for path in workspace
        if path.startswith("task_agents/") and path.count("/") >= 2 and path.split("/")[1] not in {"staged", "shared"}
    }
    return canonical_json({
        "depth": 0,
        "repository": [],
        "roles": sorted(map(str, generator.task_pools)),
        "reads": {str(role): channel for role, channel in generator.reads.items()},
        "prepare": {},
        "slots": sorted(map(str, scorers)),
        "judges": {
            str(slot): {
                "scorer": "fixed" if isinstance(spec.scorer, FixedInstrument) else "node",
                "over": sorted(map(str, jurisdiction[slot])),
            }
            for slot, spec in scorers.items()
        },
        "seeded_agents": sorted(agents),
        "dataset_to_rows": {
            f"{role}/{task}": {
                "measured": half(generator.task_sets.get(task)),
                "practice": half(None if train is None else train.task_sets.get(task)),
            }
            for role, tasks in sorted(generator.task_pools.items())
            for task in tasks
        },
        "io": {role: dict(sorted(arms.items())) for role, arms in sorted(io_.items())},
    }).encode()


def read_form(*, name: str, home: Path, document: Mapping, images: Mapping[str, str]) -> DomainManifest:
    """Return one form's manifest from its `form.json` document, whose image recipes are already digests."""
    dataset = read_dataset(home.parent.parent / "data")
    roles = document["roles"]
    episodes = {RoleId(role): _episode(one["episode"]) for role, one in roles.items()}
    generator = GeneratorSpec(
        task_pools={RoleId(role): tuple(map(TaskId, one["tasks"])) for role, one in roles.items()},
        episodes=episodes,
        reads={RoleId(role): one["reads"] for role, one in roles.items()},
        task_sets={TaskId(task): sets["search"] for one in roles.values() for task, sets in one["tasks"].items()},
    )
    train = TrainSpec(task_sets={
        TaskId(task): sets["train"] for one in roles.values() for task, sets in one["tasks"].items() if "train" in sets
    })
    scorers: dict[EvaluatorSlot, ScorerSpec] = {}
    for slot, judge in document["judges"].items():
        settle = ProcessInstrument(argv=("python", "settle.py", slot), home=str(home), deadline_seconds=judge["deadline_seconds"])
        scorers[EvaluatorSlot(slot)] = ScorerSpec(
            rule=ScoringRuleId(judge["rule"]),
            scorer=episodes.get(RoleId(slot)) or FixedInstrument(implementation=settle),
            stage=StageSpec(
                identity=judge["identity"], reads=frozenset(judge["reads"]), prepare=dict(judge.get("prepare", {})),
                finish=settle, settles=frozenset(judge["settles"]),
            ),
            interpretation=judge.get("interpretation"),
        )
    jurisdiction = {EvaluatorSlot(slot): frozenset(map(RoleId, judge["over"])) for slot, judge in document["judges"].items()}
    coordinates = document["coordinates"]
    transitions = {CriterionCoordinate(one): _policy(rule) for one, rule in document["transitions"].items()}
    workspace = compose(genesis=home / "genesis", meta=home.name)
    weights = {BindingTag(tag): dict(axes) for tag, axes in document["weights"].items()}
    tags = {role: episode.tag for role, episode in episodes.items() if episode.tag is not None}
    tags |= {RoleId(slot): BindingTag(judge["tag"]) for slot, judge in document["judges"].items() if "tag" in judge}
    return DomainManifest(
        name=name,
        domain=DomainId(document["domain"]),
        dataset=dataset,
        generator=generator,
        scorers=scorers,
        jurisdiction=jurisdiction,
        transition=TransitionSpec(
            coordinates=transitions, scopes=dict.fromkeys(transitions, frozenset({SEARCH})),
        ),
        coordinates=CoordinateDeclarations(
            currency={CriterionCoordinate(k): v for k, v in coordinates["currency"].items()},
            sample=CriterionCoordinate(coordinates["sample"]),
            dataset=CriterionCoordinate(coordinates["dataset"]),
            interpretation=CriterionCoordinate(coordinates["interpretation"]) if coordinates.get("interpretation") else None,
        ),
        interpretations=frozenset(document["interpretations"]),
        pins={CriterionCoordinate(k): v for k, v in document["pins"].items()},
        train=train if train.task_sets else None,
        meta=_episode(document["expansion"]),
        workspace=workspace,
        contract=_contract(
            generator=generator, scorers=scorers, jurisdiction=jurisdiction, train=train, dataset=dataset,
            workspace=workspace,
        ),
        agent_slots=frozenset(slot for slot, spec in scorers.items() if isinstance(spec.scorer, EpisodeBinding)),
        tags=tags,
        bindings={BindingTag(tag): BindingConfig(**one) for tag, one in document["bindings"].items()},
        weights=weights,
        axes=tuple(sorted({axis for one in weights.values() for axis in one})),
        cpus=document["cpus"],
        memory_bytes=document["memory_bytes"],
        images=dict(images),
    )
