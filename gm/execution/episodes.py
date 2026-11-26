# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""One episode, for every job kind: stage a workspace, run it in a container, harvest and record it.

`Episodes` runs generation and judgment jobs here; `expansion.py` runs expansions on the same parts.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from collections.abc import Mapping
from contextlib import AbstractContextManager, contextmanager
from operator import itemgetter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, cast

import pygit2
from pygit2.enums import DiffOption, RepositoryOpenFlag

from gm import experience, pointer
from gm.domain import EpisodeBinding
from gm.domain.identity import EvaluatorSlot, SampleKey
from gm.domain.results import (
    Artifact,
    ArtifactRef,
    EvaluatorJudgment,
    FailureReason,
    InlineArtifact,
    InvalidJudgment,
    JudgmentResult,
    TaskFailure,
    TaskSuccess,
)
from gm.domain.seams import (
    GENERATION_KEYS,
    MEMBER_KEY,
    MEMBERS_KEY,
    WEATHER_EXIT,
    WIRE_KEYS,
    ScoredMember,
    ScorerRefusalError,
    batch_answers,
)
from gm.domain import minted_origin
from gm.ids import UNBOUNDED, ActionId, ArtifactOid, Ceiling, InfraWeather, NodeId, require_safe_relpath
from gm.ledger import ActionKind, LedgerError
from gm.ledger.events import FILE_EFFECTS_ARTIFACT, SEAL_OMITTED
from gm.spend import SpendRows, merged
from gm.execution.docker import TRANSCRIPT_CAP
from gm.execution.gateway import Attribution, filed

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from gm.domain import DomainManifest, StageSpec
    from gm.domain.dataset import DatasetSpec
    from gm.git_store import NodeStore
    from gm.ledger import SqliteEventLog
    from gm.work_queue import GenerationJob, JudgmentJob, Pins
    from gm.execution.docker import Docker
    from gm.execution.gateway import Gateway, GrantCredential

ALLOTMENT_ENV = "RQGM_AGENT_TOKEN_ALLOTMENT"
EVIDENCE_CAP = 4096
INPUT_ROOT = ".rqgm/input"
OUTPUT_ROOT = ".rqgm/output"
RECORD_ROOT = ".rqgm/record"
JOB_PATH = ".rqgm/job.json"
DISPATCH_PATH = ".rqgm/dispatch.json"
JUDGMENT_PATH = ".rqgm/judgment.json"
GENERATION_PATH = ".rqgm/generation.json"
FAILURE_PATH = ".rqgm/failure.txt"
MESSAGE_PATH = ".rqgm/commit-message"
MESSAGE_LIMIT = 4096
ERROR_PATH = "error.json"
STAGING_ARTIFACT = "staging.json"
ACCOUNT_FOLLOWS = "the episode's own account follows:"
_REFUSAL_MARKS = ("provider refused, not the seat:", "provider refused, not the agent:")
#: What a production or judgment settles as its own failure rather than raising.
_SETTLES: tuple[type[Exception], ...] = (
    ArithmeticError, LedgerError, LookupError, OSError, RuntimeError, ScorerRefusalError, TypeError, ValueError,
)


# The episode contract.

@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ExecSpec:
    """One episode: what runs, in which image, under which limits and grant."""

    argv: tuple[str, ...]
    env: Mapping[str, str]
    token_allotment: Ceiling
    cpus: float | str
    memory_bytes: int | str
    image: str
    grant: GrantCredential | None = None
    agent_tag: str | None = None
    outputs: tuple[str, ...] = ()
    pool_socket: Path | None = None
    #: The sample's own image and the repository path in it, for an episode that works inside a sample.
    sample_image: str | None = None
    checkout: str | None = None


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Transcript:
    """Bounded combined process output."""

    data: bytes
    truncated: bool


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Ran:
    """A closed episode: its exit code, routed files, transcript, spend and dialogue."""

    code: int
    files: Mapping[str, Artifact]
    transcript: Transcript
    spend: SpendRows
    dialogue: bytes | None = None


class EpisodeRunError(Exception):
    """A runner failure, carrying the spend and dialogue it already retired."""

    def __init__(self, error: Exception, spend: SpendRows, *, dialogue: bytes | None = None) -> None:
        super().__init__(str(error) or type(error).__name__)
        self.spend, self.dialogue = spend, dialogue


def clip(text: str, ceiling: int = EVIDENCE_CAP) -> str:
    """Return a text bounded to a ceiling: its head, an elision marker, its tail."""
    if len(text) <= ceiling:
        return text
    keep = ceiling // 2
    return text[:keep] + f"\n[... {len(text) - 2 * keep} characters elided ...]\n" + text[-keep:]


def failure_text(error: Exception | str) -> str:
    """Render one failure as the bounded message a settled record carries."""
    return clip(str(error)) or type(error).__name__


def last_words(transcript: Transcript) -> str:
    """Return the end of one episode transcript, for a refusal that must name its cause."""
    text = transcript.data.decode(errors="replace")
    return f"; the episode's own last words were: {text[-2000:].strip()}" if text.strip() else ""


def claim_message(workspace: Path) -> str | None:
    """Take the message one episode wrote for the commit about to be sealed, and clear it."""
    control = workspace / MESSAGE_PATH
    try:
        with control.open("rb") as held:
            raw = held.read(MESSAGE_LIMIT + 1)
    except FileNotFoundError:
        return None
    control.unlink(missing_ok=True)
    if len(raw) > MESSAGE_LIMIT:
        raise ValueError(f"commit message is over the {MESSAGE_LIMIT} byte ceiling")
    return raw.decode("utf-8") or None


def host_write(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def keep_out_of_git(*, workspace: Path, paths: Sequence[str]) -> None:
    """Keep the host's own files out of the agent's `git status` and `git add -A`."""
    dot_git = workspace / ".git"
    if not dot_git.exists():
        return
    exclude = dot_git / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    present = exclude.read_text(encoding="utf-8").splitlines() if exclude.is_file() else []
    if missing := [one for one in paths if one not in present]:
        with exclude.open("a", encoding="utf-8") as sink:
            sink.write("".join(f"{one}\n" for one in missing))
    subprocess.run(["git", "-C", str(workspace), "update-index", "--skip-worktree", "--", *paths],
                   capture_output=True, text=True, timeout=20, check=False)


def allotment_of(*, binding: EpisodeBinding, envelope: Ceiling, kind: ActionKind) -> Ceiling:
    """The form's declared allotment, unless the run's envelope is tighter; a judgment keeps its own."""
    declared = binding.token_allotment
    if declared is None:
        return envelope
    if kind is ActionKind.JUDGMENT or envelope == UNBOUNDED:
        return declared
    return envelope if declared == UNBOUNDED or envelope < declared else declared


def failure_of(*, artifacts: SqliteEventLog, ran: Ran, actor: str, detailed: bool = True) -> tuple[str, FailureReason] | None:
    """Return what failed and why, or None when the episode exited 0.

    Exit 75 is the node's statement that its provider, not its code, failed; that is weather.
    """
    if ran.code == 0:
        return None
    note = ran.files.get(FAILURE_PATH)
    text = (ran.transcript.data if note is None else artifacts.resolve_artifact(note)).decode(errors="replace")
    if ran.code == WEATHER_EXIT:
        lines = text.splitlines()
        at = next((i for i in reversed(range(len(lines))) if lines[i].strip()), None)
        if at and any(mark in lines[at] for mark in _REFUSAL_MARKS):
            rest = "\n".join(lines[:at] + lines[at + 1:]).strip()
            text = f"{lines[at].strip()}\n\n{ACCOUNT_FOLLOWS}\n{rest}"
        weather = InfraWeather(clip(text) or f"{actor} exited {WEATHER_EXIT}")
        weather.weathered_dialogue = ran.dialogue
        raise weather
    error = f"{actor} exited {ran.code}"
    described = (clip(text) or error) if detailed else error
    if described != error:
        described = clip(f"{error}\n\n{ACCOUNT_FOLLOWS}\n{described}")
    return described, FailureReason.AGENT_ERROR


# File effects: what an episode did to its tree, recorded after its container is gone.

def _head(workspace: Path) -> str | None:
    try:
        return str(pygit2.Repository(str(workspace), RepositoryOpenFlag.NO_SEARCH).head.peel(pygit2.Commit).id)
    except pygit2.GitError:
        return None


def _effects(*, workspace: Path, before: str | None, route: Callable[[bytes], ArtifactRef]) -> bytes:
    """Compare the starting commit with the final checkout, untracked and ignored outputs included."""
    if before is None:
        roots = {"workspace": {"read": False, "refusal": "no starting Git commit was available for this episode"}}
        return json.dumps({"v": 1, "roots": roots, "effects": []}, sort_keys=True, separators=(",", ":")).encode()
    diff = pygit2.Repository(str(workspace), RepositoryOpenFlag.NO_SEARCH).diff(
        before, flags=DiffOption.INCLUDE_UNTRACKED | DiffOption.RECURSE_UNTRACKED_DIRS | DiffOption.INCLUDE_IGNORED
        | DiffOption.RECURSE_IGNORED_DIRS | DiffOption.INCLUDE_TYPECHANGE,
    )
    effects: list[dict[str, object]] = []
    for delta in diff.deltas:
        status = delta.status_char()
        rel = delta.old_file.path if status == "D" else delta.new_file.path
        if experience.unpublished(rel=rel) or (status == "I" and not experience.required(rel=rel)):
            continue
        row: dict[str, object] = {"path": rel, "root": "workspace",
                                  "change": "deleted" if status == "D" else "added" if status in "A?I" else "modified"}
        if status != "D":
            path = workspace / rel
            if path.is_symlink():
                row |= {"size": 0, SEAL_OMITTED: "symlink"}
            else:
                payload = path.read_bytes()
                row |= {"size": len(payload), "oid": str(route(payload).oid)}
        effects.append(row)
    roots = {"workspace": {"read": True, "refusal": None, "base_commit": before}}
    return json.dumps({"v": 1, "roots": roots, "effects": sorted(effects, key=itemgetter("path"))},
                      sort_keys=True, separators=(",", ":")).encode()


def _harvest(*, workspace: Path, outputs: Sequence[str]) -> dict[str, bytes]:
    """Read every declared output that is a regular file inside the workspace."""
    root = workspace.resolve()
    files: dict[str, bytes] = {}
    for rel in outputs:
        path = workspace / rel
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            continue
        if (size := path.stat().st_size) > TRANSCRIPT_CAP:
            raise InfraWeather(f"the declared output {rel!r} is {size} bytes, over this host's"
                               f" {TRANSCRIPT_CAP}-byte carry ceiling")
        files[rel] = path.read_bytes()
    return files


def run_episode(*, spec: ExecSpec, workspace: Path, docker: Docker, gateway: Gateway | None,
                artifacts: SqliteEventLog) -> Ran:
    """Run one episode and retire its grant on every path; a failure carries what it spent and said."""
    spend: SpendRows = ()
    dialogue = None
    try:
        try:
            before = _head(workspace)
            code, transcript = docker.run(spec, workspace)
        finally:
            if gateway is not None and spec.grant is not None:
                spend, dialogue = gateway.retire(spec.grant)
        route = artifacts.route_artifact
        harvested = _harvest(workspace=workspace, outputs=(JOB_PATH, DISPATCH_PATH, *spec.outputs))
        files = {name: route(pointer.pointerized(data=data, route=route, fetch=artifacts.get_artifact))
                 for name, data in harvested.items()}
        files[FILE_EFFECTS_ARTIFACT] = route(_effects(workspace=workspace, before=before, route=route))
    except InfraWeather as weather:
        weather.weathered_spend = merged(weather.weathered_spend or (), spend)
        weather.weathered_dialogue = weather.weathered_dialogue or dialogue
        raise
    except Exception as exc:
        raise EpisodeRunError(exc, spend, dialogue=dialogue) from exc
    return Ran(code=code, files=files, transcript=transcript, spend=spend, dialogue=dialogue)


# The evidence an episode leaves.

def staged_record(*, members: Mapping[str, Sequence[tuple[str, str, bytes]]],
                  route: Callable[[bytes], Artifact]) -> dict[str, Artifact]:
    """Route the `staging.json` record of every (name, path, bytes) the host staged, per member."""
    document = {"schema": "rqgm.staging.v1", "members": {
        member: {"schema": "rqgm.staging.v1", "mode": "numbered",
                 "root": f"{INPUT_ROOT}/{member}",
                 "cells": [{"name": name, "path": path, "oid": hashlib.sha256(data).hexdigest(), "size": len(data),
                            "consumed": []} for name, path, data in cells]}
        for member, cells in sorted(members.items())
    }}
    return {STAGING_ARTIFACT: route(json.dumps(document, sort_keys=True, separators=(",", ":")).encode())}


def bundle(*, artifacts: SqliteEventLog, ran: Ran | None, evidence: Mapping[str, Artifact]) -> dict[str, Artifact]:
    """Return the evidence, staged inputs, workspace outputs, transcript and file effects of one episode."""
    files = dict(evidence)
    if (staged := evidence.get(STAGING_ARTIFACT)) is not None:
        for member in json.loads(artifacts.resolve_artifact(staged))["members"].values():
            for cell in member["cells"]:
                files[f"inputs/{cell['path']}"] = ArtifactRef(oid=ArtifactOid(cell["oid"]), size=int(cell["size"]))
    if ran is None:
        return files
    files |= {f"workspace/{path}": value for path, value in ran.files.items() if path != FILE_EFFECTS_ARTIFACT}
    files["transcript.txt"] = artifacts.route_artifact(ran.transcript.data)
    effects = ran.files[FILE_EFFECTS_ARTIFACT]
    files[FILE_EFFECTS_ARTIFACT] = effects
    for effect in json.loads(artifacts.resolve_artifact(effects))["effects"]:
        if "oid" in effect:
            files[f"{effect['root']}/{effect['path']}"] = ArtifactRef(oid=ArtifactOid(effect["oid"]), size=effect["size"])
    return files


def failure_bundle(*, artifacts: SqliteEventLog, ran: Ran | None, failure: tuple[str, FailureReason],
                   outputs: Mapping[str, str], evidence: Mapping[str, Artifact]) -> dict[str, Artifact]:
    """Return `bundle` plus an `error.json` standing in for every declared output."""
    files = bundle(artifacts=artifacts, ran=ran, evidence=evidence)
    verdict = None if ran is None else {"kind": "exited", "code": ran.code, "oom": None}
    error = artifacts.route_artifact(json.dumps({
        "schema": "rqgm.error.v1", "reason": str(failure[1]), "message": failure[0], "verdict": verdict,
        "transcript_truncated": None if ran is None else ran.transcript.truncated, "outputs": dict(outputs),
    }, sort_keys=True).encode() + b"\n")
    files[ERROR_PATH] = error
    for path in outputs.values():
        require_safe_relpath(path=path)
        files[f"outputs/{path}"] = error
    return files


def failure_fields(*, files: Mapping[str, Artifact], resolve: Callable[[Artifact], bytes]) -> dict[str, str]:
    """Return the failure reason and detail an `error.json` in these files states, or nothing."""
    if (error := files.get(ERROR_PATH)) is None:
        return {}
    document = json.loads(resolve(error))
    return {"failure_reason": str(FailureReason(document.get("reason"))), "failure_detail": document["message"]}


def _routed(*, artifacts: SqliteEventLog, judgment: EvaluatorJudgment) -> EvaluatorJudgment:
    files = {name: artifacts.route_artifact(one.data) if isinstance(one, InlineArtifact) else one
             for name, one in judgment.files.items()}
    return dataclasses.replace(judgment, files=files)


# Generation and judgment episodes.

@dataclasses.dataclass(slots=True, kw_only=True)
class Episodes:
    """Stage, run and settle one generation or judgment job; expansions build on the same parts."""

    domain: DomainManifest
    nodes: NodeStore
    artifacts: SqliteEventLog
    docker: Docker
    gateway: Gateway
    envelope: Ceiling
    nudge_at: float | None
    episode_dir: Path
    #: Lends the action's CPU while it waits on another job's build of the same node tree.
    lend: Callable[[ActionId], AbstractContextManager[None]] | None = None

    def corpus(self, *, pins: Pins) -> DatasetSpec:
        return self.domain.dataset.build(version=pins[self.domain.coordinates.dataset])

    @contextmanager
    def workspace(self, *, prefix: str, node: NodeId | None,
                  waiting: Callable[[], AbstractContextManager[None]] | None = None) -> Iterator[Path]:
        """A scratch workspace, holding the node's task tree when `node` is given."""
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=f"{prefix}-", dir=self.episode_dir) as scratch:
            workspace = Path(scratch)
            if node is not None:
                self.nodes.materialize_task(node=node, path=workspace, on_wait=waiting)
            yield workspace

    def agent_env(self, *, allotment: Ceiling) -> dict[str, str]:
        """The budget contract every episode is told about."""
        return {ALLOTMENT_ENV: str(allotment), "RQGM_AGENT_BUDGET_EXPOSE": "",
                "RQGM_AGENT_BUDGET_NUDGE": "off" if self.nudge_at is None else str(self.nudge_at)}

    def contract(self, *, workspace: Path, depth: int | None) -> None:
        """Stage the form's I/O contract at `.rqgm/contract.json`, stating the episode's depth."""
        (workspace / experience.ENVIRONMENT).unlink(missing_ok=True)
        if self.domain.contract:
            declared = json.loads(self.domain.contract)
            if depth is not None:
                declared = {**declared, "depth": depth}
            host_write(workspace / experience.CONTRACT, json.dumps(declared, sort_keys=True).encode())

    def run(
        self, *, binding: EpisodeBinding, workspace: Path, kind: ActionKind, outputs: Sequence[str],
        actions: tuple[ActionId, ...], node: NodeId | None, depth: int | None, attempt_no: int = 1,
        sample_image: str | None = None, pool_socket: Path | None = None, writing: object = None,
        on_call: Callable[[int], None] | None = None,
    ) -> Ran:
        """Mint the grant a model-bound binding declares, and run one episode in its container."""
        allotment = allotment_of(binding=binding, envelope=self.envelope, kind=kind)
        env, grant = self.agent_env(allotment=allotment), None
        seat = f"{kind.value}:{','.join(str(one) for one in actions)}"
        if binding.tag is not None:
            grant = self.gateway.issue(
                tag=binding.tag, binding=self.domain.bindings[binding.tag], allotment=allotment,
                weights=self.domain.weights, attribution=Attribution(kind=kind, actions=actions, attempt_no=attempt_no),
                mirror=workspace / experience.node_dialogue(depth=experience.staged_depth(workspace=workspace)),
                node=node, seat=seat, writing=writing, on_call=on_call,
            )
            env["RQGM_AGENT_MODEL"] = str(binding.tag)
            env |= self.domain.bindings[binding.tag].agent_env()
        self.contract(workspace=workspace, depth=depth)
        spec = ExecSpec(
            argv=tuple(binding.argv), agent_tag=seat, env=env, token_allotment=allotment, cpus=self.domain.cpus,
            memory_bytes=self.domain.memory_bytes, grant=grant, pool_socket=pool_socket, image=str(binding.image),
            outputs=tuple(dict.fromkeys(outputs)), sample_image=sample_image, checkout=binding.checkout,
        )
        return run_episode(spec=spec, workspace=workspace, docker=self.docker, gateway=self.gateway,
                           artifacts=self.artifacts)

    def _stage(self, *, workspace: Path, files: Mapping[str, Artifact], member: str) -> list[tuple[str, str, bytes]]:
        """Write each staged file, numbered under the member's input root."""
        cells = []
        for index, (name, artifact) in enumerate(sorted(files.items())):
            relative = f"{INPUT_ROOT}/{member}/{index:04d}"
            data = pointer.materialized(data=self.artifacts.resolve_artifact(artifact), fetch=self.artifacts.get_artifact)
            host_write(workspace / relative, data)
            self.artifacts.route_artifact(data)
            cells.append((name, relative, data))
        return cells

    def _write_job(self, *, workspace: Path, payload: Mapping[str, object]) -> None:
        """Write the full dispatch for the runtime and the member-only job for the agent."""
        public = {MEMBERS_KEY: [{MEMBER_KEY: one[MEMBER_KEY] if isinstance(one, Mapping) else one}
                                for one in cast("Sequence[object]", payload[MEMBERS_KEY])]}
        for target, stated in ((DISPATCH_PATH, payload), (JOB_PATH, public)):
            host_write(workspace / target, json.dumps(stated, sort_keys=True, separators=(",", ":")).encode())
        keep_out_of_git(workspace=workspace, paths=(DISPATCH_PATH, JOB_PATH))

    def _channels(self, *, job: JudgmentJob, sample: SampleKey | None = None) -> Mapping[str, bytes]:
        """One sample's hidden dataset row (a minted sample reads its origin's)."""
        if job.source_channels is not None and sample is None:
            return {name: self.artifacts.resolve_artifact(one) for name, one in job.source_channels.items()}
        sample = sample or job.source_sample or job.sample
        rows = self.corpus(pins=job.pins).channels
        row = rows.get(sample)
        if row is None and (origin := minted_origin(member=sample)) is not None:
            row = rows.get(origin)
        return row or {}

    def generate(self, *, job: GenerationJob) -> TaskSuccess | TaskFailure:
        """Run one producer member in its node's task tree and settle its outputs."""
        binding = self.domain.generator.episodes[job.role]
        member = str(job.action)
        places = {"outputs": {name: f"{OUTPUT_ROOT}/{member}/{path}" for name, path in binding.outputs.items()},
                  "evidence": {name: f"{RECORD_ROOT}/{member}/{path}" for name, path in binding.evidence.items()}}
        sample = SampleKey(job.pins[self.domain.coordinates.sample])
        with self.workspace(prefix="generation", node=job.node, waiting=self.waiting(job.action)) as workspace:
            cells = self._stage(workspace=workspace, files=dict(job.inputs), member=member)
            staged = staged_record(members={member: cells}, route=self.artifacts.route_artifact)
            spec = self.domain.scorers.get(EvaluatorSlot(str(job.role)))
            self._write_job(workspace=workspace, payload={
                "mode": "generate", "role": str(job.role), "rule": None if spec is None else str(spec.rule),
                MEMBERS_KEY: [{MEMBER_KEY: member, "seed": job.seed,
                               "inputs": {name: path for name, path, _ in cells}, **places}],
            })
            try:
                ran = self.run(
                    binding=binding, workspace=workspace, kind=ActionKind.GENERATION, actions=(job.action,),
                    outputs=(GENERATION_PATH, *places["outputs"].values(), *places["evidence"].values(), FAILURE_PATH),
                    node=job.node, depth=experience.staged_depth(workspace=workspace),
                    sample_image=None if binding.checkout is None else self.corpus(pins=job.pins).images[sample],
                )
            except InfraWeather:
                raise
            except (*_SETTLES, EpisodeRunError) as exc:
                spent = exc.spend if isinstance(exc, EpisodeRunError) else ()
                spoken = exc.dialogue if isinstance(exc, EpisodeRunError) else None
                evidence = filed(dialogue=spoken, route=self.artifacts.route_artifact) | staged
                return TaskFailure(
                    files=failure_bundle(artifacts=self.artifacts, ran=None, failure=(str(exc), FailureReason.UNKNOWN),
                                         outputs=binding.outputs, evidence=evidence),
                    error=str(exc), reason=FailureReason.UNKNOWN, spend=spent,
                )
            evidence = (filed(dialogue=ran.dialogue, route=self.artifacts.route_artifact) | staged
                        | {FILE_EFFECTS_ARTIFACT: ran.files[FILE_EFFECTS_ARTIFACT]})
            failure = failure_of(artifacts=self.artifacts, ran=ran, actor="producer")
            if failure is None:
                failure = self._membership(ran=ran, member=member, outputs=places["outputs"])
            if failure is not None:
                return TaskFailure(
                    files=failure_bundle(artifacts=self.artifacts, ran=ran, failure=failure, outputs=binding.outputs,
                                         evidence=evidence),
                    error=failure[0], reason=failure[1], spend=ran.spend,
                )
            declared = {name: ran.files[path] for name, path in places["evidence"].items() if path in ran.files}
            return TaskSuccess(
                files={name: ran.files[path] for name, path in places["outputs"].items()}, spend=ran.spend,
                metadata=bundle(artifacts=self.artifacts, ran=ran, evidence=declared | evidence),
            )

    def _membership(self, *, ran: Ran, member: str, outputs: Mapping[str, str]) -> tuple[str, FailureReason] | None:
        """Refuse a producer that did not state this member or omitted a declared output."""
        if (manifest := ran.files.get(GENERATION_PATH)) is None:
            return f"producer stated no membership at {GENERATION_PATH}", FailureReason.MALFORMED_OUTPUT
        try:
            batch_answers(returned=json.loads(self.artifacts.resolve_artifact(manifest)), requested={member},
                          declared=GENERATION_KEYS, called="the producer")
        except ValueError as exc:
            return str(exc), FailureReason.MALFORMED_OUTPUT
        if missing := set(outputs.values()) - ran.files.keys():
            return f"producer omitted declared outputs for {member}: {sorted(missing)}", FailureReason.MALFORMED_OUTPUT
        return None

    def _staged_subject(self, *, job: JudgmentJob, stage: StageSpec) -> dict[str, Artifact]:
        """The produced files a judge reads, plus the hidden channels it prepares."""
        files = {path: job.produced.files[path] for path in stage.reads & job.produced.files.keys()}
        if stage.prepare:
            row = self._channels(job=job)
            files |= {path: InlineArtifact(data=row[channel]) for channel, path in stage.prepare.items()}
        return files

    def judge(self, *, job: JudgmentJob) -> JudgmentResult:
        """Score one subject: a node's own judge episode, or the slot's fixed process."""
        spec = self.domain.scorers[job.slot]
        if job.role is not None:
            actor = cast("EpisodeBinding", self.domain.scorers[EvaluatorSlot(str(job.role))].scorer)
            return self._judge_episode(job=job, binding=actor, stage=spec.stage)
        try:
            files = self._staged_subject(job=job, stage=spec.stage)
            inline = {name: InlineArtifact(data=self.artifacts.resolve_artifact(one)) for name, one in files.items()}
            verdict = spec.scorer.implementation.score([ScoredMember(
                member=str(job.action), sample=job.sample if spec.stage.identity else None,
                produced=TaskSuccess(files=inline, spend=()), channels=self._channels(job=job),
            )], fetch=self.artifacts.get_artifact)[str(job.action)]
        except _SETTLES as exc:
            return InvalidJudgment(files={}, spend=(), error=failure_text(exc))
        slot = self.artifacts.route_artifact(json.dumps({
            "schema": "rqgm.judgment-slot.v1", "slot": str(job.slot), "rule": str(verdict.scoring_rule),
            "score": verdict.score, "outcome": verdict.outcome, "kind": str(verdict.kind),
            "refusal_code": None if verdict.refusal_code is None else str(verdict.refusal_code), "reads": sorted(files),
        }, sort_keys=True).encode() + b"\n")
        staged = {f"inputs/{name}": held for name, held in files.items()}
        return _routed(artifacts=self.artifacts, judgment=dataclasses.replace(
            verdict, files={"slot.json": slot, **staged, **dict(verdict.files)}))

    def _judge_episode(self, *, job: JudgmentJob, binding: EpisodeBinding, stage: StageSpec) -> JudgmentResult:
        """Run the node's judge seat over the staged subject, then the slot's finisher over its raw score."""
        member = str(job.action)
        ran: Ran | None = None
        staged: dict[str, Artifact] = {}
        with self.workspace(prefix="judgment", node=job.node, waiting=self.waiting(job.action)) as workspace:
            try:
                files = self._staged_subject(job=job, stage=stage)
                cells = self._stage(workspace=workspace, files=files, member=member)
                staged = staged_record(members={member: cells}, route=self.artifacts.route_artifact)
                self._write_job(workspace=workspace, payload={
                    "mode": "judge", "role": str(job.role), "inputs": {name: path for name, path, _ in cells},
                    "seed": int.from_bytes(hashlib.sha256(member.encode()).digest()[:8]),
                    "rule": str(self.domain.scorers[job.slot].rule), "evidence": dict(binding.evidence),
                    "outputs": dict(binding.outputs), MEMBERS_KEY: [member],
                })
                ran = self.run(
                    binding=binding, workspace=workspace, kind=ActionKind.JUDGMENT, actions=(job.action,),
                    outputs=(JUDGMENT_PATH, *binding.evidence.values(), *binding.outputs.values(), FAILURE_PATH),
                    node=job.node, depth=experience.staged_depth(workspace=workspace),
                )
                if failure := failure_of(artifacts=self.artifacts, ran=ran, actor="evaluator"):
                    return self._invalid(ran=ran, error=failure[0], reason=failure[1], binding=binding, staged=staged)
                wire = json.loads(self.artifacts.resolve_artifact(ran.files[JUDGMENT_PATH]))
                stated = batch_answers(returned=wire, requested={member}, declared=WIRE_KEYS, called="the evaluator")
                declared = {**dict(binding.outputs), **dict(binding.evidence)}
                evidence = ({name: ran.files[path] for name, path in declared.items() if path in ran.files}
                            | filed(dialogue=ran.dialogue, route=self.artifacts.route_artifact)
                            | {FILE_EFFECTS_ARTIFACT: ran.files[FILE_EFFECTS_ARTIFACT]} | staged)
                raw = EvaluatorJudgment(
                    files=bundle(artifacts=self.artifacts, ran=ran, evidence=evidence), spend=ran.spend,
                    score=cast("float", stated[member]["score"]), outcome=0.0, scoring_rule=self.domain.scorers[job.slot].rule,
                    utilities={}, units={}, labels={}, declared_outputs=dict(binding.outputs),
                )
                if stage.finish is None:
                    return raw
                settled = stage.finish.finish(job.sample, raw, channels=self._channels(job=job, sample=job.sample),
                                              member=member)
                return _routed(artifacts=self.artifacts, judgment=dataclasses.replace(
                    settled, declared_outputs=dict(binding.outputs),
                    files=dict(raw.files) | dict(settled.files) | staged
                    | {FILE_EFFECTS_ARTIFACT: ran.files[FILE_EFFECTS_ARTIFACT]},
                ))
            except InfraWeather:
                raise
            except (*_SETTLES, EpisodeRunError) as exc:
                return self._invalid(ran=ran, error=exc, reason=FailureReason.MALFORMED_OUTPUT, binding=binding,
                                     staged=staged)

    def _invalid(self, *, ran: Ran | None, error: Exception | str, reason: FailureReason, binding: EpisodeBinding,
                 staged: Mapping[str, Artifact]) -> InvalidJudgment:
        message = failure_text(error)
        evidence: dict[str, Artifact] = {}
        spent: SpendRows = ()
        if ran is not None:
            spent = ran.spend
            evidence = {"evaluator.log": self.artifacts.route_artifact(ran.transcript.data or message.encode())}
            evidence |= filed(dialogue=ran.dialogue, route=self.artifacts.route_artifact)
            evidence[FILE_EFFECTS_ARTIFACT] = ran.files[FILE_EFFECTS_ARTIFACT]
        elif isinstance(error, EpisodeRunError):
            spent = error.spend
            evidence = {"evaluator.log": self.artifacts.route_artifact(message.encode())}
            evidence |= filed(dialogue=error.dialogue, route=self.artifacts.route_artifact)
        evidence |= staged
        return InvalidJudgment(
            files=failure_bundle(artifacts=self.artifacts, ran=ran, failure=(message, reason),
                                 outputs=binding.outputs, evidence=evidence),
            spend=spent, error=message,
        )

    def waiting(self, action: ActionId) -> Callable[[], AbstractContextManager[None]] | None:
        """Lend this action's CPU while its node's tree is being built by another job."""
        lend = self.lend
        return None if lend is None else lambda: lend(action)
