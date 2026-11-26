# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""One expansion: the meta episode over its parent's experience, mechanical admission of the tree
it leaves, and publication of the child."""

from __future__ import annotations

import hashlib
import json
import posixpath
import stat
import threading
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from gm import experience
from gm.domain.results import FailureReason, TaskFailure, TaskSuccess
from gm.git_store import EmptyTreeError, NodeExistsError, experience_view
from gm.ids import NodeId, RunId, canonical_json
from gm.ledger import ActionKind
from gm.ledger.events import FILE_EFFECTS_ARTIFACT, SEAL_MANIFEST_ARTIFACT, SEAL_OMITTED
from gm.tree_copy import copy_tree
from gm.work_queue import ExpansionCompletion, ExpansionJob, Pins
from gm.execution.episodes import (
    JOB_PATH,
    EpisodeRunError,
    Episodes,
    ExecSpec,
    Transcript,
    allotment_of,
    claim_message,
    failure_of,
    failure_text,
    keep_out_of_git,
    last_words,
    run_episode,
)
from gm.execution.gateway import dialogue_calls, filed
from gm.execution.train import stage_train_data

if TYPE_CHECKING:
    from collections.abc import Callable

    from gm.execution.train import Pool

PROBE_PATH = ".rqgm/admission_probe.py"
PROBE_PROGRAM = Path(__file__).with_name("admission_probe.py").read_bytes()
OUTCOME_KEY = "expansion.outcome"
_EXITED = "the meta agent ended the episode itself"
_ALIGNED = "the chain answered aligned on every spec and the gates passed"


@dataclass(slots=True)
class _Ending:
    """How one expansion episode ended, for the node-state record its final turn writes."""

    node: NodeId | None = None
    reason: str | None = None
    done: bool = False
    grounds: str | None = None


class Expansions:
    """Run expansion jobs over the shared episode parts and the run's train pool."""

    def __init__(self, *, episodes: Episodes, pool: Pool, run: RunId, admit: Callable[..., None]) -> None:
        self.episodes, self.pool, self.run, self.admit = episodes, pool, run, admit
        self._shared: dict[str, Path] = {}
        self._lock = threading.Lock()

    def expand(self, *, job: ExpansionJob) -> ExpansionCompletion:
        """Run one expansion episode to a published child, a decline, or a refusal."""
        episodes, nodes, artifacts = self.episodes, self.episodes.nodes, self.episodes.artifacts
        pins = job.export.pins
        depth = nodes.depth(job.parent) + 1
        with experience_view(nodes, job.parent, workdir=episodes.episode_dir) as workspace:
            chain = nodes.open_chain(run=self.run, parent=job.parent, action=str(job.action))
            admit = self.admit
            chain.on_first_turn = lambda key, action=str(job.action): admit(action=action, node=key)
            socket = self.pool.open_episode(action=job.action, checkout=workspace, depth=depth)
            docker = episodes.docker

            def turn(call: int) -> None:
                # The container waits on this call's reply, so its tree is still; a call that edited nothing
                # reuses the chain's tip.
                if docker.carry_out(socket=str(socket), checkout=workspace):
                    chain.turn(checkout=workspace, message=f"model call {call}", claim=claim_message(workspace))

            ended = _Ending()
            try:
                completion = self._settle(job=job, pins=pins, workspace=workspace, depth=depth, socket=socket,
                                          ended=ended, on_call=turn)
            finally:
                self.pool.close_episode(action=job.action)
            manifest = _seal_manifest(pending=workspace / experience.generation(depth=depth),
                                      route=artifacts.route_artifact)
            sealed = {SEAL_MANIFEST_ARTIFACT: artifacts.route_artifact(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())}
            self.pool.published(action=job.action)
            node, done = ended.node, ended.done
            if node is None and chain.announced:
                node, done = NodeId(chain.key), False
            if node is not None:
                nodes.final_turn(node=node, chain=chain.node_ref, reason=ended.reason or FailureReason.UNKNOWN.value,
                                 done=done, grounds=ended.grounds or "the episode ended before anything judged it",
                                 action=str(job.action))
            result = completion.result
            folded = (replace(result, metadata={**result.metadata, **sealed}) if isinstance(result, TaskSuccess)
                      else replace(result, files={**result.files, **sealed}))
            return ExpansionCompletion(child=completion.child, result=folded)

    def _shared_data(self, *, pins: Pins) -> Path:
        """Stage this curriculum version's train data once, shared by every expansion that reads it."""
        build = self.episodes.corpus(pins=pins)
        version = str(build.version)
        with self._lock:
            if version not in self._shared:
                shared = self.episodes.episode_dir / f"shared-data-{hashlib.sha256(version.encode()).hexdigest()}"
                if (train := self.episodes.domain.train) is not None:
                    stage_train_data(build=build, artifacts=self.episodes.artifacts, sets=frozenset(train.every),
                                     into=shared)
                self._shared[version] = shared
            return self._shared[version]

    def _settle(self, *, job: ExpansionJob, pins: Pins, workspace: Path, depth: int, socket: Path,
                ended: _Ending, on_call: Callable[[int], None]) -> ExpansionCompletion:
        episodes, nodes, artifacts = self.episodes, self.episodes.nodes, self.episodes.artifacts
        route = artifacts.route_artifact
        (workspace / experience.generation(depth=depth)).mkdir(parents=True, exist_ok=True)
        (workspace / JOB_PATH).parent.mkdir(parents=True, exist_ok=True)
        (workspace / JOB_PATH).write_text(canonical_json(
            {"mode": "expand", "action": str(job.action), "depth": depth, "members": [{"member": str(job.action)}]},
        ), encoding="utf-8")
        keep_out_of_git(workspace=workspace, paths=(JOB_PATH,))
        _link_tree(shared=self._shared_data(pins=pins), workspace=workspace)
        try:
            ran = episodes.run(
                binding=job.binding, workspace=workspace, kind=ActionKind.EXPANSION, outputs=(),
                actions=(job.action,), node=job.parent, depth=depth, attempt_no=job.attempt_no,
                pool_socket=socket, writing=self.pool.episode_writing(action=job.action), on_call=on_call,
            )
        except EpisodeRunError as exc:
            metadata = dict(filed(dialogue=exc.dialogue, route=route))
            metadata["episode.error"] = route(failure_text(exc).encode())
            transcript = Transcript(data=failure_text(exc).encode(), truncated=False)
            metadata["episode.log"] = route(transcript.data)
            dialogue, spent = exc.dialogue, exc.spend
            ending = f"the episode's last call did not finish: {failure_text(exc)}"
            ended.reason = ending
        else:
            metadata = dict(filed(dialogue=ran.dialogue, route=route))
            metadata["episode.log"] = route(ran.transcript.data)
            metadata[FILE_EFFECTS_ARTIFACT] = ran.files[FILE_EFFECTS_ARTIFACT]
            failure = failure_of(artifacts=artifacts, ran=ran, actor="meta episode", detailed=False)
            ending = None if failure is None else failure[0]
            ended.reason = _EXITED if failure is None else failure[1].value
            transcript, dialogue, spent = ran.transcript, ran.dialogue, ran.spend
        self.pool.drain_episode(action=job.action)
        success = TaskSuccess(files={}, spend=spent, metadata=metadata)
        try:
            if (best := self.pool.best_submission(action=job.action)) is not None:
                nodes.restore(commit=best, checkout=workspace)
                self.pool.drain_episode(action=job.action)
            admission = self._admit(job=job, workspace=workspace, success=success, depth=depth)
            if isinstance(admission, TaskFailure):
                with suppress(OSError, ValueError, EmptyTreeError, NodeExistsError):
                    marked = nodes.mark_attempt(action=str(job.action), checkout=workspace,
                                                note=str(admission.error).encode())
                    ended.node = marked if nodes.node_ref(marked) is not None else None
                nodes.repack()
                ended.grounds = f"admission refused: {admission.error}"
                return ExpansionCompletion(child=None, result=admission)
            success = admission
            self.pool.drain_episode(action=job.action)
            child = nodes.mark_attempt(action=str(job.action), checkout=workspace, note=None,
                                       message=claim_message(workspace))
            ended.node, ended.done, ended.grounds, ended.reason = child, True, _ALIGNED, _EXITED
            nodes.repack()
            if ending is not None:
                success = _noted(success=success, route=route, note=(
                    "published: the candidate tree stands and is kept, but the episode's own"
                    f" process did not exit cleanly ({ending}){last_words(transcript)}"
                ))
        except EmptyTreeError:
            child = None
            ended.grounds = "declined: the candidate tree is unchanged from its parent"
            silent = dialogue is not None and not dialogue_calls(dialogue)
            success = _noted(success=success, route=route, note=(
                "declined: the candidate tree is unchanged from its parent AND the episode made no"
                " model calls at all, so the agent never ran and this is not its statement -- the"
                " container failed before its first turn" if silent else
                "declined: the candidate tree is unchanged from its parent,"
                " so the agent produced nothing to publish"
            ) + last_words(transcript))
        except NodeExistsError as exc:
            child = exc.node
            ended.grounds = f"duplicate: the candidate tree is already published as {exc.node}"
            success = _noted(success=success, route=route, note=(
                f"duplicate: the candidate tree is already published as {exc.node};"
                " the store refused a second copy, the agent did not decline"))
        except (OSError, ValueError) as exc:
            ended.grounds = f"malformed: {failure_text(exc)}"
            return ExpansionCompletion(child=None, result=_malformed(success=success, error=exc))
        return ExpansionCompletion(child=child, result=success)

    def _admit(self, *, job: ExpansionJob, workspace: Path, success: TaskSuccess,
               depth: int) -> TaskSuccess | TaskFailure:
        """Run the host's admission probe over a copy of the candidate tree, in an isolated episode."""
        _validate(workspace)
        episodes = self.episodes
        with episodes.workspace(prefix="admission", node=None) as probe:
            copy_tree(source=workspace, into=probe, dirs_exist_ok=True, symlinks=True, ignore=_transient)
            (probe / PROBE_PATH).parent.mkdir(parents=True, exist_ok=True)
            (probe / PROBE_PATH).write_bytes(PROBE_PROGRAM)
            episodes.contract(workspace=probe, depth=depth)
            allotment = allotment_of(binding=job.binding, envelope=episodes.envelope, kind=ActionKind.EXPANSION)
            spec = ExecSpec(
                argv=("python", "-B", PROBE_PATH, *job.binding.argv), env=episodes.agent_env(allotment=allotment),
                token_allotment=allotment, cpus=episodes.domain.cpus, memory_bytes=episodes.domain.memory_bytes,
                image=str(job.binding.image),
            )
            ran = run_episode(spec=spec, workspace=probe, docker=episodes.docker, gateway=None,
                              artifacts=episodes.artifacts)
        transcript = ran.transcript.data or b"admission probe produced no transcript"
        admitted = TaskSuccess(files=success.files, spend=success.spend, metadata={
            **success.metadata, "admission.log": episodes.artifacts.route_artifact(transcript)})
        if ran.code == 0:
            return admitted
        return _malformed(success=admitted, error=ValueError(f"admission probe exited {ran.code}"))


def _link_tree(*, shared: Path, workspace: Path) -> None:
    """Hard-link the staged train data into the episode's `experience/data`."""
    root = workspace / experience.DATA_ROOT
    root.mkdir(parents=True, exist_ok=True)
    if not shared.is_dir():
        return
    for source in sorted(shared.rglob("*")):
        target = root / source.relative_to(shared)
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.hardlink_to(source)


def _seal_manifest(*, pending: Path, route: Callable[[bytes], object]) -> dict[str, object]:
    """Return the manifest of every file an episode left in its own generation folder."""
    entries: list[dict[str, object]] = []
    for path in sorted(pending.rglob("*")) if pending.is_dir() else ():
        rel = path.relative_to(pending).as_posix()
        if path.is_dir() and not path.is_symlink():
            if not any(path.iterdir()):
                entries.append({"path": rel, "kind": "dir"})
        elif path.is_symlink():
            entries.append({"path": rel, "kind": "file", "size": 0, SEAL_OMITTED: "symlink"})
        else:
            payload = path.read_bytes()
            entries.append({"path": rel, "kind": "file", "size": len(payload), "oid": str(route(payload).oid)})
    return {"v": 1, "root": pending.name, "sealed": pending.is_dir(), "entries": entries}


def _validate(workspace: Path) -> None:
    """Refuse nested repositories, escaping symlinks and special files in a candidate tree."""
    for path in sorted(workspace.rglob("*")):
        rel = path.relative_to(workspace)
        if rel.parts[0] == ".git":
            continue
        if ".git" in rel.parts:
            raise ValueError(f"nested repository refused at {rel.as_posix()}")
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            target = str(path.readlink())
            resolved = posixpath.normpath(posixpath.join(rel.parent.as_posix(), target))
            if posixpath.isabs(target) or resolved == ".." or resolved.startswith("../"):
                raise ValueError(f"escaping symlink refused at {rel.as_posix()}")
        elif not stat.S_ISDIR(mode) and not stat.S_ISREG(mode):
            raise ValueError(f"unsupported workspace entry at {rel.as_posix()}")


def _transient(directory: str, names: list[str]) -> set[str]:
    """Git's own housekeeping files, which a copy of the candidate tree skips."""
    if ".git" not in Path(directory).parts:
        return set()
    return {name for name in names if name.endswith(".lock") or name.startswith("tmp_object_")}


def _noted(*, success: TaskSuccess, route: Callable[[bytes], object], note: str) -> TaskSuccess:
    return replace(success, metadata={**success.metadata, OUTCOME_KEY: route(note.encode())})


def _malformed(*, success: TaskSuccess, error: Exception) -> TaskFailure:
    return TaskFailure(files=success.metadata, error=failure_text(error), reason=FailureReason.MALFORMED_OUTPUT,
                       spend=success.spend)
