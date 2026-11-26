# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""Store immutable candidate trees and expansion turn chains in Git."""

from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import tempfile
import threading
from collections.abc import Callable
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING, Self
from uuid import uuid4

import pygit2
from pygit2.enums import CheckoutStrategy, FileMode, ObjectType, RepositoryOpenFlag, SortMode

from gm import experience, git_subprocess
from gm.ids import NodeId, RunId, TreeId
from gm.tree_copy import copy_tree

if TYPE_CHECKING:
    from collections.abc import Generator
    from contextlib import AbstractContextManager

_NODE_NAMESPACE = "refs/rqgm/nodes"
_TURN_NAMESPACE = "refs/rqgm/turns"
MAX_FILE_BYTES = 512 * 1024 * 1024
MAX_NODE_BYTES = 4 * 1024 * 1024 * 1024
SIGNATURE = pygit2.Signature("RQGM", "rqgm@example.invalid", 946684800, 0)
_PINNED_CONFIG = {
    "core.autocrlf": "false",
    "core.excludesFile": "",
    "core.fileMode": "true",
    "core.symlinks": "true",
    "commit.gpgsign": "false",
    "core.hooksPath": "/dev/null",
    "gc.auto": "0",
    "receive.autogc": "false",
    "i18n.commitEncoding": "utf-8",
}


def _init_repo(path: Path, *, bare: bool = False) -> pygit2.Repository:
    repo = pygit2.init_repository(str(path), bare=bare)
    for key, value in _PINNED_CONFIG.items():
        repo.config[key] = value
    return repo


def _only_evidence(*, repo: pygit2.Repository, before: pygit2.Oid, after: pygit2.Oid) -> bool:
    return all(
        experience.episode_context(rel=delta.new_file.path) and experience.episode_context(rel=delta.old_file.path)
        for delta in repo[before].diff_to_tree(repo[after]).deltas
    )


def _task_tree(*, source: pygit2.Repository, target: pygit2.Repository, tree: pygit2.Tree) -> pygit2.Oid:
    old, visible = pygit2.Index(), pygit2.Index()
    old.read_tree(tree)
    for entry in old:
        if (
            entry.path.split("/", 1)[0] == experience.EXPERIENCE
            or experience.unpublished(rel=entry.path)
            or experience.episode_context(rel=entry.path)
        ):
            continue
        if entry.id not in target:
            target.create_blob(source[entry.id].data)
        visible.add(entry)
    return visible.write_tree(target)


def _restore_entry(*, checkout: Path, entry: pygit2.IndexEntry, data: bytes) -> None:
    target = checkout / entry.path
    if target.is_dir() and not target.is_symlink():
        shutil.rmtree(target)
    else:
        target.unlink(missing_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    if entry.mode == FileMode.LINK:
        target.symlink_to(os.fsdecode(data))
    else:
        target.write_bytes(data)
        target.chmod(0o755 if entry.mode == FileMode.BLOB_EXECUTABLE else 0o644)


class NodeExistsError(Exception):
    """Indicate that the same deterministic node commit is already published."""

    def __init__(self, node: NodeId) -> None:
        super().__init__(f"node already published: {node}")
        self.node = node


class EmptyTreeError(ValueError):
    """Indicate that a candidate contains no change beyond host evidence."""


def _stage(checkout: Path, repo: pygit2.Repository) -> None:
    for _ in range(8):
        index = repo.index
        index.clear()
        try:
            index.add_all()
            paths = [checkout / rel for rel in experience.HOST_CONTEXT]
            paths.extend((checkout / experience.GENERATIONS_ROOT).rglob("*"))
            for path in paths:
                rel = path.relative_to(checkout).as_posix()
                if experience.required(rel=rel) and (path.is_file() or path.is_symlink()):
                    index.add(rel)
        except (pygit2.GitError, OSError):
            continue
        index.remove_all(list(experience.UNPUBLISHED))
        return
    raise OSError(f"could not take a settled snapshot of {checkout}")


@dataclass(frozen=True, kw_only=True)
class _Snapshot:
    index: pygit2.Index
    tree: pygit2.Oid
    repo: pygit2.Repository


@dataclass(kw_only=True)
class _TurnChain:
    store: NodeStore
    run: RunId
    parent: NodeId | None
    action: str | None
    head: pygit2.Oid | None
    on_first_turn: Callable[[str], None] | None = None
    announced: bool = False
    pending_claims: list[str] = field(default_factory=list)
    agent_head: str | None = None
    turns: list[NodeId] = field(default_factory=list)
    last_turn: bool = False

    @property
    def key(self) -> str:
        basis = self.action or f"{self.run}:{self.parent or 'root'}"
        return hashlib.sha256(f"turn:{basis}".encode()).hexdigest()

    @property
    def node_ref(self) -> str:
        return f"{_TURN_NAMESPACE}/{self.key}"

    def _claims(self, repo: pygit2.Repository) -> list[str]:
        try:
            if repo.head_is_unborn:
                return []
            head = str(repo.head.target)
        except (pygit2.GitError, ValueError, KeyError):
            return []
        before = self.agent_head if self.agent_head is not None else (None if self.parent is None else str(self.parent))
        self.agent_head = head
        if before is None or before == head:
            return []
        try:
            walk = repo.walk(pygit2.Oid(hex=head), SortMode.TOPOLOGICAL | SortMode.REVERSE)
            walk.hide(pygit2.Oid(hex=before))
            return [message for commit in walk if (message := str(commit.message).strip())]
        except (pygit2.GitError, ValueError, KeyError):
            return []

    def _message(
        self, *, tree: pygit2.Oid, caption: str | None, claim: str | None, repo: pygit2.Repository | None = None
    ) -> str | None:
        if repo is not None:
            self.pending_claims.extend(self._claims(repo))
        if claim and claim.strip():
            self.pending_claims.append(claim)
        unchanged = self.head is not None and self.store._only_host(head=self.head, tree=tree)
        if unchanged or not self.pending_claims:
            return caption
        carried, self.pending_claims = self.pending_claims, []
        return "\n\n".join([*carried, *([caption] if caption else [])])

    def _commit(self, *, snapshot: _Snapshot, message: str | None) -> NodeId:
        self.store._mirror(snapshot)
        written = self.store._write_commit(
            tree=snapshot.tree, parents=[] if self.head is None else [self.head], message=message
        )
        self.head = pygit2.Oid(hex=str(written))
        self.store._repo.references.create(self.node_ref, self.head, force=True)
        if not self.announced and self.on_first_turn is not None:
            self.announced = True
            self.on_first_turn(self.key)
        return written

    def turn(
        self, *, checkout: Path, message: str | None = None, claim: str | None = None
    ) -> NodeId:
        """Commit one model call's tree, or reuse the tip when only host context changed."""
        with self.store._snapshot(checkout) as snapshot:
            self.store._mirror(snapshot)
            if self.head is not None and self.store._only_host(head=self.head, tree=snapshot.tree):
                self._message(tree=snapshot.tree, caption=message, claim=claim, repo=snapshot.repo)
                return NodeId(str(self.head))
            written = self._commit(
                snapshot=snapshot,
                message=self._message(tree=snapshot.tree, caption=message, claim=claim, repo=snapshot.repo),
            )
        self.turns.append(written)
        self.last_turn = True
        return written

    def submission(
        self, *, checkout: Path, ordinal: int, tag: str | None, claim: str | None = None
    ) -> NodeId:
        """Commit a candidate version for an in-expansion train job."""
        with self.store._snapshot(checkout) as snapshot:
            self.store._mirror(snapshot)
            written = self._commit(
                snapshot=snapshot,
                message=self._message(
                    tree=snapshot.tree, caption=f"train submit {ordinal}" + (f": {tag}" if tag else ""), claim=claim,
                    repo=snapshot.repo,
                ),
            )
        self.last_turn = False
        return written

    def seal(self, *, checkout: Path, message: str | None = None) -> NodeId:
        """Publish the chain's final candidate node."""
        with self.store._snapshot(checkout) as snapshot:
            self.pending_claims.extend(self._claims(snapshot.repo))
            self.store._mirror(snapshot)
            host_only = self.head is not None and self.store._only_host(head=self.head, tree=snapshot.tree)
        if message and message.strip():
            self.pending_claims.append(message)
        carried, self.pending_claims = self.pending_claims, []
        if host_only and carried:
            carried = [f"claimed, no file changed: {claim}" for claim in carried]
        return self.store._seal_chain(
            snapshot=snapshot,
            parent=self.parent,
            head=self.head,
            head_is_turn=self.last_turn,
            message="\n\n".join(carried) if carried else None,
            turn_ref=self.node_ref if self.action is not None else None,
        )


class NodeStore:
    """Own one exclusive writer over a bare deterministic Git node store."""

    def __init__(self, path: Path) -> None:
        primary = pygit2.Repository(str(path))
        self._git_dir = Path(primary.path)
        self._lock: IO[str] | None = _acquire_writer_lock(self._git_dir)
        self._handles = threading.local()
        self._handles.repo = primary
        self._protos: Path | None = None
        self._proto_lock = threading.Lock()
        self._proto_builds: dict[Path, threading.Lock] = {}
        self._chains: dict[str, _TurnChain] = {}
        self._states: dict[str, dict[str, object]] = {}
        self._state_lock = threading.Lock()

    @property
    def _repo(self) -> pygit2.Repository:
        repo: pygit2.Repository | None = getattr(self._handles, "repo", None)
        if repo is None:
            repo = pygit2.Repository(str(self._git_dir), RepositoryOpenFlag.NO_SEARCH)
            self._handles.repo = repo
        return repo

    @classmethod
    def init(cls, path: Path) -> NodeStore:
        _init_repo(path, bare=True)
        return cls(path)

    def close(self) -> None:
        if self._lock is not None:
            self._lock.close()
            self._lock = None
        root, self._protos = self._protos, None
        if root is not None:
            shutil.rmtree(root, ignore_errors=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def create_root(self, *, run: RunId, checkout: Path) -> NodeId:
        """Create a root; repeated calls may create independent roots."""
        chain = self.open_chain(run=run, parent=None)
        node = chain.seal(checkout=checkout)
        self.final_turn(
            node=node,
            chain=None,
            reason="the store created this node directly",
            done=True,
            grounds="the run's stated anchor, admitted by the operator who named it",
        )
        return node

    def _write_commit(self, *, tree: pygit2.Oid, parents: list[pygit2.Oid], message: str | None) -> NodeId:
        return NodeId(str(self._repo.create_commit(None, SIGNATURE, SIGNATURE, message or "", tree, parents)))

    def open_chain(self, *, run: RunId, parent: NodeId | None, action: str | None = None) -> _TurnChain:
        head = None if parent is None else self._commit(parent).id
        chain = _TurnChain(store=self, run=run, parent=parent, action=action, head=head)
        if action is not None:
            self._chains[action] = chain
        return chain

    def _seal_chain(
        self,
        *,
        snapshot: _Snapshot,
        parent: NodeId | None,
        head: pygit2.Oid | None,
        head_is_turn: bool,
        message: str | None,
        turn_ref: str | None,
    ) -> NodeId:
        if parent is not None and _only_evidence(
            repo=self._repo, before=self._commit(parent).tree_id, after=snapshot.tree
        ):
            raise EmptyTreeError(f"checkout is unchanged from parent: {parent}")
        if head is not None and head_is_turn and self._commit(NodeId(str(head))).tree_id == snapshot.tree and message is None:
            node = NodeId(str(head))
        else:
            node = self._write_commit(tree=snapshot.tree, parents=[] if head is None else [head], message=message)
        if turn_ref is not None and turn_ref in self._repo.references:
            self._repo.references.delete(turn_ref)
        ref = f"{_NODE_NAMESPACE}/{node}"
        if ref in self._repo.references:
            raise NodeExistsError(node)
        self._repo.references.create(ref, pygit2.Oid(hex=str(node)))
        return node

    def _published(self) -> set[NodeId]:
        prefix = f"{_NODE_NAMESPACE}/"
        return {NodeId(name.removeprefix(prefix)) for name in self._repo.references if name.startswith(prefix)}

    def lineage(self, node: NodeId) -> tuple[NodeId, ...]:
        """Return the published nodes on this commit's first-parent chain."""
        published, line = self._published(), []
        commit = self._commit(node)
        while True:
            current = NodeId(str(commit.id))
            if current in published:
                line.append(current)
            if not commit.parents:
                break
            commit = commit.parents[0]
        return tuple(reversed(line))

    def depth(self, node: NodeId) -> int:
        return len(self.lineage(node)) - 1

    def repack(self) -> None:
        git_subprocess.probe(f"--git-dir={self._git_dir}", "gc", "--prune=2.hours.ago", "--quiet")

    def submission(
        self, *, checkout: Path, action: str, ordinal: int, tag: str | None, claim: str | None = None
    ) -> NodeId:
        return self._chains[action].submission(checkout=checkout, ordinal=ordinal, tag=tag, claim=claim)

    def submission_context(self, *, action: str, commit: NodeId) -> tuple[TreeId, NodeId | None, int | None]:
        turns = self._chains[action].turns
        return TreeId(str(self._commit(commit).tree_id)), (turns[-1] if turns else None), (len(turns) if turns else None)

    def restore(self, *, commit: NodeId, checkout: Path) -> None:
        held = self.read_tree(node=commit)
        for path in sorted(checkout.rglob("*"), reverse=True):
            rel = path.relative_to(checkout).as_posix()
            if experience.unpublished(rel=rel) or experience.episode_context(rel=rel):
                continue
            if path.is_dir() and not path.is_symlink():
                with suppress(OSError):
                    path.rmdir()
            elif rel not in held:
                path.unlink(missing_ok=True)
        entries = pygit2.Index()
        entries.read_tree(self._commit(commit).tree)
        for entry in entries:
            if not experience.unpublished(rel=entry.path) and not experience.episode_context(rel=entry.path):
                _restore_entry(checkout=checkout, entry=entry, data=held[entry.path])

    def mark_attempt(self, *, action: str, checkout: Path, note: bytes | None, message: str | None = None) -> NodeId:
        chain = self._chains[action]
        if note is not None:
            target = checkout / experience.generation(depth=0 if chain.parent is None else self.depth(chain.parent) + 1)
            target = target / experience.VERDICT
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(note)
        return chain.seal(checkout=checkout, message=message)

    def node_ref(self, node: NodeId) -> str | None:
        ref = f"{_NODE_NAMESPACE}/{node}"
        return ref if ref in self._repo.references else None

    def final_turn(
        self, *, node: NodeId, chain: str | None, reason: str, done: bool, grounds: str, action: str | None = None
    ) -> None:
        """Hold the final-turn fact until the controller appends it to the ledger."""
        del chain
        if action is not None:
            with self._state_lock:
                self._states[action] = {
                    "node": str(node), "reason": reason, "done": done, "grounds": grounds, "action": action,
                }

    def node_state_for_action(self, action: str) -> dict[str, object] | None:
        with self._state_lock:
            return self._states.get(action)

    @contextmanager
    def _snapshot(self, checkout: Path) -> Generator[_Snapshot]:
        repo = self._workspace_repo(checkout)
        _stage(checkout, repo)
        total = 0
        for entry in repo.index:
            kind, size = repo.odb.read_header(entry.id)
            if kind != ObjectType.BLOB:
                raise ValueError(f"node path {entry.path!r} is not an ordinary file")
            if size > MAX_FILE_BYTES:
                raise ValueError(f"node path {entry.path!r} is {size} bytes, over the {MAX_FILE_BYTES}-byte ceiling")
            total += size
        if total > MAX_NODE_BYTES:
            raise ValueError(f"node files total {total} bytes, over the {MAX_NODE_BYTES}-byte ceiling")
        tree = repo.index.write_tree()
        flat = pygit2.Index()
        flat.read_tree(repo[tree])
        yield _Snapshot(index=flat, tree=tree, repo=repo)

    def _mirror(self, snapshot: _Snapshot) -> None:
        mirror = pygit2.Index()
        for entry in snapshot.index:
            if entry.id not in self._repo:
                self._repo.create_blob(snapshot.repo[entry.id].data)
            mirror.add(entry)
        mirror.write_tree(self._repo)

    @staticmethod
    def _workspace_repo(root: Path) -> pygit2.Repository:
        return pygit2.Repository(str(root), RepositoryOpenFlag.NO_SEARCH) if (root / ".git").exists() else _init_repo(root)

    def _only_host(self, *, head: pygit2.Oid, tree: pygit2.Oid) -> bool:
        before = self._commit(NodeId(str(head))).tree_id
        return before == tree or _only_evidence(repo=self._repo, before=before, after=tree)

    def read_tree(self, *, node: NodeId) -> dict[str, bytes]:
        flat = pygit2.Index()
        flat.read_tree(self._commit(node).tree)
        return {entry.path: self._repo[entry.id].read_raw() for entry in flat}

    def materialize(
        self, node: NodeId, path: Path, *, on_wait: Callable[[], AbstractContextManager[None]] | None = None
    ) -> None:
        self._materialize_cached(node=node, path=path, task=False, on_wait=on_wait)

    def materialize_task(
        self, *, node: NodeId, path: Path, on_wait: Callable[[], AbstractContextManager[None]] | None = None
    ) -> None:
        self._materialize_cached(node=node, path=path, task=True, on_wait=on_wait)

    def _materialize_cached(
        self, *, node: NodeId, path: Path, task: bool, on_wait: Callable[[], AbstractContextManager[None]] | None
    ) -> None:
        with self._proto_lock:
            if self._protos is None:
                self._protos = Path(tempfile.mkdtemp(prefix="rqgm-protos-"))
            proto = self._protos / f"{'task' if task else 'full'}-{node}"
            building = self._proto_builds.setdefault(proto, threading.Lock())
        while not building.acquire(blocking=False):
            with nullcontext() if on_wait is None else on_wait():
                with building:
                    pass
        try:
            if not proto.exists():
                staging = self._protos / f"tmp-{uuid4().hex}"
                if task:
                    self._materialize_task_cold(node=node, path=staging)
                else:
                    self._materialize_cold(node=node, path=staging)
                staging.rename(proto)
        finally:
            building.release()
        copy_tree(source=proto, into=path, symlinks=True, dirs_exist_ok=True)

    def _history(self, node: NodeId) -> tuple[pygit2.Commit, ...]:
        return tuple(self._repo.walk(self._commit(node).id, SortMode.TOPOLOGICAL))

    def _materialize_cold(self, *, node: NodeId, path: Path) -> None:
        workspace = _init_repo(path)
        pack = pygit2.PackBuilder(self._repo)
        pack.set_threads(0)
        for commit in self._history(node):
            pack.add_recur(commit.id)
        pack_dir = Path(workspace.path) / "objects" / "pack"
        pack_dir.mkdir(parents=True, exist_ok=True)
        pack.write(str(pack_dir))
        tip = pygit2.Oid(hex=str(node))
        workspace.checkout_tree(workspace[tip].tree, strategy=CheckoutStrategy.FORCE | CheckoutStrategy.RECREATE_MISSING)
        workspace.set_head(tip)
        exclude = Path(workspace.path) / "info" / "exclude"
        exclude.parent.mkdir(parents=True, exist_ok=True)
        exclude.write_text("".join(f"{one}\n" for one in experience.UNPUBLISHED), encoding="utf-8")
        for published in self.lineage(node):
            workspace.references.create(f"{_NODE_NAMESPACE}/{published}", pygit2.Oid(hex=str(published)))

    def _materialize_task_cold(self, *, node: NodeId, path: Path) -> None:
        workspace = _init_repo(path)
        mapped: dict[pygit2.Oid, pygit2.Oid] = {}
        trees: dict[pygit2.Oid, pygit2.Oid] = {}
        for ancestor in reversed(self._history(node)):
            trees.setdefault(ancestor.tree_id, _task_tree(source=self._repo, target=workspace, tree=ancestor.tree))
            mapped[ancestor.id] = workspace.create_commit(
                None, SIGNATURE, SIGNATURE, f"Task source of {ancestor.id}", trees[ancestor.tree_id],
                [mapped[parent] for parent in ancestor.parent_ids if parent in mapped],
            )
        tip = mapped[self._commit(node).id]
        workspace.checkout_tree(workspace[tip].tree, strategy=CheckoutStrategy.FORCE)
        workspace.set_head(tip)
        try:
            generations = self._repo[self._commit(node).tree[experience.GENERATIONS_ROOT].id]
        except KeyError:
            return
        if isinstance(generations, pygit2.Tree):
            for entry in generations:
                if isinstance(entry, pygit2.Tree) and entry.name.isdigit():
                    (path / experience.GENERATIONS_ROOT / entry.name).mkdir(parents=True)

    def _commit(self, node: NodeId) -> pygit2.Commit:
        return self._repo[pygit2.Oid(hex=str(node))]


@contextmanager
def experience_view(store: NodeStore, parent: NodeId, *, workdir: Path | None = None) -> Generator[Path]:
    workspace = Path(tempfile.mkdtemp(prefix="rqgm-experience-", dir=workdir))
    try:
        store.materialize(parent, workspace)
        yield workspace
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def _acquire_writer_lock(git_dir: Path) -> IO[str]:
    handle = (git_dir / "rqgm-writer.lock").open("a+")
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    handle.truncate(0)
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle
