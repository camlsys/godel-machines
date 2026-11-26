# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""Serve an expansion's own train evaluations on a socket only that expansion can reach.

Each expansion episode gets one listener socket, bind-mounted read-only at `/run/rqgm/pool.sock`;
the socket is the request's whole identity. A `submit` commits the node's tree, is admitted against
free CPU, reserves the accepted (role, sample) seats in the `train` scope, and every settled row is
delivered into the node's generation folder as `results.csv` and `summary.json`. The node-side
client is `rqgm-call`.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import socket
import stat
import threading
import time
from collections import Counter
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Final

from gm import experience, ids, pointer, sampling
from gm.domain import RoleId, SampleKey, TaskId
from gm.domain.results import TaskFailure
from gm.ledger import INPUT_CHANNEL
from gm.execution.episodes import claim_message, failure_fields, host_write

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from gm.domain.dataset import DatasetSpec
    from gm.domain.manifest import DomainManifest
    from gm.domain.results import Artifact
    from gm.ledger import SqliteEventLog
    from gm.execution.runner import Runner

MEASURED: Final = "MEASURED"
FAILED: Final = "FAILED"
REFUSED: Final = "REFUSED"
#: Fixed places for every figure a group states about itself.
PLACES: Final = 3
#: The fields one deciding judgment contributes to its row.
METRIC_FIELDS: Final = (
    "kind", "success", "score", "refusal_code", "failure_reason", "failure_detail", "judgment_action",
    "evaluator_commit",
)
#: Exactly the keys each operation may carry: no request field can name another episode.
_ALLOWED_KEYS = {
    "submit": frozenset({"op", "batch", "all", "per_role", "role", "tag", "timeout_s"}),
    "wait": frozenset({"op", "job", "timeout_s"}),
}
#: The closed error vocabulary a node can receive.
UNKNOWN_ROLE = "unknown_role"
UNDECLARED_SAMPLE = "undeclared_sample"
INSUFFICIENT_CAPACITY = "insufficient_capacity"
NOT_AN_EPISODE = "not_an_episode"
HOST_FAULT = "host_fault"
_IDLE_SECONDS = 900.0
_ACCEPT_POLL = 0.25
_WRITABLE = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


# The results files.

@dataclass(frozen=True, slots=True, kw_only=True)
class TrainRow:
    """One `results.csv` row: one sample, evaluated once, for one role, in one job. Fields are in column order."""

    job: str
    ordinal: int
    submitted_at: str
    turn_commit: str
    submission_commit: str
    evaluated_commit: str = ""
    evaluator_commit: str = ""
    provenance_error: str = ""
    seat_digest: str
    tag: str = ""
    role: str
    task: str
    sample: str
    kind: str
    success: bool | None = None
    score: float | None = None
    refusal_code: str = ""
    tokens_in: int | None = None
    tokens_out: int | None = None
    action: str = ""
    judgment_action: str = ""
    failure_reason: str = ""
    failure_detail: str = ""
    metrics: Mapping[str, Mapping[str, object]] = field(default_factory=dict)


TRAIN_COLUMNS: Final = tuple(one.name for one in fields(TrainRow))


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:.{PLACES}f}"
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _read(column: str, cell: str) -> object:
    """Read one `results.csv` cell back; an empty number reads as None."""
    if column == "metrics":
        return json.loads(cell) if cell else {}
    if column == "success":
        return cell == "1" if cell else None
    if column in {"ordinal", "tokens_in", "tokens_out"}:
        return int(cell) if cell else None
    if column == "score":
        return float(cell) if cell else None
    return cell


def stats(rows: Sequence[TrainRow]) -> dict[str, object]:
    """Count one group of rows and state its success rate and mean score over the measured ones."""
    kinds = Counter(row.kind for row in rows)
    judged = [bool(row.success) for row in rows if row.success is not None]
    scores = [float(row.score) for row in rows if row.score is not None]
    return {
        "n": len(rows), "measured": kinds[MEASURED], "failed": kinds[FAILED], "refused": kinds[REFUSED],
        "successes": sum(judged),
        "rate": round(sum(judged) / len(judged), PLACES) if judged else None,
        "score_mean": round(sum(scores) / len(scores), PLACES) if scores else None,
    }


def _by_role(rows: Sequence[TrainRow]) -> dict[str, object]:
    return {role: stats([row for row in rows if row.role == role]) for role in sorted({row.role for row in rows})}


def _by_seat(rows: Sequence[TrainRow]) -> dict[str, object]:
    """Group each role's rows by the submitted tree that produced them."""
    return {
        role: {digest: stats([row for row in rows if row.role == role and row.seat_digest == digest])
               for digest in sorted({row.seat_digest for row in rows if row.role == role})}
        for role in sorted({row.role for row in rows})
    }


def write_results(*, workspace: Path, depth: int, rows: Sequence[TrainRow], turns: Mapping[str, int | None],
                  jobs: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """Write one generation's `results.csv` and `summary.json`; return the summary written."""
    root = workspace / experience.generation(depth=depth)
    root.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow(TRAIN_COLUMNS)
    writer.writerows([_cell(getattr(row, column)) for column in TRAIN_COLUMNS] for row in rows)
    (root / experience.RESULTS).write_bytes(buffer.getvalue().encode())
    lineage: dict[int, list[TrainRow]] = {}
    for ancestor in range(depth):
        record = workspace / experience.generation(depth=ancestor) / experience.RESULTS
        if record.is_file():
            text = record.read_text(encoding="utf-8", errors="replace")
            if held := [TrainRow(**{column: _read(column, parsed.get(column, "")) for column in TRAIN_COLUMNS})
                        for parsed in csv.DictReader(io.StringIO(text))]:
                lineage[ancestor] = held
    document = {
        "jobs": {
            job: {**held, "job": job, "turn": turns.get(job), "tag": held["tag"] or None,
                  "roles": _by_role([row for row in rows if row.job == job])}
            for job, held in jobs.items()
        },
        "roles": _by_role(rows),
        "by_seat": _by_seat(rows),
        "lineage": {str(depth): _by_seat(ancestors) for depth, ancestors in sorted(lineage.items())},
    }
    (root / experience.SUMMARY).write_bytes((json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode())
    return document


def stage_train_data(*, build: DatasetSpec, artifacts: SqliteEventLog, sets: frozenset[str], into: Path) -> None:
    """Write every declared train sample's inputs and source labels under `experience/data/train/`, read-only."""
    selected = artifacts.selected_inputs(versions=(build.version,), sample_sets=sets, channel=INPUT_CHANNEL)
    inputs = {sample: tuple(artifacts.resolve_artifact(one) for one in dict.fromkeys(handles))
              for (_, sample), handles in selected.items()}
    root = into / experience.TRAIN
    root.mkdir(parents=True, exist_ok=True)
    for sample in dict.fromkeys(sample for name in sorted(sets) for sample in build.splits[name]):
        ids.require_safe_relpath(path=str(sample))
        folder = root / str(sample)
        folder.mkdir(parents=True)
        for index, payload in enumerate(inputs.get(sample, ())):
            (folder / f"input-{index}").write_bytes(payload)
        for label, data in build.source_rows.get(sample, {}).items():
            ids.require_safe_relpath(path=label)
            target = folder / experience.LABELS / label
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    for path in into.rglob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode & ~_WRITABLE)



# The pool.

class PoolError(RuntimeError):
    """Refuse one request with a code from the closed vocabulary; it becomes the reply."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


@dataclass(slots=True, eq=False)
class _Job:
    """One submission: its seats, the actions they opened, and the rows settled so far."""

    job: str
    ordinal: int
    commit: ids.NodeId
    tag: str | None
    submitted_at: str
    turn_commit: str
    turn: int | None
    seat_digest: str
    refused: list[dict[str, str]]
    requested: set[tuple[RoleId, SampleKey]]
    actions: dict[tuple[RoleId, SampleKey], ids.ActionId] = field(default_factory=dict)
    tasks: dict[tuple[RoleId, SampleKey], TaskId] = field(default_factory=dict)
    judgments: dict[tuple[RoleId, SampleKey], tuple[ids.ActionId, ...]] = field(default_factory=dict)
    settled: set[ids.ActionId] = field(default_factory=set)
    issuing: bool = True
    done: bool = False
    #: Rows already delivered into the node's files, keyed by (role, sample).
    recorded: dict[tuple[str, str], dict[str, object]] = field(default_factory=dict)
    summary: dict[str, object] | None = None
    staging_error: str | None = None

    def every(self) -> set[ids.ActionId]:
        """Every producer and every judgment reserved beside it."""
        return set(self.actions.values()) | {one for opened in self.judgments.values() for one in opened}

    def answered(self, pair: tuple[RoleId, SampleKey]) -> bool:
        """Whether one (role, sample) has its producer and every reserved judgment settled."""
        return self.actions[pair] in self.settled and all(one in self.settled for one in self.judgments.get(pair, ()))

    def pending(self) -> list[str]:
        """Samples whose rows have not been delivered yet."""
        return [str(sample) for role, sample in sorted(self.requested | set(self.actions), key=str)
                if (str(role), str(sample)) not in self.recorded]


@dataclass(slots=True, eq=False)
class _Episode:
    """One expansion episode's listener and the submissions it made."""

    action: ids.ActionId
    checkout: Path
    depth: int
    path: Path
    listener: socket.socket
    jobs: list[_Job] = field(default_factory=list)
    #: Row identities already reported to the node, per job.
    reported: dict[str, set[tuple[str, str]]] = field(default_factory=dict)
    #: Requests currently blocked with this expansion's CPU lent out.
    sleepers: int = 0
    closed: bool = False
    #: Set once the expansion sealed its generation folder; later rows reach no published tree.
    published: bool = False
    #: Held by the expansion's turn commits and by every delivery into the same folder.
    writing: threading.RLock = field(default_factory=threading.RLock)


class Pool:
    """One listener per expansion episode, and the train batches submitted on them."""

    def __init__(self, *, store: Path, controller: object, domain: DomainManifest, nodes: object,
                 runner: Runner, artifacts: SqliteEventLog) -> None:
        self._store, self._controller, self._domain = store, controller, domain
        self._nodes, self._runner, self._artifacts = nodes, runner, artifacts
        self._lock = threading.RLock()
        self._room = threading.Condition(self._lock)
        self._admitting = threading.Lock()
        self._episodes: dict[ids.ActionId, _Episode] = {}
        self._closed = False

    # Lifecycle, driven by the expansion.

    def open_episode(self, *, action: ids.ActionId, checkout: Path, depth: int) -> Path:
        """Start listening for one episode and return the socket its container mounts."""
        root = self._store / "pool"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{str(action)[:12]}.sock"
        path.unlink(missing_ok=True)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(path))
        # The container reaches the socket through its group (`--group-add`), without DAC override.
        path.chmod(0o660)
        listener.listen(8)
        listener.settimeout(_ACCEPT_POLL)
        episode = _Episode(action=action, checkout=checkout, depth=depth, path=path, listener=listener)
        with self._lock:
            self._episodes[action] = episode
        threading.Thread(target=self._accept, args=(episode,), name=f"rqgm-pool-{action}", daemon=True).start()
        return path

    def episode_writing(self, *, action: ids.ActionId) -> threading.RLock:
        """The lock the episode's transcript writes share with result delivery."""
        return self._episodes[action].writing

    def close_episode(self, *, action: ids.ActionId) -> None:
        """Stop listening for one episode; its accepted jobs keep settling into its checkout."""
        with self._room:
            self._retire(self._episodes[action])

    def drain_episode(self, *, action: ids.ActionId) -> None:
        """Wait, CPU lent, until every accepted job settled; then rewrite its files on the checkout."""
        episode = self._episodes[action]
        with self._room:
            self._retire(episode)
        self._sleep_until(episode=episode, ready=lambda: all(one.done for one in episode.jobs), timeout=None)
        if episode.jobs and not episode.published:
            # A restored submission may have overwritten the delivered files.
            self._deliver(episode)

    def best_submission(self, *, action: ids.ActionId) -> ids.NodeId | None:
        """The settled submission with the best mean measured score."""
        scores: dict[ids.NodeId, list[float]] = {}
        with self._lock:
            for job in self._episodes[action].jobs:
                for row in self._rows(job):
                    score = row.get("score")
                    if row["kind"] == MEASURED and type(score) in {int, float} and math.isfinite(score):
                        scores.setdefault(job.commit, []).append(float(score))
        return max(scores, key=lambda commit: sum(scores[commit]) / len(scores[commit])) if scores else None

    def published(self, *, action: ids.ActionId) -> None:
        """Fence delivery once this episode's generation folder has been sealed."""
        episode = self._episodes[action]
        with episode.writing:
            episode.published = True

    def close(self) -> None:
        """Retire every listener and wake every blocked request."""
        with self._room:
            self._closed = True
            for episode in self._episodes.values():
                self._retire(episode)

    def settled(self, *, action_id: ids.ActionId) -> None:
        """Record one settled action and deliver the rows it completed."""
        with self._controller.ledger_held(), self._lock:
            touched: list[tuple[_Episode, _Job, bool]] = []
            for episode in self._episodes.values():
                for job in episode.jobs:
                    if job.issuing or action_id not in job.every() or action_id in job.settled:
                        continue
                    before = sum(map(job.answered, job.actions))
                    job.settled.add(action_id)
                    touched.append((episode, job, sum(map(job.answered, job.actions)) > before))
        for episode, job, answered in touched:
            if not job.every() - job.settled:
                self._finish(job=job, episode=episode)
            elif answered:
                self._deliver(episode)
        with self._room:
            self._room.notify_all()

    def _retire(self, episode: _Episode) -> None:
        episode.closed = True
        with contextlib.suppress(OSError):
            episode.listener.close()
        episode.path.unlink(missing_ok=True)
        self._room.notify_all()

    # The wire.

    def _accept(self, episode: _Episode) -> None:
        while not episode.closed:
            try:
                connection, _ = episode.listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            threading.Thread(target=self._converse, args=(episode, connection), daemon=True).start()

    def _converse(self, episode: _Episode, connection: socket.socket) -> None:
        """Answer the one request line a connection carries, then close it."""
        with connection, contextlib.suppress(OSError):
            connection.settimeout(_IDLE_SECONDS)
            if line := connection.makefile("rb").readline():
                connection.sendall(self.handle(action=episode.action, line=line.decode("utf-8")).encode("utf-8"))

    def handle(self, *, action: ids.ActionId, line: str) -> str:
        """The reply line one request earns; every refusal is a reply, never silence."""
        try:
            request = json.loads(line)
            op = request.get("op")
            if op not in _ALLOWED_KEYS:
                raise PoolError(NOT_AN_EPISODE, f"no such operation: {op!r}")
            if extra := sorted(set(request) - _ALLOWED_KEYS[op]):
                raise PoolError(NOT_AN_EPISODE, f"request carries keys {op!r} does not declare: {extra}; identity"
                                                " is the socket and no request field names an episode")
            episode = self._episodes[action]
            body = self._submit(episode, request) if op == "submit" else self._wait(episode, request)
            body["ok"] = True
        except PoolError as refusal:
            body = {"ok": False, "error": {"code": refusal.code, "detail": refusal.detail}}
        except Exception as fault:  # a fault still answers, so the node stops retrying
            body = {"ok": False, "error": {"code": HOST_FAULT, "detail": f"{type(fault).__name__}: {fault}"}}
        return json.dumps(body, separators=(",", ":"), sort_keys=True) + "\n"

    # Waiting with the expansion's CPU lent out.

    def _lend(self, *, episode: _Episode, sleeping: bool) -> None:
        """Count one blocked request in or out; the first lends the expansion's CPU, the last takes it back."""
        with self._lock:
            episode.sleepers += 1 if sleeping else -1
            edge = episode.sleepers == (1 if sleeping else 0)
        # Reacquiring the CPU may wait on settling train work, so no pool lock is held here.
        if edge:
            (self._runner.sleep if sleeping else self._runner.wake)(str(episode.action))

    def _sleep_until(self, *, episode: _Episode, ready: Callable[[], bool], timeout: float | None) -> None:
        with self._lock:
            if ready():
                return
        self._lend(episode=episode, sleeping=True)
        deadline = None if timeout is None else time.monotonic() + timeout
        try:
            with self._room:
                while not (ready() or self._closed):
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        break
                    self._room.wait(timeout=remaining)
        finally:
            self._lend(episode=episode, sleeping=False)

    # Operations.

    def _submit(self, episode: _Episode, request: Mapping[str, object]) -> dict[str, object]:
        """Admit one batch; the episode's first submission also waits for its first result."""
        ceiling = request.get("timeout_s")
        if episode.jobs or ceiling == 0:
            return self._response(job=self._admit(episode, request), episode=episode)
        # The CPU is lent before capacity is asked for, so on a host with one free slot the batch
        # can take this expansion's.
        self._lend(episode=episode, sleeping=True)
        try:
            job = self._admit(episode, request)
            self._sleep_until(episode=episode, timeout=ceiling, ready=lambda: job.done or job.staging_error is not None
                              or any(row["kind"] != REFUSED for row in job.recorded.values()))
        finally:
            self._lend(episode=episode, sleeping=False)
        return self._response(job=job, episode=episode)

    def _wait(self, episode: _Episode, request: Mapping[str, object]) -> dict[str, object]:
        """Wait for new results from any batch, or for the named batch to complete."""
        named = request.get("job")
        with self._lock:
            jobs = list(episode.jobs)
        job = next((one for one in jobs if one.job == named), None) if named else (jobs[-1] if jobs else None)
        if job is None:
            raise PoolError(NOT_AN_EPISODE, f"no submission {named!r} to wait for")

        def ready() -> bool:
            if named:
                return job.done or job.staging_error is not None
            return all(one.done for one in episode.jobs) or any(
                set(one.recorded) - episode.reported.get(one.job, set()) or one.staging_error is not None
                for one in episode.jobs)

        if request.get("timeout_s") != 0:
            self._sleep_until(episode=episode, ready=ready, timeout=request.get("timeout_s"))
        return self._response(job=job, episode=episode)

    # Admission.

    def _build(self) -> DatasetSpec:
        return sampling.pinned_build(projection=self._controller.projection, domain=self._domain)

    def _train_sets(self, role: RoleId) -> list[tuple[TaskId, str]]:
        """Each task of `role` that declares a train set, with that set's name."""
        sets = {} if self._domain.train is None else self._domain.train.task_sets
        return [(task, str(sets[task])) for task in self._domain.generator.task_pools.get(role, ()) if task in sets]

    def _asked(self, request: Mapping[str, object], build: DatasetSpec) -> dict[RoleId, list[SampleKey]]:
        """The (role, sample) pairs one request asks for; a `role/task` batch key is read as the role."""
        if (batch := request.get("batch")) is not None:
            return {RoleId(str(role).partition("/")[0]): [SampleKey(str(one)) for one in samples]
                    for role, samples in batch.items()}
        roles = self._domain.generator.task_pools
        named = request.get("role")
        wanted = [RoleId(str(named))] if named is not None else sorted(roles, key=str)
        drawn = {role: tuple(dict.fromkeys(sample for _, name in self._train_sets(role)
                                           for sample in build.splits.get(name, ()))) for role in wanted}
        # Absent a size, take the smallest population any drawing role declares.
        count = request.get("per_role") or min(len(members) for members in drawn.values() if members)
        return {role: list(drawn[role][:count]) for role in wanted if drawn[role]}

    def _task_of(self, *, role: RoleId, sample: SampleKey, build: DatasetSpec) -> TaskId:
        """The task binding whose declared train set contains this sample."""
        for task, name in self._train_sets(role):
            if sample in build.splits.get(name, ()):
                return task
        raise PoolError(UNDECLARED_SAMPLE, f"sample {str(sample)!r} is in no train binding of role {str(role)!r}")

    def _admit(self, episode: _Episode, request: Mapping[str, object]) -> _Job:
        """Seat the rows free CPU can run now, dealt round-robin across roles; refuse the rest."""
        action, granted = episode.action, 0
        try:
            with self._admitting:
                build = self._build()
                asked = self._asked(request, build)
                refused: list[dict[str, str]] = []
                for role in sorted(asked, key=str):
                    if role not in self._domain.generator.task_pools:
                        refused += [{"role": str(role), "sample": str(one), "code": UNKNOWN_ROLE, "task": ""}
                                    for one in asked.pop(role)]
                candidates: list[tuple[RoleId, SampleKey, TaskId]] = []
                known = sorted(asked, key=str)
                for index in range(max((len(asked[role]) for role in known), default=0)):
                    for role in known:
                        if index < len(asked[role]):
                            sample = asked[role][index]
                            try:
                                candidates.append((role, sample, self._task_of(role=role, sample=sample, build=build)))
                            except PoolError as refusal:
                                refused.append({"role": str(role), "sample": str(sample), "code": refusal.code,
                                                "task": ""})
                if candidates:
                    granted = self._runner.grant(owner=str(action), requested=len(candidates))
                accepted: dict[RoleId, list[SampleKey]] = {}
                for index, (role, sample, task) in enumerate(candidates):
                    if index < granted:
                        accepted.setdefault(role, []).append(sample)
                    else:
                        refused.append({"role": str(role), "sample": str(sample), "task": str(task),
                                        "code": INSUFFICIENT_CAPACITY})
                ordinal = len(episode.jobs) + 1
                tag = None if request.get("tag") is None else str(request["tag"])
                with episode.writing:
                    self._runner.carry_out(socket=str(episode.path), checkout=episode.checkout)
                    commit = self._nodes.submission(checkout=episode.checkout, action=str(action), ordinal=ordinal,
                                                    tag=tag, claim=claim_message(episode.checkout))
                return self._issue(episode=episode, ordinal=ordinal, commit=commit, accepted=accepted,
                                   refused=refused, tag=tag, build=build)
        finally:
            # Issued seats consumed their units at dispatch; whatever remains goes back.
            if granted:
                self._runner.release(owner=str(action), count=granted)

    def _issue(self, *, episode: _Episode, ordinal: int, commit: ids.NodeId, accepted: Mapping[RoleId, list[SampleKey]],
               refused: list[dict[str, str]], tag: str | None, build: DatasetSpec) -> _Job:
        """Reserve the accepted seats under the `train` scope and track their settlement."""
        digest, turn_commit, turn = self._nodes.submission_context(action=str(episode.action), commit=commit)
        job = _Job(
            job=f"t-{str(episode.action)[:8]}-{ordinal}", ordinal=ordinal, commit=commit, tag=tag,
            submitted_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            turn_commit="" if turn_commit is None else str(turn_commit), turn=turn, seat_digest=str(digest),
            refused=refused, requested={(role, sample) for role, samples in accepted.items() for sample in samples},
        )
        with self._lock:
            episode.jobs.append(job)
        # The ledger lock orders this against `settled`: no settlement is seen before its job is known.
        with self._controller.ledger_held():
            if accepted:
                job.actions, job.judgments = self._controller.issue_train_batch(
                    action=episode.action, commit=commit, batch=dict(accepted))
                job.tasks = {pair: self._task_of(role=pair[0], sample=pair[1], build=build) for pair in job.actions}
            with self._lock:
                job.settled = job.every() - set(self._controller.projection.open_actions)
                job.issuing = False
        if not job.every() - job.settled:
            self._finish(job=job, episode=episode)
        return job

    # Delivery into the node's generation folder.

    def _finish(self, *, job: _Job, episode: _Episode) -> None:
        self._deliver(episode)
        with self._room:
            job.done = True
            self._room.notify_all()

    def _deliver(self, episode: _Episode) -> None:
        """Rewrite the episode's result files from every settled row and carry them into its container."""
        files = list(_files_of(episode.depth).values())
        with episode.writing:
            with self._lock:
                jobs = list(episode.jobs)
                for one in jobs:
                    # Append-only: a row enters the files once, in the order its pair settled.
                    for row in self._rows(one):
                        one.recorded.setdefault((str(row["role"]), str(row["sample"])), row)
                rows = [row for one in jobs for row in one.recorded.values()]
                timing = {one.job: self._timing(job=one, episode=episode) for one in jobs}
            summary, error = None, None
            if episode.published:
                error = "the expansion sealed its generation folder before these rows were delivered"
            else:
                try:
                    evidence = self._lay_evaluations(rows=rows, episode=episode)
                    summary = write_results(workspace=episode.checkout, depth=episode.depth,
                                            rows=[TrainRow(**row) for row in rows],
                                            turns={one.job: one.turn for one in jobs}, jobs=timing)
                    if not episode.closed and not self._runner.carry_in(
                            socket=str(episode.path), checkout=episode.checkout, paths=(*files, *evidence)):
                        error = "the container did not confirm the carry"
                except Exception as broken:  # reported to the node as owed delivery
                    error = f"{type(broken).__name__}: {broken}"
            with self._room:
                for one in jobs:
                    one.summary = summary if summary is not None else one.summary
                    one.staging_error = None if episode.published else error
                self._room.notify_all()

    def _response(self, *, job: _Job, episode: _Episode) -> dict[str, object]:
        """One job's reply, with every job's unreported updates and the batches still pending."""
        with episode.writing:
            self._deliver(episode)
            with self._lock:
                updates = [one for one in episode.jobs if set(one.recorded) - episode.reported.get(one.job, set())
                           or one.staging_error is not None]
                body = self._reply(job=job, episode=episode)
                body["updates"] = [self._reply(job=one, episode=episode) for one in updates]
                body["pending_jobs"] = [{"job": one.job, "ordinal": one.ordinal, "tag": one.tag,
                                         "submission_commit": str(one.commit), "pending": one.pending()}
                                        for one in episode.jobs if one.pending()]
                for one in updates:
                    if one.staging_error is None:
                        episode.reported.setdefault(one.job, set()).update(one.recorded)
                return body

    def _reply(self, *, job: _Job, episode: _Episode) -> dict[str, object]:
        if job.summary is not None:
            body = dict(job.summary["jobs"][job.job])
        else:
            body = {"job": job.job, **self._timing(job=job, episode=episode), "turn": job.turn, "roles": {}}
        body["submission_commit"] = str(job.commit)
        pending = job.pending()
        body["partial"] = bool(pending)
        if pending:
            body["pending"] = pending
        else:
            body.pop("pending", None)
        if job.staging_error is not None:
            body["staging_refused"] = job.staging_error
        return body

    @staticmethod
    def _timing(*, job: _Job, episode: _Episode) -> dict[str, object]:
        pending = job.pending()
        return {"submission_commit": str(job.commit), "ordinal": job.ordinal, "tag": job.tag,
                "submitted_at": job.submitted_at, "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "partial": bool(pending), "pending": pending, "files": _files_of(episode.depth)}

    def _lay_evaluations(self, *, rows: Sequence[Mapping[str, object]], episode: _Episode) -> tuple[str, ...]:
        """Lay every settled seat's whole record, and its judges' files, under `evaluations/<action>/`."""
        actions = {name for row in rows if (name := str(row.get("action", "")))}
        if not actions:
            return ()
        projection = self._controller.projection
        produced = {event: one for event, one in dict(projection.outcomes).items() if str(one.action) in actions}
        laid: dict[str, dict[str, Artifact]] = {}
        for one in produced.values():
            result = one.result
            laid[str(one.action)] = dict(result.files) if isinstance(result, TaskFailure) else {
                **{f"outputs/{name}": art for name, art in result.files.items()}, **dict(result.metadata)}
        for judgment in tuple(projection.judgments):
            if (subject := getattr(judgment.subject, "outcome", None)) not in produced:
                continue
            declared = dict(getattr(judgment.judgment, "declared_outputs", None) or {})
            laid[str(produced[subject].action)].update({
                f"judgment/{judgment.action}/{f'outputs/{declared[name]}' if name in declared else name}": art
                for name, art in judgment.judgment.files.items()
            })
        paths: list[str] = []
        for action, held in laid.items():
            root = f"{experience.generation(depth=episode.depth)}/evaluations/{action}"
            for name, artifact in held.items():
                ids.require_safe_relpath(path=name)
                data = pointer.materialized(data=self._artifacts.resolve_artifact(artifact),
                                            fetch=self._artifacts.get_artifact)
                host_write(episode.checkout / root / name, data)
                paths.append(f"{root}/{name}")
        return tuple(paths)

    def _rows(self, job: _Job) -> list[dict[str, object]]:
        """One results row per refused or fully settled (role, sample) of a job, refusals first."""
        common = {"job": job.job, "ordinal": job.ordinal, "submitted_at": job.submitted_at,
                  "turn_commit": job.turn_commit, "submission_commit": str(job.commit), "tag": job.tag or "",
                  "seat_digest": job.seat_digest}
        rows: list[dict[str, object]] = [
            {**common, "role": one["role"], "task": one["task"], "sample": one["sample"], "kind": REFUSED,
             "refusal_code": one["code"], "provenance_error": "not executed: refused before dispatch"}
            for one in job.refused
        ]
        outcomes = {one.action: (event, one) for event, one in self._controller.projection.outcomes.items()}
        for (role, sample), action in sorted(job.actions.items(), key=lambda pair: str(pair[0])):
            if not job.answered((role, sample)):
                continue
            reserved = job.judgments.get((role, sample), ())
            row = {**common, "role": str(role), "task": str(job.tasks.get((role, sample), "")), "sample": str(sample),
                   "action": str(action), **({"judgment_action": str(reserved[0])} if reserved else {})}
            if (held := outcomes.get(action)) is None:
                row |= {"kind": FAILED, "provenance_error": "producer outcome unavailable"}
            else:
                row |= {"evaluated_commit": str(job.commit), **self._outcome(event=held[0], outcome=held[1], role=str(role))}
            rows.append(row)
        return rows

    def _outcome(self, *, event: object, outcome: object, role: str) -> dict[str, object]:
        """What one settled train action produced, as the results columns."""
        judging = {str(slot) for slot, roles in self._domain.jurisdiction.items() if role in {str(one) for one in roles}}
        agents = {str(slot) for slot in self._domain.agent_slots}
        # A program scorer decides over an agent judge where both judge the role.
        deciding = frozenset(judging - agents or judging) if judging else None
        result = outcome.result
        row: dict[str, object] = {"kind": FAILED, **failure_fields_of(result)}
        for _, axis, amount in result.spend:
            if str(axis) in {"tokens_in", "tokens_out"}:
                row[str(axis)] = row.get(str(axis), 0) + int(amount)
        metrics: dict[str, dict[str, object]] = {
            slot: {"kind": FAILED, "evaluator_commit": "", "provenance_error": "no settled judging outcome"}
            for slot in sorted(deciding or ())
        }
        # Projection order decides retries; action ids do not encode settlement order.
        for judgment in self._controller.projection.judgments:
            if getattr(judgment.subject, "outcome", None) != event:
                continue
            slot = str(judgment.slot)
            if deciding is not None and slot not in deciding:
                continue
            metrics[slot] = self._metric(judgment=judgment, result=result, slot=slot, agent=slot in agents)
        if len(metrics) == 1:
            metric = next(iter(metrics.values()))
            row |= {key: metric[key] for key in METRIC_FIELDS if key in metric}
        elif metrics:
            # Coverage is a sample property; scores and success opinions belong to their slots.
            row |= {"kind": MEASURED if {one["kind"] for one in metrics.values()} == {MEASURED} else FAILED,
                    "judgment_action": ""}
        if metrics:
            row["metrics"] = dict(sorted(metrics.items()))
        return row

    def _metric(self, *, judgment: object, result: object, slot: str, agent: bool) -> dict[str, object]:
        """One verdict's measured or failed fields, with its failure feedback and evaluator identity."""
        verdict = judgment.judgment
        row: dict[str, object] = {
            "kind": FAILED, "judgment_action": str(judgment.action), "provenance_error": "",
            "evaluator_commit": str(judgment.pins.get(ids.CriterionCoordinate(slot), "")) if agent else "",
            **failure_fields_of(result),
        }
        if not isinstance(result, TaskFailure):
            kind = getattr(verdict, "kind", None)
            judged = MEASURED if kind is None else str(getattr(kind, "value", kind))
            if judged != MEASURED:
                row["kind"] = judged
                if (code := getattr(verdict, "refusal_code", None)) is not None:
                    row["refusal_code"] = str(getattr(code, "value", code))
            elif (score := getattr(verdict, "score", None)) is not None:
                row |= {"kind": MEASURED, "score": float(score)}
                if (settled := getattr(verdict, "outcome", None)) is not None:
                    row["success"] = bool(float(settled) > 0)
        if "failure_reason" not in row:
            row |= failure_fields(files=verdict.files, resolve=self._artifacts.resolve_artifact)
        return row


def failure_fields_of(result: object) -> dict[str, str]:
    """A producing failure's feedback, without turning it into a refusal."""
    if isinstance(result, TaskFailure):
        return {"failure_reason": str(result.reason), "failure_detail": result.error}
    return {}


def _files_of(depth: int) -> dict[str, str]:
    """The two node paths one job's numbers land in, relative to the checkout."""
    folder = experience.generation(depth=depth)
    return {"results": f"{folder}/{experience.RESULTS}", "summary_json": f"{folder}/{experience.SUMMARY}"}
