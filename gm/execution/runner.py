# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""The job runner: CPU admission by class priority, one thread per started job, bounded retries.

Expansion precedes train work, which precedes validation; accepted work keeps its place in its
class. An expansion that waits on its own train batch lends its CPU and takes it back first. An
infrastructure failure is retried a bounded number of times; an exhausted job settles as a skip
(a judgment as a host-minted `0.0`).
"""

from __future__ import annotations

import json
import math
import queue
import threading
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

from gm import ids
from gm import work_queue as wq
from gm.domain.identity import RoleId
from gm.domain.results import FORFEIT_NOTE, FailureReason, TaskFailure, TaskSuccess, forfeit
from gm.ids import Ceiling, InfraWeather, RunId
from gm.ledger.events import AttemptDialogue, attempt_dialogue_name
from gm.spend import SpendRows, canonical, merged
from gm.work_queue import Completion, CompletionEnvelope, ExpansionJob, GenerationJob, Job, JudgmentJob, Skipped
from gm.execution.docker import Docker, bridge_address, run_network
from gm.execution.episodes import EpisodeRunError, Episodes
from gm.execution.expansion import Expansions
from gm.execution.gateway import Gateway, Provider
from gm.execution.train import Pool

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from gm.domain import DomainManifest
    from gm.git_store import NodeStore
    from gm.ledger import SqliteEventLog


@dataclass(frozen=True, slots=True, kw_only=True)
class Envelope:
    """The token allotment and nudge point every episode of a run is told about."""

    allotment: Ceiling
    nudge_at: float | None


@dataclass(slots=True, kw_only=True)
class _Held:
    """One submitted job's place in admission: its class, owner and CPU, and whether it runs or sleeps."""

    job: Job
    work_class: str
    order: int
    owner: str | None
    accepted: bool
    cpu: int
    running: bool = False
    asleep: bool = False

    @property
    def kind(self) -> str:
        return "expansion" if isinstance(self.job, ExpansionJob) else (
            "judgment" if isinstance(self.job, JudgmentJob) else "generation")

    def rank(self) -> tuple[int, bool, bool, int]:
        return wq.CLASS_ORDER[self.work_class], not self.accepted, self.kind != "judgment", self.order


class Runner:
    """Run every job kind through one completion queue; implements `gm.work_queue.WorkQueue`."""

    def __init__(self, *, domain: DomainManifest, log: SqliteEventLog, episodes: Episodes, cpus: int,
                 attempts: int = 2) -> None:
        self._domain, self._log, self._episodes = domain, log, episodes
        self._cpus, self._cpu, self._attempts = cpus, math.ceil(domain.cpus), attempts
        self._expansions: Expansions | None = None
        self._lock = threading.Lock()
        self._room = threading.Condition(self._lock)
        self._held: dict[str, _Held] = {}
        self._used = 0
        self._order = 0
        self._waking: set[str] = set()
        self._granted: dict[str, int] = {}
        self._finished: set[ids.ActionId] = set()
        self._threads: dict[str, threading.Thread] = {}
        self._done: queue.Queue[CompletionEnvelope | BaseException] = queue.Queue()
        self._pending = 0
        self._closed = False
        episodes.lend = self.lending

    def serve(self, controller: object, *, store: Path) -> None:
        """Bind the controller: its admit marks first turns, and the train pool reserves through it."""
        pool = Pool(store=store, controller=controller, domain=self._domain, nodes=self._episodes.nodes,
                    runner=self, artifacts=self._log)
        controller.on_settlement(pool.settled)
        controller.on_stop(pool.close)
        self._expansions = Expansions(episodes=self._episodes, pool=pool, run=RunId(str(self._domain.domain)),
                                      admit=controller.admit)

    # The WorkQueue protocol.

    def submit(self, job: Job, *, ticket: wq.Ticket | None = None) -> None:
        """Queue one job; it counts as pending until its completion is consumed."""
        expansion = isinstance(job, ExpansionJob)
        ticket = ticket or wq.Ticket(kind=wq.job_class(scope=wq.SEARCH_SCOPE, expansion=expansion))
        with self._lock:
            if self._closed:
                return
            self._pending += 1
            self._order += 1
            held = _Held(job=job, work_class=ticket.kind, order=self._order, owner=ticket.owner,
                         accepted=isinstance(job, JudgmentJob) or ticket.owner is not None, cpu=self._cpu)
            self._held[str(job.action)] = held
            if held.work_class == wq.TRAIN_CLASS and held.kind == "generation" and held.owner and held.accepted:
                self._spend_grant(owner=held.owner, count=1)
            started = self._select()
        self._launch(started)

    def next_completion(self, timeout: float | None) -> CompletionEnvelope | None:
        """Return the next arrival-ordered completion, or None when `timeout` passes first."""
        try:
            item = self._done.get(timeout=timeout) if timeout is None or timeout > 0 else self._done.get_nowait()
        except queue.Empty:
            return None
        if isinstance(item, BaseException):
            raise item
        key = str(item.job.action)
        with self._lock:
            self._pending -= 1
            self._finished.discard(item.job.action)
            if (held := self._held.pop(key, None)) is not None and held.running and not held.asleep:
                self._used -= held.cpu
            self._room.notify_all()
            started = self._select()
        self._launch(started)
        return item

    def expansion_execution(self, *, actions: frozenset[ids.ActionId]) -> wq.ExpansionExecution:
        """Each open expansion's state, and whether another expansion could start now."""
        with self._lock:
            states = {
                action: wq.EXPANSION_FINISHED if action in self._finished
                else wq.EXPANSION_STARTED if (held := self._held.get(str(action))) is not None and held.running
                else wq.EXPANSION_QUEUED
                for action in actions
            }
            return wq.ExpansionExecution(states=states, available=not self._closed)

    def close(self) -> None:
        """Stop admissions; join the workers when every completion has been consumed."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for key in [key for key, held in self._held.items() if not held.running]:
                del self._held[key]
            self._room.notify_all()
            running = [] if self._pending else list(self._threads.values())
        for thread in running:
            thread.join()

    # CPU lending, and grants for train batches.

    def sleep(self, key: str) -> None:
        """Lend a running job's CPU while it waits on work it submitted."""
        with self._lock:
            held = self._held.get(key)
            if held is None or not held.running or held.asleep:
                return
            held.asleep = True
            self._used -= held.cpu
            self._room.notify_all()
            started = self._select()
        self._launch(started)

    def wake(self, key: str) -> None:
        """Take a lent CPU back, ahead of newly queued work of lower priority."""
        with self._lock:
            held = self._held.get(key)
            if held is None or not held.asleep:
                return
            self._waking.add(key)
            try:
                while not self._closed and self._held.get(key) is held:
                    first = min(self._waking, key=lambda one: self._held[one].rank())
                    if first == key and self._used + held.cpu <= self._cpus:
                        held.asleep = False
                        self._used += held.cpu
                        return
                    self._room.wait()
            finally:
                self._waking.discard(key)
                self._room.notify_all()

    @contextmanager
    def lending(self, action: ids.ActionId) -> Iterator[None]:
        """Lend `action`'s CPU for the duration of a wait."""
        self.sleep(str(action))
        try:
            yield
        finally:
            self.wake(str(action))

    def grant(self, *, owner: str, requested: int) -> int:
        """Reserve the currently free part of one train request, in seats; the excess is refused."""
        if not requested:
            return 0
        with self._lock:
            free = 0
            queued = [held for held in self._held.values() if not held.running]
            if not self._closed and not any(held.work_class == wq.EXPANSION_CLASS for held in queued):
                waiting = sum(held.cpu for held in queued
                              if held.work_class == wq.TRAIN_CLASS and held.kind == "generation")
                unissued = sum(self._granted.values()) * self._cpu
                free = max(0, (self._free() - unissued - waiting) // self._cpu)
            granted = min(requested, free)
            if granted:
                self._granted[owner] = self._granted.get(owner, 0) + granted
            return granted

    def release(self, *, owner: str, count: int) -> None:
        """Return the unissued part of an owner's grant and offer the room to queued work."""
        with self._lock:
            self._spend_grant(owner=owner, count=count)
            self._room.notify_all()
            started = self._select()
        self._launch(started)

    def carry_out(self, *, socket: str, checkout: Path) -> bool:
        """Bring a pooled episode's live tree onto its checkout; False once its container is gone."""
        return self._episodes.docker.carry_out(socket=socket, checkout=checkout)

    def carry_in(self, *, socket: str, checkout: Path, paths: Sequence[str]) -> bool:
        """Carry host-written files into a pooled episode's live tree; False once its container is gone."""
        return self._episodes.docker.carry_in(socket=socket, checkout=checkout, paths=paths)

    # Admission, under the lock.

    def _spend_grant(self, *, owner: str, count: int) -> None:
        remaining = self._granted.get(owner, 0) - count
        if remaining > 0:
            self._granted[owner] = remaining
        else:
            self._granted.pop(owner, None)

    def _free(self, held: _Held | None = None) -> int:
        """Free CPU, less what sleeping jobs of at least this priority are waking to take back."""
        owed = sum(self._held[key].cpu for key in self._waking
                   if held is None or self._held[key].rank() <= held.rank())
        return self._cpus - self._used - owed

    def _select(self) -> list[_Held]:
        """Start queued jobs in priority order until the first that cannot start now."""
        started: list[_Held] = []
        if self._closed:
            return started
        for held in sorted((one for one in self._held.values() if not one.running), key=_Held.rank):
            refused = (held.work_class == wq.VALIDATION_CLASS and self._granted) or held.cpu > self._free(held)
            if refused:
                break
            held.running = True
            self._used += held.cpu
            started.append(held)
        return started

    def _launch(self, started: Sequence[_Held]) -> None:
        for held in started:
            thread = threading.Thread(target=self._run, args=(held.job,), daemon=True,
                                      name=f"rqgm-{held.work_class}-{str(held.job.action)[:12]}")
            with self._lock:
                self._threads[str(held.job.action)] = thread
            thread.start()

    def _run(self, job: Job) -> None:
        item: CompletionEnvelope | BaseException
        try:
            item = self._settled(job)
        except BaseException as exc:  # carried to the applying thread, then raised there
            item = exc
        with self._lock:
            self._threads.pop(str(job.action), None)
            if isinstance(job, ExpansionJob):
                self._finished.add(job.action)
        self._done.put(item)

    # Running one job, with bounded retries.

    def _settled(self, job: Job) -> CompletionEnvelope:
        """Produce one completion, retrying infrastructure weather up to the declared attempts."""
        domain, log = self._domain, self._log
        if isinstance(job, JudgmentJob) and isinstance(job.produced, TaskFailure):
            # The producer's failure owes its cover a host-minted 0.0; no episode runs.
            failure = replace(job.produced, spend=())
            if FORFEIT_NOTE not in failure.files:
                note = _forfeit_note(attempts=0, error=failure.error, reason=failure.reason, log=log)
                failure = replace(failure, files={**failure.files, FORFEIT_NOTE: note})
            return CompletionEnvelope(job=job, outcome=forfeit(
                failure=failure, scoring_rule=domain.scorers[job.slot].rule, interpretations=domain.interpretations))
        weather: InfraWeather | None = None
        spend: SpendRows = ()
        spoken: tuple[AttemptDialogue, ...] = ()
        for attempt in range(1, self._attempts + 1):
            try:
                match job:
                    case ExpansionJob():
                        attempted = replace(job, attempt_no=attempt)
                        done = self._expansions.expand(job=attempted)
                        result = _carried(result=done.result, tag=None, prior=spend, spoken=spoken)
                        return CompletionEnvelope(job=attempted, outcome=replace(done, result=result))
                    case JudgmentJob(slot=slot):
                        result = self._episodes.judge(job=job)
                        return CompletionEnvelope(job=job, outcome=_carried(
                            result=result, tag=self._tag(RoleId(str(slot))), prior=spend, spoken=spoken))
                    case GenerationJob(role=role):
                        result = self._episodes.generate(job=job)
                        return CompletionEnvelope(job=job, outcome=_carried(
                            result=result, tag=self._tag(role), prior=spend, spoken=spoken))
            except Exception as exc:  # a container may not end a search; an expansion's failure ends the run
                if isinstance(job, ExpansionJob) or isinstance(exc, EpisodeRunError):
                    raise
                if isinstance(exc, InfraWeather):
                    weather = exc
                else:
                    weather = InfraWeather(f"{type(exc).__name__}: {exc}")
                    weather.weathered_spend = getattr(exc, "spend", None)
            if weather.weathered_spend is not None:
                spend = merged(spend, weather.weathered_spend)
            if weather.weathered_dialogue is not None:
                spoken = (*spoken, AttemptDialogue(attempt_no=attempt,
                                                   spoken=log.route_artifact(weather.weathered_dialogue)))
            log.record_retry(action=job.action, attempts=attempt, error=str(weather), spend=spend)
        error = str(weather)
        if isinstance(job, JudgmentJob):
            files = {FORFEIT_NOTE: _forfeit_note(attempts=self._attempts, error=error,
                                                 reason=FailureReason.RETRY_EXHAUSTED, log=log)}
            files |= {attempt_dialogue_name(attempt_no=one.attempt_no): one.spoken for one in spoken}
            outcome: Completion = forfeit(
                failure=TaskFailure(files=files, error=error, reason=FailureReason.RETRY_EXHAUSTED, spend=spend),
                scoring_rule=domain.scorers[job.slot].rule, interpretations=domain.interpretations)
        elif weather.by_candidate:
            outcome = _carried(result=TaskFailure(files={}, error=error, reason=FailureReason.AGENT_ERROR, spend=spend),
                               tag=None, prior=(), spoken=spoken)
        else:
            outcome = Skipped(attempts=self._attempts, error=error, spend=spend, dialogues=spoken)
        return CompletionEnvelope(job=job, outcome=cast("Completion", outcome))

    def _tag(self, role: RoleId) -> ids.BindingTag | None:
        return self._domain.tags.get(role)


def _carried[T](*, result: T, tag: ids.BindingTag | None, prior: SpendRows, spoken: tuple[AttemptDialogue, ...]) -> T:
    """Attribute untagged spend to a position, then add what earlier attempts spent and said."""
    spend = result.spend
    if tag is not None and spend:
        spend = canonical((tag if current is None else current, axis, amount) for current, axis, amount in spend)
    if prior:
        spend = merged(spend, prior)
    if spend is not result.spend:
        result = replace(result, spend=spend)
    if not spoken:
        return result
    filed = {attempt_dialogue_name(attempt_no=one.attempt_no): one.spoken for one in spoken}
    if isinstance(result, TaskSuccess):
        return replace(result, metadata={**filed, **dict(result.metadata)})
    return replace(result, files={**filed, **dict(result.files)})


def _forfeit_note(*, attempts: int, error: str, reason: FailureReason, log: SqliteEventLog) -> object:
    return log.route_artifact(json.dumps({"attempts": attempts, "error": error, "reason": reason.value},
                                         sort_keys=True).encode())


@contextmanager
def open_runner(*, domain: DomainManifest, nodes: NodeStore, log: SqliteEventLog, envelope: Envelope,
                provider: Provider, store_path: Path, cpus: int, edge: str, surface: str) -> Iterator[Runner]:
    """Open the gateway, the run network and its relay, and one runner over them; close all on exit.

    Episodes that bind a model join the internal run network, whose only route out is the relay
    to the gateway; every other episode runs with `--network none`.
    """
    (store_path / "episodes").mkdir(parents=True, exist_ok=True)
    with (
        Gateway(provider, bind=(bridge_address(), 0)) as gateway,
        run_network(gateway_host=bridge_address(), gateway_port=gateway.port, edge=edge) as (network, url),
    ):
        episodes = Episodes(
            domain=domain, nodes=nodes, artifacts=log, docker=Docker(network=network, gateway_url=url, surface=surface),
            gateway=gateway, envelope=envelope.allotment, nudge_at=envelope.nudge_at,
            episode_dir=store_path / "episodes",
        )
        runner = Runner(domain=domain, log=log, episodes=episodes, cpus=cpus)
        try:
            yield runner
        finally:
            runner.close()
