# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""The controller: the single writer that loops decide, reserve, dispatch and settle over one run's ledger."""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import hashlib
import threading
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, assert_never

from gm import ids, ledger, sampling, settlement, transitions
from gm import work_queue as wq
from gm.domain import (
    Artifact,
    DomainManifest,
    EpisodeBinding,
    EvaluatorSlot,
    InlineArtifact,
    RoleId,
    SampleKey,
    TaskFailure,
    TaskId,
    TaskSuccess,
)
from gm.ledger.projection import CoverIndex
from gm.ledger.question import question_digest
from gm.policy import (
    Action,
    Evaluation,
    EvidenceInReach,
    Expansion,
    ExpansionWaiting,
    Hgm,
    NoAction,
    Policy,
    Snapshot,
    pick,
    searchable,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterator, Mapping, Sequence

    from gm.git_store import NodeStore


#: One reservation request: kind, node, role, task, scope, sample, owed, slot. The trailing two
#: are `ActionReserved.owed` and `ActionReserved.slot`, both stated per request so the request and
#: the record carry one vocabulary and nothing about a cell is defaulted. `slot` is the seat that
#: took a judging reservation and `None` everywhere else, which the event's own validator enforces:
#: a non-judging producer states it absent structurally rather than leaving it unset.
_Reservation = tuple[
    ledger.ActionKind,
    ids.NodeId,
    RoleId,
    TaskId | None,
    str,
    SampleKey | None,
    bool,
    EvaluatorSlot | None,
]

#: Reservation role for cell-less kinds; never in balance.
_META_ROLE = RoleId("meta")
_SNAPSHOT = sampling.SEARCH

_Generation = tuple[ids.NodeId, RoleId, TaskId, SampleKey]

# How long the controller waits for one completion before rechecking its state.
SERVING_POLL_SECONDS = 0.5


class Stall(StrEnum):
    """Why the run loop stopped asking for work."""

    BUDGET = "budget"
    NO_ELIGIBLE_CELL = "no-eligible-cell"
    EXPANSION_DECLINED = "expansion-declined"
    ISSUANCE_REFUSED = "issuance-refused"
    NO_UNWALKED_SAMPLE = "no-unwalked-sample"
    EVIDENCE_IN_REACH = "evidence-in-reach"


class AdmissionError(ValueError):
    """Refuse a reservation group that cannot acquire its declared evaluation units."""


def _cell(*, one: ledger.ActionReserved) -> tuple:
    return one.node, one.role, one.task, str(one.scope), one.sample


class Allocations:
    """Rebuild outstanding evaluation obligations without changing ledger retention.

    A producer holds one unit until failure or its required judging cover finishes. Required
    judges inherit that unit; a fresh judging cover acquires its own. Skips release allocations
    without manufacturing spend. Off-clock work never acquires a clock allocation.
    """

    def __init__(self, *, scopes: frozenset[str], cover: sampling.JudgingCover) -> None:
        """Initialize the allocation fold at an empty ledger prefix."""
        self.scopes, self.cover = scopes, cover
        self._covers = CoverIndex.empty()
        self._open: dict[ids.ActionId, ledger.ActionReserved] = {}
        self._held: dict[ids.ActionId, str] = {}
        self._groups: dict[ids.ActionId, ids.ActionId | None] = {}
        self._subjects: dict[ids.ActionId, ids.EventId] = {}
        self.waiting: dict[ids.EventId, ledger.ActionReserved] = {}

    def counts(self) -> Counter[str]:
        """Return held evaluation counts by their original issuing scope."""
        return Counter(self._held.values())

    def committed(self, *, scopes: Collection[str]) -> int:
        """Count accepted evaluation obligations once across the requested scopes."""
        return sum(scope in scopes for scope in self._held.values())

    def additional(self, *, reservations: Sequence[ledger.ActionReserved]) -> Mapping[str, int]:
        """Price a whole atomic reservation group, including shared judging covers."""
        proposed: Allocations = copy.copy(self)
        proposed._covers = CoverIndex(
            cover=dict(self._covers.cover), outstanding=dict(self._covers.outstanding),
            issuing=self._covers.issuing,
        )
        proposed._open = dict(self._open)
        proposed._held = dict(self._held)
        proposed._groups = dict(self._groups)
        proposed._subjects = dict(self._subjects)
        proposed.waiting = dict(self.waiting)
        before = self.counts()
        for one in reservations:
            proposed._reserve(one=one)
        return proposed.counts() - before

    def _reserve(self, *, one: ledger.ActionReserved) -> None:
        self._covers.witness(one)
        if str(one.scope) not in self.scopes or one.kind is ledger.ActionKind.EXPANSION:
            return
        self._open[one.action] = one
        if one.kind is ledger.ActionKind.GENERATION:
            self._held[one.action] = str(one.scope)
            return
        group = self._covers.cover[one.action]
        if group in self._groups:
            return
        matches = [ref for ref, source in self.waiting.items()
                   if one.owed and _cell(one=source) == _cell(one=one)]
        if len(matches) > 1:
            raise ValueError("required judgment has ambiguous unsettled evaluation subjects")
        if matches:
            (ref,) = matches
            source = self.waiting.pop(ref)
            self._subjects[group] = ref
            self._groups[group] = source.action if source.action in self._held else None
        elif one.task is not None:
            self._groups[group] = group
            self._held[group] = str(one.scope)
        else:
            self._groups[group] = None

    def advance(self, *, stored: ledger.StoredEvent) -> None:
        """Apply the same event order on live append and controller reconstruction."""
        event = stored.event
        if isinstance(event, ledger.ActionReserved):
            self._reserve(one=event)
            return
        self._covers.witness(event)
        if not isinstance(event, ledger.TaskOutcome | ledger.JudgmentRecorded | ledger.SoftSkip):
            return
        source = self._open.pop(event.action, None)
        if source is None:
            self._covers.close(event.action)
            return
        if source.kind is ledger.ActionKind.GENERATION:
            if isinstance(event, ledger.TaskOutcome) and self.cover[event.role]:
                self.waiting[stored.event_id] = source
                if isinstance(event.result, TaskSuccess):
                    return
            self._held.pop(event.action)
            return
        group = self._covers.cover[event.action]
        if (isinstance(event, ledger.JudgmentRecorded) and group in self._subjects
                and event.subject != ledger.OutcomeRef(self._subjects[group])):
            raise ValueError("required judgment differs from its inherited evaluation subject")
        if self._covers.remaining(event.action) == 1:
            allocation = self._groups.pop(group)
            self._subjects.pop(group, None)
            if allocation is not None:
                self._held.pop(allocation)
        self._covers.close(event.action)


@dataclass(frozen=True, slots=True, kw_only=True)
class RunReport:
    """A finished run: its id, node count, ledger event ids, the final pick by BB_eps, and why the loop stopped."""

    run: ids.RunId
    nodes: int
    event_ids: tuple[ids.EventId, ...]
    pick: ids.NodeId | None = None
    stall: Stall | None = None


class Controller:
    """The single state writer over one composed run."""

    def __init__(
        self,
        domain: DomainManifest,
        meta: EpisodeBinding,
        log: ledger.SqliteEventLog,
        nodes: NodeStore,
        queue: wq.WorkQueue,
        policy: Policy | None = None,
    ) -> None:
        """Create the sole writer for one freshly anchored run; without a policy, the shipped HGM at the run's settings."""
        self._settlement_callbacks: list[Callable[..., None]] = []
        self._stop_callbacks: list[Callable[..., None]] = []
        # The pool's handler threads reserve train batches while the loop thread reserves its own:
        # reading `decisions`, minting ids and appending is one critical section. Re-entrant,
        # because a settlement holding it chains judgments.
        self._ledger_lock = threading.RLock()
        self.domain, self._meta = domain, meta
        self.log, self._nodes, self.queue = log, nodes, queue
        self._triggers = transitions.Triggers(domain.transition.coordinates)
        self._fold_ledger()
        self.settings = self.projection.settings
        self._policy = policy or Hgm(
            workers=self.projection.manifest.workers, alpha=self.settings.alpha, bootstrap=self.settings.bootstrap,
        )
        #: The shipped policy's widening gates the kernel also enforces at issue; a custom policy owns its own.
        self._hgm = self._policy if isinstance(self._policy, Hgm) else None
        self._lockstep = self.projection.manifest.workers == 1
        self.sampler = sampling.Sampler(domain=domain, projection=self.projection)

    def _fold_ledger(self) -> None:
        """Fold the anchored ledger into the live caches."""
        self.events = list(self.log.events())
        self.projection = ledger.Fold()
        #: Bumped by every append, so one decision snapshot serves one ledger head.
        self._head = 0
        self._evaluations = Allocations(scopes=_SNAPSHOT, cover=self.judging_cover())
        self._childless: dict[tuple[ids.NodeId, ids.Sha256Hex], tuple[int, int]] = {}
        self._settled = 0
        self._stall: Stall | None = None
        self._expansions: dict[ids.ActionId, tuple[ids.NodeId, ids.Sha256Hex]] = {}
        for stored in self.events:
            self.projection.apply(stored)
            self._evaluations.advance(stored=stored)
            self._triggers.advance(stored)
            self._remember_childless(stored=stored)
        self._decision: tuple[int, wq.ExpansionExecution | None, Snapshot] | None = None
        self._train_judgments: dict[ids.ActionId, tuple[ids.NodeId, tuple[ids.ActionId, ...], str]] = {}

    def run(self) -> RunReport:
        """Return the terminal report of the whole run loop."""
        try:
            return self._loop()
        finally:
            for end in self._stop_callbacks:
                end()

    def _loop(self) -> RunReport:
        while True:  # every decision sees the freshest applied state
            if not self._lockstep:
                self._drain_ready()  # apply ready work immediately
            if self._exhausted(scope=ledger.SEARCH):
                # Accepted episodes keep their normal turns and native training admissions.
                self._stall = Stall.BUDGET
                if self._settle_one():
                    continue
                return self._after_drain()
            if (fired := self._triggers.due(self.projection)) is not None:
                transitions.checkpoint(self, None, fired)
            elif not (self._dispatch() or self._settle_one()):
                break  # nothing dispatchable and nothing in flight
        self._drain_ready()  # everything already here, batched
        while self._settle_one():
            self._drain_ready()
        return self._after_drain()

    def _after_drain(self) -> RunReport:
        if self.projection.open_actions:
            return self.report()
        fired, scope = self._triggers.due(self.projection), None
        if fired is None and (owed := self._triggers.owed()):
            fired, scope = transitions.RECOMPARISON, owed
        if fired is not None:
            transitions.checkpoint(self, scope, fired)
        return self.report()

    def report(self) -> RunReport:
        """Return this controller's terminal facts, whatever brought it here."""
        return RunReport(
            run=self.projection.manifest.run,
            nodes=1 + len(self.projection.lineage),
            event_ids=tuple(stored.event_id for stored in self.events),
            pick=pick(self.sampler.snapshot(execution=None, commitments=0)),
            stall=self._stall,
        )

    def _searching(self) -> bool:
        return any(
            self.sampler.room(kind=ledger.ActionKind.GENERATION, split=samples) > 0
            for samples in sorted({str(name) for name in self.domain.generator.task_sets.values()})
        )

    def _dispatch(self) -> bool:
        try:
            return self._dispatch_next()
        except (EvidenceInReach, ExpansionWaiting) as refused:
            self._stall = (Stall.EVIDENCE_IN_REACH
                           if isinstance(refused, EvidenceInReach)
                           else Stall.ISSUANCE_REFUSED)
            return False

    def _dispatch_next(self) -> bool:
        if self._judge_waiting():
            return True
        # Evaluation holds cannot close an expansion that acquires no clock units.
        if self._exhausted(scope=ledger.SEARCH):
            self._stall = Stall.BUDGET
            return False
        try:
            action = self._decide()
        except NoAction as wait:
            if isinstance(wait, ExpansionWaiting):
                self._stall = Stall.ISSUANCE_REFUSED
                return False
            self._stall = Stall.EVIDENCE_IN_REACH if isinstance(wait, EvidenceInReach) else Stall.NO_ELIGIBLE_CELL
            # A refused expansion can still leave an eligible validation reading.
            issued = self._searching() and self._measure()
            if issued:
                self._stall = None
            return issued
        if not self._admits(action=action):
            self._stall = Stall.ISSUANCE_REFUSED
            issued = self._searching() and self._measure()
        else:
            match action:
                case Expansion(parent=parent):
                    self._stall = Stall.EXPANSION_DECLINED
                    issued = self._expand(parent)
                    if not issued:
                        issued = self._searching() and self._measure()
                case Evaluation():
                    self._stall = Stall.NO_UNWALKED_SAMPLE
                    issued = self._searching() and self._measure(action=action)
                case _:
                    assert_never(action)
        if issued:
            self._stall = None
        return issued

    def _search_answered(self, *, node: ids.NodeId, role: RoleId, task: TaskId | None, sample: SampleKey | None) -> bool:
        """Return whether some counted search outcome already answers this search cell and sample."""
        generator = self.domain.generator
        if (
            sample is None or task is None
            or task not in generator.task_pools.get(role, ())
            or generator.task_sets[task] not in self.projection.manifest.snapshot_sets
            or sample not in sampling.pinned_build(projection=self.projection, domain=self.domain).splits[
                generator.task_sets[task]]
            or node not in searchable(root=self.projection.manifest.source, lineage=self.projection.lineage, done=frozenset(self.projection.done))
        ):
            return False
        axis = self.projection.manifest.coordinates.sample
        return any(
            outcome.pins.get(axis) == sample
            and self.projection.scopes[outcome.action] in _SNAPSHOT
            and self.projection.counted_in(outcome)
            for _, outcome in self.projection.cell_outcomes.get(ledger.Cell(node=node, role=role, task=task), ())
        )

    def _bootstrap_issuing(self) -> bool:
        return self._hgm is not None and self._hgm.bootstrap_remaining(self._snapshot()) > 0

    def _admits(self, *, action: Action) -> bool:
        # Evaluations always issue. An expansion waits until every evaluable node has a settled reading, except the
        # bootstrap cohort, which holds the root closed; budget admission still judges it at dispatch.
        if not isinstance(action, Expansion) or self._bootstrap_issuing():
            return True
        return self._snapshot().admits_expansion()

    def _decide(
        self,
        *,
        expandable: bool = True,
        closed: frozenset[ledger.Cell] = frozenset(),
    ) -> Action:
        snapshot = self._snapshot()
        snapshot = dataclasses.replace(snapshot, closed=snapshot.closed | closed, expandable=expandable)
        return self._policy.decide(snapshot)

    def _measure(self, *, action: Action | None = None) -> bool:
        # A full cell can remain the least loaded while other cells have capacity. Exclude
        # only this turn's refusals, leaving evidence, sample order and later turns unchanged.
        closed: set[ledger.Cell] = set()
        while True:
            if action is None:
                try:
                    action = self._decide(expandable=False, closed=frozenset(closed))
                except NoAction:
                    return False
            if not isinstance(action, Evaluation):
                return False
            cell = ledger.Cell(node=action.node, role=action.role, task=action.task)
            if cell in closed or not self._admits(action=action):
                return False
            if self._evaluate(action.node, action.role, action.task):
                return True
            closed.add(cell)
            action = None

    def _evaluate(self, node: ids.NodeId, role: RoleId, task: TaskId) -> bool:
        try:
            return self._evaluate_cell(node=node, role=role, task=task)
        except AdmissionError:
            self._stall = Stall.BUDGET
            return False
        except (EvidenceInReach, ExpansionWaiting) as refused:
            self._stall = (Stall.EVIDENCE_IN_REACH
                           if isinstance(refused, EvidenceInReach)
                           else Stall.ISSUANCE_REFUSED)
            return False

    def _evaluate_cell(self, *, node: ids.NodeId, role: RoleId, task: TaskId) -> bool:
        if task not in self.domain.generator.task_pools.get(role, ()):
            return transitions.labeled_cell(host=self, node=node, role=role, task=task)
        if transitions.rescore(self, node, role, task):
            return True
        # The kernel, not the policy, chooses the sample.
        sample = self.sampler.next_sample(ledger.Cell(node=node, role=role, task=task))
        if sample is None:
            return False
        if self.sampler.room(kind=ledger.ActionKind.GENERATION, split=self.sampler.split(role=role, sample=sample)) <= 0:
            return False
        self._generate_all(requests=[(node, role, task, sample)])
        return True

    def judge(
        self,
        ref: ids.EventId,
        outcome: ledger.TaskOutcome,
        scope: str | None = None,
        *,
        required: bool = False,
    ) -> bool:
        """Return whether this chained any judgment reservation to a produced success."""
        produced = outcome.result
        scope = ledger.SEARCH if scope is None else scope
        required = required or ref in self._evaluations.waiting
        held = sampling.reserved_members(projection=self.projection, scopes={scope})
        cell = ledger.Cell(node=outcome.node, role=outcome.role, task=outcome.task)
        sample = SampleKey(outcome.pins[self.projection.manifest.coordinates.sample])
        if sample in held.get(cell, frozenset()):
            return False
        if not required and self._closed(scope=scope):
            return False
        owed = settlement.owed_judgments(
            projection=self.projection,
            domain=self.domain,
            ref=ref,
            outcome=outcome,
        )
        required_slots = self.judging_cover()[outcome.role]
        owed = tuple(one for one in owed if one.slot in required_slots)
        try:
            actions = self.reserve_all([
                (
                    ledger.ActionKind.JUDGMENT,
                    outcome.node,
                    outcome.role,
                    outcome.task,
                    scope,
                    one.sample,
                    required,  # the settlement path; `rescore` chooses and states False
                    one.slot,
                )
                for one in owed
            ])
        except (AdmissionError, EvidenceInReach, ExpansionWaiting):
            return False
        self._dispatch_judgments(
            owed=owed,
            actions=actions,
            produced=produced,
            ticket=None,
        )
        return bool(owed)

    def _judge_waiting(self) -> bool:
        """Reserve the judgments a settled producer still owes, when its settlement could not."""
        issued = False
        for ref, source in tuple(self._evaluations.waiting.items()):
            outcome = self.projection.outcomes[ref]
            issued = self.judge(ref, outcome, str(source.scope), required=True) or issued
        return issued

    def _dispatch_judgments(
        self,
        *,
        owed: Sequence[settlement.OwedJudgment],
        actions: Sequence[ids.ActionId],
        produced: TaskSuccess | TaskFailure,
        ticket: wq.Ticket | None,
    ) -> None:
        for judgment, action in zip(owed, actions, strict=True):
            job = wq.JudgmentJob(
                action=action,
                node=judgment.node,
                slot=judgment.slot,
                subject=judgment.subject,
                sample=judgment.sample,
                produced=produced,
                role=judgment.agent,
                pins=judgment.pins,
            )
            if ticket is None:
                self.queue.submit(job)
            else:
                self.queue.submit(job, ticket=dataclasses.replace(ticket, key=str(action)))

    def _judge_train(
        self,
        *,
        ref: ids.EventId,
        outcome: ledger.TaskOutcome,
    ) -> bool:
        reserved = self._train_judgments.pop(outcome.action, None)
        if reserved is None:
            return False
        commit, opened, owner = reserved
        seated = dict(zip(self._train_judges(role=outcome.role), opened, strict=True))
        owed = settlement.owed_judgments(
            projection=self.projection,
            domain=self.domain,
            ref=ref,
            outcome=outcome,
            seated_at=dict.fromkeys(seated, commit).__getitem__,
        )
        matched = [(one, seated[one.slot]) for one in owed if one.slot in seated
                   and seated[one.slot] in self.projection.open_actions]
        self._dispatch_judgments(
            owed=[one for one, _ in matched],
            actions=[action for _, action in matched],
            produced=outcome.result,
            ticket=wq.Ticket(
                key="",
                kind=wq.job_class(scope=wq.TRAIN_SCOPE, expansion=False),
                owner=owner,
                batch=str(commit),
            ),
        )
        self._release_train_seats(
            actions=[one for slot, one in seated.items() if one in self.projection.open_actions
                     and slot not in {two.slot for two, _ in matched}],
            reason="its subject already carries a current verdict by this seat",
        )
        return bool(matched)

    def _release_train_seats(self, *, actions: Sequence[ids.ActionId], reason: str) -> None:
        for action in actions:
            self.apply_events(events=[ledger.SoftSkip(action, 1, reason, ())])
            for observe in self._settlement_callbacks:
                observe(action_id=action)

    def judging_cover(self) -> sampling.JudgingCover:
        """Return the judging seats this run requires for each declared source role."""
        return sampling.judging_cover(self.domain)

    def reserve_all(self, requests: Sequence[_Reservation]) -> tuple[ids.ActionId, ...]:
        """Return the action ids of N reservations appended in one transaction."""
        if not requests:
            # An empty rescore probe can precede the actual issue in this dispatch.
            return ()
        with self._ledger_lock:
            base = self.projection.decisions + 1
            reservations = [
                ledger.ActionReserved(
                    ids.ActionId(f"a{base + offset}"),
                    kind,
                    node,
                    role,
                    task,
                    scope,
                    sample,
                    owed,
                    slot,
                )
                for offset, (kind, node, role, task, scope, sample, owed, slot) in enumerate(requests)
            ]
            added = self._evaluations.additional(reservations=reservations)
            for scope, count in added.items():
                if scope == ledger.SEARCH and count * ledger.CLOCK_RESERVE > self.headroom():
                    raise AdmissionError(f"the budget has no room for {count} evaluations")
            if self._hgm is not None:
                fresh = [
                    one for one in reservations
                    if str(one.scope) in _SNAPSHOT
                    and one.kind is not ledger.ActionKind.EXPANSION
                    # A later judgment settles a producer this run already accepted.
                    and not (one.kind is ledger.ActionKind.JUDGMENT and self._search_answered(
                        node=one.node, role=one.role, task=one.task, sample=one.sample,
                    ))
                ]
                if additional := sum(self._evaluations.additional(reservations=fresh).values()):
                    snapshot = self._snapshot()
                    nodes = {one.node for one in fresh}
                    self._hgm.admit(dataclasses.replace(snapshot, closed=snapshot.closed | frozenset(
                        cell for node in snapshot.parents if node not in nodes for cell in snapshot.cells(node)
                    )), additional)
            self.apply_events(events=reservations)
            return tuple(one.action for one in reservations)

    def headroom(self, *, held: bool = True) -> float:
        """Return the search budget left after settled trials and, unless `held` is false, open evaluations."""
        left = self.projection.manifest.budget - self.projection.total(ledger.CLOCK, {ledger.SEARCH})
        return left - ledger.CLOCK_RESERVE * self._evaluations.counts()[ledger.SEARCH] if held else left

    def _closed(self, *, scope: str) -> bool:
        return scope == ledger.SEARCH and self.headroom() < ledger.CLOCK_RESERVE

    def _exhausted(self, *, scope: str) -> bool:
        return scope == ledger.SEARCH and self.headroom(held=False) < ledger.CLOCK_RESERVE

    def _apply(self, envelope: wq.CompletionEnvelope) -> None:
        job, outcome = envelope.job, envelope.outcome
        scope = self.projection.scopes[job.action]
        arrival = ledger.Arrival(
            action=job.action,
            payload=ids.canonical_json({"outcome": type(outcome).__name__}),
        )
        event = settlement.settle(job, outcome, self.projection)
        settled, *_states = self.apply_events(
            events=[event, *self._node_states(event=event)],
            arrivals=(arrival,),
        )
        stored = settled
        if isinstance(stored.event, ledger.TaskOutcome):
            if not self._judge_train(ref=stored.event_id, outcome=stored.event):
                self.judge(stored.event_id, stored.event, scope, required=True)
        elif (held := self._train_judgments.pop(job.action, None)) is not None:
            self._release_train_seats(
                actions=held[1],
                reason=ledger.cascaded_skip(producer=job.action),
            )
        for observe in self._settlement_callbacks:
            observe(action_id=job.action)

    def _node_states(self, *, event: ledger.LedgerEvent) -> list[ledger.LedgerEvent]:
        if not isinstance(event, ledger.ExpansionRecorded | ledger.SoftSkip):
            return []
        # Identical candidates share a child commit, but each episode owns its final turn.
        state = self._nodes.node_state_for_action(str(event.action))
        if state is None:
            return []
        node = ids.NodeId(str(state["node"]))
        return [
            ledger.NodeStateRecorded(
                action=event.action,
                node=node,
                state=ledger.NodeState.FINAL_TURN,
                reason=str(state["reason"]),
                done=bool(state["done"]),
                grounds=str(state["grounds"]),
            ),
        ]

    def admit(self, *, action: str, node: str) -> None:
        """Record that an episode's first turn landed, so its node is on the ledger from now."""
        self.apply_events(
            events=[
                ledger.NodeStateRecorded(
                    action=ids.ActionId(action),
                    node=ids.NodeId(node),
                    state=ledger.NodeState.ADMITTED,
                ),
            ],
        )

    def on_settlement(self, observe: Callable[..., None]) -> None:
        """Register one reader to be told, by action id, that a settlement committed."""
        self._settlement_callbacks.append(observe)

    def on_stop(self, end: Callable[..., None]) -> None:
        """Register one collaborator to be stopped when the run loop ends, however it ends."""
        self._stop_callbacks.append(end)

    @contextlib.contextmanager
    def ledger_held(self) -> Iterator[None]:
        """Hold the ledger lock across a read of the events and the apply that depends on it."""
        with self._ledger_lock:
            yield

    def apply_events(
        self,
        *,
        events: Sequence[ledger.LedgerEvent],
        phase: ledger.CheckpointPhase | None = None,
        arrivals: Sequence[ledger.Arrival] = (),
        scope: frozenset[ids.CriterionCoordinate] | None = None,
        fired: str | None = None,
    ) -> tuple[ledger.StoredEvent, ...]:
        """Return the stored events, atomically appended, with the live folds advanced."""
        with self._ledger_lock:
            stored = self.log.apply(events, phase=phase, scope=scope, fired=fired, arrivals=arrivals)
            self.events.extend(stored)
            for one in stored:
                self.projection.apply(one)
                self._evaluations.advance(stored=one)
                self._triggers.advance(one)
                self._remember_childless(stored=one)
            self._head += 1
            return stored

    def _drain_ready(self) -> None:
        while (envelope := self.queue.next_completion(0.0)) is not None:
            self._apply(envelope)

    def _settle_one(self) -> bool:
        if not self.projection.open_actions:
            return False
        envelope = self.queue.next_completion(SERVING_POLL_SECONDS)
        if envelope is not None:
            self._apply(envelope)
        return True

    def _snapshot(self) -> Snapshot:
        execution = None
        if (reader := getattr(self.queue, "expansion_execution", None)) is not None:
            execution = reader(actions=frozenset(
                action for action, held in self.projection.open_actions.items()
                if held.kind is ledger.ActionKind.EXPANSION and str(held.scope) in _SNAPSHOT
            ))
        # One snapshot per ledger head and execution state: every call within a dispatch reuses it.
        held = self._decision
        if held is not None and held[0] == self._head and held[1] == execution:
            return held[2]
        head = self._head
        snapshot = self.sampler.snapshot(execution=execution, commitments=self._evaluations.committed(scopes=_SNAPSHOT))
        hgm = self._hgm
        bootstrap = hgm is not None and hgm.bootstrap_remaining(snapshot) > 0
        blocked = frozenset(
            parent for parent in {question[0] for question in self._childless}
            if not bootstrap and parent in snapshot.parents
            and self._closed_question(parent=parent, digest=self._export(parent=parent).expected_hash)
        )
        refresh = blocked
        # Once the widening allowance is spent, the settled-reading gate still needs one reading of each unread node.
        if (
            hgm is not None and not hgm.bootstrap_pending(snapshot) and not snapshot.admits_expansion()
            and hgm.evidence_in_reach(snapshot)
        ):
            refresh |= frozenset(node for node in snapshot.parents if snapshot.open_cells(node) and snapshot.unread(node))
        snapshot = dataclasses.replace(snapshot, blocked_expansions=blocked, evaluation_refresh=refresh)
        self._decision = (head, execution, snapshot)
        return snapshot

    def _expand(self, parent: ids.NodeId) -> bool:
        export = self._export(parent=parent)
        if self._filed(parent=parent, digest=export.expected_hash):
            self._stall = Stall.EXPANSION_DECLINED
            return False
        reservations: list[_Reservation] = [
            (
                ledger.ActionKind.EXPANSION,
                parent,
                _META_ROLE,
                None,
                ledger.SEARCH,
                None,
                False,
                None,
            ),
        ]
        (action,) = self.reserve_all(reservations)
        self.queue.submit(
            wq.ExpansionJob(
                action=action,
                parent=parent,
                export=export,
                binding=self._meta,
                allowed_tags=frozenset(self.domain.bindings),
                train=self.domain.train,
            ),
            ticket=wq.Ticket(key=str(action), kind=wq.EXPANSION_CLASS),
        )
        return True

    def _filed(self, *, parent: ids.NodeId, digest: ids.Sha256Hex) -> bool:
        # Bootstrap owes a fixed number of attempts; a failed first attempt cannot
        # require a root reading while that same cohort is keeping the root closed.
        return self._closed_question(parent=parent, digest=digest) and not self._bootstrap_issuing()

    def _closed_question(self, *, parent: ids.NodeId, digest: ids.Sha256Hex) -> bool:
        # Snapshot construction supplies its own bootstrap state to avoid recursive snapshots.
        filed = self._childless.get((parent, digest))
        if filed is None:
            return False
        answers, stamped = filed
        if not answers:
            return True
        if not self.settings.reasks:
            return True
        if answers <= self.settings.reasks:
            return False
        return self._settled - stamped < (answers - self.settings.reasks) * self.settings.reasks

    def _remember_childless(self, *, stored: ledger.StoredEvent) -> None:
        event = stored.event
        if isinstance(event, ledger.ActionReserved) and event.kind is ledger.ActionKind.EXPANSION:
            issued = self._export(parent=event.node).expected_hash
            self._expansions[event.action] = (event.node, issued)
        elif isinstance(event, ledger.ExpansionRecorded):
            self._expansions.pop(event.action, None)
            if event.child is None:
                key = (event.parent, event.experience_v4)
                answered = isinstance(event.result, TaskSuccess)
                self._childless[key] = (
                    self._childless.get(key, (0, 0))[0] + answered,
                    self._settled,
                )
            else:
                self._childless = {q: a for q, a in self._childless.items() if q[0] != event.parent}
        elif isinstance(event, ledger.SoftSkip) and event.action in self._expansions:
            key = self._expansions.pop(event.action)
            # Cancellation says nothing about the question, including any earlier answer.
            self._childless[key] = (self._childless.get(key, (0, 0))[0], self._settled)
        elif isinstance(event, ledger.TaskOutcome | ledger.JudgmentRecorded):
            self._settled += 1
            self._childless = {q: a for q, a in self._childless.items() if a[0]}

    def _export(self, *, parent: ids.NodeId) -> wq.ExportSpec:
        chain = self._nodes.lineage(parent)
        digest = question_digest(self.projection, chain, self.projection.active_pins, self.domain.axes)
        return wq.ExportSpec(pins=self.projection.active_pins, expected_hash=digest)

    def _staged_input(self, *, role: RoleId, sample: SampleKey) -> dict[str, Artifact]:
        corpus = sampling.pinned_build(projection=self.projection, domain=self.domain)
        channel = self.domain.generator.reads.get(role)
        if channel is None:
            return {}
        if channel == ledger.INPUT_CHANNEL:
            return {str(index): artifact for index, artifact in enumerate(corpus.inputs[sample])}
        return {str(channel): InlineArtifact(data=corpus.channels[sample][channel])}

    def _generate_all(self, *, requests: Sequence[_Generation]) -> None:
        reservations: list[_Reservation] = [
            (
                ledger.ActionKind.GENERATION,
                node,
                role,
                task,
                ledger.SEARCH,
                sample,
                False,
                None,
            )
            for node, role, task, sample in requests
        ]
        actions = self.reserve_all(reservations)
        for (node, role, task, sample), action in zip(requests, actions, strict=True):
            self.queue.submit(
                wq.GenerationJob(
                    action=action,
                    role=role,
                    inputs=self._staged_input(role=role, sample=sample),
                    seed=int.from_bytes(hashlib.sha256(str(action).encode()).digest()[:8]),
                    node=node,
                    task=task,
                    pins=settlement.generation_pins(
                        projection=self.projection,
                        domain=self.domain,
                        sample=sample,
                        role=role,
                    ),
                ),
                ticket=wq.Ticket(
                    key=str(action),
                    kind=wq.job_class(scope=ledger.SEARCH, expansion=False),
                ),
            )

    def _train_task(self, *, role: RoleId, sample: SampleKey) -> TaskId:
        train = self.domain.train
        if train is None:
            msg = f"this domain declares no train split, so role {str(role)!r} has none to draw"
            raise KeyError(msg)
        splits = sampling.pinned_build(projection=self.projection, domain=self.domain).splits
        for task in self.domain.generator.task_pools.get(role, ()):
            if (named := train.task_sets.get(task)) and sample in splits.get(named, ()):
                return task
        msg = f"sample {str(sample)!r} is in no train binding of role {str(role)!r}"
        raise KeyError(msg)

    def _train_judges(self, *, role: RoleId) -> tuple[EvaluatorSlot, ...]:
        return tuple(slot for slot, roles in self.domain.jurisdiction.items() if role in roles)

    def issue_train_batch(
        self,
        *,
        action: ids.ActionId,
        commit: ids.NodeId,
        batch: Mapping[RoleId, Sequence[SampleKey]],
    ) -> tuple[dict[tuple[RoleId, SampleKey], ids.ActionId], dict[tuple[RoleId, SampleKey], tuple[ids.ActionId, ...]]]:
        """Issue one submission's whole batch on the run's pool; return its producer and judgment actions."""
        pairs = [(role, sample) for role, samples in batch.items() for sample in samples]
        # Keyed by role and sample: two samples of one role from different sets are two tasks.
        tasks = {(role, sample): self._train_task(role=role, sample=sample) for role, sample in pairs}
        requests: list[_Reservation] = []
        for role, sample in pairs:
            requests.append(
                (
                    ledger.ActionKind.GENERATION,
                    commit,
                    role,
                    tasks[role, sample],
                    ledger.TRAIN,
                    sample,
                    False,
                    None,
                ),
            )
            requests.extend(
                (
                    ledger.ActionKind.JUDGMENT,
                    commit,
                    role,
                    tasks[role, sample],
                    ledger.TRAIN,
                    sample,
                    True,
                    slot,
                )
                for slot in self._train_judges(role=role)
            )
        staged = {(role, sample): self._staged_input(role=role, sample=sample) for role, sample in pairs}
        with self._ledger_lock:
            templates = [
                wq.GenerationJob(
                    action=ids.ActionId("unreserved"), role=role, inputs=staged[role, sample], seed=0,
                    node=commit, task=tasks[role, sample],
                    pins=settlement.generation_pins(
                        projection=self.projection, domain=self.domain, sample=sample, role=role,
                    ),
                )
                for role, sample in pairs
            ]
            opened = iter(self.reserve_all(requests))
        actions, judgments, jobs = {}, {}, []
        for template, (role, sample) in zip(templates, pairs, strict=True):
            produced = next(opened)
            actions[role, sample] = produced
            reserved = tuple(next(opened) for _ in self._train_judges(role=role))
            judgments[role, sample] = reserved
            self._train_judgments[produced] = (commit, reserved, str(action))
            ticket = wq.Ticket(key=str(produced), kind=wq.TRAIN_CLASS, owner=str(action), batch=str(commit))
            job = dataclasses.replace(
                template, action=produced, seed=int.from_bytes(hashlib.sha256(str(produced).encode()).digest()[:8]),
            )
            jobs.append((job, ticket))
        for job, ticket in jobs:
            self.queue.submit(job, ticket=ticket)
        return actions, judgments
