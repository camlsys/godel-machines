# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""The search policy seam: the frozen snapshot a policy reads, the actions it returns, the shipped HGM policy,
and `pick`, the final selection by epsilon-best-belief.

A policy chooses which node to evaluate or expand; the kernel chooses which sample an evaluation reads, so every
record stays valid whichever policy issued it.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, NamedTuple, Protocol

from scipy.special import betaincinv

from gm.ids import NodeId, canonical_json
from gm.ledger import Cell
from gm.work_queue import EXPANSION_STARTED

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from gm.domain import SurfaceSide
    from gm.domain.identity import RoleId, TaskId
    from gm.ledger import CellSlot
    from gm.ledger.trial_index import Trial
    from gm.work_queue import ExpansionExecution

#: The quantile of the epsilon-best-belief score.
EPSILON = 0.05
#: The draw-key version a schema-8 run's policy pin keyed its draws under.
SAMPLER_VERSION = "hash-beta-v1"


@dataclass(frozen=True, slots=True)
class Evidence:
    """Beta pseudo-counts from fractional outcomes in [0, 1]: S = sum o, and trials = S + F."""

    score_sum: float = 0.0
    trials: float = 0.0

    @property
    def shortfall(self) -> float:
        """F, the trial mass not assigned to success."""
        return self.trials - self.score_sum

    def __add__(self, other: Evidence) -> Evidence:
        """Return two independent evidence summaries pooled."""
        return Evidence(self.score_sum + other.score_sum, self.trials + other.trials)


def paper_aggregation(trials: Sequence[Trial]) -> Evidence:
    """Compress a cell's trial stream, with counted failures mapped to zero."""
    return Evidence(
        math.fsum(0.0 if t.value is None else t.value * t.mass for t in trials), math.fsum(t.mass for t in trials),
    )


def best_belief(score_sum: float, shortfall: float, epsilon: float = EPSILON) -> float:
    """BB_eps = I^-1_eps(1 + S, 1 + F): the epsilon-quantile of the Beta posterior, in [0, 1]."""
    return float(betaincinv(1.0 + score_sum, 1.0 + shortfall, epsilon))


@dataclass(frozen=True, slots=True)
class Draws:
    """The seeded randomness of one decision, keyed by the run seed and the decision index.

    A run this schema starts names nodes by publication ordinal; a schema-8 run keyed its draws by its recorded
    policy pin and node commits.
    """

    seed: int
    index: int
    ordinals: Mapping[str, int]
    policy: str | None = None

    def uniform(self, purpose: str, arm: str = "") -> float:
        """Return a keyed draw strictly inside (0, 1)."""
        if self.policy is None:
            key = canonical_json([self.seed, self.index, purpose, self.ordinals.get(arm, arm)])
        else:
            key = canonical_json([SAMPLER_VERSION, self.seed, self.policy, self.index, purpose, arm])
        return (int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") + 0.5) / 2.0**64

    def thompson(self, purpose: str, arms: Mapping[NodeId, Evidence]) -> NodeId:
        """Return the arm with the highest Beta-posterior draw."""
        return self.argmax({arm: best_belief(ev.score_sum, ev.shortfall, self.uniform(purpose, arm)) for arm, ev in arms.items()})

    def argmax[A](self, scores: Mapping[A, float]) -> A:
        """Return the highest-scoring arm, ties to the earliest published (by name on a schema-8 run)."""
        return argmax(scores, ordinals=self.ordinals, by_name=self.policy is not None)


def argmax[A](scores: Mapping[A, float], *, ordinals: Mapping[str, int], by_name: bool = False) -> A:
    """Return the highest-scoring arm, ties to the earliest published node (or, `by_name`, the smallest name)."""
    def order(arm: A) -> tuple:
        return (ordinals[str(arm)],) if str(arm) in ordinals else (1 << 62, str(arm))

    return max(sorted(scores, key=str if by_name else order), key=scores.__getitem__)


def searchable(*, root: NodeId, lineage: Mapping[NodeId, NodeId], done: frozenset[NodeId]) -> dict[NodeId, NodeId | None]:
    """Return the parent map a policy may act on: the root, plus done nodes whose parent it reaches."""
    reachable: dict[NodeId, NodeId | None] = {root: None}
    pending = dict(lineage)
    while admitted := {child: parent for child, parent in pending.items() if child in done and parent in reachable}:
        reachable.update(admitted)
        for child in admitted:
            del pending[child]
    return reachable


@dataclass(frozen=True, slots=True, kw_only=True)
class Snapshot:
    """One frozen decision state, built from the fold; nothing a run would not know."""

    #: The archive: each searchable node's parent (None for a root).
    parents: Mapping[NodeId, NodeId | None]
    #: Each role's evaluation cells (tasks), in declared order.
    roles: Mapping[RoleId, tuple[TaskId, ...]]
    #: Which side of a seat objective each cell answers on.
    sides: Mapping[CellSlot, SurfaceSide]
    #: Settled and open sample positions per cell.
    completed: Mapping[Cell, int]
    reserved: Mapping[Cell, int]
    #: Each cell's counted evidence.
    evidence: Mapping[Cell, Evidence]
    #: Evaluations counted against the budget so far, and the budget.
    clock_completed: float
    budget: int
    action_index: int
    seed: int
    ordinals: Mapping[str, int]
    policy: str | None = None
    expansion_execution: ExpansionExecution | None = None
    #: Terminal status of every expansion, in admission order.
    expansion_completed: tuple[bool, ...] = ()
    #: Evaluation units the open actions hold.
    evaluation_commitments: int = 0
    #: Nodes owed a reading before widening continues; empty for none.
    evaluation_refresh: frozenset[NodeId] = frozenset()
    #: Cells that admit no further evaluation.
    closed: frozenset[Cell] = frozenset()
    expandable: bool = True
    #: Parents whose expansion question is closed until new evidence arrives.
    blocked_expansions: frozenset[NodeId] = frozenset()

    @property
    def draws(self) -> Draws:
        """This decision's seeded draws."""
        return Draws(self.seed, self.action_index, self.ordinals, self.policy)

    @property
    def roots(self) -> tuple[NodeId, ...]:
        """Every node without a parent, in declared order."""
        return tuple(node for node, parent in self.parents.items() if parent is None)

    @property
    def published(self) -> int:
        """The searchable expansion width."""
        return len(self.parents) - 1

    @property
    def executing_expansions(self) -> int:
        """Started, nonterminal expansion episodes."""
        return sum(state == EXPANSION_STARTED for state in self.expansion_execution.states.values())

    def cell_load(self, cell: Cell) -> int:
        """Completed plus reserved positions at one cell."""
        return self.completed.get(cell, 0) + self.reserved.get(cell, 0)

    def cells(self, node: NodeId) -> tuple[Cell, ...]:
        """Every cell a node declares, in declared order."""
        return tuple(Cell(node=node, role=role, task=task) for role, tasks in self.roles.items() for task in tasks)

    def open_cells(self, node: NodeId) -> tuple[Cell, ...]:
        """A node's cells that admit another evaluation, in declared order."""
        return tuple(cell for cell in self.cells(node) if cell not in self.closed)

    def unread(self, node: NodeId) -> bool:
        """Whether a node has no settled reading yet; an evaluation in flight leaves it unread."""
        return not any(self.completed.get(cell, 0) for cell in self.cells(node))

    def admits_expansion(self) -> bool:
        """Whether every node that can still be evaluated has been read (a lone root always admits)."""
        return len(self.parents) <= 1 or not any(self.open_cells(node) and self.unread(node) for node in self.parents)


class EvidenceMaps(NamedTuple):
    """Each node's own evidence, and the same pooled through its clade."""

    node: dict[NodeId, Evidence]
    clade: dict[NodeId, Evidence]


def evidence_maps(snapshot: Snapshot) -> EvidenceMaps:
    """Return per-node evidence and the descendant-pooled clade evidence (CMP)."""
    node: dict[NodeId, Evidence] = {}
    for cell, value in snapshot.evidence.items():
        node[cell.node] = node.get(cell.node, Evidence()) + value
    clade = dict.fromkeys(snapshot.parents, Evidence())
    for member in snapshot.parents:
        if (own := node.get(member)) is None:
            continue
        cursor: NodeId | None = member
        while cursor is not None:
            clade[cursor] += own
            cursor = snapshot.parents.get(cursor)
    return EvidenceMaps(node, clade)


def pick(snapshot: Snapshot, epsilon: float = EPSILON) -> NodeId | None:
    """Return the node a search returns: the highest BB_eps over its own evidence, None before any evaluation."""
    scores = {
        node: best_belief(ev.score_sum, ev.shortfall, epsilon)
        for node, ev in evidence_maps(snapshot).node.items() if ev.trials and node in snapshot.parents
    }
    return snapshot.draws.argmax(scores) if scores else None


@dataclass(frozen=True, slots=True)
class Expansion:
    """Run the meta agent on `parent`."""

    parent: NodeId


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Evaluate one node on one role and task cell; the kernel picks the sample."""

    node: NodeId
    role: RoleId
    task: TaskId


type Action = Expansion | Evaluation


class Policy(Protocol):
    """A search policy: one action from one frozen snapshot."""

    def decide(self, snapshot: Snapshot) -> Action:
        """Return the next action, or raise `NoAction` when there is nothing to do now."""
        ...


class NoAction(Exception):
    """No eligible cell or expansion; the kernel waits on completions."""


class EvidenceInReach(NoAction):
    """Completed and committed evaluations already cover the next expansion's need."""


class ExpansionWaiting(NoAction):
    """Widening is owed, so new evaluation waits for expansion execution to progress."""


@dataclass(frozen=True, slots=True)
class Hgm:
    """The shipped HGM policy: Thompson draws over clade evidence to expand, node evidence to evaluate.

    It widens the tree while completed evaluations ** alpha reach the published width, after a bootstrap cohort of
    floor(workers ** alpha) root expansions (none when `bootstrap` is off).
    """

    workers: int = 1
    alpha: float = 2 / 3
    bootstrap: bool = True

    def _cohort(self) -> int:
        return math.floor(self.workers**self.alpha) if self.bootstrap else 0

    def bootstrap_pending(self, snapshot: Snapshot) -> bool:
        """Whether the root stays closed until every member of the bootstrap cohort is terminal."""
        cohort = snapshot.expansion_completed[: self._cohort()]
        return len(cohort) < self._cohort() or not all(cohort)

    def bootstrap_remaining(self, snapshot: Snapshot) -> int:
        """How many bootstrap expansions are still to issue."""
        return max(0, self._cohort() - len(snapshot.expansion_completed))

    def _refresh(self, snapshot: Snapshot) -> frozenset[NodeId]:
        return frozenset(node for node in snapshot.evaluation_refresh if snapshot.open_cells(node))

    def evidence_in_reach(self, snapshot: Snapshot) -> bool:
        """Whether completed and committed evaluations cover the next publication's need."""
        if self._refresh(snapshot):
            return snapshot.evaluation_commitments >= 1
        return (snapshot.clock_completed + snapshot.evaluation_commitments) ** self.alpha >= snapshot.published + 1

    def admit(self, snapshot: Snapshot, additional: int) -> None:
        """Refuse `additional` new evaluation units once the evidence in reach covers the next expansion."""
        if not additional:
            return
        if additional > 1:
            snapshot = replace(snapshot, evaluation_commitments=snapshot.evaluation_commitments + additional - 1)
        if self.evidence_in_reach(snapshot):
            raise EvidenceInReach("completed and committed evaluations cover the next expansion need")

    def decide(self, snapshot: Snapshot) -> Action:
        """Return the next expansion or evaluation."""
        draws = snapshot.draws
        bar = max(snapshot.published + snapshot.executing_expansions, self._cohort())
        node_evidence, clade_evidence = evidence_maps(snapshot)
        scored = {node for node, own in node_evidence.items() if own.score_sum > 0}
        candidates = {
            n for n in snapshot.parents
            if (n in snapshot.roots or n in scored) and n not in snapshot.blocked_expansions
        }
        pending = self.bootstrap_pending(snapshot)
        wants = bool(candidates) and (
            self.bootstrap_remaining(snapshot) > 0 if pending else snapshot.clock_completed**self.alpha >= bar
        )
        if wants:
            execution = snapshot.expansion_execution
            if not execution.available or (
                not pending and any(state != EXPANSION_STARTED for state in execution.states.values())
            ):
                raise ExpansionWaiting("widening waits for expansion execution capacity or progress")
            if snapshot.expandable:
                arms = {node: clade_evidence[node] for node in snapshot.parents if node in candidates}
                return Expansion(draws.thompson("expand", arms))
        targets = {
            node: node_evidence.get(node, Evidence())
            for node in snapshot.parents
            if snapshot.open_cells(node) and (not pending or node not in snapshot.roots)
        }
        if refresh := self._refresh(snapshot):
            targets = {node: evidence for node, evidence in targets.items() if node in refresh}
        if not targets:
            raise NoAction("bootstrap expansions are incomplete" if pending else "every eligible cell is exhausted")
        self.admit(snapshot, 1)
        unclaimed = {
            node: evidence for node, evidence in targets.items()
            if not any(snapshot.cell_load(cell) for cell in snapshot.cells(node))
        }
        return _least_loaded(snapshot, draws.thompson("evaluate", unclaimed or targets))


def _least_loaded(snapshot: Snapshot, node: NodeId) -> Evaluation:
    """The node's least-loaded open cell: by role, then objective side, then cell."""
    mine = snapshot.open_cells(node)
    every = snapshot.cells(node)

    def load(cells: Sequence[Cell]) -> int:
        return sum(snapshot.cell_load(cell) for cell in cells)

    open_roles = {cell.role for cell in mine}
    role = min((r for r in snapshot.roles if r in open_roles), key=lambda r: load([c for c in every if c.role == r]))
    mine = [cell for cell in mine if cell.role == role]
    sides = dict.fromkeys(snapshot.sides.get(c.slot) for c in mine)
    side = min(sides, key=lambda s: load([c for c in every if c.role == role and snapshot.sides.get(c.slot) == s]))
    least = min((c for c in mine if snapshot.sides.get(c.slot) == side), key=snapshot.cell_load)
    return Evaluation(node, role, least.task)
