# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""The declared evaluation standard: objectives, curricula, adversarial pools, and the sets they mint."""

from __future__ import annotations

import hashlib
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from gm.domain.dataset import SampleSetName
from gm.domain.identity import EvaluatorSlot, SampleKey, TaskId
from gm.ids import CriterionCoordinate, EventId, NodeId, canonical_json


@dataclass(frozen=True, slots=True, kw_only=True)
class UtilitySurface:
    """Judgments from one slot over one sample set."""

    judge: EvaluatorSlot
    samples: SampleSetName


class SurfaceSide(StrEnum):
    """Which answer a utility surface rewards."""

    ACCEPT = "accept"
    REJECT = "reject"


@dataclass(frozen=True, slots=True, kw_only=True)
class CompositeObjective:
    """Promote the seat whose evidence over these sided surfaces is best."""

    surfaces: Mapping[UtilitySurface, SurfaceSide]
    min_trials: int


_MINTED_FIELDS = 3


def minted_origin(*, member: SampleKey) -> SampleKey | None:
    """Return the built sample a minted member was drawn from, or None if it names none."""
    parts = str(member).split("/", _MINTED_FIELDS)
    return SampleKey(parts[_MINTED_FIELDS]) if len(parts) > _MINTED_FIELDS else None


@dataclass(frozen=True, slots=True, kw_only=True)
class MintedSet:
    """A sample set a committed transition created: its members, and the outcome each member stages."""

    name: SampleSetName
    members: tuple[SampleKey, ...]
    sources: Mapping[SampleKey, EventId] = field(default_factory=dict)
    judge: EvaluatorSlot | None = None
    interpretation: str | None = None
    drawn: tuple[str, ...] = ()

    def require_declared(
        self, *, splits: Mapping[SampleSetName, tuple[SampleKey, ...]], live: Mapping[str, MintedSet],
    ) -> None:
        """Refuse a set whose members collide with built sets or other live sets, or re-mint its own."""
        if self.judge is not None:
            TaskId(str(self.name))
            if unstaged := set(self.members) - self.sources.keys():
                msg = f"judge-bearing minted set {self.name!r} leaves members unstaged: {sorted(unstaged)}"
                raise ValueError(msg)
        taken = {key for keys in splits.values() for key in keys}
        taken.update(member for name, one in live.items() if name != self.name for member in one.members)
        if hit := taken & set(self.members):
            msg = f"minted set {self.name!r} members collide with built splits or other live sets: {sorted(hit)}"
            raise ValueError(msg)
        standing = live.get(self.name)
        if standing is not None and (kept := set(standing.members) & set(self.members)):
            msg = (
                f"minted set {self.name!r} is closed once minted and cannot re-mint over"
                f" members its live set already committed: {sorted(kept)}"
            )
            raise ValueError(msg)


@dataclass(frozen=True, slots=True, kw_only=True)
class JudgedWork:
    """Store one settled judgment fact for transition rules."""

    slot: EvaluatorSlot
    pins: Mapping[CriterionCoordinate, str]
    score: float
    outcome: float
    node: NodeId
    sample: SampleKey
    source: EventId | None = None
    labels: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class TransitionFacts:
    """Collect settled projection facts at a drained checkpoint."""

    clock: float
    epoch: int
    pins: Mapping[CriterionCoordinate, str] = field(default_factory=dict)
    judged: tuple[JudgedWork, ...] = ()
    labeled: Mapping[EvaluatorSlot, tuple[JudgedWork, ...]] = field(default_factory=dict)
    minted: Mapping[SampleSetName, MintedSet] = field(default_factory=dict)
    retained: Mapping[SampleSetName, tuple[MintedSet, ...]] = field(default_factory=dict)
    #: Each node's publication ordinal, which names it in minted member keys; None names nodes by commit (schema 8).
    ordinals: Mapping[str, int] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Curriculum:
    """Advance one coordinate to a declared successor at a declared clock."""

    at: int
    source: str
    target: str

    def crossing(self, clock: float, incumbent: str) -> str | None:
        """Return the successor when the schedule is due, and otherwise nothing."""
        return self.target if incumbent == self.source and clock >= self.at else None


@dataclass(frozen=True, slots=True, kw_only=True)
class PoolCriterion:
    """State which settled judgment puts its sample into a pool."""

    slot: EvaluatorSlot
    low: float
    high: float

    def admits(self, work: JudgedWork) -> bool:
        """Return whether one settled judgment puts its sample into the pool."""
        return work.slot == self.slot and self.low <= work.score <= self.high


def _membership_digest(*, members: Collection[SampleKey]) -> str:
    return hashlib.sha256("\n".join(sorted(members)).encode()).hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class PoolMint:
    """Mint a sample set from the judgments one declared criterion admits."""

    name: SampleSetName
    criterion: PoolCriterion
    signer: CriterionCoordinate
    judge: EvaluatorSlot | None = None
    interpretation: str | None = None

    def crossing(
        self, facts: TransitionFacts, incumbent: str, *, displaced: str | None
    ) -> tuple[str | None, MintedSet | None]:
        """Restore or first-mint the pool of a committed displaced reviewer."""
        if displaced is None:
            return None, None
        ordinals = facts.ordinals
        node = str if ordinals is None else (lambda one: f"n{ordinals[one]}")
        label = node(str(displaced))
        family = tuple(facts.retained.get(self.name, ()))
        for one in family:
            if label in self._drawn_from(one=one):
                digest = self._digest(one=one)
                return (None, None) if digest == incumbent else (digest, one)
        prior = facts.minted.get(self.name)
        already = set(prior.drawn) if prior else set()
        already.update(signer for one in family for signer in one.drawn)
        mined: dict[tuple, tuple[SampleKey, EventId]] = {}
        for work in facts.judged:
            if work.pins.get(self.signer) != displaced:
                continue
            if work.source is None or not self.criterion.admits(work):
                continue
            order = (str(work.node) if ordinals is None else ordinals[str(work.node)], str(work.sample))
            mined.setdefault(order, (SampleKey(f"{self.name}/{label}/{node(str(work.node))}/{work.sample}"), work.source))
        members = [mined[key] for key in sorted(mined)]
        one = MintedSet(
            name=self.name,
            members=tuple(member for member, _ in members),
            sources=dict(members),
            judge=self.judge,
            interpretation=self.interpretation,
            drawn=tuple(sorted(already | {label})),
        )
        digest = self._digest(one=one)
        return (None, None) if digest == incumbent else (digest, one)

    @staticmethod
    def _digest(*, one: MintedSet) -> str:
        if one.members:
            return _membership_digest(members=one.members)
        identity = {"name": str(one.name), "drawn": sorted(one.drawn)}
        return hashlib.sha256(canonical_json(identity).encode()).hexdigest()

    def _drawn_from(self, *, one: MintedSet) -> frozenset[str]:
        if not one.members:
            return frozenset(one.drawn)
        prefix = f"{self.name}/"
        return frozenset(str(member).removeprefix(prefix).partition("/")[0] for member in one.members)


type TransitionPolicy = CompositeObjective | Curriculum | PoolMint


@dataclass(frozen=True, slots=True, kw_only=True)
class TransitionSpec:
    """Each transitionable coordinate's policy, and the fact scopes whose evidence it reads."""

    coordinates: Mapping[CriterionCoordinate, TransitionPolicy]
    scopes: Mapping[CriterionCoordinate, frozenset[str]] = field(default_factory=dict)
