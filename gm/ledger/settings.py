# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""The decision settings a run's own manifest fixes, read by one function.

A run this schema starts decides by constants. A schema-8 ledger decides by the settings it recorded
(`resolved_config`, its tags' provenance coordinates and its policy pin) and by that schema's rules:
draws keyed by the policy pin and node commits, epoch-permuted sample walks, per-cell caps, and the
fixed-evaluator, withheld-pool and pool-mass arms.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field

from gm.ledger.events import LEGACY_SCHEMA, ActionKind, RunManifest


@dataclass(frozen=True, slots=True, kw_only=True)
class Settings:
    """What the policy, sampling, caps and transition rules read about one run."""

    alpha: float = 2 / 3
    #: Whether a cohort of floor(workers ** alpha) root expansions runs before the root is evaluated.
    bootstrap: bool = True
    #: How often a childless expansion question may be asked again before it waits on new evidence.
    reasks: int
    #: Open actions allowed per (kind, sample set) cell; a cell not named allows one.
    caps: Mapping[tuple[ActionKind, str | None], int] = field(default_factory=dict)
    #: Seeded permutation passes over a cell's samples after the walk in declared order.
    epochs: int = 0
    #: The policy pin a schema-8 run keyed its draws by, naming nodes by commit; None names them by publication ordinal.
    policy: str | None = None
    #: Never replace an evaluator seat, though its objective is still compared.
    fixed_evaluator: bool = False
    #: Pool coordinates mined but never committed.
    withheld_pools: frozenset[str] = frozenset()
    #: The trial mass one judgment on a minted pool surface carries in a promotion objective.
    pool_mass: float = 1.0
    #: Each role's model-provenance coordinate, pinned on what that role produced or judged.
    provenance: Mapping[str, str] = field(default_factory=dict)
    #: Expansions allowed to execute at once; None leaves it to CPU admission.
    expansion_concurrency: int | None = None


def settings(manifest: RunManifest) -> Settings:
    """Return the settings one run decides by, from its recorded manifest."""
    if manifest.version != LEGACY_SCHEMA:
        return Settings(reasks=min(max(5, 3 * manifest.workers), 64))
    config = json.loads(manifest.recorded["resolved_config"])
    caps = {}
    for spelled, cap in config["in_flight_by_cell"].items():
        kind, _, split = spelled.partition("/")
        caps[ActionKind(kind), None if split == "*" else split] = cap
    widened = max(config["reask_budget"], config["reask_per_worker"] * config["parallelism"])
    return Settings(
        alpha=config["alpha"],
        bootstrap=config["bootstrap"] != "root-only",
        reasks=min(widened, config["reask_ceiling"]),
        caps=caps,
        epochs=config.get("sampling_epochs") or 0,
        policy=manifest.pins["policy"],
        fixed_evaluator=config["fixed_evaluator"],
        withheld_pools=frozenset(config.get("withheld_pools") or ()),
        pool_mass=float(config.get("pool_surface_mass", 1.0)),
        provenance={role: one["provenance"] for role, one in manifest.recorded["tags"].items()},
        expansion_concurrency=config["expansion_concurrency"],
    )
