# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""Start a run: read a form, write its manifest as the ledger's first event, and drive the controller."""

from __future__ import annotations

import contextlib
import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from gm import git_subprocess, ledger
from gm.controller import Controller, RunReport
from gm.domain import DomainManifest, RoleId
from gm.domain.manifest import read_form
from gm.genesis_tree import experience_data, seed_tree
from gm.git_store import NodeStore
from gm.ids import CriterionCoordinate, NodeId, RunId
from gm.policy import Policy

if TYPE_CHECKING:
    from gm.execution.gateway import Provider
    from gm.execution.runner import Envelope

DOMAINS = Path(__file__).resolve().parents[1] / "domains"

def forms() -> list[str]:
    """Every `domain/form` folder holding `form.json` and `genesis/`."""
    return sorted(
        f"{domain.name}/{form.name}"
        for domain in DOMAINS.iterdir() if (domain / "forms").is_dir()
        for form in (domain / "forms").iterdir()
        if (form / "form.json").is_file() and (form / "genesis").is_dir()
    )


def load_form(name: str) -> DomainManifest:
    """Read one form's declaration, data and genesis tree, with its image recipes resolved to digests."""
    from gm.execution.docker import resolved_images

    if name not in forms():
        raise SystemExit(f"unknown domain {name!r}; runnable forms are: {', '.join(forms())}")
    domain, _, form = name.partition("/")
    home = DOMAINS / domain / "forms" / form
    document, images = resolved_images(document=json.loads((home / "form.json").read_text(encoding="utf-8")))
    return read_form(name=name, home=home, document=document, images=images)


def initial_pins(*, domain: DomainManifest, source: NodeId) -> dict[CriterionCoordinate, str]:
    """The criterion vector a run starts on: the dataset build, the genesis seats, the form's declared pins."""
    pins = {domain.coordinates.dataset: str(domain.dataset.version)}
    seats = {str(slot) for slot in domain.agent_slots} | {
        str(coordinate) for coordinate in domain.transition.coordinates if RoleId(str(coordinate)) in domain.generator.task_pools
    }
    pins |= {CriterionCoordinate(seat): str(source) for seat in sorted(seats)}
    return pins | dict(domain.pins)


def start(
    domain: DomainManifest,
    *,
    log: ledger.SqliteEventLog,
    nodes: NodeStore,
    queue: object,
    seed: int,
    workers: int,
    budget: int,
    policy: Policy | None = None,
) -> Controller:
    """Seed the root node and the dataset tables, write `run-configured`, and return the controller."""
    for build in (domain.dataset, *domain.dataset.successors):
        files = {(str(key), ledger.INPUT_CHANNEL): tuple(row) for key, row in build.inputs.items()}
        log.put_dataset(version=build.version, sets=build.splits, files=files, images=build.images)
    run = RunId(str(domain.domain))
    with tempfile.TemporaryDirectory() as checkout:
        tree = seed_tree(domain=domain, checkout=Path(checkout), experience_tree=experience_data(domain=domain, artifacts=log))
        source = nodes.create_root(run=run, checkout=tree)
    if (commit := git_subprocess.clean_head(repo=DOMAINS.parent)) is None:
        raise ValueError("a run records the source commit, so it must start from a clean checkout")
    manifest = ledger.RunManifest(
        run=run,
        form=domain.name,
        seed=seed,
        workers=workers,
        budget=budget,
        source=source,
        source_commit=NodeId(commit),
        pins=initial_pins(domain=domain, source=source),
        coordinates=domain.coordinates,
        snapshot_sets=frozenset(str(name) for name in domain.generator.task_sets.values()),
        seat_interpretations={
            slot: spec.interpretation for slot, spec in domain.scorers.items() if spec.interpretation is not None
        },
        bindings=domain.bindings,
        tags=domain.tags,
        weights=domain.weights,
        images=domain.images,
        train=domain.train,
    )
    log.apply([ledger.RunConfigured(manifest)])
    return Controller(domain, domain.meta, log, nodes, queue, policy)


def run(
    domain: DomainManifest,
    *,
    store: Path,
    provider: Provider,
    envelope: Envelope,
    seed: int = 0,
    workers: int = 1,
    cpus: int | None = None,
    budget: int,
    edge: str,
    surface: str = "responses",
    policy: Policy | None = None,
) -> RunReport:
    """Run one search to its end on a fresh store; return the report (ledger and node store stay in `store`)."""
    from gm.execution.runner import open_runner

    store.mkdir(parents=True, exist_ok=True)
    with (
        NodeStore.init(store / "nodes") as nodes,
        ledger.SqliteEventLog(store / "ledger.sqlite") as log,
        contextlib.ExitStack() as stack,
    ):
        queue = stack.enter_context(open_runner(
            domain=domain, nodes=nodes, log=log, envelope=envelope, provider=provider, store_path=store,
            cpus=cpus or workers, edge=edge, surface=surface,
        ))
        controller = start(domain, log=log, nodes=nodes, queue=queue, seed=seed, workers=workers, budget=budget, policy=policy)
        queue.serve(controller, store=store)
        try:
            return controller.run()
        finally:
            stack.close()
