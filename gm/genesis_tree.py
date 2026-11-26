# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""Build the genesis tree a run starts from: the domain's genesis workspace, the root's empty
generation folder and the shared train data. The run config is the ledger's record, never the node's.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from gm import experience
from gm.domain import materialize

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gm.domain import DomainManifest


#: What the root's own generation folder holds, and why it holds nothing else.
ROOT_NOTE = "README.md"
ROOT_NOTE_TEXT = b"""\
# Generation 0

This folder is empty because no expansion produced this node.

Every other node in a run is the product of a turn: a meta agent edited its
parent's tree, and this folder carries what that turn wrote -- its dialogue, the
prompt it was given, the rows its episodes settled, and the summary of what it
scored. The root has none of that. It is the tree the run starts from, assembled
from the domain's genesis workspace before any agent was asked anything.

Descendants carry their ancestors' generation folders beside their own, so this
folder appears, still empty, in every node below the root.
"""


def experience_data(*, domain: DomainManifest, artifacts: object) -> dict[str, bytes]:
    """Return the run's shared train data as tree paths, or nothing where the form declares none.

    Every other node commits `experience/data/` because the host links it into the workspace
    before the episode runs; the root is published before any episode, so it carries it from
    here. Staged and read back, so the bytes and layout are the stager's.
    """
    from gm.execution.train import stage_train_data

    if domain.train is None:
        return {}
    with tempfile.TemporaryDirectory() as shared:
        root = Path(shared)
        stage_train_data(build=domain.dataset, artifacts=artifacts, sets=frozenset(domain.train.every), into=root)
        return {
            f"{experience.DATA_ROOT}/{path.relative_to(root).as_posix()}": path.read_bytes()
            for path in sorted(root.rglob("*")) if path.is_file()
        }


def seed_tree(*, domain: DomainManifest, checkout: Path, experience_tree: Mapping[str, bytes]) -> Path:
    """Stage the root checkout into ``checkout`` and return it."""
    workspace = dict(domain.workspace)
    # The root's own generation folder, empty but present with a note saying why, as every
    # other node carries the generation folders above it.
    workspace.setdefault(f"{experience.generation(depth=0)}/{ROOT_NOTE}", ROOT_NOTE_TEXT)
    workspace.update(experience_tree)
    return materialize.stage(declared=workspace, into=checkout)
