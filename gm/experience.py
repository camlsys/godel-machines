# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""The experience tree: every path the host writes into a node, spelled once.

`experience/data/` is shared by the whole chain and staged once per run.
`experience/generations/<depth>/` is one node's own folder, its ancestors' folders beside it
read-only.
Node-side code cannot import this module, so `task_agents/shared/TRAIN.md` and the shared meta
prompt state the same paths in words.
"""

from __future__ import annotations

from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from pathlib import Path

EXPERIENCE: Final = "experience"
DATA: Final = "data"
GENERATIONS: Final = "generations"
TRAIN: Final = "train"
LABELS: Final = "labels"
DATA_ROOT: Final = f"{EXPERIENCE}/{DATA}"
GENERATIONS_ROOT: Final = f"{EXPERIENCE}/{GENERATIONS}"
TRAIN_ROOT: Final = f"{DATA_ROOT}/{TRAIN}"

#: Files published in a node's generation folder.
RESULTS: Final = "results.csv"
SUMMARY: Final = "summary.json"
VERDICT: Final = "watchdog/verdict.md"
PROMPT: Final = "prompt.json"

NODE_DIALOGUE_MD: Final = "node-dialogue.md"


ARRIVING_NAME: Final = ".node-dialogue.arriving"

ENVIRONMENT: Final = ".rqgm/environment.json"
CONTRACT: Final = ".rqgm/contract.json"
LARGE_FILE_TOOL: Final = ".rqgm/large-file.py"
RUNTIME_CONTEXT: Final = (ENVIRONMENT, CONTRACT, LARGE_FILE_TOOL)
HOST_CONTEXT: Final = (
    *RUNTIME_CONTEXT, ".rqgm/job.json", ".rqgm/dispatch.json",
)

#: The host-private evidence an expansion's host driver writes beside its tree.
HOST_EVIDENCE: Final = (
    ".rqgm/host-prompt.json", ".rqgm/host-conversation.json", ".rqgm/host-model-runtime.json",
)

#: Git internals, scratch, caches and unfinished host writes never enter a snapshot.
UNPUBLISHED: Final = (
    *HOST_EVIDENCE,
    f"{GENERATIONS_ROOT}/*/{ARRIVING_NAME}",
    ".rqgm-staging.arriving", "*/.rqgm-staging.arriving",
    ".rqgm/.*.tmp",
    f"{GENERATIONS_ROOT}/*/{RESULTS}.tmp",
    f"{GENERATIONS_ROOT}/*/{SUMMARY}.tmp",
    ".git", ".git/*", "*/.git", "*/.git/*",
    ".rqgm-edit", ".rqgm-edit/*",
    "*.pyc", "*.pyo", "__pycache__/*", "*/__pycache__/*",
    ".rqgm/large-files", ".rqgm/large-files/*",
)


def required(*, rel: str) -> bool:
    """Identify declared context that ordinary Git ignores may not hide."""
    return episode_context(rel=rel) and not unpublished(rel=rel)


def host_context(*, rel: str) -> bool:
    """Identify current host evidence that a live checkout cannot replace."""
    if rel in HOST_CONTEXT:
        return True
    member = _generation_member(rel=rel)
    return member is not None and (
        member in {NODE_DIALOGUE_MD, RESULTS, SUMMARY}
        or member.split("/", 1)[0] in {"evaluations", "observations", "watchdog"}
    )


def episode_context(*, rel: str) -> bool:
    """Identify current episode context retained when selecting an earlier candidate's code."""
    return host_context(rel=rel) or _generation_member(rel=rel) == PROMPT


def _generation_member(*, rel: str) -> str | None:
    prefix, separator, tail = rel.partition(f"{GENERATIONS_ROOT}/")
    if prefix or not separator:
        return None
    depth, separator, member = tail.partition("/")
    return member if separator and depth.isdigit() else None


def generation(*, depth: int) -> str:
    """Return `experience/generations/<depth>`, one generation folder relative to the workspace."""
    return f"{GENERATIONS_ROOT}/{depth}"


def node_dialogue(*, depth: int) -> str:
    """Return `experience/generations/<depth>/node-dialogue.md`, one node's gathered transcript."""
    return f"{generation(depth=depth)}/{NODE_DIALOGUE_MD}"


def staged_depth(*, workspace: Path) -> int:
    """Return the deepest generation folder a workspace carries, which is the node's own.

    The host creates the node's folder before its episode starts and its ancestors' ride along
    beside it, so the highest number present is this node's depth.

    Args:
        workspace (Path): The episode's tree.

    Returns:
        int: That depth, 0 for a tree carrying no generation folder.
    """
    folders = (workspace / GENERATIONS_ROOT).glob("*")
    return max((int(one.name) for one in folders if one.name.isdigit() and one.is_dir()), default=0)


def unpublished(*, rel: str) -> bool:
    """Say whether a path is Git metadata, scratch, a cache or an unfinished host write.

    Args:
        rel (str): The path, posix-spelled, relative to the workspace.

    Returns:
        bool: True where `UNPUBLISHED` names it.
    """
    return any(fnmatchcase(rel, spec) for spec in UNPUBLISHED)
