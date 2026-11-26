# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""Run bounded Git queries in an isolated environment."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

GIT_TIMEOUT_SECONDS = 60.0
_GIT_ENV = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
}


def probe(
    *args: str, repo: Path | str | None = None, timeout: float = GIT_TIMEOUT_SECONDS
) -> subprocess.CompletedProcess[bytes]:
    """Return one bounded Git result, including nonzero exit status."""
    argv = ["git", *(("-C", str(repo)) if repo is not None else ()), *args]
    try:
        return subprocess.run(
            argv, capture_output=True, check=False, env=dict(_GIT_ENV), timeout=timeout
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise subprocess.SubprocessError(f"git {' '.join(args)} could not run: {exc}") from exc


def clean_head(*, repo: Path | str) -> str | None:
    """Return the checked-out commit only when the working tree is clean."""
    head = probe("rev-parse", "HEAD", repo=repo)
    status = probe("status", "--porcelain", repo=repo)
    if head.returncode != 0 or status.returncode != 0 or status.stdout.strip():
        return None
    return head.stdout.decode().strip()
