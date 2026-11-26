# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""Write the files a node carries into a checkout, so a caller can take a commit from it."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gm.ids import require_safe_relpath

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


NODE_COMMANDS = frozenset({"shared/rqgm-call"})

_COMMAND_MODE = 0o755


def stage(*, declared: Mapping[str, bytes], into: Path) -> Path:
    """Fill a checkout with exactly the declared files."""
    for destination, data in declared.items():
        require_safe_relpath(path=destination)
        target = into / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        if destination in NODE_COMMANDS:
            target.chmod(_COMMAND_MODE)
    return into
