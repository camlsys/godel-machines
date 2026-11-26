# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""Copy a tree, hardlinking git objects that are already read-only (on one filesystem) and copying everything else.

A git object is content-addressed and replaced rather than edited, so a link to a read-only one
gives the destination the same isolation a copy would (`git clone --local` does the same).
"""

from __future__ import annotations

import errno
import os
import shutil
import stat
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def copy_tree(
    *,
    source: Path,
    into: Path,
    symlinks: bool = True,
    dirs_exist_ok: bool = False,
    ignore: Callable[[str, list[str]], set[str]] | None = None,
) -> None:
    """Copy one tree; read-only files under `.git/objects` (outside `objects/info`) are hardlinked when they can be."""
    objects = os.path.join(os.fspath(source), ".git", "objects") + os.sep
    info = objects + "info" + os.sep

    def place(one: str, other: str, *, follow_symlinks: bool = True) -> str:
        if one.startswith(objects) and not one.startswith(info) and os.sep in one[len(objects):]:
            if not os.stat(one, follow_symlinks=False).st_mode & stat.S_IWUSR:
                try:
                    os.link(one, other)
                except OSError as exc:  # across filesystems a link is impossible; a copy is as good
                    if exc.errno != errno.EXDEV:
                        raise
                else:
                    return other
        return shutil.copy2(one, other, follow_symlinks=follow_symlinks)

    shutil.copytree(source, into, symlinks=symlinks, dirs_exist_ok=dirs_exist_ok, ignore=ignore, copy_function=place)
