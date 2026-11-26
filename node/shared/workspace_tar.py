# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""Carry a tree or selected files into a container as root-owned archives."""

from __future__ import annotations

import io
import posixpath
import shlex
import subprocess
import tarfile
import tempfile
import uuid
from contextlib import suppress
from pathlib import Path
from collections.abc import Sequence


class TransferRefused(RuntimeError):
    """A tree could not be loaded into its container."""


def _root(entry: tarfile.TarInfo) -> tarfile.TarInfo:
    entry.uid = entry.gid = 0
    entry.uname = entry.gname = "root"
    entry.mode |= 0o755 if entry.isdir() else 0o644
    return entry


def send(*, workspace: Path, name: str, target: str, exclude: Sequence[str], docker: str = "docker") -> None:
    source = workspace.resolve()
    if not source.is_dir():
        raise TransferRefused(f"episode workspace is not a host directory: {source}")
    parent, leaf = posixpath.split(target.rstrip("/"))
    excluded = frozenset(exclude)
    def chosen(entry: tarfile.TarInfo) -> tarfile.TarInfo | None:
        head = entry.name[len(leaf):].lstrip("/").split("/", 1)[0]
        return None if head in excluded else _root(entry)
    try:
        process = subprocess.Popen([docker, "cp", "-", f"{name}:{parent or '/'}"], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as error:
        raise TransferRefused("docker CLI is not installed") from error
    assert process.stdin is not None
    try:
        with tarfile.open(fileobj=process.stdin, mode="w|") as archive:
            archive.add(str(source), arcname=leaf, filter=chosen)
    except BrokenPipeError:
        pass
    finally:
        with suppress(BrokenPipeError, OSError):
            process.stdin.close()
    detail = process.stderr.read().decode(errors="replace").replace("\n", " ")[:500] if process.stderr else ""
    if process.wait():
        raise TransferRefused(f"docker cp into {name}:{target} failed (exit {process.returncode}): {detail or 'no daemon output'}")


def send_files(*, workspace: Path, paths: Sequence[str], name: str, target: str, timeout: float) -> None:
    selected = []
    for relative in paths:
        if posixpath.isabs(relative) or ".." in relative.split("/"):
            raise TransferRefused(f"delivery path escapes its workspace: {relative!r}")
        source = (workspace / relative).resolve()
        if source.is_file():
            selected.append((relative, source))
    if not selected:
        return
    leaf = ".rqgm-staging-" + uuid.uuid4().hex
    staging = posixpath.join(target, leaf)
    script = ["set -eu", "umask 022", "landing=", "trap 'rm -f -- \"${landing:-$1/unused}\"; rm -rf -- \"$1\"' 0"]
    for index, (relative, _) in enumerate(selected):
        destination = posixpath.join(target, relative)
        parent = posixpath.dirname(destination)
        landing = posixpath.join(parent, f"{leaf}-{index}")
        script += [f"mkdir -p -- {shlex.quote(parent)}", f"landing={shlex.quote(landing)}", f'mv -- {shlex.quote(staging + "/" + str(index))} "$landing"', f'mv -- "$landing" {shlex.quote(destination)}', "landing="]
    body = ("\n".join(script) + "\n").encode()
    try:
        with tempfile.TemporaryFile() as stream:
            with tarfile.open(fileobj=stream, mode="w") as archive:
                directory = tarfile.TarInfo(leaf)
                directory.type, directory.mode = tarfile.DIRTYPE, 0o755
                archive.addfile(_root(directory))
                for index, (_, source) in enumerate(selected):
                    archive.add(str(source), arcname=f"{leaf}/{index}", filter=_root)
                entry = tarfile.TarInfo(f"{leaf}/deliver.sh")
                entry.size = len(body)
                archive.addfile(_root(entry), io.BytesIO(body))
            stream.seek(0)
            commands = [(["docker", "cp", "-", f"{name}:{target}"], stream), (["docker", "exec", name, "/bin/sh", f"{staging}/deliver.sh", staging], subprocess.DEVNULL)]
            for argv, incoming in commands:
                done = subprocess.run(argv, stdin=incoming, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=timeout, check=False)
                if done.returncode:
                    raise TransferRefused(f"batched delivery into {name}:{target} failed: {done.stderr.decode(errors='replace')[:500]}")
    except subprocess.TimeoutExpired as error:
        raise TransferRefused(f"batched delivery into {name}:{target} timed out after {timeout}s") from error
    except OSError as error:
        raise TransferRefused(f"batched delivery into {name}:{target} failed: {error}") from error
    except TransferRefused:
        with suppress(OSError, subprocess.SubprocessError):
            subprocess.run(["docker", "exec", name, "rm", "-rf", "--", staging], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout, check=False)
        raise
