# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""Everything that talks to Docker: pinned images, the run network and its gateway relay, and one
contained container per episode with its tree carried in and out as tar archives."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import posixpath
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager, suppress
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

from gm import experience
from gm.ids import UNBOUNDED, InfraWeather
from gm.node_runtime import workspace_tar


if TYPE_CHECKING:
    from collections.abc import Generator, Iterator, Mapping, Sequence

    from pydantic import JsonValue

    from gm.execution.episodes import ExecSpec, Transcript

REPO = Path(__file__).resolve().parents[2]
POOL_SOCKET_TARGET = "/run/rqgm/pool.sock"
CHECKOUT_VAR = "RQGM_AGENT_CHECKOUT"
CALL_SECONDS_VAR = "RQGM_AGENT_CALL_SECONDS"
#: Keys the host sets per episode; a declared environment never carries them in.
RESERVED_ENV = frozenset({"OPENAI_BASE_URL", "OPENAI_API_KEY", "RQGM_AGENT_SURFACE", "RQGM_AGENT_TAG"})
TRANSCRIPT_CAP = 10 << 20
_IMAGE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_EPISODE_DEFAULTS = {"PYTHONDONTWRITEBYTECODE": "1"} | {
    f"GIT_{role}_{part}": value
    for role in ("AUTHOR", "COMMITTER")
    for part, value in (("NAME", "rqgm episode"), ("EMAIL", "episode@rqgm.invalid"))
}
_LAYER_SECONDS = 1800.0
_EDGE_PORT = 8000
#: The run network's one route out: a TCP relay from the edge port to the host gateway.
_EDGE_PROGRAM = """
import asyncio, os
async def pump(reader, writer):
    try:
        while data := await reader.read(65536):
            writer.write(data); await writer.drain()
    except OSError:
        pass
    finally:
        writer.close()
async def relay(reader, writer):
    try:
        upstream = await asyncio.wait_for(asyncio.open_connection(
            "host.docker.internal", int(os.environ["RQGM_GATEWAY_PORT"])), timeout=60)
    except (OSError, TimeoutError):
        writer.close(); return
    await asyncio.gather(pump(reader, upstream[1]), pump(upstream[0], writer))
async def main():
    server = await asyncio.start_server(relay, "0.0.0.0", int(os.environ["RQGM_EDGE_PORT"]))
    async with server:
        await server.serve_forever()
asyncio.run(main())
"""
#: Run inside the episode image to describe its interpreter and tools into `.rqgm/environment.json`.
_ENVIRONMENT_PROGRAM = """
import json, os, platform, shutil, sys
from importlib import metadata
print(json.dumps({
    "interpreter": {"executable": sys.executable, "implementation": platform.python_implementation(),
                    "version": platform.python_version()},
    "packages": {d.metadata["Name"]: d.version for d in metadata.distributions() if d.metadata["Name"]},
    "programs": {name: shutil.which(name) for name in (
        "bash", "curl", "file", "git", "jq", "node", "npm", "rg", "sqlite3", "time", "xz")},
    "shell": {"/bin/sh": os.path.realpath("/bin/sh")},
}, indent=1, sort_keys=True))
"""
_ENVIRONMENT_FINDER = 'for py in python3 python; do command -v "$py" >/dev/null 2>&1 && exec "$py" -c "$0"; done'
#: Each named image recipe's build script, which prints the tag it applies when asked with `--reference`.
RECIPES: Mapping[str, tuple[str, ...]] = {
    "meta-arm/python": ("images/build_meta_arm_images.sh", "python"),
    "meta-arm/mini-swe": ("images/build_meta_arm_images.sh", "mini-swe"),
    "meta-arm/runtime": ("images/build_meta_arm_images.sh", "runtime"),
}


# Docker CLI.

def _run(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["docker", *args], capture_output=True, timeout=60.0, check=False)


def docker(*args: str) -> None:
    """Run one Docker command, raising weather when it fails."""
    try:
        done = _run(*args)
    except subprocess.TimeoutExpired as exc:
        raise InfraWeather(f"docker {args[0]} timed out") from exc
    if done.returncode != 0:
        said = done.stderr.decode(errors="replace").strip().replace("\n", " ")[:500]
        raise InfraWeather(f"docker {args[0]} failed (exit {done.returncode}): {said or 'no daemon output'}")


def _capture(*args: str) -> str:
    """Return one Docker command's stdout, or empty when it failed."""
    try:
        done = _run(*args)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return done.stdout.decode(errors="replace") if done.returncode == 0 else ""


# Images.

def resolved_images(*, document: Mapping[str, JsonValue]) -> tuple[dict[str, JsonValue], dict[str, str]]:
    """Replace every `image_recipe` in a run document by the digest this daemon holds for it.

    Returns the rewritten document and, per recipe site, `<recipe> via <tag> -> <digest>`.
    """
    digests: dict[str, tuple[str, str]] = {}
    images: dict[str, str] = {}

    def rewrite(value: JsonValue, path: str) -> JsonValue:
        if isinstance(value, list):
            return [rewrite(entry, f"{path}[{index}]") for index, entry in enumerate(value)]
        if not isinstance(value, dict):
            return value
        out: dict[str, JsonValue] = {}
        for key, entry in value.items():
            here = f"{path}.{key}" if path else str(key)
            if key == "image_recipe" and isinstance(entry, str):
                if entry not in digests:
                    said = subprocess.run([*RECIPES[entry], "--reference"], capture_output=True, cwd=REPO,
                                          timeout=60.0, check=False).stdout.decode()
                    tag = next((line.strip() for line in said.splitlines() if line.strip()), "")
                    found = _capture("image", "inspect", tag, "--format", "{{.Id}}").strip()
                    if not found:
                        raise ValueError(f"this daemon holds no image for {here}, which names recipe {entry!r}")
                    digests[entry] = (tag, found)
                tag, found = digests[entry]
                images[here] = f"{entry} via {tag} -> {found}"
                out["image"] = found
            else:
                out[key] = rewrite(entry, here)
        return out

    return rewrite(dict(document), ""), images


def _sample_image(*, image: str, runtime: str) -> str:
    """Return a sample's own image with the agent runtime copied in at `/opt/agent-runtime`, built once."""
    tag = f"rqgm-sample:{hashlib.sha256(f'{image}\0{runtime}'.encode()).hexdigest()[:32]}"
    if _capture("image", "inspect", "--format", "{{.Id}}", tag).strip():
        return tag
    if not _capture("image", "inspect", "--format", "{{.Id}}", image).strip():
        _slow("pulling", image, ["docker", "pull", image])
    # A build stage cannot name an image by id, so the runtime is named by a tag first.
    source = f"rqgm-runtime:{runtime.removeprefix('sha256:')[:32]}"
    docker("image", "tag", runtime, source)
    recipe = f"FROM {source} AS runtime\nFROM {image}\nCOPY --from=runtime /opt/agent-runtime /opt/agent-runtime\n"
    _slow("layering the runtime onto", image,
          ["docker", "build", "--network=none", "--pull=false", "--tag", tag, "-"], stdin=recipe.encode())
    return tag


def _slow(doing: str, image: str, argv: list[str], *, stdin: bytes | None = None) -> None:
    try:
        done = subprocess.run(argv, input=stdin, capture_output=True, timeout=_LAYER_SECONDS, check=False)
    except subprocess.TimeoutExpired as exc:
        raise InfraWeather(f"{doing} {image} passed {_LAYER_SECONDS:g}s") from exc
    if done.returncode != 0:
        raise InfraWeather(f"{doing} {image} failed: {done.stderr.decode(errors='replace')[-500:]}")


@cache
def _image_env(*, image: str) -> dict[str, str]:
    """Return the environment an image declares, so an episode in a sample's image keeps its toolchain."""
    declared = json.loads(_capture("image", "inspect", "--format", "{{json .Config.Env}}", image) or "[]") or []
    return dict(entry.split("=", 1) for entry in declared if "=" in entry)


@cache
def _described(*, image: str) -> str:
    """Return one image's own account of the interpreter and packages it holds."""
    return _capture(
        "run", "--rm", "--entrypoint", "sh", image, "-c", _ENVIRONMENT_FINDER, _ENVIRONMENT_PROGRAM,
    ).strip()


# The run network and its edge.

def bridge_address() -> str:
    """Return the host side of Docker's default bridge, the one address the gateway listens on."""
    address = _capture("network", "inspect", "bridge", "--format", "{{range .IPAM.Config}}{{.Gateway}}{{end}}").strip()
    if not address:
        raise InfraWeather("the Docker daemon reports no bridge gateway address")
    return address


@contextmanager
def run_network(*, gateway_host: str, gateway_port: int, edge: str) -> Iterator[tuple[str, str]]:
    """Create the run's internal network and its relay to the gateway; yield (network, gateway URL).

    The network is `--internal`: the relay, also joined to the default bridge, is its only route out.
    """
    network = f"rqgm-{uuid.uuid4().hex[:12]}"
    name = f"rqgm-gw-{uuid.uuid4().hex}"
    docker("network", "create", "--internal", network)
    try:
        docker(
            "run", "-d", "--name", name, "--network", network,
            f"--add-host=host.docker.internal:{gateway_host}",
            "-e", f"RQGM_EDGE_PORT={_EDGE_PORT}", "-e", f"RQGM_GATEWAY_PORT={int(gateway_port)}",
            edge, "python", "-c", _EDGE_PROGRAM,
        )
        docker("network", "connect", "bridge", name)
        probe = f"import socket; socket.create_connection(('127.0.0.1', {_EDGE_PORT}), timeout=1)"
        deadline = time.monotonic() + 30.0
        while _run("exec", name, "python", "-c", probe).returncode:
            if time.monotonic() >= deadline:
                raise InfraWeather("gateway edge never started listening on the run network")
            time.sleep(0.2)
        yield network, f"http://{name}:{_EDGE_PORT}"
    finally:
        if containers := _capture("ps", "-aq", "--filter", f"network={network}").split():
            _capture("rm", "-f", *containers)
        _capture("network", "rm", network)


# One episode's container.

class _CappedReader:
    """Drain process output in the background, keeping its first and last `TRANSCRIPT_CAP / 2` bytes."""

    def __init__(self, pipe: object) -> None:
        self._pipe = pipe
        self._head, self._tail, self._seen = bytearray(), bytearray(), 0
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        half = TRANSCRIPT_CAP // 2
        while block := self._pipe.read1(65536):
            self._seen += len(block)
            room = half - len(self._head)
            self._head.extend(block[:max(0, room)])
            self._tail.extend(block[max(0, room):])
            if len(self._tail) > half:
                del self._tail[:-half]

    def result(self) -> tuple[bytes, bool]:
        """Return the captured output and whether it was clipped, once the stream closed."""
        self._thread.join(timeout=30.0)
        if not self._thread.is_alive():
            self._pipe.close()
        return bytes(self._head + self._tail), self._seen > TRANSCRIPT_CAP


@dataclasses.dataclass(slots=True, kw_only=True)
class Docker:
    """Run episodes in fresh contained containers, and carry a live expansion's tree in and out of its own."""

    network: str
    gateway_url: str
    surface: str
    #: Each pooled expansion's running container and the lock serialising transfers into it, by socket.
    _live: dict[str, tuple[str, threading.Lock]] = dataclasses.field(default_factory=dict)

    def run(self, spec: ExecSpec, workspace: Path) -> tuple[int, Transcript]:
        """Run one episode to its exit; the tree comes back onto `workspace`. Returns the exit code and transcript."""
        from gm.execution.episodes import Transcript

        if spec.checkout is not None and spec.sample_image is not None:
            spec = dataclasses.replace(spec, image=_sample_image(image=spec.sample_image, runtime=spec.image))
        self._describe(workspace=workspace, spec=spec)
        name = f"rqgm-tools-{uuid.uuid4().hex}"
        try:
            env = {"HOME": "/tmp"} | self._create(name=name, spec=spec)
            _send(workspace=workspace, name=name)
            docker("start", name)
            with self._watched(spec=spec, name=name):
                argv = ["docker", "exec", "-w", "/workspace", name, "/usr/bin/env", "-i"]
                argv += [f"{key}={value}" for key, value in env.items()]
                argv += list(spec.argv)
                with subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, close_fds=True) as process:
                    output = _CappedReader(process.stdout)
                    process.wait()
                    docker("stop", "-t", "0", name)
                    data, clipped = output.result()
                fetch(name=name, workspace=workspace)
            return process.returncode, Transcript(data=data, truncated=clipped)
        finally:
            _capture("rm", "-f", name)

    @contextmanager
    def _watched(self, *, spec: ExecSpec, name: str) -> Generator[None]:
        """Register a pooled expansion's container so the train pool can carry files in and out of it."""
        key = None if spec.pool_socket is None else str(spec.pool_socket)
        if key is not None:
            self._live[key] = (name, threading.Lock())
        try:
            yield
        finally:
            if key is not None:
                _, lock = self._live[key]
                with lock:
                    del self._live[key]

    def carry_out(self, *, socket: str, checkout: Path) -> bool:
        """Bring a pooled episode's live tree onto its host checkout; False once its container is gone."""
        if (held := self._live.get(socket)) is None:
            return False
        with held[1]:
            if self._live.get(socket) is not held:
                return False
            fetch(name=held[0], workspace=checkout)
            return True

    def carry_in(self, *, socket: str, checkout: Path, paths: Sequence[str]) -> bool:
        """Carry host-written files into a pooled episode's live tree; False once its container is gone."""
        if (held := self._live.get(socket)) is None:
            return False
        with held[1]:
            if self._live.get(socket) is not held:
                return False
            try:
                workspace_tar.send_files(workspace=checkout, paths=paths, name=held[0], target="/workspace", timeout=60.0)
            except workspace_tar.TransferRefused as refused:
                raise InfraWeather(str(refused)) from refused
            return True

    def _create(self, *, name: str, spec: ExecSpec) -> dict[str, str]:
        """Create one container of this episode's shape, and return the environment it carries."""
        home = "/workspace"
        env = {**_EPISODE_DEFAULTS, "PYTHONPATH": home, "PATH": f"{home}/shared:{_IMAGE_PATH}"}
        if spec.checkout is not None and spec.sample_image is not None:
            image = _image_env(image=spec.sample_image)
            path = image.get("PATH", _IMAGE_PATH)
            env = image | env | {"HOME": "/root", "PATH": f"{home}/shared:{path}", CHECKOUT_VAR: spec.checkout}
        env = env | {CALL_SECONDS_VAR: UNBOUNDED} | dict(spec.env) | dict.fromkeys(sorted(RESERVED_ENV), "")
        network = "none"
        if spec.grant is not None:
            network = self.network
            env |= {"OPENAI_BASE_URL": self.gateway_url, "OPENAI_API_KEY": str(spec.grant),
                    "RQGM_AGENT_SURFACE": self.surface, "RQGM_AGENT_TAG": str(spec.agent_tag)}
        argv = ["create", "--name", name, "--network", network, "--entrypoint", "/usr/bin/env"]
        if spec.cpus != UNBOUNDED:
            argv += [f"--cpus={spec.cpus}"]
        if spec.memory_bytes != UNBOUNDED:
            argv += [f"--memory={spec.memory_bytes}b"]
        argv += ["-w", "/workspace"]
        if spec.pool_socket is not None:
            argv += ["--group-add", str(spec.pool_socket.stat().st_gid),
                     "-v", f"{spec.pool_socket}:{POOL_SOCKET_TARGET}:ro"]
        argv += ["--pull=never", "--cap-drop=ALL", "--security-opt=no-new-privileges=true", "--ipc=private"]
        argv += [word for key, value in env.items() for word in ("-e", f"{key}={value}")]
        docker(*argv, spec.image, "/bin/sleep", "infinity")
        return env

    @staticmethod
    def _describe(*, workspace: Path, spec: ExecSpec) -> None:
        """Write what the host knows about this episode's machine into `.rqgm/environment.json`."""
        stated: dict[str, object] = {}
        if described := _described(image=spec.image):
            with suppress(json.JSONDecodeError):
                stated = json.loads(described)
        limits: dict[str, object] = {
            name: bound for name, bound in (("cpus", spec.cpus), ("memory_bytes", spec.memory_bytes))
            if bound != UNBOUNDED
        }
        if spec.token_allotment != UNBOUNDED:
            limits["episode_token_allotment"] = spec.token_allotment
        stated["limits"] = limits
        target = workspace / experience.ENVIRONMENT
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"{json.dumps(stated, indent=1, sort_keys=True)}\n", encoding="utf-8")


# Tree transfer.

def _send(*, workspace: Path, name: str) -> None:
    try:
        workspace_tar.send(workspace=workspace, name=name, target="/workspace", exclude=())
    except (workspace_tar.TransferRefused, OSError) as refused:
        raise InfraWeather(f"episode workspace could not be carried into its container: {refused}") from refused


def fetch(*, name: str, workspace: Path) -> None:
    """Copy a container's tree back onto the host checkout, keeping the host's `.git` and own files.

    The archive is validated by a separate interpreter (this file run as a script) in private scratch
    first, so an escaping path or link never touches host data.
    """
    with tempfile.TemporaryDirectory(prefix="rqgm-harvest-", dir=workspace.parent) as scratch:
        incoming = Path(scratch) / "checkout"
        try:
            done = subprocess.run(
                [sys.executable, __file__, name, str(incoming)],
                env={key: value for key, value in os.environ.items() if key in _TRANSFER_ENV},
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False,
            )
        except OSError as exc:
            raise InfraWeather(f"sandbox archive worker could not start: {exc}") from exc
        if done.returncode:
            detail = done.stderr.decode(errors="replace").strip()[-4096:]
            raise InfraWeather(detail or f"sandbox archive worker exited {done.returncode}")
        _mirror(source=incoming, target=workspace, root=workspace)


def _mirror(*, source: Path, target: Path, root: Path) -> None:
    """Make `target` match `source`, leaving `.git` and host-owned context alone."""
    target.mkdir(parents=True, exist_ok=True)
    incoming = {path.name: path for path in source.iterdir()}
    for name in sorted(incoming.keys() | {path.name for path in target.iterdir()}):
        into = target / name
        rel = into.relative_to(root).as_posix()
        if rel == ".git" or experience.host_context(rel=rel):
            continue
        came = incoming.get(name)
        if came is None:
            _prune(target=into, root=root)
        elif came.is_dir() and not came.is_symlink():
            if into.is_symlink() or (into.exists() and not into.is_dir()):
                into.unlink()
            _mirror(source=came, target=into, root=root)
        else:
            if into.is_dir() and not into.is_symlink():
                shutil.rmtree(into)
            else:
                into.unlink(missing_ok=True)
            if came.is_symlink():
                into.symlink_to(came.readlink())
            else:
                shutil.copy2(came, into)


def _prune(*, target: Path, root: Path) -> None:
    if experience.host_context(rel=target.relative_to(root).as_posix()):
        return
    if target.is_dir() and not target.is_symlink():
        for path in target.iterdir():
            _prune(target=path, root=root)
        if not any(target.iterdir()):
            target.rmdir()
    else:
        target.unlink(missing_ok=True)


_TRANSFER_ENV = frozenset({
    "PATH", "PYTHONPATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "LD_LIBRARY_PATH",
    "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH",
})


def _member(member: tarfile.TarInfo, destination: str) -> tarfile.TarInfo | None:
    """Refuse escaping paths and escaping or `.git`-reaching links; otherwise apply tar's data filter."""
    name = posixpath.normpath(member.name)
    if posixpath.isabs(name) or name == ".." or name.startswith("../"):
        raise tarfile.FilterError(f"sandbox archive has an escaping path: {member.name!r}")
    if member.issym():
        link = posixpath.normpath(posixpath.join(posixpath.dirname(name), member.linkname))
        if posixpath.isabs(member.linkname) or link == ".." or link.startswith("../") or ".git" in Path(link).parts:
            raise tarfile.FilterError(f"sandbox archive has an escaping symlink: {member.name!r} -> {member.linkname!r}")
    return tarfile.data_filter(member, destination)


def _receive(*, container: str, into: Path) -> None:
    """Copy `/workspace` out of a container into `into`, minus its `.git`."""
    into.mkdir(parents=True)
    with tempfile.TemporaryFile() as archive:
        try:
            done = subprocess.run(["docker", "cp", f"{container}:/workspace/.", "-"], stdout=archive,
                                  stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, timeout=60, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise InfraWeather("sandbox archive could not be received") from exc
        if done.returncode:
            detail = done.stderr.decode(errors="replace").strip()[:2048]
            raise InfraWeather(f"Docker refused the sandbox archive {container}:/workspace: {detail}")
        archive.seek(0)
        try:
            with tarfile.open(fileobj=archive) as opened:
                opened.extractall(into, filter=_member)
        except (OSError, tarfile.TarError) as exc:
            raise InfraWeather(f"sandbox archive refused before host data was changed: {exc}") from exc
    shutil.rmtree(into / ".git", ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="receive one container's /workspace into a fresh directory")
    parser.add_argument("container")
    parser.add_argument("into", type=Path)
    args = parser.parse_args()
    try:
        _receive(container=args.container, into=args.into)
    except (InfraWeather, OSError) as refused:
        sys.stderr.write(str(refused)[:4096] + "\n")
        raise SystemExit(1) from refused
