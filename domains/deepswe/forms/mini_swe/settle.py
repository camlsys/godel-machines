# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""The deepswe form's scoring law: the task's own verifier for the solver, agreement for the reviewer."""

from __future__ import annotations

import base64
import io
import json
import pathlib
import re
import subprocess
import sys
import tarfile
from pathlib import PurePosixPath

FORM = pathlib.Path(__file__).parent
sys.path.insert(0, str(FORM.parents[2] / "common"))

import settlement

VERDICT, PATCH, LABEL, ACCEPT = "verdict.txt", "model.patch", "review_label", "accept"
VERIFIER, IMAGE = "verifier", "image"
#: The task's verifier timeout (every DeepSWE task declares 1800 s) and its declared machine.
VERIFIER_SECONDS, CPUS, MEMORY = 1800, 2, 8 << 30
#: Test infrastructure a solver may not change. A path matching one of these is dropped from the
#: patch unless the task's reference solution changes it too; a path the task's own test patch
#: owns is always dropped, since the verifier restores it before running.
PROTECTED = (
    "**/test/**", "**/tests/**", "**/__tests__/**", "**/testdata/**", "**/fixtures/**", "**/*_test.*",
    "**/test_*.py", "**/*.test.*", "**/*.spec.*", "**/conftest.py", "**/pytest.ini", "**/setup.cfg",
    "**/tox.ini", "**/jest.config.*", "**/vitest.config.*", ".github/**",
)
HEAD = re.compile(rb"^diff --git a/(.+?) b/(.+?)$", re.M)
#: Untar the verifier and the patch, run the task's `test.sh`, and send `/logs/verifier` back as a tar.
RUN = (
    "tar -x -C / && mkdir -p /logs/verifier && bash /tests/test.sh > /logs/verifier/test-stdout.txt 2>&1;"
    " tar -c -C /logs verifier"
)


def guarded(*, patch: bytes, owned: set[str], reference: set[str]) -> tuple[bytes, list[str]]:
    """Return the patch without its changes to test infrastructure, and the paths it dropped."""
    starts = [match.start() for match in HEAD.finditer(patch)] + [len(patch)]
    kept, dropped = [patch[: starts[0]]], []
    for start, end in zip(starts, starts[1:]):
        chunk = patch[start:end]
        paths = {path.decode(errors="replace") for match in HEAD.finditer(chunk) for path in match.groups()}
        protected = any(PurePosixPath(path).full_match(glob) for path in paths - reference for glob in PROTECTED)
        if paths & owned or protected:
            dropped += sorted(paths)
        else:
            kept.append(chunk)
    return b"".join(kept), dropped


def _tar(files: dict[str, bytes]) -> bytes:
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w") as archive:
        for name, data in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o755 if name.endswith(".sh") else 0o644
            archive.addfile(info, io.BytesIO(data))
    return out.getvalue()


def verified(*, patch: bytes, verifier: dict, image: str) -> tuple[float, dict[str, bytes]]:
    """Run the task's verifier over the patch in a fresh container; return the reward and its evidence."""
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(verifier["tests"])), mode="r:gz") as tests:
        files = {member.name: tests.extractfile(member).read() for member in tests.getmembers() if member.isfile()}
    owned = {path.decode() for match in HEAD.finditer(files["tests/test.patch"]) for path in match.groups()}
    kept, dropped = guarded(patch=patch, owned=owned, reference=set(verifier["reference_paths"]))
    guard = json.dumps({"dropped": dropped}, sort_keys=True).encode()
    if not kept.strip():
        return 0.0, {"guard.json": guard}
    argv = [
        "docker", "run", "--rm", "-i", "--network=none", "--cap-drop=ALL",
        "--security-opt=no-new-privileges=true", "--ipc=private", "--pull=never",
        f"--cpus={CPUS}", f"--memory={MEMORY}b", "--entrypoint", "/bin/sh", image, "-c", RUN,
    ]
    staged = _tar({**{name: data for name, data in files.items()}, "logs/artifacts/model.patch": kept})
    try:
        done = subprocess.run(argv, input=staged, capture_output=True, timeout=VERIFIER_SECONDS, check=False)
    except subprocess.TimeoutExpired as late:
        raise settlement.HostWeatherError(f"the verifier passed its {VERIFIER_SECONDS}s bound") from late
    try:
        with tarfile.open(fileobj=io.BytesIO(done.stdout), mode="r:") as returned:
            logs = {member.name: returned.extractfile(member).read() for member in returned.getmembers() if member.isfile()}
        reward = json.loads(logs["verifier/reward.json"])
    except (tarfile.TarError, KeyError, ValueError) as lost:
        said = done.stderr.decode(errors="replace")[-500:]
        raise settlement.HostWeatherError(f"the verifier left no reward (exit {done.returncode}): {said}") from lost
    stdout = logs.get("verifier/test-stdout.txt", b"")
    return float(reward["reward"]), {"guard.json": guard, "reward.json": logs["verifier/reward.json"], "test-stdout.txt": stdout[-(1 << 20):]}


def raw_score(*, slot: str, request: dict) -> float | tuple[float, dict[str, bytes]]:
    """Return this form's raw score for one judgment, with the evidence it observed."""
    if slot == "reviewer":
        return float(request["score"])
    if slot == "grader":
        patch = settlement.staged(request=request, key=PATCH) or b""
        verifier = json.loads(settlement.channel(request=request, key=VERIFIER))
        return verified(patch=patch, verifier=verifier, image=settlement.channel(request=request, key=IMAGE))
    verdict = settlement.staged(request=request, key=VERDICT)
    if verdict is None:
        raise ValueError(f"the calls anchor was staged {sorted(request['files'])}, not {VERDICT!r}")
    label = settlement.channel(request=request, key=LABEL)
    return float(settlement.stated_score(artifact=verdict) == float(label == ACCEPT))


def meant(*, slot: str, raw: float, member: dict, form: dict) -> float:
    """Return what a raw score is worth on `[0, 1]`: an unreadable verdict is worth nothing."""
    return 0.0 if raw == settlement.MALFORMED else raw


LAW = settlement.Law(form=FORM, raw_score=raw_score, meant=meant)
settled, main, WEATHER_EXIT = LAW.settled, LAW.main, settlement.WEATHER_EXIT

if __name__ == "__main__":
    main()
