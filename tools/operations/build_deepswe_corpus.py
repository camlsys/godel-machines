# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""Build the `deepswe` corpus under `domains/deepswe/data/build/` from DeepSWE v1.1.

Downloads the DeepSWE repository at a pinned revision and the 226 review patches `review.json`
names, checking every sha256. `split.json` assigns each of the 113 tasks to train, validation or
test by whole repository; `images.json` pins each task's image by digest. Every task gives one
solver row and two reviewer rows: a model patch that passed the task's tests (`accept`) and one
that failed them (`reject`). A solver row carries the task statement, the image and the verifier:
the task's own `tests/` folder plus the paths its reference solution touches, which settlement
alone reads. Every written member is checked against its pinned sha256.

    python tools/operations/build_deepswe_corpus.py
"""

from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import io
import json
import re
import sys
import tarfile
import tomllib
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "domains" / "deepswe" / "data"
REVISION = "6db64a40f3318d8659238ff34a8cc4b491c49205"
UPSTREAM = f"https://github.com/datacurve-ai/deep-swe/archive/{REVISION}.tar.gz"
UPSTREAM_SHA256 = "14822589c296a0d9b4a934f77d3507ae6c0b929a2e97909ddd898cabaffeffb9"
MEMBERS = {
    "test": "975046f35cae83bd045670d4e00e74b1df10fd5ea4526627b0f90dd1f6060f7b",
    "test_negative": "c298904a8e8c673a039fcc825f2d27aadb67d278e9a0145b9cdaef0605bc36b0",
    "test_positive": "02ea46363d03830a8d22772bf100491c2c19c80182c1f3d7e85b6aadffd6d04d",
    "train": "9049979cc2efe95709e6436c95699f9787d24091e4a897811cf598b0b5ef9f98",
    "train_negative": "8f8678805af9536a81fffe00071010e617a91a829a5812fecabb551308f2eea2",
    "train_positive": "8681a27bf2607925c085f219ef6704d28363120ade444896e61ad9589e1e059b",
    "validation": "6204bfa1f6b5e65dff59a5977c1ce7618ebd1578965a317054c8d2557390a6bf",
    "validation_negative": "419188c88534bd6958f1262eb8cb1895b8f26dd4f1e91fecb3d20570f948bab9",
    "validation_positive": "ae33049845631a3f90d1e79afae24c54c1e860514345ed9c4716e7cfc60f09ae",
}
SOLVER = ("sample", "task", "image", "verifier")
REVIEWER = ("sample", "assessed", "image", "review_label")
#: The reviewer's envelope, as `task_agents/reviewer` composes it from a statement and a diff.
PROBLEM_MARK, DIFF_MARK = "### PROBLEM\n", "\n\n### DIFF\n"
HEAD = re.compile(r"^diff --git a/(?:.+?) b/(.+?)$", re.M)

csv.field_size_limit(sys.maxsize)


def fetched(url: str, sha256: str) -> bytes:
    """Return the bytes at `url`, refusing any that are not the pinned bytes."""
    with urllib.request.urlopen(url, timeout=300) as response:
        body = response.read()
    if (seen := hashlib.sha256(body).hexdigest()) != sha256:
        raise SystemExit(f"{url} has sha256 {seen}, expected {sha256}")
    return body


def tasks(archive: bytes) -> dict[str, dict[str, bytes]]:
    """Return every task folder in the upstream archive, as its files' relative paths and bytes."""
    found: dict[str, dict[str, bytes]] = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as opened:
        for member in opened.getmembers():
            parts = member.name.split("/")
            if member.isfile() and len(parts) > 3 and parts[1] == "tasks":
                found.setdefault(parts[2], {})["/".join(parts[3:])] = opened.extractfile(member).read()
    return found


def touched(patch: str) -> list[str]:
    """Return the paths a unified diff changes, in order."""
    return list(dict.fromkeys(HEAD.findall(patch)))


def verifier(files: dict[str, bytes]) -> str:
    """Return a task's verifier as one column: its `tests/` folder, gzipped tar in base64, and the reference paths."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name in sorted(path for path in files if path.startswith("tests/")):
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(files[name]), 0o755 if name.endswith(".sh") else 0o644
            archive.addfile(info, io.BytesIO(files[name]))
    tests = gzip.compress(raw.getvalue(), mtime=0)
    reference = touched(files["solution/solution.patch"].decode())
    return json.dumps({"tests": base64.b64encode(tests).decode(), "reference_paths": reference}, sort_keys=True)


def statement(files: dict[str, bytes]) -> str:
    """Return the cell the solver is staged: the statement and where the work happens."""
    meta = tomllib.loads(files["task.toml"].decode())["metadata"]
    return json.dumps({
        "instruction": files["instruction.md"].decode(),
        "repository": meta["repository_url"].removeprefix("https://github.com/"),
        "language": meta["language"],
        "path": "/app",
    }, sort_keys=True)


def write(name: str, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    """Write one set under `build/` and check it against its pin."""
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=list(columns))
    writer.writeheader()
    writer.writerows(rows)
    body = out.getvalue().encode("utf-8")
    if (seen := hashlib.sha256(body).hexdigest()) != MEMBERS.get(name):
        raise SystemExit(f"build/{name}.csv has sha256 {seen}, expected {MEMBERS.get(name)}")
    (DATA / "build" / f"{name}.csv").write_bytes(body)


def main() -> None:
    upstream = tasks(fetched(UPSTREAM, UPSTREAM_SHA256))
    split = json.loads((DATA / "split.json").read_bytes())
    images = json.loads((DATA / "images.json").read_bytes())
    reviews = json.loads((DATA / "review.json").read_bytes())
    (DATA / "build").mkdir(exist_ok=True)
    for name, members in split.items():
        write(name, SOLVER, [
            {"sample": task, "task": statement(upstream[task]), "image": images[task], "verifier": verifier(upstream[task])}
            for task in members
        ])
        for label, half in (("accept", "positive"), ("reject", "negative")):
            write(f"{name}_{half}", REVIEWER, [
                {
                    "sample": f"{task}.{label}",
                    "assessed": PROBLEM_MARK + upstream[task]["instruction.md"].decode() + DIFF_MARK
                    + fetched(**reviews[task][label]).decode(),
                    "image": images[task],
                    "review_label": label,
                }
                for task in members
            ])
    print(f"wrote {len(MEMBERS)} sets to {DATA / 'build'}")


if __name__ == "__main__":
    main()
