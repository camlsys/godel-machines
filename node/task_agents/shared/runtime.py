# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""Run one declared task-agent job and write its host-facing answer."""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.util
import json
import os
import stat
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from task_agents.shared import io_contract

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from types import ModuleType

CONTROL = Path(".rqgm")
DISPATCH, JUDGMENT, GENERATION = (CONTROL / name for name in ("dispatch.json", "judgment.json", "generation.json"))
CONTRACT = CONTROL / "contract.json"
CHECKOUT = "RQGM_AGENT_CHECKOUT"
PROVIDER_EXIT = 75
_HARVEST = threading.local()


@contextlib.contextmanager
def harvesting() -> Iterator[list[dict[str, Any]]]:
    outer = getattr(_HARVEST, "rows", None)
    _HARVEST.rows = rows = []
    try:
        yield rows
    finally:
        _HARVEST.rows = outer


def _record(job: dict, files: dict[str, bytes]) -> None:
    stated, rows = job.get("provenance"), getattr(_HARVEST, "rows", None)
    if rows is None or not isinstance(stated, dict) or not all(stated.get(n) for n in ("call", "dataset", "task", "samples", "basis")):
        return
    arrived = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    selected = {str(sample): dict(keys) for sample, keys in dict(stated["samples"]).items()}
    received = sorted(sample for sample, keys in selected.items() if keys and all(arrived.get(str(k)) == str(d) for k, d in keys.items()))
    rows.append({
        "call": str(stated["call"]), "role": str(job["role"]), "task": str(stated["task"]),
        "dataset": str(stated["dataset"]), "selected": sorted(selected), "samples": received,
        "missing": sorted(selected.keys() - received), "basis": str(stated["basis"]),
        "received": {name: len(data) for name, data in sorted(files.items())}, "carried": len(received),
    })


def review_capability(*, role: str) -> Callable[..., tuple[float, str]] | None:
    try:
        judges = json.loads(CONTRACT.read_text(encoding="utf-8")).get("judges", {})
    except (OSError, ValueError, AttributeError):
        return None
    seats = [name for name, spec in judges.items() if isinstance(spec, dict) and spec.get("scorer") == "node" and role in (spec.get("over") or ())]
    if len(seats) != 1:
        return None
    try:
        return getattr(importlib.import_module(f"task_agents.{seats[0]}"), "review", None)
    except ImportError:
        return None


def chat_capability() -> Callable[[str], str] | None:
    if importlib.util.find_spec("shared") is None:
        return None
    client = importlib.import_module("shared.gateway.entrypoint").client_from_env()
    return None if client is None else lambda prompt: client.chat(messages=[{"role": "user", "content": prompt}]).content


def solve_capability() -> Callable[[str], dict] | None:
    if importlib.util.find_spec("meta") is None or importlib.util.find_spec("meta.solve") is None:
        return None
    return lambda task: importlib.import_module("meta.solve").solve(
        task=task, workspace=Path(os.environ.get(CHECKOUT) or Path.cwd()), notes=Path.cwd(),
    )


def _inputs(job: dict) -> dict[str, bytes]:
    raw = job.get("inputs", {})
    raw = dict(enumerate(raw)) if isinstance(raw, list) else raw
    if not isinstance(raw, dict):
        raise ValueError("job inputs must be a list or object")
    return {str(name): Path(str(path)).read_bytes() for name, path in raw.items()}


def _write(files: dict[str, bytes], declared: dict) -> None:
    if extra := files.keys() - declared.keys():
        raise ValueError(f"agent returned undeclared files {sorted(extra)}; contract declares {sorted(declared)}")
    for name, data in files.items():
        path = Path(str(declared[name]))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def _forward(agent: ModuleType, job: dict) -> dict:
    role, mode = str(job["role"]), job.get("mode")
    if not isinstance(mode, str) or not mode.strip():
        raise io_contract.IoRefusalError(f"the job for {role!r} states no `mode`, so no arm declares its contract")
    files = _inputs(job)
    arm = io_contract.declared_arm(role=role, mode=mode)
    io_contract.check_inputs(role=role, mode=mode, files=files, arm=arm)
    _record(job, files)
    answer = agent.forward({
        "files": files, "seed": int(job["seed"]), "chat": chat_capability(),
        "solve": solve_capability(), "review": review_capability(role=role),
        "evidence": sorted(job.get("evidence") or ()),
    })
    if not isinstance(answer, dict):
        raise TypeError(f"task_agents.{role}.forward returned {type(answer).__name__}; expected a dict")
    io_contract.check_output(role=role, mode=mode, answer=answer, arm=arm)
    return answer


def _job() -> dict | None:
    if not DISPATCH.is_file():
        sys.stderr.write(f"{DISPATCH} is absent; ask the host with `rqgm-call submit`.\n")
        return None
    try:
        opened = os.open(DISPATCH, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        sys.stderr.write(f"{DISPATCH} is not a host-owned regular file ({error}).\n")
        return None
    with os.fdopen(opened, encoding="utf-8") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            sys.stderr.write(f"{DISPATCH} is not a regular file.\n")
            return None
        job = json.load(handle)
    DISPATCH.unlink()
    return job


def _score(member: str, data: bytes) -> dict[str, object]:
    try:
        stated = json.loads(data)
    except ValueError as error:
        raise ValueError(f"member {member!r} wrote a verdict this runtime cannot parse: {error}") from error
    score = stated.get("score") if isinstance(stated, dict) else None
    if score is None:
        raise ValueError(f"member {member!r} wrote a verdict that names no 'score': {stated!r}")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError(f"member {member!r} wrote a verdict whose score is not a JSON number: {score!r}")
    return {"member": member, "score": score}


def run() -> int:
    job = _job()
    if job is None:
        return 1
    agent = importlib.import_module(f"task_agents.{job['role']}")
    mode = str(job["mode"])
    if mode == "judge":
        members = job.get("members")
        if not isinstance(members, list) or not all(isinstance(m, str) and m.strip() for m in members):
            raise ValueError(f"job names no members for this episode to answer for: {members!r}")
        outputs = job.get("outputs")
        if not isinstance(outputs, dict) or len(outputs) != 1:
            raise ValueError(f"a judgment job must declare exactly one output channel; it declared {outputs!r}")
        verdict = next(iter(outputs))
        answered = [(member, _forward(agent, job)) for member in members]
        scores, evidence = [], {}
        for member, answer in answered:
            files = dict(answer["files"])
            if verdict not in files:
                raise TypeError(f"agent result for member {member!r} states no {verdict!r}")
            scores.append(_score(member, files[verdict]))
            for name, data in dict(answer.get("evidence", {})).items():
                if name in evidence:
                    raise ValueError(f"member {member!r} duplicates evidence {name!r}")
                evidence[name] = data
            _write(files, outputs)
        _write(evidence, dict(job.get("evidence", {})))
        JUDGMENT.write_text(json.dumps(scores, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        return 0
    if mode == "generate":
        members = job.get("members")
        if not isinstance(members, list) or not members:
            raise ValueError(f"job names no members for this episode to produce for: {members!r}")
        stated = []
        for body in members:
            if not isinstance(body, dict) or not isinstance(body.get("member"), str):
                raise ValueError(f"job states a member naming no member: {body!r}")
            if missing := [key for key in ("seed", "inputs", "outputs", "evidence") if key not in body]:
                raise ValueError(f"member {body['member']!r} states no {missing}, which it owes")
            answer = _forward(agent, job | body)
            _write(dict(answer["files"]), dict(body["outputs"]))
            _write(dict(answer.get("evidence", {})), dict(body["evidence"]))
            stated.append({"member": body["member"], "outputs": sorted(answer["files"]), "evidence": sorted(answer.get("evidence", {}))})
        GENERATION.write_text(json.dumps(stated, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        return 0
    raise TypeError(f"the job declares mode {mode!r}, which this runtime does not serve")


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except RuntimeError as failure:
        if type(failure).__name__ != "GatewayRefused":
            raise
        from shared.gateway.client import weathered
        if not weathered(status=getattr(failure, "status", 0), spoken=str(failure)):
            raise
        sys.stderr.write(f"[task-agents] provider refused, not the seat: {failure}\n")
        raise SystemExit(PROVIDER_EXIT) from failure
