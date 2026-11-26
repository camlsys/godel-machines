# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""Ask one question of every candidate node: do its declared operators work?"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CONTROL = Path(".rqgm")
PROBE_ROLE = "rqgm_admission_probe"
PAYLOAD = b"admission probe payload"

_IMPORT_SEATS = (
    "import importlib, pathlib, sys;"
    "seats = sorted(p.name for p in pathlib.Path('task_agents').iterdir()"
    " if (p / '__init__.py').is_file());"
    "[importlib.import_module('task_agents.' + name) for name in seats];"
    "sys.stdout.write('imported %d seat(s): %s' % (len(seats), ', '.join(seats) or '(none)'))"
)

_ATTEMPT_COST = (
    "import pathlib, sys;"
    "sys.path.insert(0, '.');"
    "sys.exit(0) if not pathlib.Path('shared/expansion_batch.py').is_file() else None;"
    "exec('''\n"
    "import shared.expansion_batch as b\n"
    "try:\n"
    "    have = b.limits()\n"
    "    said = 'one graded attempt claims %d thread(s) and %d byte(s)' % (\n"
    "        b.attempt_cpus(have=have), b.attempt_memory(have=have))\n"
    "except b.BatchRefused as refusal:\n"
    "    sys.stdout.write(' '.join(str(refusal).split()))\n"
    "    sys.exit(3)\n"
    "sys.stdout.write(said)\n"
    "''')"
)

UNBOOKABLE_MARK = "admission probe: no graded attempt can be booked"
PROBE_MEMBER = "probe"
ALLOTMENT = 90_000
ALLOTMENT_ENV = "RQGM_AGENT_TOKEN_ALLOTMENT"
PROBE_WAIT_SECONDS = 900.0
CONTROL_TIMEOUT_MARK = "rqgm-admission-probe-control-timeout"


def allowed() -> str:
    """Return the token allotment this probe declares to the operator it drives.

    Returns:
        str: The declared episode allotment when it exceeds `ALLOTMENT`, and
    """
    try:
        declared = int(os.environ.get(ALLOTMENT_ENV, ""))
    except ValueError:
        return str(ALLOTMENT)
    return str(max(ALLOTMENT, declared))


SURFACE = "chat"
SURFACE_ENV = "RQGM_AGENT_SURFACE"

_REPLY = json.dumps(
    {
        "id": "admission-probe",
        "object": "chat.completion",
        "model": "admission-probe",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "probe acknowledged"},
                "finish_reason": "stop",
            },
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    },
).encode()


class _Mock(BaseHTTPRequestHandler):

    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        """Consume the request and return the canned completion."""
        self.rfile.read(int(self.headers.get("Content-Length", "0") or 0))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(_REPLY)))
        self.end_headers()
        self.wfile.write(_REPLY)

    def log_message(self, format: str, *args: object) -> None:
        """Keep the probe transcript to admission evidence only."""
        del format, args


def _operator_env(*, port: int) -> dict[str, str]:
    return os.environ | {
        "OPENAI_BASE_URL": f"http://127.0.0.1:{port}",
        "OPENAI_API_KEY": "admission-probe",
        "RQGM_AGENT_MODEL": "admission-probe",
        ALLOTMENT_ENV: allowed(),
        "RQGM_AGENT_BUDGET_EXPOSE": "",
        "RQGM_AGENT_BUDGET_NUDGE": "off",
        "RQGM_AGENT_MAX_SEQUENCE_TOKENS": "unbounded",
        "RQGM_AGENT_MAX_OUTPUT_TOKENS": "unbounded",
        SURFACE_ENV: SURFACE,
    }


def _run(*, argv: list[str], port: int) -> tuple[int, str]:
    try:
        done = subprocess.run(
            argv,
            env=_operator_env(port=port),
            capture_output=True,
            check=False,
            timeout=PROBE_WAIT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return 1, f"{CONTROL_TIMEOUT_MARK}: no answer within {PROBE_WAIT_SECONDS:g}s"
    except (OSError, subprocess.SubprocessError) as broken:
        return 1, f"{type(broken).__name__}: {broken}"
    tail = (done.stdout + done.stderr).decode(errors="replace")[-600:].strip()
    return done.returncode, tail


def _observed(*, name: str, argv: list[str], status: int, tail: str) -> str:
    said = "\n".join(f"    {line}" for line in tail.splitlines()) if tail else "    wrote nothing"
    return f"  {name}: exit {status}\n    $ {' '.join(argv)}\n{said}"


CONTRACT = CONTROL / "contract.json"

IO_KEY = "io"


def _declare_probe_seat() -> str:
    if not CONTRACT.is_file():
        return f"{CONTRACT} is absent, so this tree states no seat vocabulary at all"
    try:
        stated = json.loads(CONTRACT.read_text(encoding="utf-8"))
    except ValueError as broken:
        return f"{CONTRACT} is not readable JSON: {broken}"
    if not isinstance(stated, dict):
        return f"{CONTRACT} states {type(stated).__name__}, not an object"
    if IO_KEY not in stated:
        return f"{CONTRACT} carries no {IO_KEY!r}, so no seat of this tree can be checked"
    block = stated[IO_KEY]
    if not isinstance(block, dict):
        return f"{CONTRACT} states an {IO_KEY!r} that is not an object: {block!r}"
    if PROBE_ROLE in block:
        return (
            f"{CONTRACT} already declares {PROBE_ROLE!r}, which is the host's own probe role and"
            " no form declares it. A candidate authoring that name is claiming the instrument"
            " that measures it"
        )
    block[PROBE_ROLE] = {
        "generate": {
            "reads": [],
            "positional": True,
            "writes": ["probe"],
            "evidence": [],
        },
    }
    CONTRACT.write_text(json.dumps(stated, sort_keys=True, separators=(",", ":")))
    return ""


def _task_job() -> str:
    if refused := _declare_probe_seat():
        return refused
    role = Path("task_agents") / PROBE_ROLE
    role.mkdir(parents=True, exist_ok=True)
    (role / "__init__.py").write_bytes(
        b"def forward(subject):\n    return {'files': {'probe': subject['files']['0']}}\n",
    )
    (CONTROL / "probe-input").write_bytes(PAYLOAD)
    declaration = {
        "mode": "generate",
        "role": PROBE_ROLE,
        "members": [
            {
                "member": PROBE_MEMBER,
                "seed": 0,
                "inputs": [(CONTROL / "probe-input").as_posix()],
                "outputs": {"probe": (CONTROL / "probe-output").as_posix()},
                "evidence": {},
            },
        ],
    }
    (CONTROL / "dispatch.json").write_text(
        json.dumps(declaration, sort_keys=True, separators=(",", ":")),
    )
    (CONTROL / "job.json").write_text(
        json.dumps(
            {"members": [{"member": PROBE_MEMBER}]},
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    return ""


def main() -> int:
    """Exercise the declared expansion and task-agent operators once each.

    Returns:
        int: 0 when both operators ran and answered, and 1 otherwise, with the
    """
    expansion = sys.argv[1:]
    if not expansion:
        sys.stdout.write("admission refused: no declared expansion operator\n")
        return 1
    CONTROL.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Mock)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        status, tail = _run(argv=expansion, port=port)
        if status != 0:
            sys.stdout.write(f"admission refused: expansion operator exited {status}: {tail}\n")
            return 1
        seen = [_observed(name="expansion", argv=expansion, status=status, tail=tail)]
        if refused := _task_job():
            sys.stdout.write(f"admission refused: {refused}\n")
            return 1
        task = [sys.executable, "-B", "-m", "task_agents.shared.runtime"]
        status, tail = _run(argv=task, port=port)
        if status != 0:
            sys.stdout.write(f"admission refused: task operator exited {status}: {tail}\n")
            return 1
        seen.append(_observed(name="task", argv=task, status=status, tail=tail))
        produced = CONTROL / "probe-output"
        if not produced.is_file() or produced.read_bytes() != PAYLOAD:
            sys.stdout.write("admission refused: task operator produced no declared output\n")
            return 1
        seen.append(f"  output: {len(PAYLOAD)} bytes, matching the declared payload")
        status, tail = _run(argv=[sys.executable, "-B", "-c", _IMPORT_SEATS], port=port)
        if status != 0:
            sys.stdout.write(f"admission refused: a declared seat will not import: {tail}\n")
            return 1
        seen.append(_observed(name="seats", argv=["<import every task_agents seat>"], status=status, tail=tail))
        status, tail = _run(argv=[sys.executable, "-B", "-c", _ATTEMPT_COST], port=port)
        seen.append(
            f"  attempt: {tail}"
            if status == 0 and tail
            else f"  {UNBOOKABLE_MARK}: {tail or f'the cost probe exited {status}'}",
        )
    finally:
        server.shutdown()
        server.server_close()
    sys.stdout.write("admission probe: declared operators ran and answered\n")
    sys.stdout.write("\n".join(seen) + "\n")
    sys.stdout.write(
        "  note: the probe's provider answers fixed prose rather than an action, so an"
        " agent that parses actions ends on a format stop. That is this probe passing --"
        " the operator ran, loaded candidate code, reached a provider and exited 0.\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
