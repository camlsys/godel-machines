# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""Submit this tree for train evaluation and wait for results."""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
from pathlib import Path

SOCKET_PATH = Path("/run/rqgm/pool.sock")
EXIT_OK, EXIT_REFUSED, EXIT_NO_SOCKET = 0, 1, 2


def _positive(text: str) -> int:
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"--per-role takes a whole number: {text!r}") from error
    if value < 1:
        raise argparse.ArgumentTypeError(f"--per-role takes at least one sample per role, not {value}")
    return value


def _ceiling(text: str) -> float:
    try:
        value = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"--timeout-s takes a number of seconds: {text!r}") from error
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(f"--timeout-s takes zero seconds or more, not {text!r}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rqgm-call")
    operations = parser.add_subparsers(dest="op", required=True)
    submit = operations.add_parser("submit")
    selection = submit.add_mutually_exclusive_group()
    selection.add_argument("--batch", type=Path)
    selection.add_argument("--role")
    submit.add_argument("--all", action="store_true")
    submit.add_argument("--per-role", type=_positive)
    submit.add_argument("--tag")
    submit.add_argument("--timeout-s", type=_ceiling)
    wait = operations.add_parser("wait")
    wait.add_argument("--job")
    wait.add_argument("--timeout-s", type=_ceiling)
    return parser


def _batch(parser: argparse.ArgumentParser, named: Path) -> dict:
    try:
        text = named.read_text(encoding="utf-8")
    except OSError:
        text = str(named)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        parser.error(f"--batch {named} is not JSON: {error}")
    if not isinstance(value, dict):
        parser.error(f"--batch {named} must be an object of role to sample ids")
    return value


def request(parser: argparse.ArgumentParser, parsed: argparse.Namespace) -> dict[str, object]:
    asked: dict[str, object] = {"op": parsed.op}
    if parsed.timeout_s is not None:
        asked["timeout_s"] = parsed.timeout_s
    if parsed.op == "wait":
        if parsed.job is not None:
            asked["job"] = parsed.job
        return asked
    if parsed.tag is not None:
        asked["tag"] = parsed.tag
    if parsed.batch is not None:
        if parsed.all or parsed.per_role is not None:
            parser.error("--batch cannot be combined with --all or --per-role")
        asked["batch"] = _batch(parser, parsed.batch)
        return asked
    if parsed.all and parsed.role is not None:
        parser.error("--all cannot be combined with --role")
    if parsed.all:
        asked["all"] = True
    if parsed.role is not None:
        asked["role"] = parsed.role
    if parsed.per_role is not None:
        asked["per_role"] = parsed.per_role
    return asked


def line(asked: dict[str, object]) -> bytes:
    return json.dumps(asked, separators=(",", ":"), sort_keys=True).encode() + b"\n"


def call(asked: dict[str, object], where: Path) -> str:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as pool:
        pool.connect(str(where))
        pool.sendall(line(asked))
        with pool.makefile("rb") as replies:
            return replies.readline().decode().rstrip("\n")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    asked = request(parser, parser.parse_args(sys.argv[1:] if argv is None else argv))
    try:
        available = SOCKET_PATH.is_socket()
    except OSError:
        available = False
    if not available:
        print(f"no pool at {SOCKET_PATH}: this tree is not running as an expansion episode", file=sys.stderr)
        return EXIT_NO_SOCKET
    try:
        reply = call(asked, SOCKET_PATH)
    except OSError as error:
        print(f"pool exchange failed at {SOCKET_PATH}: {error}", file=sys.stderr)
        return EXIT_REFUSED
    if not reply:
        print(f"no reply at {SOCKET_PATH}", file=sys.stderr)
        return EXIT_REFUSED
    print(reply)
    try:
        answered = json.loads(reply)
    except json.JSONDecodeError:
        return EXIT_REFUSED
    return EXIT_OK if isinstance(answered, dict) and answered.get("ok") is True else EXIT_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
