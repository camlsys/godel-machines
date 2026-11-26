"""Command line: `python -m gm run|verify`."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from gm.ids import UNBOUNDED


def _provider(args: argparse.Namespace):
    if args.provider == "scripted":
        from gm.execution.providers import ScriptedProvider

        return ScriptedProvider(seed=args.seed)
    from gm.execution.providers import Deployment, LiveProvider

    keys = args.provider_key_env * len(args.provider_url) if len(args.provider_key_env) == 1 else args.provider_key_env
    return LiveProvider(
        deployments=tuple(
            Deployment(base_url=url, api_key=None if key == "none" else os.environ[key])
            for url, key in zip(args.provider_url, keys, strict=True)
        ),
    )


def command_run(args: argparse.Namespace) -> int:
    from gm.execution.runner import Envelope
    from gm.run import load_form, run

    form = load_form(args.domain)
    store = Path(args.store)
    if args.fresh and store.exists():
        shutil.rmtree(store)
    store.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="gm-run-", dir=Path.home() / "tmp" if (Path.home() / "tmp").is_dir() else None))
    os.environ["TMPDIR"] = str(scratch)
    tempfile.tempdir = str(scratch)
    report = run(
        form,
        store=store,
        provider=_provider(args),
        envelope=Envelope(
            allotment=args.episode_tokens if args.episode_tokens == UNBOUNDED else int(args.episode_tokens),
            nudge_at=None if args.nudge_at == "off" else float(args.nudge_at),
        ),
        edge=args.edge,
        surface=args.surface,
        seed=args.seed,
        workers=args.workers,
        cpus=args.cpus,
        budget=args.budget,
    )
    print(json.dumps({"run": str(report.run), "events": len(report.event_ids), "nodes": report.nodes, "pick": report.pick}))
    return 0


def command_verify(args: argparse.Namespace) -> int:
    from gm.verify import verify

    result = verify(store=Path(args.store))
    print(json.dumps(result) if args.json else "\n".join(f"{k}: {v}" for k, v in result.items()))
    return 0 if result["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gm")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run one search")
    run.add_argument("--domain", required=True, help="domain/form, e.g. toy/mini_swe")
    run.add_argument("--store", required=True)
    run.add_argument("--budget", type=int, required=True, help="evaluations the search may count")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--workers", type=int, default=1, help="concurrent jobs; 1 is serial and reproducible")
    run.add_argument("--cpus", type=int, default=None, help="host CPUs jobs are admitted against (default: workers)")
    run.add_argument("--surface", default="responses", choices=("chat", "responses"))
    run.add_argument("--provider", default="scripted", choices=("scripted", "live"))
    run.add_argument("--provider-url", action="append", default=[])
    run.add_argument("--provider-key-env", action="append", default=[])
    run.add_argument("--episode-tokens", default=UNBOUNDED)
    run.add_argument("--nudge-at", default="off")
    run.add_argument("--edge", required=True, help="image id of the gateway relay container")
    run.add_argument("--fresh", action="store_true")
    run.set_defaults(handler=command_run)
    verify = sub.add_parser("verify", help="check a store's ledger: codec identity, checkpoint chain, fold")
    verify.add_argument("store")
    verify.add_argument("--json", action="store_true")
    verify.set_defaults(handler=command_verify)
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
