# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""Run upstream mini-swe-agent as the editable expansion arm."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from solve import EXPANSION_INSTANCE, GUARDRAIL_EXIT, LOOP, SUBMISSION, build_agent, episode_state, revision, untracked

from shared.gateway.budget import Budget
from shared.gateway.client import GatewayRefused, weathered
from shared.gateway.entrypoint import client_from_env
from shared.gateway.prompts import filled, resolve, shared_meta

__all__ = ["main"]

PROVIDER_EXIT = 75

ADDITION = filled(name="expansion_addition.txt", fields={"LOOP": LOOP, "SUBMISSION": SUBMISSION})

PROMPT = f"{shared_meta()}\n\n{ADDITION}"
INSTRUCTION = Path("meta/instruction.txt")
TASK = "Improve this tree."




def main() -> int:
    """Run one bounded mini-swe episode over the candidate workspace."""
    client = client_from_env()
    if client is None:
        return 1
    agent = build_agent(
        client=client,
        budget=Budget.declared(),
        system_template=resolve(default=PROMPT, carried=INSTRUCTION),
        workspace=Path.cwd(),
        # The shell states where this episode started and what it was asked, so a self-review
        # reads the work against that commit; without it the review seat refuses every ask.
        state=episode_state(baseline=revision(workspace=Path.cwd()), carried=untracked(workspace=Path.cwd()), task=TASK, tree=Path.cwd()),
        instance=EXPANSION_INSTANCE,
    )
    try:
        outcome = agent.run(TASK)
    except GatewayRefused as refused:
        if not weathered(status=refused.status, spoken=str(refused)):
            sys.stderr.write(f"[mini-swe] provider refused this request: {refused}\n")
            return 0
        sys.stderr.write(f"[mini-swe] provider refused, not the agent: {refused}\n")
        return PROVIDER_EXIT
    except Exception as failure:
        sys.stderr.write(f"[mini-swe] episode ended on {type(failure).__name__}: {failure}\n")
        return 0
    stop = str(outcome.get("exit_status", "unknown"))
    if stop == GUARDRAIL_EXIT:
        sys.stderr.write(
            "[mini-swe] a safety guardrail ended the episode; the tree still publishes\n",
        )
    sys.stderr.write(f"[mini-swe] stop={stop} turns={agent.n_calls} tokens={agent.charged}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
