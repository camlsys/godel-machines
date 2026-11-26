# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""Read the episode environment the host states, and build a gateway client from it."""

from __future__ import annotations

import os
import sys

from shared.gateway.client import Client

REQUIRED = (
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "RQGM_AGENT_MODEL",
    "RQGM_AGENT_TOKEN_ALLOTMENT",
    "RQGM_AGENT_SURFACE",
)

DECLARED = (
    "RQGM_AGENT_BUDGET_EXPOSE",
    "RQGM_AGENT_BUDGET_NUDGE",
    "RQGM_AGENT_MAX_SEQUENCE_TOKENS",
    "RQGM_AGENT_MAX_OUTPUT_TOKENS",
)

NOT_DISPATCHED = "rqgm-episode-not-dispatched"

UNBOUNDED = "unbounded"


def declared_budget(*, raw: str) -> int | None:
    """Return a declared token figure, or `None` only for a host that wrote `unbounded`."""
    return None if raw == UNBOUNDED else int(raw)


def declared_nudge(*, raw: str) -> float | None:
    """Return the warning threshold as a used-fraction, or `None` for `off`."""
    return None if raw == "off" else float(raw)


def client_from_env() -> Client | None:
    """Return a gateway client, or `None` after naming the episode variables the host left unstated."""
    missing = [key for key in REQUIRED if not os.environ.get(key)]
    missing += [key for key in DECLARED if key not in os.environ]
    if missing:
        sys.stderr.write(f"missing required episode env: {missing}\n{NOT_DISPATCHED}\n")
        return None
    return Client(
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
        model=os.environ["RQGM_AGENT_MODEL"],
        reasoning=os.environ.get("RQGM_AGENT_REASONING"),
        surface=os.environ["RQGM_AGENT_SURFACE"],
    )
