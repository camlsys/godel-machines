# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""Recover an episode from a provider safety guardrail instead of discarding its work."""

from __future__ import annotations

import enum

from shared.gateway.client import GatewayRefused, moderated


class Rung(enum.Enum):
    """One recovery step, named by how much of the conversation it gives up."""

    COMPACT = "compact"
    CLEAR = "clear"
    PUBLISH = "publish"


class Ladder:
    """Per-episode guardrail escalation state. Monotonic."""

    def __init__(self) -> None:
        """Start one episode at the foot of the ladder, having taken no rung."""
        self._taken: list[Rung] = []

    def next(self) -> Rung:
        """Hand out the rung this refusal escalates to, and record it."""
        rung = Rung.PUBLISH
        if not self._taken:
            rung = Rung.COMPACT
        elif len(self._taken) == 1:
            rung = Rung.CLEAR
        self._taken.append(rung)
        return rung

    @property
    def cleared(self) -> bool:
        """Whether this episode already restarted from its starting prompt."""
        return Rung.CLEAR in self._taken

    @property
    def taken(self) -> tuple[Rung, ...]:
        """The rungs handed out so far, in the order they were handed out."""
        return tuple(self._taken)


def guardrail_refusal(*, error: BaseException) -> bool:
    """Answer whether one raised error is a provider SAFETY GUARDRAIL refusal."""
    if not isinstance(error, GatewayRefused):
        return False
    return moderated(spoken=str(error))
