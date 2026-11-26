# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""Hold one episode's declared spend policy, and the sentences it puts to the agent."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

from shared.gateway.entrypoint import declared_budget, declared_nudge
from shared.gateway.prompts import carried

AXES = frozenset({"tokens"})

WARNING = carried(name="budget_warning.txt")
TURNS = carried(name="budget_turns.txt")


@dataclass(frozen=True)
class Budget:
    """The spend policy one host declared: the allotment, what the agent may see, when to warn."""

    tokens: int | None
    exposed: frozenset[str]
    nudge_at: float | None
    _at_last_turn: dict[str, float] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        """Refuse a policy this module couldn't act on."""
        if unknown := self.exposed - AXES:
            raise ValueError(f"unknown exposed budget axes: {sorted(unknown)}")
        if self.nudge_at is not None and not 0 < self.nudge_at <= 1:
            raise ValueError(f"nudge_at must be within (0, 1]: {self.nudge_at!r}")

    @classmethod
    def declared(cls) -> Budget:
        """Return the policy in this episode's environment; the allotment is the key the meter cuts on."""
        return cls(
            tokens=declared_budget(raw=os.environ["RQGM_AGENT_TOKEN_ALLOTMENT"]),
            exposed=frozenset(filter(None, os.environ["RQGM_AGENT_BUDGET_EXPOSE"].split(","))),
            nudge_at=declared_nudge(raw=os.environ["RQGM_AGENT_BUDGET_NUDGE"]),
        )

    def remaining(self, *, served: float | None, spent: int) -> float | None:
        """Return what remains of the allotment, preferring the meter's own reading."""
        if served is not None:
            return served
        return None if self.tokens is None else self.tokens - spent

    def used(self, *, remaining: float | None, spent: int) -> float:
        """Return how much of the allotment is gone."""
        if self.tokens is not None and remaining is not None:
            return self.tokens - remaining
        return spent

    def exposure(self, *, remaining_tokens: float | None) -> list[str]:
        """Return the remainder fragments the agent may see, in tokens and in turns."""
        fragments = []
        if "tokens" in self.exposed and remaining_tokens is not None:
            fragments.append(f"~{remaining_tokens} budget tokens remain")
            if (turns := self.turns_left(remaining_tokens=remaining_tokens)) is not None:
                fragments.append(TURNS.format(turns=turns))
        return fragments

    def turns_left(self, *, remaining_tokens: float) -> int | None:
        """Return the turns the last turn's cost leaves, or `None` before a cost is known."""
        if self.tokens is None or (previous := self._at_last_turn.get("tokens")) is None:
            return None
        step = self.tokens - remaining_tokens - previous
        return None if step <= 0 else math.ceil(max(0.0, remaining_tokens) / step)

    def binding(self, *, taken: int, step_limit: int, remaining_tokens: float | None) -> str:
        """Return the one sentence naming the limit that ends this episode first."""
        turns = (
            None if remaining_tokens is None else self.turns_left(remaining_tokens=remaining_tokens)
        )
        steps_left = step_limit - taken if step_limit else None
        if turns is not None and (steps_left is None or turns < steps_left):
            return (
                f"~{turns} turn(s) left. The token allotment is what ends this episode, not"
                " the step count; the last turn starts inside the allotment and is billed"
                " whole, so the episode stops past the allotment rather than on it"
            )
        if steps_left is None:
            return ""
        return (
            f"{steps_left} turn(s) left: step {taken} of {step_limit}, and the step limit is"
            " what ends this episode first"
        )

    def nudge(self, *, used_tokens: float, warned: set[str]) -> list[str]:
        """Record this turn's spend, and warn once when the next turn would cross the threshold."""
        if self.tokens is None:
            return []
        step = max(0.0, used_tokens - self._at_last_turn.get("tokens", used_tokens))
        self._at_last_turn["tokens"] = used_tokens
        if self.nudge_at is None or "tokens" in warned:
            return []
        if (used_tokens + step) / self.tokens < self.nudge_at:
            return []
        warned.add("tokens")
        return [
            WARNING.format(
                axis="tokens",
                fraction=self.nudge_at,
                turns=int((self.tokens - used_tokens) // step) if step else 0,
            ),
        ]
