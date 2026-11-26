# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""Estimate prompt tokens from characters, and learn the ratio from the endpoint."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SEED_CHARS_PER_TOKEN = 3.0

RATIO_FLOOR = 2.0
RATIO_CEILING = 6.0

SMOOTHING = 0.25


def chars(*, value: object) -> int:
    """Return the serialized size of one wire value, exactly as the client sends it."""
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


class Estimator:
    """Convert characters to prompt tokens on a ratio learned from the endpoint."""

    def __init__(self, *, seed: float = SEED_CHARS_PER_TOKEN) -> None:
        """Start at a stated ratio, having seen nothing."""
        self._ratio = seed
        self.observations = 0
        self.rejected = 0

    @property
    def ratio(self) -> float:
        """The current characters-per-token figure, held inside its band."""
        return min(max(self._ratio, RATIO_FLOOR), RATIO_CEILING)

    def observe(self, *, sent: int, counted: int) -> bool:
        """Fold one endpoint-reported prompt count into the ratio."""
        if counted <= 0:
            self.rejected += 1
            return False
        seen = sent / counted
        if not RATIO_FLOOR <= seen <= RATIO_CEILING:
            self.rejected += 1
            return False
        self._ratio += SMOOTHING * (seen - self._ratio)
        self.observations += 1
        return True

    def tokens(self, *, text: str) -> int:
        """Return the token cost of one raw string, rounded up."""
        return math.ceil(len(text) / self.ratio)

    def payload(self, *, value: object) -> int:
        """Return the token cost of one whole wire value, rounded up."""
        return math.ceil(chars(value=value) / self.ratio)

    def messages(self, *, messages: Sequence[Mapping[str, object]]) -> int:
        """Return the summed token cost of a conversation."""
        return sum(self.payload(value=message) for message in messages)
