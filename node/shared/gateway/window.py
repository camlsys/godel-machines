# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""Fit a growing conversation to the context window a binding declared, under one stated policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from shared.gateway.compaction import CompactionConfig, CompactionReport, compact, limit_tokens

if TYPE_CHECKING:
    from collections.abc import Sequence

    from shared.gateway.compaction import Message, SummarizeSpan
    from shared.gateway.estimate import Estimator

TARGET_RATIO = 0.5
EAGER_RATIO = 0.6
MAX_MSG_TOKENS = 400
KEEP_TAIL_MSGS = 6
EMERGENCY_MSG_TOKENS = 96
CLEAR_AT_LEAST = 512


class Window:
    """Apply one episode's compaction policy to a conversation."""

    def __init__(
        self,
        *,
        sequence: int | None,
        output: int | None,
        estimator: Estimator,
        summarizer: SummarizeSpan | None = None,
    ) -> None:
        """Bind the policy to one episode's declared ceilings and estimator."""
        self.sequence = sequence
        self.output = output
        self.estimator = estimator
        self.summarizer = summarizer

    @property
    def declared(self) -> bool:
        """Whether the binding stated a context window."""
        return self.sequence is not None

    def config(self, *, overhead: int) -> CompactionConfig | None:
        """Return the ceilings one compaction runs under, or `None` for no window."""
        limit = limit_tokens(sequence=self.sequence, output=self.output, overhead=overhead)
        if limit is None:
            return None
        return CompactionConfig(
            limit_tokens=limit,
            estimator=self.estimator,
            summarizer=self.summarizer,
            target_ratio=TARGET_RATIO,
            eager_ratio=EAGER_RATIO,
            max_msg_tokens=MAX_MSG_TOKENS,
            keep_tail_msgs=KEEP_TAIL_MSGS,
            emergency_msg_tokens=EMERGENCY_MSG_TOKENS,
            clear_at_least=CLEAR_AT_LEAST,
        )

    def fitted(
        self,
        *,
        messages: Sequence[Message],
        overhead: int = 0,
        eager: bool = False,
    ) -> tuple[list[Message], CompactionReport | None]:
        """Return the conversation fitted to this episode's window, and what it cost."""
        config = self.config(overhead=overhead)
        if config is None:
            return list(messages), None
        return compact(messages=messages, config=config, eager=eager)
