# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""Fit conversations inside a declared context window without touching pinned turns."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from shared.gateway.prompts import carried

Message = dict[str, object]
PIN = "rqgm_pin"
SUMMARY_SYSTEM = carried(name="summary_system.txt")
SUMMARY_PREFIX = carried(name="summary_prefix.txt")
_MARKER = "[context-compaction: {n} earlier messages elided]"
_MIN_TAIL = 2


class ContextOverflowError(RuntimeError):
    def __init__(self, *, measured: int, limit: int, pinned: int, floor: int) -> None:
        super().__init__(f"context overflow: {measured} estimated tokens > limit {limit}; {pinned} held by the pinned tier and a floor of {floor}")
        self.measured, self.limit, self.pinned, self.floor = measured, limit, pinned, floor


@dataclass(frozen=True)
class CompactionReport:
    fired: bool
    clips: int
    clears: int
    dropped: int
    emergency_clips: int
    summarized: bool
    before: int
    after: int


@dataclass(frozen=True)
class CompactionConfig:
    limit_tokens: int
    estimator: object
    target_ratio: float = 0.5
    max_msg_tokens: int = 400
    keep_tail_msgs: int = 6
    emergency_msg_tokens: int = 96
    clear_at_least: int = 512
    eager_ratio: float = 0.6
    summarizer: Callable[..., str] | None = None

    def __post_init__(self) -> None:
        if self.limit_tokens <= 0 or not 0 < self.target_ratio < 1:
            raise ValueError("compaction limits must be positive fractions")
        if not self.target_ratio <= self.eager_ratio < 1:
            raise ValueError("eager_ratio must be in [target_ratio, 1)")
        if self.keep_tail_msgs < _MIN_TAIL or not self.max_msg_tokens >= self.emergency_msg_tokens > 0:
            raise ValueError("invalid compaction message limits")

    @property
    def target(self) -> int:
        return int(self.limit_tokens * self.target_ratio)


def pin(*, message: Message) -> Message:
    return {**message, PIN: True}


def _text(message: Mapping[str, object]) -> str:
    value = message.get("content")
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)


def clip(*, text: str, max_tokens: int, estimator: object) -> str:
    if estimator.tokens(text=text) <= max_tokens:
        return text
    budget = max(0, int(max_tokens * estimator.ratio))
    marker = f"\n[compaction: {max(0, len(text) - budget)} chars elided]\n"
    room = max(0, budget - len(marker))
    head = (room + 1) // 2
    return text[:head] + marker + (text[-(room - head):] if room > head else "")


def units(*, messages: Sequence[Mapping[str, object]]) -> list[list[int]]:
    groups, pending = [], None
    for index, message in enumerate(messages):
        if message.get("role") == "assistant" and message.get("tool_calls"):
            groups.append([index])
            pending = len(groups) - 1
        elif message.get("role") == "tool" and pending is not None:
            groups[pending].append(index)
        else:
            groups.append([index])
            pending = None
    return groups


def pairing_valid(*, messages: Sequence[Mapping[str, object]]) -> bool:
    expected: set[str] = set()
    for message in messages:
        if message.get("role") == "assistant":
            expected = {str(call.get("id")) for call in message.get("tool_calls", []) if isinstance(call, dict)}
        elif message.get("role") == "tool":
            if str(message.get("tool_call_id")) not in expected:
                return False
            expected.discard(str(message.get("tool_call_id")))
        elif expected:
            return False
    return not expected


def _idle(estimate: int) -> CompactionReport:
    return CompactionReport(False, 0, 0, 0, 0, False, estimate, estimate)


def compact(*, messages: Sequence[Message], config: CompactionConfig, eager: bool = False) -> tuple[list[Message], CompactionReport]:
    estimator = config.estimator
    before = estimator.messages(messages=messages)
    threshold = int(config.eager_ratio * config.limit_tokens) if eager else config.limit_tokens
    if before <= threshold:
        return list(messages), _idle(before)
    grouped = units(messages=messages)
    pinned = {i for i, message in enumerate(messages) if message.get(PIN)}
    keep_groups = [group for group in grouped if pinned & set(group)]
    tail = next((group for group in reversed(grouped) if not pinned & set(group)), [])
    keep = sorted({i for group in keep_groups for i in group} | set(tail))
    dropped = [i for i in range(len(messages)) if i not in keep]
    history = "\n\n".join(_text(messages[i]) for i in dropped)
    summary = config.summarizer(history=history).strip() if config.summarizer and history.strip() else ""
    marker = _MARKER.format(n=len(dropped)) + (f"\n{summary}" if summary else "")
    result = [messages[i] for i in keep if i < (tail[0] if tail else len(messages))]
    if dropped:
        result.append({"role": "user", "content": marker})
    result.extend(messages[i] for i in tail)
    after = estimator.messages(messages=result)
    emergency = 0
    if after > config.limit_tokens:
        for index, message in enumerate(result):
            if message.get(PIN):
                continue
            shorter = clip(text=_text(message), max_tokens=config.emergency_msg_tokens, estimator=estimator)
            if shorter != _text(message):
                result[index] = {**message, "content": shorter}
                emergency += 1
        after = estimator.messages(messages=result)
    if after > config.limit_tokens:
        held = sum(estimator.payload(value=message) for message in messages if message.get(PIN))
        raise ContextOverflowError(measured=after, limit=config.limit_tokens, pinned=held, floor=config.emergency_msg_tokens)
    report = CompactionReport(True, 0, 0, len(dropped), emergency, bool(summary), before, after)
    if before <= config.limit_tokens and before - after < config.clear_at_least:
        return list(messages), _idle(before)
    return result, report


class Summarizer:
    def __init__(self, *, chat: Callable[..., object], note: Callable[..., None]) -> None:
        self._chat, self._note, self.failed, self.calls = chat, note, False, 0

    def __call__(self, *, history: str) -> str:
        if self.failed or not history.strip():
            return ""
        self.calls += 1
        try:
            reply = self._chat(messages=[{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": history}])
            text = str(getattr(reply, "content", "") or "").strip()
        except Exception as error:
            self._note(entry="compaction", body=f"summarizer failed, falling back to dropping: {error}")
            self.failed = True
            return ""
        if not text:
            self.failed = True
            return ""
        self._note(entry="compaction summary", body=text)
        return f"{SUMMARY_PREFIX}\n\n{text}"


def limit_tokens(*, sequence: int | None, output: int | None, overhead: int) -> int | None:
    if sequence is None:
        return None
    return max(sequence - (math.ceil(sequence / 4) if output is None else output) - overhead, 1)
