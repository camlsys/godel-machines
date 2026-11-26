# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""Call the local OpenAI-compatible gateway with standard-library HTTP."""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

CALL_SECONDS_VAR = "RQGM_AGENT_CALL_SECONDS"
BUDGET_HEADER = "X-RQGM-Budget-Remaining"
UNBOUNDED = "unbounded"
TRANSIENT = frozenset({429, 500, 502, 503, 504})
TRANSIENT_ATTEMPTS = 8
BACKOFF_CEILING = 60.0
MODERATED = "contentpolicyviolation"
REFUSAL_BODY = 2000
_SUMMARY = "detailed"
_CAPACITY = "capacity_unavailable"
_NOT_ACCEPTED = "provider capacity is assigned to accepted requests; this request was not accepted"


def _call_seconds() -> float | None:
    declared = os.environ.get(CALL_SECONDS_VAR)
    if declared == UNBOUNDED:
        return None
    try:
        seconds = float(declared)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{CALL_SECONDS_VAR} must declare positive finite seconds or {UNBOUNDED}") from error
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{CALL_SECONDS_VAR} must declare positive finite seconds or {UNBOUNDED}")
    return seconds


def _capacity_refused(*, status: int, spoken: bytes) -> bool:
    if status != HTTPStatus.TOO_MANY_REQUESTS:
        return False
    try:
        return json.loads(spoken) == {"error": {
            "type": _CAPACITY, "code": _CAPACITY, "message": _NOT_ACCEPTED,
        }}
    except (ValueError, UnicodeError):
        return False


def moderated(*, spoken: str) -> bool:
    """Answer whether a refusal is a moderation verdict on the content sent."""
    return MODERATED in "".join(spoken.split("_")).replace(" ", "").lower()


def weathered(*, status: int, spoken: str) -> bool:
    """Answer whether one refusal is the environment's rather than the candidate's."""
    return status == 0 or status in TRANSIENT or moderated(spoken=spoken)


class GatewayRefused(RuntimeError):
    """The gateway answered with a refusal this episode can't use."""

    def __init__(self, message: str, *, status: int) -> None:
        """Bind the message a reader sees to the status a caller branches on."""
        super().__init__(message)
        self.status = status


@dataclass(frozen=True, slots=True, kw_only=True)
class Surface:
    """One OpenAI API, named by the wire vocabulary it spells things with."""

    name: str
    path: str
    turns: str
    usage_in: str
    usage_out: str
    ceiling: str


CHAT = Surface(
    name="chat",
    path="/chat/completions",
    turns="messages",
    usage_in="prompt_tokens",
    usage_out="completion_tokens",
    ceiling="max_completion_tokens",
)
RESPONSES = Surface(
    name="responses",
    path="/responses",
    turns="input",
    usage_in="input_tokens",
    usage_out="output_tokens",
    ceiling="max_output_tokens",
)
SURFACES: Mapping[str, Surface] = {api.name: api for api in (CHAT, RESPONSES)}


@dataclass(frozen=True)
class Reply:
    """One parsed assistant completion."""

    content: str
    tool_calls: tuple[tuple[str, str, str], ...]
    total_tokens: int
    budget_remaining: float | None = None
    truncated: bool = False
    prompt_tokens: int = 0
    reasoning_tokens: int = 0
    reasoning: str = ""
    prompt_chars: int = 0


def _backoff(*, attempt: int) -> float:
    return min(2.0**attempt, BACKOFF_CEILING)


def _pause(*, attempt: int, refusal: urllib.error.HTTPError) -> float:
    try:
        return min(float(refusal.headers.get("Retry-After") if refusal.headers else None), BACKOFF_CEILING)
    except (TypeError, ValueError):
        return _backoff(attempt=attempt)


def _fetch(*, request: urllib.request.Request, timeout: float | None) -> tuple[bytes, str | None]:
    waited = 0.0
    attempt = 1
    while True:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read(), response.headers.get(BUDGET_HEADER)
        except urllib.error.HTTPError as refusal:
            with refusal:
                spoken, status = refusal.read()[:REFUSAL_BODY], refusal.code
                deferred = _capacity_refused(status=status, spoken=spoken)
                pause = _backoff(attempt=0) if deferred else _pause(attempt=attempt, refusal=refusal)
            if deferred:
                # No request was accepted; keep the draw and recheck its grant after a short wait.
                time.sleep(pause)
                waited += pause
                continue
            if status in TRANSIENT and attempt < TRANSIENT_ATTEMPTS:
                time.sleep(pause)
                waited += pause
                attempt += 1
                continue
            ladder = f" after {attempt} attempts over {waited:.0f}s" if attempt > 1 else ""
            raise GatewayRefused(f"gateway refused ({status}){ladder}: {spoken!r}", status=status) from refusal


def _ceiling_note(*, dropped: OSError, sent: int, timeout: float | None) -> str:
    if not isinstance(dropped, TimeoutError) or timeout is None:
        return f"posted {sent}B"
    return f"waited the {timeout:g}s client socket ceiling for a {sent}B request, so the gateway did not answer within it"


class Client:
    """Send chat requests through the route one grant serves."""

    def __init__(self, *, base_url: str, api_key: str, model: str, reasoning: str | None = None, surface: str) -> None:
        """Bind one gateway credential, model selection and API surface."""
        if (api := SURFACES.get(surface)) is None:
            raise ValueError(f"declared surface {surface!r} is not one of {sorted(SURFACES)}; a binding states which API it speaks")
        self._api, self._url = api, base_url.rstrip("/") + api.path
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self._model, self._reasoning = model, reasoning

    def chat(
        self,
        *,
        messages: Sequence[Mapping[str, object]],
        tools: Sequence[Mapping[str, object]] | None = None,
        output_tokens: int | None = None,
        **kwargs: object,
    ) -> Reply:
        """Send one chat request; a lost reply cannot establish that the gateway refused it."""
        timeout = _call_seconds()
        stated = kwargs if output_tokens is None else {**kwargs, self._api.ceiling: output_tokens}
        tag = os.environ["RQGM_AGENT_TAG"]
        payload = _request(
            api=self._api, messages=messages, tools=tools, model=self._model, reasoning=self._reasoning, agent_tag=tag, kwargs=stated,
        )
        body = json.dumps(payload).encode()
        request = urllib.request.Request(self._url, body, self._headers)
        try:
            raw, served = _fetch(request=request, timeout=timeout)
        except OSError as dropped:
            raise GatewayRefused(
                f"gateway transport failed: {type(dropped).__name__}: {dropped}"
                f" ({_ceiling_note(dropped=dropped, sent=len(body), timeout=timeout)});"
                " acceptance is unknown, so this request was not resent",
                status=0,
            ) from dropped
        return _parse(raw=raw, served=served, sent=len(body), api=self._api)


def _request(
    *,
    api: Surface,
    messages: Sequence[Mapping[str, object]],
    tools: Sequence[Mapping[str, object]] | None,
    model: str,
    reasoning: str | None,
    agent_tag: str,
    kwargs: Mapping[str, object],
) -> dict[str, object]:
    responses = api is RESPONSES
    turns = list(_items(messages=messages)) if responses else list(messages)
    payload: dict[str, object] = {api.turns: turns, "model": model, "agent_tag": agent_tag, **kwargs}
    if responses and (asked := payload.pop(CHAT.ceiling, None)) is not None:
        payload[api.ceiling] = asked
    if reasoning is not None:
        payload["reasoning" if responses else "reasoning_effort"] = {"effort": reasoning, "summary": _SUMMARY} if responses else reasoning
    if tools:
        payload["tools"] = [_tool(schema=schema) for schema in tools] if responses else list(tools)
    return payload


def _document(*, value: object) -> Mapping[str, object]:
    return {str(key): item for key, item in value.items()} if isinstance(value, dict) else {}


def _listed(*, value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _tool(*, schema: Mapping[str, object]) -> Mapping[str, object]:
    inner = schema.get("function")
    return {"type": "function", **_document(value=inner)} if isinstance(inner, dict) else schema


def _items(*, messages: Sequence[Mapping[str, object]]) -> Iterator[Mapping[str, object]]:
    for message in messages:
        if message.get("role") == "tool":
            yield {"type": "function_call_output", "call_id": message.get("tool_call_id"), "output": str(message.get("content") or "")}
            continue
        calls = message.get("tool_calls")
        if message.get("content") or not calls:
            yield {"role": message.get("role"), "content": message.get("content") or ""}
        for entry in _listed(value=calls):
            call = _document(value=entry)
            function = _document(value=call.get("function"))
            arguments = function.get("arguments") or "{}"
            yield {"type": "function_call", "call_id": call.get("id"), "name": function.get("name"), "arguments": arguments}


def _budget(*, served: str | None) -> float | None:
    if served is None or served == UNBOUNDED:
        return None
    try:
        return float(served)
    except ValueError:
        return 0.0


def _count(*keys: str, usage: object) -> int:
    for key in keys:
        if not isinstance(usage, dict):
            return 0
        usage = usage.get(key)
    return max(usage, 0) if isinstance(usage, int) and not isinstance(usage, bool) else 0


def _read_chat(*, data: Mapping[str, object]) -> tuple[str, str, tuple, bool, tuple[str, ...]]:
    choice = _document(value=_listed(value=data["choices"])[0])
    message = _document(value=choice["message"])
    calls = tuple(
        (str(call.get("id") or f"call_{n}"), str(function["name"]), str(function.get("arguments") or "{}"))
        for n, entry in enumerate(_listed(value=message.get("tool_calls")))
        for call in (_document(value=entry),)
        for function in (_document(value=call["function"]),)
    )
    content, reasoning = str(message.get("content") or ""), str(message.get("reasoning_content") or "")
    truncated = choice.get("finish_reason") == "length"
    return content, reasoning, calls, truncated, ("completion_tokens_details", "reasoning_tokens")


def _read_responses(*, data: Mapping[str, object]) -> tuple[str, str, tuple, bool, tuple[str, ...]]:
    content, reasoning, calls = [], [], []
    for entry in _listed(value=data.get("output")):
        item = _document(value=entry)
        kind = item.get("type")
        if kind == "function_call":
            calls.append((str(item.get("call_id") or f"call_{len(calls)}"), str(item.get("name") or ""), str(item.get("arguments") or "{}")))
        elif kind == "reasoning":
            reasoning += [str(_document(value=part).get("text") or "") for part in _listed(value=item.get("summary"))]
        elif kind == "message":
            parts = (_document(value=raw) for raw in _listed(value=item.get("content")))
            content += [str(part.get("text") or "") for part in parts if part.get("type") == "output_text"]
    truncated = _document(value=data.get("incomplete_details")).get("reason") == RESPONSES.ceiling
    thinking = ("output_tokens_details", "reasoning_tokens")
    return "".join(content), "\n".join(filter(None, reasoning)), tuple(calls), truncated, thinking


def _parse(*, raw: bytes, served: str | None = None, sent: int = 0, api: Surface = CHAT) -> Reply:
    data = json.loads(raw)
    content, reasoning, calls, truncated, thinking = (_read_responses if api is RESPONSES else _read_chat)(data=data)
    usage = data.get("usage")
    return Reply(
        content=content,
        tool_calls=calls,
        total_tokens=int(data["usage"]["total_tokens"]),
        budget_remaining=_budget(served=served),
        truncated=truncated,
        prompt_tokens=_count(api.usage_in, usage=usage),
        reasoning_tokens=_count(*thinking, usage=usage),
        reasoning=reasoning,
        prompt_chars=sent,
    )
