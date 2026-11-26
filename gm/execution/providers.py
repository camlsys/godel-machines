# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""The two providers behind the gateway: a seeded scripted one for offline runs, LiteLLM for live models."""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING

from gm.ids import InfraWeather, canonical_json
from gm.execution.gateway import MAX_BODY, SURFACES, Api, declared_api, error_body

if TYPE_CHECKING:
    from pydantic import JsonValue

_SUBMISSION = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
_EDIT_COMMAND = (
    "mkdir -p task_agents && printf '%s\\n' 'stand-in step {step}' >> task_agents/STAND_IN.md"
)


def _surface(*, payload: Mapping[str, object]) -> Api:
    try:
        return declared_api(payload=payload)
    except ValueError:
        return SURFACES["chat"]


def _body(*, content: str, commands: tuple[str, ...], spend: tuple[int, int], digest: str, api: Api) -> bytes:
    """Render one scripted reply in its surface's wire shape, with usage."""
    arguments = [json.dumps({"command": command}, sort_keys=True) for command in commands]
    if api is SURFACES["responses"]:
        output: list[dict[str, object]] = [
            {"type": "function_call", "call_id": f"scripted-call-{position}", "name": "bash", "arguments": one}
            for position, one in enumerate(arguments)
        ]
        output.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": content}]})
        body: dict[str, object] = {"id": f"scripted-{digest[:12]}", "object": "response", "model": "scripted",
                                   "status": "completed", "output": output}
    else:
        message: dict[str, object] = {"role": "assistant", "content": content}
        if arguments:
            message["tool_calls"] = [
                {"id": f"scripted-call-{position}", "type": "function", "function": {"name": "bash", "arguments": one}}
                for position, one in enumerate(arguments)
            ]
        body = {"id": f"scripted-{digest[:12]}", "object": "chat.completion", "model": "scripted",
                "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if arguments else "stop"}]}
    tokens_in, tokens_out = spend
    body["usage"] = {api.usage_in: tokens_in, api.usage_out: tokens_out, "total_tokens": tokens_in + tokens_out}
    return canonical_json(body).encode()


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ScriptedProvider:
    """A seed and the request digest fix every reply: the first turn edits the tree, the next submits."""

    seed: int

    def complete(self, payload: Mapping[str, object]) -> tuple[int, bytes]:
        """Render the reply for one canonical request."""
        digest = hashlib.sha256(canonical_json(dict(payload)).encode()).hexdigest()
        hexed = hashlib.sha256(f"{self.seed}|{payload.get('model', '')}|{digest}".encode()).hexdigest()
        spend = (int(hexed[:4], 16) % 512 + 1, int(hexed[4:8], 16) % 512 + 1)
        api = _surface(payload=payload)
        turns = payload.get(api.turns)
        answered = sum(
            1 for turn in turns if isinstance(turn, Mapping) and turn.get("role") == "assistant"
        ) if isinstance(turns, list) else 0
        command = _EDIT_COMMAND.format(step=hexed[:16]) if answered == 0 else f"echo {_SUBMISSION}"
        return 200, _body(content=f"editing:{hexed[:16]}", commands=(command,), spend=spend, digest=digest, api=api)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Deployment:
    """One OpenAI-compatible endpoint and the credential it takes."""

    base_url: str
    api_key: str | None = dataclasses.field(repr=False)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class LiveProvider:
    """Send each call to the declared deployments in turn through LiteLLM; no retries, no admission control."""

    deployments: tuple[Deployment, ...]
    timeout_seconds: float = 600.0
    turns: itertools.count = dataclasses.field(default_factory=itertools.count, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Refuse an empty deployment list."""
        if not self.deployments:
            raise ValueError("a live provider requires declared deployments")

    def complete(self, payload: Mapping[str, JsonValue]) -> tuple[int, bytes]:
        """Send one governed request and return the upstream status and body.

        Raises:
            InfraWeather: The transport failed before the endpoint answered.
        """
        import litellm

        deployment = self.deployments[next(self.turns) % len(self.deployments)]
        api = declared_api(payload=payload)
        sent = dict(payload)
        # The ceiling and effort also travel in extra_body: some SDKs rewrite top-level limits.
        controlled: dict[str, JsonValue] = {}
        if api.ceiling in sent:
            controlled[api.ceiling] = sent[api.ceiling]
        if (effort := sent.pop("reasoning_effort", None)) is not None:
            if api.name == "responses":
                options = sent.pop("reasoning", None)
                controlled["reasoning"] = {**(options if isinstance(options, dict) else {}),
                                           "effort": str(effort), "summary": "detailed"}
            else:
                controlled["reasoning_effort"] = effort
        if controlled:
            extra = sent.get("extra_body")
            sent["extra_body"] = {**(extra if isinstance(extra, dict) else {}), **controlled}
        call = litellm.responses if api.name == "responses" else litellm.completion
        try:
            reply = call(**sent, api_base=deployment.base_url, api_key=deployment.api_key,
                         timeout=self.timeout_seconds, max_retries=0)
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if not isinstance(status, int):
                raise InfraWeather(f"provider transport failure: {exc}") from exc
            return status, error_body(message=str(exc), kind="upstream_error")
        body = (json.dumps(reply) if isinstance(reply, Mapping) else reply.model_dump_json()).encode()
        if len(body) > MAX_BODY:
            return 502, error_body(message="upstream response exceeded the size cap", kind="response_too_large")
        return 200, body
