# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""Funnel every episode's model calls through one host-metered gateway.

OpenAI-compatible `POST /chat/completions` and `POST /responses` front one `Provider`. A grant is an
opaque bearer credential over one binding and a weighted token allotment; every call is recorded on
its grant's dialogue (`dialogue.log`) and rendered into the node's `node-dialogue.md`.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import secrets
import threading
from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from enum import StrEnum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Final, Protocol, Self

from pydantic import JsonValue

from gm import experience
from gm.ids import UNBOUNDED, ActionId, BindingTag, Ceiling, Ident, InfraWeather, NodeId, canonical_json
from gm.ledger import ActionKind
from gm.ledger.events import DIALOGUE_ARTIFACT
from gm.spend import AxisName, SpendRows, Weight, canonical, merged, weighted

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from gm.domain import BindingConfig
    from gm.domain.results import Artifact

MAX_BODY = 16 * 1024 * 1024
BUDGET_HEADER = "X-RQGM-Budget-Remaining"
#: Past this many characters one call's block enters the gathered transcript truncated.
GATHERED_BLOCK_CAP: Final = 256 * 1024
_PAID = range(200, 300)
_UPSTREAM_REFUSAL = 502


@dataclass(frozen=True, slots=True, kw_only=True)
class Api:
    """One OpenAI API, named by the wire vocabulary it spells things with."""

    name: str
    path: str
    turns: str
    usage_in: str
    usage_out: str
    ceiling: str


SURFACES: Mapping[str, Api] = {api.name: api for api in (
    Api(name="chat", path="/chat/completions", turns="messages", usage_in="prompt_tokens",
        usage_out="completion_tokens", ceiling="max_completion_tokens"),
    Api(name="responses", path="/responses", turns="input", usage_in="input_tokens",
        usage_out="output_tokens", ceiling="max_output_tokens"),
)}
_ROUTES: Mapping[str, Api] = {prefix + api.path: api for api in SURFACES.values() for prefix in ("", "/v1")}
_CEILING_KEYS = frozenset({api.ceiling for api in SURFACES.values()} | {"max_tokens"})


def declared_api(*, payload: Mapping[str, object]) -> Api:
    """Return the one API a governed payload declares by the turns field it carries."""
    declared = [api for api in SURFACES.values() if api.turns in payload]
    if len(declared) != 1:
        raise ValueError(f"payload names no single API: expected one of {sorted(api.turns for api in SURFACES.values())}")
    return declared[0]


def error_body(*, message: str, kind: str) -> bytes:
    """Render the OpenAI-shaped error body every refusal on this gateway answers with."""
    return json.dumps({"error": {"message": message, "type": kind, "code": kind}}).encode()


class Provider(Protocol):
    """Answer one validated governed payload with an HTTP status and body."""

    def complete(self, payload: Mapping[str, JsonValue]) -> tuple[int, bytes]:
        """Forward one validated provider payload."""
        ...


class GrantCredential(Ident):
    """The only opaque bearer credential an episode receives."""


class _State(StrEnum):
    ACTIVE = "active"
    CLOSING = "closing"
    RETIRED = "retired"


@dataclass(frozen=True, slots=True, kw_only=True)
class Attribution:
    """The actions one grant's calls belong to, and which attempt on them the grant served."""

    kind: ActionKind
    actions: tuple[ActionId, ...]
    attempt_no: int = 1


# The dialogue record.

def dialogue_bytes(*, attribution: Attribution, node: str | None, pool: Mapping[str, JsonValue],
                   calls: Sequence[Mapping[str, JsonValue]]) -> bytes:
    """Return one grant's canonical `dialogue.log`: who it served, the turn pool and every call."""
    return canonical_json({
        "attribution": {"kind": str(attribution.kind), "actions": [str(one) for one in attribution.actions],
                        "attempt_no": attribution.attempt_no},
        "node": node, "pool": dict(pool), "calls": list(calls),
    }).encode()


def dialogue_calls(data: bytes) -> int:
    """Return how many calls one `dialogue.log` records."""
    return len(json.loads(data)["calls"])


def filed(*, dialogue: bytes | None, route: Callable[[bytes], Artifact]) -> dict[str, Artifact]:
    """Return the evidence mapping that carries one episode's dialogue, empty where it reached no model."""
    return {} if dialogue is None else {DIALOGUE_ARTIFACT: route(dialogue)}


def _outcome(status: int) -> str:
    if status == 0:
        return "abandoned"
    if status in _PAID:
        return "completed"
    return "refused" if status < 500 else "errored"


def _call_record(*, tag: BindingTag, agent_tag: str, status: int, invocation_id: int, call_seq: int,
                 digests: Sequence[str], request: Mapping[str, JsonValue], reply: bytes) -> dict[str, JsonValue]:
    try:
        document = json.loads(reply)
    except ValueError:
        document = None
    packed: dict[str, JsonValue] = {"reply_bytes": base64.b64encode(reply).decode("ascii")}
    if isinstance(document, dict | list):
        packed = {"reply": document} if canonical_json(document).encode() == reply else {"reply": document, **packed}
    return {
        "tag": str(tag), "agent_tag": agent_tag, "status": status, "outcome": _outcome(status),
        "invocation_id": invocation_id, "call_seq": call_seq, "digests": list(digests),
        "request": dict(request), "reply_digest": hashlib.sha256(reply).hexdigest(), **packed,
    }


_HEADER = re.compile(r"(?m)^(?=## call \d+ · .+ · .+ · \d+$)|^(?=\*\*[A-Za-z_][\w-]*\*\*$)")


def _quoted(body: str) -> str:
    """Escape a line that would read as one of this document's own headers."""
    return _HEADER.sub(lambda _: "\\", body)


def _turn_text(turn: Mapping[str, object]) -> str:
    """Return what one recorded turn said: its content, tool arguments or tool output."""
    for key in ("content", "arguments", "output", "text"):
        found = turn.get(key)
        if isinstance(found, str) and found:
            return found
        if found is not None and key != "content":
            return json.dumps(found, indent=1)
    if (body := turn.get("content")) is not None:
        return json.dumps(body, indent=1)
    return json.dumps({k: v for k, v in turn.items() if k not in {"role", "type"}}, indent=1)


def _spoken(reply: bytes) -> str | None:
    """Return what the model said in one reply, on either surface, or None."""
    try:
        document = json.loads(reply)
    except ValueError:
        return None
    if not isinstance(document, dict):
        return None
    found: list[Mapping[str, object]] = []
    for choice in document.get("choices") or []:
        if isinstance(choice, Mapping) and isinstance(message := choice.get("message"), Mapping):
            found.append(message)
    for item in document.get("output") or []:
        parts = item.get("content") if isinstance(item, Mapping) else None
        found.extend(part for part in (parts if isinstance(parts, list) else []) if isinstance(part, Mapping))
    spoken = [text for item in found for key in ("content", "text") if isinstance(text := item.get(key), str) and text]
    return "\n".join(spoken) or None


def _block(*, ordinal: int, conversation: str, tag: str, status: int, turns: Sequence[object], reply: bytes) -> str:
    """Render one call as its `node-dialogue.md` block: the header, the turns first seen here, the reply."""
    lines = [f"\n## call {ordinal} · {conversation} · {tag} · {status}\n"]
    for turn in turns:
        if isinstance(turn, dict):
            lines.append(f"\n**{turn.get('role') or turn.get('type') or '?'}**\n\n{_quoted(_turn_text(turn))}\n")
    if (answered := _spoken(reply)) is not None:
        lines.append(f"\n**assistant**\n\n{_quoted(answered)}\n")
    return "".join(lines)


# Grants.

@dataclass(slots=True, kw_only=True)
class _Grant:
    """One weighted allotment over one binding, and everything its calls recorded."""

    credential: GrantCredential
    tag: BindingTag
    model: str
    binding: BindingConfig
    allotment: Ceiling
    weights: Mapping[BindingTag, Mapping[AxisName, Weight]]
    attribution: Attribution
    mirror: Path | None = None
    node: NodeId | None = None
    seat: str = "seat"
    writing: AbstractContextManager[object] | None = None
    #: Told the grant's call count after each recorded call, with the transcript already written.
    on_call: Callable[[int], None] | None = None
    recording: threading.RLock = field(default_factory=threading.RLock)
    pool: dict[str, JsonValue] = field(default_factory=dict)
    calls: list[dict[str, JsonValue]] = field(default_factory=list)
    per_caller: dict[str, int] = field(default_factory=dict)
    conversations: dict[str, tuple[str, ...]] = field(default_factory=dict)
    gathered: str = ""
    arrivals: int = 0
    inflight: int = 0
    used: SpendRows = ()
    state: _State = _State.ACTIVE
    provider_refusal: str | None = None

    def exhausted(self) -> bool:
        return self.allotment != UNBOUNDED and self._spent() >= self.allotment

    def budget(self) -> dict[str, str]:
        """The remaining-allowance header served with every answer on this grant."""
        if self.allotment == UNBOUNDED:
            return {BUDGET_HEADER: UNBOUNDED}
        return {BUDGET_HEADER: repr(max(self.allotment - self._spent(), 0.0))}

    def _spent(self) -> float:
        return float(sum(sorted(weighted(entries=self.used, weights=self.weights).values())))

    def dialogue(self) -> bytes:
        node = None if self.node is None or not self.calls else str(self.node)
        return dialogue_bytes(attribution=self.attribution, node=node, pool=self.pool, calls=self.calls)


class _Refused(Exception):
    def __init__(self, *, status: int, message: str, kind: str, headers: Mapping[str, str]) -> None:
        super().__init__(status)
        self.reply = (status, error_body(message=message, kind=kind), dict(headers))


def _count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _usage(*, body: bytes, tag: BindingTag, api: Api) -> tuple[SpendRows | None, SpendRows, bytes]:
    """Read one reply's billable rows, its cached-input row, and the body served with `total_tokens`."""
    try:
        raw = json.loads(body)
    except ValueError:
        return None, (), body
    usage = raw.get("usage") if isinstance(raw, dict) else None
    if not isinstance(usage, dict):
        return None, (), body
    cached = None
    for key in ("input_tokens_details", "prompt_tokens_details"):
        details = usage.get(key)
        if isinstance(details, dict) and (cached := _count(details.get("cached_tokens"))) is not None:
            break
    observed = () if cached is None else canonical(((tag, "tokens_in_cached", cached),))
    prompt, completion = _count(usage.get(api.usage_in)), _count(usage.get(api.usage_out))
    if prompt is None or completion is None:
        return None, observed, body
    usage["total_tokens"] = prompt + completion
    return canonical(((tag, "tokens_in", prompt), (tag, "tokens_out", completion))), observed, json.dumps(raw).encode()


def _unfinished(*, body: bytes, api: Api) -> str | None:
    """Explain why a paid reply is unfinished, or None where it declares completion."""
    try:
        raw = json.loads(body)
    except ValueError:
        return "the provider returned no JSON response"
    if not isinstance(raw, dict):
        return "the provider returned no response object"
    if raw.get("error"):
        return "the provider returned an error in its response"
    if api.name == "responses":
        if (status := raw.get("status")) != "completed":
            return f"the provider response status is {status!r}, not 'completed'"
        details = raw.get("incomplete_details")
        if isinstance(details, dict) and details.get("reason") is not None:
            return "the provider declared incomplete_details on a completed response"
        output = raw.get("output")
        if not isinstance(output, list):
            return "the provider returned no output array"
        if any(not isinstance(item, dict) or item.get("status") not in (None, "completed") for item in output):
            return "the provider returned an unfinished output item"
        return None
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        return "the provider returned no completion choices"
    for choice in choices:
        reason = choice.get("finish_reason") if isinstance(choice, dict) else None
        if reason not in {"stop", "tool_calls", "function_call"}:
            return f"the provider choice ended with finish_reason={reason!r}"
    return None


def _payload(*, request: dict[str, JsonValue], grant: _Grant, api: Api) -> dict[str, JsonValue]:
    """Bind one request to its grant's binding: model, effort, kwargs and the output ceiling.

    A caller may name the binding's own model or effort, and may ask for fewer output tokens than
    the binding allows, never more.
    """
    payload = dict(request)
    for key, value in grant.binding.kwargs.items():
        payload.setdefault(key, value)
    payload.pop("agent_tag", None)
    effort = grant.binding.effort
    extra = payload.pop("extra_body", None)
    if extra is not None and not isinstance(extra, dict):
        raise ValueError("extra_body must be a JSON object")
    extra = dict(extra or {})
    reasoning: dict[str, JsonValue] = {}
    for fields in (payload, extra):
        options = fields.pop("reasoning", None)
        if options is not None and not isinstance(options, dict):
            raise ValueError("reasoning must be a JSON object")
        options = dict(options or {})
        named = fields.pop("model", None)
        levels = (fields.pop("reasoning_effort", None), options.pop("effort", None))
        if (named is not None and (not isinstance(named, str) or named not in {str(grant.tag), grant.model})) or any(
            level is not None and level != effort for level in levels
        ):
            raise _Refused(status=403, message="model/reasoning is not allowed", kind="model_not_permitted",
                           headers=grant.budget())
        reasoning.update(options)
    payload["model"] = grant.model
    if effort is not None:
        payload["reasoning_effort"] = effort
    if reasoning:
        payload["reasoning"] = reasoning
    asked = [fields.pop(key) for fields in (payload, extra) for key in sorted(_CEILING_KEYS) if key in fields]
    limits = [value for value in asked if isinstance(value, int) and not isinstance(value, bool) and value > 0]
    if grant.binding.output_tokens != UNBOUNDED:
        limits.append(grant.binding.output_tokens)
    if limits:
        payload[api.ceiling] = min(limits)
    elif asked:
        raise ValueError("output token limit must be a positive integer")
    if extra:
        payload["extra_body"] = extra
    return payload


class Gateway:
    """The gateway's HTTP server on a daemon thread, and every grant it serves."""

    def __init__(self, provider: Provider, *, bind: tuple[str, int] = ("127.0.0.1", 0)) -> None:
        """Start serving one provider on `bind`."""
        self._provider = provider
        self._grants: dict[str, _Grant] = {}
        self._lock = threading.Lock()
        self._drained = threading.Condition(self._lock)
        self._server = ThreadingHTTPServer(bind, _Handler)
        self._server.daemon_threads = True
        self._server.gateway = self
        self._thread = threading.Thread(target=lambda: self._server.serve_forever(poll_interval=0.01), daemon=True)
        self._thread.start()

    @property
    def port(self) -> int:
        """The bound TCP port."""
        return self._server.server_address[1]

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()

    def issue(
        self, *, tag: BindingTag, binding: BindingConfig, allotment: Ceiling,
        weights: Mapping[BindingTag, Mapping[AxisName, Weight]], attribution: Attribution,
        mirror: Path | None = None, node: NodeId | None = None, seat: str = "seat",
        writing: AbstractContextManager[object] | None = None, on_call: Callable[[int], None] | None = None,
    ) -> GrantCredential:
        """Mint one active grant; `mirror` is the `node-dialogue.md` its calls are gathered into, and `on_call`
        is told each recorded call's ordinal once its transcript is written."""
        endpoint = str(binding.endpoint)
        grant = _Grant(
            credential=GrantCredential(f"rqgm-{secrets.token_urlsafe(24)}"),
            tag=tag, model=endpoint.partition(":")[2] or endpoint, binding=binding, allotment=allotment,
            weights=weights, attribution=attribution, mirror=mirror, node=node, seat=seat, writing=writing,
            on_call=on_call,
        )
        with self._lock:
            self._grants[grant.credential] = grant
        return grant.credential

    def retire(self, credential: GrantCredential) -> tuple[SpendRows, bytes]:
        """Refuse new requests, wait for accepted calls; return the grant's usage and its dialogue."""
        with self._drained:
            grant = self._grants[credential]
            grant.state = _State.RETIRED
            while grant.inflight:
                self._drained.wait()
        with grant.recording:
            return grant.used, grant.dialogue()

    def _admitted(self, *, credential: str | None, body: bytes) -> tuple[_Grant, str, dict[str, JsonValue]]:
        with self._lock:
            grant = self._grants.get(credential or "")
            if grant is None:
                raise _Refused(status=401, message="unknown grant credential", kind="invalid_request_error", headers={})
            budget = grant.budget()
            if grant.provider_refusal is not None:
                raise _Refused(status=_UPSTREAM_REFUSAL, message=grant.provider_refusal, kind="upstream_error",
                               headers=budget)
            if grant.state is not _State.ACTIVE:
                raise _Refused(status=401, message="inactive grant credential", kind="invalid_request_error",
                               headers=budget)
        try:
            request = json.loads(body)
        except ValueError:
            request = None
        if not isinstance(request, dict):
            raise _Refused(status=400, message="request body is not valid JSON", kind="invalid_request_error",
                           headers=budget)
        agent_tag = request.get("agent_tag")
        if not isinstance(agent_tag, str) or not agent_tag.strip():
            raise _Refused(status=400, message="request states no agent_tag", kind="agent_tag_missing", headers=budget)
        return grant, agent_tag.strip(), request

    def handle(self, *, path: str, credential: str | None, body: bytes) -> tuple[int, bytes, dict[str, str]]:
        """Validate, route, meter, record and answer one request."""
        api = _ROUTES.get(path)
        try:
            if api is None:
                raise _Refused(status=404, message="no such route on this gateway", kind="invalid_request_error",
                               headers={})
            grant, caller, request = self._admitted(credential=credential, body=body)
            payload = _payload(request=request, grant=grant, api=api)
        except _Refused as refused:
            return refused.reply
        except ValueError as invalid:
            return 400, error_body(message=str(invalid), kind="invalid_request_error"), grant.budget()
        with self._lock:
            grant.arrivals += 1
            invocation_id = grant.arrivals
            refusal = None
            if grant.provider_refusal is not None:
                refusal = _UPSTREAM_REFUSAL, error_body(message=grant.provider_refusal, kind="upstream_error")
            elif grant.state is not _State.ACTIVE or grant.exhausted():
                grant.state = _State.CLOSING if grant.state is _State.ACTIVE else grant.state
                refusal = 401, error_body(message="grant is retired", kind="invalid_request_error")
            else:
                grant.inflight += 1
        if refusal is not None:
            self._record(grant=grant, caller=caller, api=api, payload=payload, answer=refusal, invocation_id=invocation_id)
            return *refusal, grant.budget()
        usage: SpendRows | None = None
        visible = None
        try:
            try:
                status, original = self._provider.complete(payload)
            except InfraWeather as weather:
                answer = _UPSTREAM_REFUSAL, error_body(message=f"upstream provider failure: {weather}",
                                                       kind="upstream_error")
                cached: SpendRows = ()
            else:
                usage, cached, visible = _usage(body=original, tag=grant.tag, api=api)
                answer = status, original
                if status in _PAID and (reason := _unfinished(body=original, api=api)):
                    message = f"{reason}; this grant cannot continue after an unfinished provider response"
                    with self._lock:
                        grant.provider_refusal = grant.provider_refusal or message
                    answer, visible = (_UPSTREAM_REFUSAL, original), error_body(message=message, kind="upstream_error")
                elif usage is None and status in _PAID:
                    answer = _UPSTREAM_REFUSAL, original
                    visible = error_body(message="no valid usage in provider response", kind="usage_missing")
            self._record(grant=grant, caller=caller, api=api, payload=payload, answer=answer, invocation_id=invocation_id)
        finally:
            self._release(grant=grant, usage=None if usage is None else merged(usage, cached))
        return answer[0], answer[1] if visible is None else visible, grant.budget()

    def _record(self, *, grant: _Grant, caller: str, api: Api, payload: Mapping[str, JsonValue],
                answer: tuple[int, bytes], invocation_id: int) -> None:
        """Record one call on its grant's dialogue and gather its block into the node's transcript."""
        turns = payload.get(api.turns)
        turns = turns if isinstance(turns, list) else []
        status, reply = answer
        digests = tuple(hashlib.sha256(canonical_json(turn).encode()).hexdigest() for turn in turns)
        with grant.recording, grant.writing if grant.writing is not None else nullcontext():
            for digest, turn in zip(digests, turns, strict=True):
                grant.pool.setdefault(digest, turn)
            call_seq = grant.per_caller.get(caller, 0)
            grant.per_caller[caller] = call_seq + 1
            grant.calls.append(_call_record(
                tag=grant.tag, agent_tag=caller, status=status, invocation_id=invocation_id, call_seq=call_seq,
                digests=digests, request={key: value for key, value in payload.items() if key != api.turns},
                reply=reply,
            ))
            # A conversation resends everything it said so far: the first known chain that prefixes
            # these turns names it, and only the turns past that prefix are rendered.
            named, written = next(
                ((label, len(chain)) for label, chain in grant.conversations.items()
                 if digests[:len(chain)] == chain), (f"seat-{len(grant.conversations)}", 0),
            )
            grant.conversations[named] = digests
            rendered = _block(ordinal=len(grant.calls) - 1, conversation=named, tag=str(grant.tag), status=status,
                              turns=turns[written:], reply=reply)
            if grant.mirror is not None:
                if len(rendered) > GATHERED_BLOCK_CAP:
                    rendered = (f"{rendered[:GATHERED_BLOCK_CAP]}\n[{len(rendered) - GATHERED_BLOCK_CAP}"
                                " bytes of this call elided; the run's ledger holds it whole]\n")
                grant.gathered += f"\n### {grant.seat}\n{rendered}"
                grant.mirror.parent.mkdir(parents=True, exist_ok=True)
                landing = grant.mirror.parent / experience.ARRIVING_NAME
                landing.write_text(grant.gathered, encoding="utf-8")
                landing.replace(grant.mirror)
            if grant.on_call is not None:
                grant.on_call(len(grant.calls))

    def _release(self, *, grant: _Grant, usage: SpendRows | None) -> None:
        """Release one accepted call, meter its usage, and close a spent grant."""
        with self._drained:
            grant.inflight -= 1
            if usage is not None:
                grant.used = merged(grant.used, usage)
                sequence = grant.binding.sequence_tokens
                if sequence != UNBOUNDED and sum(amount for _, _, amount in usage) > sequence:
                    grant.state = _State.CLOSING
            if grant.exhausted() and grant.state is _State.ACTIVE:
                grant.state = _State.CLOSING
            if grant.inflight == 0:
                self._drained.notify_all()


class _Handler(BaseHTTPRequestHandler):
    """Translate HTTP requests into gateway calls, without request logging."""

    timeout = 60.0

    def do_POST(self) -> None:
        auth = self.headers.get("Authorization", "")
        credential = auth.removeprefix("Bearer ") if auth.startswith("Bearer ") else None
        raw = self.headers.get("Content-Length", "0")
        if not raw.isdigit() or int(raw) > MAX_BODY:
            status, reply, headers = 400, error_body(message="invalid or oversized Content-Length",
                                                     kind="invalid_request_error"), {}
        else:
            try:
                body = self.rfile.read(int(raw))
            except OSError:
                return
            status, reply, headers = self.server.gateway.handle(path=self.path, credential=credential, body=body)
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(reply)
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, format: str, *args: object) -> None:
        """Suppress per-request logging."""
