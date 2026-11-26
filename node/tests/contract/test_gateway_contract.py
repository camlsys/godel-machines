# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""The gateway seam: the environment a grant is read from, the client, and the prompt files."""

import http.server
import json
import math
import threading

import pytest

from shared import reply
from shared.gateway import client as wire
from shared.gateway import entrypoint, estimate, prompts

CHAT_REPLY = {
    "choices": [{"message": {"content": "hello", "reasoning_content": "why"}, "finish_reason": "stop"}],
    "usage": {"total_tokens": 9, "prompt_tokens": 6, "completion_tokens_details": {"reasoning_tokens": 2}},
}
RESPONSES_REPLY = {
    "output": [
        {"type": "reasoning", "summary": [{"text": "thinking"}]},
        {"type": "message", "content": [{"type": "output_text", "text": "hi "}, {"type": "output_text", "text": "there"}]},
        {"type": "function_call", "call_id": "c1", "name": "bash", "arguments": '{"command": "ls"}'},
    ],
    "usage": {"total_tokens": 12, "input_tokens": 8, "output_tokens_details": {"reasoning_tokens": 3}},
    "incomplete_details": {"reason": "max_output_tokens"},
}


class _Gateway(http.server.BaseHTTPRequestHandler):
    """Answer each POST with the next scripted (status, body, headers), remembering the request."""

    script: list = []
    seen: list = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.seen.append((self.path, json.loads(self.rfile.read(length))))
        status, body, headers = self.script.pop(0)
        if status is None:
            self.close_connection = True
            return
        payload = json.dumps(body).encode()
        self.send_response(status)
        for name, value in headers.items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_):
        """Keep the test output quiet."""


@pytest.fixture
def gateway(monkeypatch):
    """A loopback gateway; the test scripts its answers and reads what it was sent."""
    _Gateway.script, _Gateway.seen = [], []
    server = http.server.HTTPServer(("127.0.0.1", 0), _Gateway)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("RQGM_AGENT_TAG", "test-tag")
    monkeypatch.setenv(wire.CALL_SECONDS_VAR, "1")
    monkeypatch.setattr(wire.time, "sleep", lambda _: None)
    try:
        yield _Gateway, f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()


def test_a_grant_is_read_whole_or_refused_naming_every_missing_key(grant, monkeypatch, capsys):
    """client_from_env builds a client from the stated environment, else prints the gaps and a marker."""
    client = entrypoint.client_from_env()
    assert isinstance(client, wire.Client) and client._url == "http://gateway.invalid/v1/chat/completions"
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.delenv("RQGM_AGENT_BUDGET_NUDGE")
    assert entrypoint.client_from_env() is None
    said = capsys.readouterr().err
    assert "['OPENAI_API_KEY', 'RQGM_AGENT_BUDGET_NUDGE']" in said and entrypoint.NOT_DISPATCHED in said
    assert entrypoint.declared_budget(raw="unbounded") is None and entrypoint.declared_budget(raw="12") == 12
    assert entrypoint.declared_nudge(raw="off") is None and entrypoint.declared_nudge(raw="0.5") == 0.5
    with pytest.raises(ValueError, match="not one of"):
        wire.Client(base_url="http://x", api_key="k", model="m", surface="grpc")


def test_a_chat_call_carries_the_tag_and_parses_the_reply(gateway):
    """One POST on the chat surface: the agent tag rides along, and content, usage and budget come back."""
    served, url = gateway
    served.script.append((200, CHAT_REPLY, {wire.BUDGET_HEADER: "100"}))
    answer = wire.Client(base_url=url, api_key="k", model="m", reasoning="low", surface="chat").chat(
        messages=[{"role": "user", "content": "hi"}],
        output_tokens=50,
    )
    assert answer.content == "hello" and answer.reasoning == "why" and answer.tool_calls == ()
    assert (answer.total_tokens, answer.prompt_tokens, answer.reasoning_tokens) == (9, 6, 2)
    assert answer.budget_remaining == 100.0 and answer.prompt_chars > 0 and not answer.truncated
    path, sent = served.seen[0]
    assert path == "/v1/chat/completions"
    assert sent["agent_tag"] == "test-tag" and sent["model"] == "m"
    assert sent["max_completion_tokens"] == 50 and sent["reasoning_effort"] == "low"


def test_the_responses_surface_speaks_its_own_vocabulary(gateway):
    """The same call on the responses surface: items instead of messages, and calls read back out."""
    served, url = gateway
    served.script.append((200, RESPONSES_REPLY, {}))
    client = wire.Client(base_url=url, api_key="k", model="m", reasoning="high", surface="responses")
    answer = client.chat(
        messages=[
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c0", "function": {"name": "bash", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c0", "content": "out"},
        ],
        tools=[{"type": "function", "function": {"name": "bash", "parameters": {}}}],
        output_tokens=20,
    )
    assert answer.content == "hi there" and answer.reasoning == "thinking" and answer.truncated
    assert answer.tool_calls == (("c1", "bash", '{"command": "ls"}'),)
    assert (answer.total_tokens, answer.prompt_tokens, answer.reasoning_tokens) == (12, 8, 3)
    path, sent = served.seen[0]
    assert path == "/v1/responses" and sent["max_output_tokens"] == 20
    assert [item["type"] for item in sent["input"]] == ["function_call", "function_call_output"]
    assert sent["reasoning"] == {"effort": "high", "summary": "detailed"}
    assert sent["tools"] == [{"type": "function", "name": "bash", "parameters": {}}]


def test_a_transient_refusal_is_retried_and_a_verdict_is_not(gateway):
    """503 with Retry-After is re-sent; a 400 is a refusal carrying its status, not weather."""
    served, url = gateway
    served.script += [(503, {"error": "busy"}, {"Retry-After": "0"}), (200, CHAT_REPLY, {})]
    client = wire.Client(base_url=url, api_key="k", model="m", surface="chat")
    assert client.chat(messages=[]).content == "hello" and len(served.seen) == 2
    served.script.append((400, {"error": "bad request"}, {}))
    with pytest.raises(wire.GatewayRefused, match=r"gateway refused \(400\)") as refused:
        client.chat(messages=[])
    assert refused.value.status == 400 and not wire.weathered(status=400, spoken=str(refused.value))
    assert wire.weathered(status=503, spoken="") and wire.weathered(status=0, spoken="")
    assert wire.moderated(spoken="ContentPolicyViolationError: flagged")
    assert wire.weathered(status=400, spoken="content_policy_violation")


@pytest.mark.parametrize("surface", ["chat", "responses"])
def test_an_uncertain_transport_is_weather_without_resending_the_request(gateway, surface):
    """The server consumes the request and drops its reply; acceptance is unknown, so nothing is resent."""
    served, url = gateway
    served.script += [(None, None, {}), (200, CHAT_REPLY if surface == "chat" else RESPONSES_REPLY, {})]
    client = wire.Client(base_url=url, api_key="k", model="m", surface=surface)
    with pytest.raises(wire.GatewayRefused, match="acceptance is unknown, so this request was not resent") as refused:
        client.chat(messages=[])
    assert len(served.seen) == 1 and len(served.script) == 1
    assert served.seen[0][1]["agent_tag"] == "test-tag"
    assert refused.value.status == 0 and wire.weathered(status=0, spoken=str(refused.value))


def test_the_estimator_learns_its_ratio_from_the_endpoint_inside_a_band():
    """Characters per token start at the seed, move toward observations, and never leave the band."""
    counter = estimate.Estimator()
    assert counter.ratio == estimate.SEED_CHARS_PER_TOKEN
    assert counter.observe(sent=400, counted=100) and counter.observations == 1
    assert not counter.observe(sent=1000, counted=1), "a ratio outside the band is not learned"
    assert not counter.observe(sent=10, counted=0) and counter.rejected == 2
    assert estimate.RATIO_FLOOR <= counter.ratio <= estimate.RATIO_CEILING
    assert counter.tokens(text="x" * 30) == math.ceil(30 / counter.ratio)
    assert counter.payload(value={"a": "b"}) == math.ceil(len('{"a":"b"}') / counter.ratio)
    assert counter.messages(messages=[{"content": "x" * 300}]) == counter.payload(value={"content": "x" * 300})


def test_carried_prompts_load_by_name_and_fill_or_refuse(tmp_path):
    """A carried prompt is bytes beside the module; a field it lacks refuses rather than vanishing."""
    assert prompts.carried(name="act_loop.txt") == prompts.filled(name="act_loop.txt", fields={})
    assert "{SUBMISSION}" not in prompts.filled(name="solve_instance.txt", fields={"SUBMISSION": "DONE"})
    with pytest.raises(KeyError, match="nothing to substitute"):
        prompts.filled(name="act_loop.txt", fields={"NO_SUCH_FIELD": "x"})
    body = prompts.shared_meta()
    assert body and "{{" not in body and "{%" not in body
    carried = tmp_path / "instruction.txt"
    assert prompts.resolve(default="base", carried=None) == "base"
    for marker, expected in ((prompts.APPEND, "base\n\nmore"), (prompts.PREPEND, "more\n\nbase"), (prompts.REPLACE, "more")):
        carried.write_text(f"{marker}\nmore\n", encoding="utf-8")
        assert prompts.resolve(default="base", carried=carried) == expected
    carried.write_text("whole\n", encoding="utf-8")
    assert prompts.resolve(default="base", carried=carried) == "whole"


def test_a_json_field_is_read_out_of_a_reply_however_it_was_wrapped():
    """A seat reads the block it asked for from prose, a fence, or the bare object."""
    assert reply.field(reply='{"score": 1}', name="score") == [1]
    fenced = reply.field(reply='Sure.\n```json\n{"score": 2}\n```\nDone.', name="score")
    assert fenced and set(fenced) == {2}, "the fence and the outer braces are two candidates for one block"
    assert reply.field(reply="nothing here", name="score") == []
    assert reply.candidates(reply="[1, 2]") == ["[1, 2]"]
