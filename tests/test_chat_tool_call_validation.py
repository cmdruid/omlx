# SPDX-License-Identifier: Apache-2.0
"""Tests that malformed tool-call output produces a structured 502, not a dropped connection."""

from unittest.mock import MagicMock, AsyncMock

import pytest
from fastapi.testclient import TestClient

from omlx.server import app, _server_state


def _make_engine_with_bad_tool_args(tool_name="edit", bad_args="{not valid json"):
    """Build a mocked engine whose chat() returns a tool_call with malformed JSON args."""
    fake_output = MagicMock()
    fake_output.text = ""
    fake_output.tool_calls = [{"name": tool_name, "arguments": bad_args}]
    fake_output.prompt_tokens = 10
    fake_output.completion_tokens = 5
    fake_output.cached_tokens = 0
    fake_output.finish_reason = "tool_calls"

    engine = MagicMock()
    engine.model_type = "gpt_oss"
    engine.tokenizer = MagicMock()
    # has_tool_calling False so we don't get sucked into native-tool-calling parser
    engine.tokenizer.has_tool_calling = False
    engine.message_extractor = None
    engine.count_chat_tokens = MagicMock(return_value=10)
    engine.chat = AsyncMock(return_value=fake_output)
    return engine


def test_malformed_tool_args_returns_structured_502(monkeypatch):
    """A model that emits malformed tool-call JSON should produce a 502 with structured body."""
    fake_engine = _make_engine_with_bad_tool_args()

    async def _fake_get_engine(_model):
        return fake_engine

    monkeypatch.setattr("omlx.server.get_engine_for_model", _fake_get_engine)
    monkeypatch.setattr("omlx.server.resolve_model_id", lambda m: m)
    monkeypatch.setattr("omlx.server.validate_context_window", lambda *a, **kw: None)
    monkeypatch.setattr("omlx.server.get_engine_pool", lambda: MagicMock(get_entry=lambda _: None))

    original_key = _server_state.api_key
    _server_state.api_key = None
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-oss-20b",
                    "messages": [{"role": "user", "content": "use a tool"}],
                    "stream": False,
                },
            )
    finally:
        _server_state.api_key = original_key

    assert resp.status_code == 502
    body = resp.json()
    err = body.get("error") or {}
    assert err.get("type") == "tool_call_validation_error"
    details = err.get("details", {})
    assert details.get("tool_name") == "edit"
    assert "raw_arguments" in details
    assert "parse_error" in details


def test_well_formed_tool_call_still_succeeds(monkeypatch):
    """Regression: a clean tool call must still produce a 200 with tool_calls in the body."""
    fake_output = MagicMock()
    fake_output.text = ""
    fake_output.tool_calls = [{"name": "edit", "arguments": '{"path": "/x"}'}]
    fake_output.prompt_tokens = 10
    fake_output.completion_tokens = 5
    fake_output.cached_tokens = 0
    fake_output.finish_reason = "tool_calls"

    engine = MagicMock()
    engine.model_type = "gpt_oss"
    engine.tokenizer = MagicMock()
    engine.tokenizer.has_tool_calling = False
    engine.message_extractor = None
    engine.count_chat_tokens = MagicMock(return_value=10)
    engine.chat = AsyncMock(return_value=fake_output)

    async def _fake_get_engine(_model):
        return engine

    monkeypatch.setattr("omlx.server.get_engine_for_model", _fake_get_engine)
    monkeypatch.setattr("omlx.server.resolve_model_id", lambda m: m)
    monkeypatch.setattr("omlx.server.validate_context_window", lambda *a, **kw: None)
    monkeypatch.setattr("omlx.server.get_engine_pool", lambda: MagicMock(get_entry=lambda _: None))

    original_key = _server_state.api_key
    _server_state.api_key = None
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-oss-20b",
                    "messages": [{"role": "user", "content": "use a tool"}],
                    "stream": False,
                },
            )
    finally:
        _server_state.api_key = original_key

    assert resp.status_code == 200
    body = resp.json()
    tool_calls = body["choices"][0]["message"].get("tool_calls") or []
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "edit"


def test_streaming_malformed_tool_args_emits_sse_error(monkeypatch):
    """Streaming path: malformed tool-call JSON emits a structured SSE error event + [DONE]."""

    # Build a final output object with a malformed arguments field.
    fake_output = MagicMock()
    fake_output.text = None
    fake_output.new_text = None
    fake_output.tool_calls = [{"name": "edit", "arguments": "{bad"}]
    fake_output.finished = True
    fake_output.finish_reason = "tool_calls"
    fake_output.prompt_tokens = 10
    fake_output.completion_tokens = 5
    fake_output.cached_tokens = 0

    # stream_chat must be an async generator, not an AsyncMock.
    async def _stream_chat(messages, **kwargs):
        yield fake_output

    engine = MagicMock()
    engine.model_type = "gpt_oss"
    engine.tokenizer = MagicMock()
    engine.tokenizer.has_tool_calling = False
    engine.message_extractor = None
    engine.count_chat_tokens = MagicMock(return_value=10)
    engine.stream_chat = _stream_chat

    async def _fake_get_engine(_model):
        return engine

    monkeypatch.setattr("omlx.server.get_engine_for_model", _fake_get_engine)
    monkeypatch.setattr("omlx.server.resolve_model_id", lambda m: m)
    monkeypatch.setattr("omlx.server.validate_context_window", lambda *a, **kw: None)
    monkeypatch.setattr("omlx.server.get_engine_pool", lambda: MagicMock(get_entry=lambda _: None))

    import json as _json

    original_key = _server_state.api_key
    _server_state.api_key = None
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-oss-20b",
                    "messages": [{"role": "user", "content": "use a tool"}],
                    "stream": True,
                },
            )
    finally:
        _server_state.api_key = original_key

    # Parse SSE events
    events = []
    for line in resp.text.split("\n"):
        if line.startswith("data: "):
            payload = line[len("data: "):]
            events.append(payload)

    # [DONE] must appear as the final event
    assert events, "Expected at least one SSE event"
    assert events[-1] == "[DONE]", f"Last event was not [DONE]: {events[-1]!r}"

    # At least one event must be a structured error with type tool_call_validation_error
    error_events = []
    for payload in events:
        if payload == "[DONE]":
            continue
        try:
            parsed = _json.loads(payload)
        except _json.JSONDecodeError:
            continue
        if "error" in parsed:
            error_events.append(parsed)

    assert error_events, "No error event found in SSE stream"
    err = error_events[0]["error"]
    assert err.get("type") == "tool_call_validation_error"
    details = err.get("details", {})
    assert details.get("tool_name") == "edit"
    assert "raw_arguments" in details
    assert "parse_error" in details
