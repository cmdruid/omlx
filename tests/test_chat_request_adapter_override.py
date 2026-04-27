# SPDX-License-Identifier: Apache-2.0
"""Tests for ChatCompletionRequest.adapter_id — schema (B8) and behavior (B9).

B8 tests: schema-level (no HTTP, no engine).
B9 tests: behavior wiring in the route handler (per-request adapter swap).
"""
import asyncio

import pytest
from unittest.mock import MagicMock, AsyncMock

from omlx.api.openai_models import ChatCompletionRequest


def _base_kwargs():
    return {
        "model": "x",
        "messages": [{"role": "user", "content": "hi"}],
    }


def test_request_without_adapter_id_field_has_default_none_and_unset():
    """Absent adapter_id: field defaults to None AND not in model_fields_set."""
    req = ChatCompletionRequest(**_base_kwargs())
    assert req.adapter_id is None
    assert "adapter_id" not in req.model_fields_set


def test_request_with_explicit_null_adapter_id_is_in_model_fields_set():
    """Explicit null: field is None AND in model_fields_set."""
    req = ChatCompletionRequest(**_base_kwargs(), adapter_id=None)
    assert req.adapter_id is None
    assert "adapter_id" in req.model_fields_set


def test_request_with_adapter_id_string_round_trips():
    """String value round-trips through pydantic."""
    req = ChatCompletionRequest(**_base_kwargs(), adapter_id="rnd-005")
    assert req.adapter_id == "rnd-005"
    assert "adapter_id" in req.model_fields_set


def test_request_adapter_id_field_is_optional_in_serialization():
    """Default-None field is excluded by exclude_none and exclude_unset."""
    req = ChatCompletionRequest(**_base_kwargs())
    dumped = req.model_dump(exclude_none=True)
    assert "adapter_id" not in dumped
    dumped_unset = req.model_dump(exclude_unset=True)
    assert "adapter_id" not in dumped_unset


def test_request_adapter_id_explicit_null_in_dump_when_not_excluded():
    """Explicit null appears in model_dump unless excluded."""
    req = ChatCompletionRequest(**_base_kwargs(), adapter_id=None)
    dumped = req.model_dump()
    assert "adapter_id" in dumped
    assert dumped["adapter_id"] is None
    # exclude_unset=False is default; explicit null is "set" so it appears.
    dumped_unset = req.model_dump(exclude_unset=True)
    assert "adapter_id" in dumped_unset


def test_request_validates_adapter_id_must_be_string_or_none():
    """Non-string, non-null adapter_id is rejected by pydantic validation."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ChatCompletionRequest(**_base_kwargs(), adapter_id=123)


# =============================================================================
# B9 — behavior tests (route handler wiring)
# =============================================================================

def _make_engine(text="hello"):
    """Mock engine with chat() returning a simple text response."""
    fake_output = MagicMock()
    fake_output.text = text
    fake_output.tool_calls = None
    fake_output.prompt_tokens = 10
    fake_output.completion_tokens = 5
    fake_output.cached_tokens = 0
    fake_output.finish_reason = "stop"

    engine = MagicMock()
    engine.model_type = "llama"
    engine.tokenizer = MagicMock()
    engine.tokenizer.has_tool_calling = False
    engine.message_extractor = None
    engine.count_chat_tokens = MagicMock(return_value=10)
    engine.chat = AsyncMock(return_value=fake_output)
    return engine


def _stub_engine_chat_path(monkeypatch, *, engine, available_adapters=None,
                            loaded_adapter_id=None):
    """Set up monkeypatching for a chat-completion test against a stubbed engine.

    Returns (fake_pool, fake_entry) so callers can assert swap calls.
    """
    available_adapters = available_adapters or []

    async def _fake_get_engine(_model):
        return engine

    fake_entry = MagicMock()
    fake_entry.available_adapters = available_adapters
    fake_entry.engine = engine
    fake_entry.preserve_thinking_default = None
    # _loaded_adapter_id reads engine._adapter_path; set it to control the helper.
    if loaded_adapter_id:
        engine._adapter_path = f"/fake/path/{loaded_adapter_id}"
    else:
        engine._adapter_path = None
    fake_entry.swap_lock = asyncio.Lock()

    fake_pool = MagicMock()
    fake_pool.get_entry = MagicMock(return_value=fake_entry)
    fake_pool.swap_adapter_for_request = AsyncMock()

    monkeypatch.setattr("omlx.server.get_engine_for_model", _fake_get_engine)
    monkeypatch.setattr("omlx.server.resolve_model_id", lambda m: m)
    monkeypatch.setattr("omlx.server.validate_context_window", lambda *a, **kw: None)
    monkeypatch.setattr("omlx.server.get_engine_pool", lambda: fake_pool)
    return fake_pool, fake_entry


def _post_chat(client, *, model="test-model", adapter_id=..., stream=False):
    """POST /v1/chat/completions with optional adapter_id."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": stream,
    }
    if adapter_id is not ...:
        body["adapter_id"] = adapter_id
    return client.post("/v1/chat/completions", json=body)


def test_request_with_adapter_id_string_triggers_swap(monkeypatch):
    """Override with a known adapter triggers swap_adapter_for_request."""
    from fastapi.testclient import TestClient
    from omlx.server import app, _server_state

    engine = _make_engine()
    adapter = MagicMock()
    adapter.adapter_id = "rnd-005"
    adapter.path = "/fake/path/rnd-005"
    fake_pool, _ = _stub_engine_chat_path(
        monkeypatch, engine=engine, available_adapters=[adapter],
        loaded_adapter_id=None,  # nothing currently loaded
    )

    original = _server_state.api_key
    _server_state.api_key = None
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = _post_chat(client, adapter_id="rnd-005")
    finally:
        _server_state.api_key = original

    assert resp.status_code == 200
    fake_pool.swap_adapter_for_request.assert_awaited_once_with("test-model", "rnd-005")


def test_request_without_adapter_id_does_not_swap(monkeypatch):
    """Absent adapter_id: no swap_adapter_for_request call."""
    from fastapi.testclient import TestClient
    from omlx.server import app, _server_state

    engine = _make_engine()
    fake_pool, _ = _stub_engine_chat_path(monkeypatch, engine=engine)

    original = _server_state.api_key
    _server_state.api_key = None
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = _post_chat(client)  # no adapter_id kwarg
    finally:
        _server_state.api_key = original

    assert resp.status_code == 200
    fake_pool.swap_adapter_for_request.assert_not_awaited()


def test_request_with_adapter_id_null_swaps_to_none(monkeypatch):
    """Explicit null adapter_id: swap_adapter_for_request called with None."""
    from fastapi.testclient import TestClient
    from omlx.server import app, _server_state

    engine = _make_engine()
    adapter = MagicMock()
    adapter.adapter_id = "rnd-005"
    adapter.path = "/fake/path/rnd-005"
    fake_pool, _ = _stub_engine_chat_path(
        monkeypatch, engine=engine, available_adapters=[adapter],
        loaded_adapter_id="rnd-005",  # currently loaded with rnd-005
    )

    original = _server_state.api_key
    _server_state.api_key = None
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = _post_chat(client, adapter_id=None)
    finally:
        _server_state.api_key = original

    assert resp.status_code == 200
    fake_pool.swap_adapter_for_request.assert_awaited_once_with("test-model", None)


def test_request_with_unknown_adapter_returns_400(monkeypatch):
    """Unknown adapter_id: 400 with adapter_not_found, no swap attempted."""
    from fastapi.testclient import TestClient
    from omlx.server import app, _server_state

    engine = _make_engine()
    adapter = MagicMock()
    adapter.adapter_id = "rnd-005"
    fake_pool, _ = _stub_engine_chat_path(
        monkeypatch, engine=engine, available_adapters=[adapter],
    )

    original = _server_state.api_key
    _server_state.api_key = None
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = _post_chat(client, adapter_id="ghost")
    finally:
        _server_state.api_key = original

    assert resp.status_code == 400
    body = resp.json()
    # JSONResponse returns the body directly, no global-handler wrap.
    err = body.get("error") or {}
    assert err.get("type") == "adapter_not_found"
    fake_pool.swap_adapter_for_request.assert_not_awaited()


def test_request_with_adapter_already_loaded_skips_swap(monkeypatch):
    """If override matches currently-loaded adapter, no swap is performed."""
    from fastapi.testclient import TestClient
    from omlx.server import app, _server_state

    engine = _make_engine()
    adapter = MagicMock()
    adapter.adapter_id = "rnd-005"
    adapter.path = "/fake/path/rnd-005"
    fake_pool, _ = _stub_engine_chat_path(
        monkeypatch, engine=engine, available_adapters=[adapter],
        loaded_adapter_id="rnd-005",  # already loaded
    )

    original = _server_state.api_key
    _server_state.api_key = None
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = _post_chat(client, adapter_id="rnd-005")
    finally:
        _server_state.api_key = original

    assert resp.status_code == 200
    fake_pool.swap_adapter_for_request.assert_not_awaited()
