# SPDX-License-Identifier: Apache-2.0
"""Tests that the chat-completions handler stamps log records with the response id."""

import logging
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from fastapi.testclient import TestClient

from omlx.logging_config import _request_id, RequestContextFilter, set_request_id
from omlx.server import app, _server_state


def test_set_request_id_is_called_at_top_of_handler(monkeypatch):
    """create_chat_completion must call set_request_id before any inner work runs.

    We patch set_request_id and assert it was called with a chatcmpl-XXX id
    BEFORE get_engine_for_model is hit.
    """
    captured: list[str] = []

    def _capture(rid):
        captured.append(rid)

    monkeypatch.setattr("omlx.server.set_request_id", _capture)

    # Force get_engine_for_model to raise so we don't need a real engine.
    async def _boom(_model):
        # By the time this is hit, set_request_id should already have been called.
        raise RuntimeError("stopping early")

    monkeypatch.setattr("omlx.server.get_engine_for_model", _boom)
    # Disable api-key check
    original_key = _server_state.api_key
    _server_state.api_key = None
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            client.post(
                "/v1/chat/completions",
                json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
            )
    finally:
        _server_state.api_key = original_key

    assert len(captured) == 1
    assert captured[0].startswith("chatcmpl-")


def test_response_id_matches_request_id_in_log_filter():
    """The same chatcmpl-XXX id used for the response should appear in the
    RequestContextFilter's request_id field of any log record emitted during
    the request."""
    seen_ids: list[str] = []

    class _CaptureHandler(logging.Handler):
        def emit(self, record):
            seen_ids.append(getattr(record, "request_id", None))

    handler = _CaptureHandler()
    handler.addFilter(RequestContextFilter())
    omlx_logger = logging.getLogger("omlx")
    original_level = omlx_logger.level
    omlx_logger.setLevel(logging.INFO)
    omlx_logger.addHandler(handler)
    try:
        # Simulate a request: directly invoke set_request_id, log, then unset.
        set_request_id("chatcmpl-deadbeef")
        logging.getLogger("omlx.test").info("during-request log")
        set_request_id(None)
        logging.getLogger("omlx.test").info("outside-request log")
    finally:
        omlx_logger.removeHandler(handler)
        omlx_logger.setLevel(original_level)

    assert "chatcmpl-deadbeef" in seen_ids
    # The outside-request log should be stamped with "-" (the default).
    assert "-" in seen_ids
