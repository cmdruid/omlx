# SPDX-License-Identifier: Apache-2.0
"""Tests for the per-request trace sink."""

import json
from pathlib import Path

import pytest

from omlx.trace_sink import TraceSink, _format_uptime_filename


def test_format_uptime_filename_uses_iso_with_dashes():
    """Filename is ISO timestamp, milliseconds, dashes for filesystem safety."""
    name = _format_uptime_filename(epoch_seconds=1714089825.123)
    # e.g. "2024-04-25T22-03-45-123Z.jsonl"
    assert name.endswith("Z.jsonl")
    assert ":" not in name
    # Format: YYYY-MM-DDTHH-MM-SS-mmmZ.jsonl
    parts = name[:-len(".jsonl")].split("-")
    assert len(parts) >= 6  # date(3) + time(3) + ms(1)


def test_trace_sink_writes_one_record_per_line(tmp_path: Path):
    """Each write produces a single JSON line in the active file."""
    sink = TraceSink(traces_dir=tmp_path)
    try:
        sink.write({"request_id": "chatcmpl-abc", "tokens": 5})
        sink.write({"request_id": "chatcmpl-def", "tokens": 7})
    finally:
        sink.close()

    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"request_id": "chatcmpl-abc", "tokens": 5}
    assert json.loads(lines[1]) == {"request_id": "chatcmpl-def", "tokens": 7}


def test_trace_sink_creates_traces_dir_if_missing(tmp_path: Path):
    """If traces_dir doesn't exist, sink creates it."""
    target = tmp_path / "traces"
    assert not target.exists()
    sink = TraceSink(traces_dir=target)
    sink.write({"x": 1})
    sink.close()
    assert target.is_dir()
    assert any(target.glob("*.jsonl"))


def test_trace_sink_write_errors_do_not_raise(tmp_path: Path, monkeypatch):
    """A broken file handle must not propagate exceptions out of write()."""
    sink = TraceSink(traces_dir=tmp_path)
    # Forcibly close the underlying file so the next write attempts I/O on a closed handle.
    sink._fh.close()
    # Should NOT raise — sink swallows the error and logs a warning.
    sink.write({"x": 1})
    sink.close()


def test_trace_sink_close_is_idempotent(tmp_path: Path):
    """close() twice must not raise."""
    sink = TraceSink(traces_dir=tmp_path)
    sink.close()
    sink.close()  # second close is a no-op


def test_reset_trace_sink_replaces_singleton(tmp_path: Path):
    """reset_trace_sink closes any existing sink and opens a fresh one."""
    from omlx.trace_sink import reset_trace_sink, get_trace_sink, shutdown_trace_sink

    try:
        reset_trace_sink(traces_dir=tmp_path / "first")
        first = get_trace_sink()
        assert first is not None
        first_path = first.path

        reset_trace_sink(traces_dir=tmp_path / "second")
        second = get_trace_sink()
        assert second is not None
        assert second.path != first_path
        # First sink should be closed.
        assert first._closed
    finally:
        shutdown_trace_sink()


def test_reset_trace_sink_with_none_disables_singleton(tmp_path: Path):
    """Passing None to reset_trace_sink leaves the singleton as None."""
    from omlx.trace_sink import reset_trace_sink, get_trace_sink, shutdown_trace_sink

    try:
        reset_trace_sink(traces_dir=tmp_path)
        assert get_trace_sink() is not None
        reset_trace_sink(traces_dir=None)
        assert get_trace_sink() is None
    finally:
        shutdown_trace_sink()


def test_chat_completion_emits_trace_record(tmp_path: Path, monkeypatch):
    """A successful chat-completion writes one JSONL trace record with the right shape."""
    from datetime import datetime
    from unittest.mock import MagicMock, AsyncMock

    from omlx.trace_sink import reset_trace_sink, get_trace_sink, shutdown_trace_sink
    from omlx.server import app, _server_state
    from fastapi.testclient import TestClient

    reset_trace_sink(traces_dir=tmp_path)

    # Mock the engine to return a fixed response without loading a real model.
    fake_tokenizer = MagicMock()
    fake_tokenizer.has_tool_calling = False

    fake_engine = MagicMock()
    fake_engine.model_type = "llama"
    fake_engine.tokenizer = fake_tokenizer
    fake_engine.message_extractor = None
    fake_engine.count_chat_tokens = MagicMock(return_value=10)
    fake_output = MagicMock()
    fake_output.text = "hi"
    fake_output.tool_calls = None
    fake_output.prompt_tokens = 10
    fake_output.completion_tokens = 5
    fake_output.cached_tokens = 0
    fake_output.finish_reason = "stop"
    fake_engine.chat = AsyncMock(return_value=fake_output)

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
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        response_id = body["id"]
    finally:
        _server_state.api_key = original_key
        shutdown_trace_sink()

    # Find the JSONL file and verify the trace record.
    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["request_id"] == response_id
    assert record["model_id"] == "test-model"
    assert "adapter_id" in record  # may be empty string
    assert "version" in record
    assert record["prompt_tokens"] == 10
    assert record["completion_tokens"] == 5
    assert "elapsed_seconds" in record
    assert "timestamp" in record


def test_build_chat_trace_record_includes_sampling_params():
    """Sampling params from request body land in the trace record."""
    from omlx.trace_sink import build_chat_trace_record

    rec = build_chat_trace_record(
        request_id="chatcmpl-xyz",
        model_id="gpt-oss-20b-MXFP4-Q8",
        adapter_id=None,
        prompt_tokens=10,
        completion_tokens=5,
        elapsed_seconds=0.5,
        temperature=0.7,
        top_p=0.95,
        top_k=0,
        max_tokens=2048,
        seed=42,
    )

    assert rec["temperature"] == 0.7
    assert rec["top_p"] == 0.95
    assert rec["top_k"] == 0
    assert rec["max_tokens"] == 2048
    assert rec["seed"] == 42


def test_build_chat_trace_record_omits_seed_when_unset():
    """seed is None on the wire when the client doesn't pass it; record must reflect that."""
    from omlx.trace_sink import build_chat_trace_record

    rec = build_chat_trace_record(
        request_id="chatcmpl-xyz",
        model_id="m",
        adapter_id=None,
        prompt_tokens=1,
        completion_tokens=1,
        elapsed_seconds=0.1,
        temperature=1.0,
        top_p=0.95,
        top_k=0,
        max_tokens=32768,
        seed=None,
    )
    assert rec["seed"] is None
    assert rec["temperature"] == 1.0
