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
