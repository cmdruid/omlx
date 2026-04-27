# SPDX-License-Identifier: Apache-2.0
"""Per-request trace sink for chat-completions.

Writes one JSONL record per chat-completion to ~/.omlx/traces/<timestamp>.jsonl.
One file per `omlx serve` lifetime; restart writes a new file.

Errors during write are logged and swallowed — trace is best-effort and must
never block or fail a request.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TextIO

logger = logging.getLogger(__name__)


def _format_uptime_filename(epoch_seconds: Optional[float] = None) -> str:
    """Return an ISO-timestamped filename safe for any filesystem.

    Format: ``YYYY-MM-DDTHH-MM-SS-mmmZ.jsonl`` (millisecond precision, dashes
    instead of colons, suffix ``Z`` for UTC).
    """
    if epoch_seconds is None:
        epoch_seconds = time.time()
    dt = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
    # millisecond precision
    ms = int(dt.microsecond / 1000)
    return f"{dt.strftime('%Y-%m-%dT%H-%M-%S')}-{ms:03d}Z.jsonl"


class TraceSink:
    """Append-only JSONL writer for per-request traces.

    One sink per server uptime. On construction, opens a new file at
    ``traces_dir / <iso-timestamp>.jsonl`` and holds the handle for the
    sink's lifetime. ``write()`` appends one record per call; failures are
    logged and swallowed.
    """

    def __init__(self, traces_dir: Path) -> None:
        self.traces_dir = Path(traces_dir)
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.traces_dir / _format_uptime_filename()
        self._fh: Optional[TextIO] = open(self.path, "a", encoding="utf-8")
        self._closed = False
        logger.info("Trace sink opened at %s", self.path)

    def write(self, record: dict[str, Any]) -> None:
        """Append a single JSON record to the trace file. Best-effort."""
        if self._closed or self._fh is None:
            return
        try:
            line = json.dumps(record, ensure_ascii=False)
            self._fh.write(line)
            self._fh.write("\n")
            self._fh.flush()
        except Exception as exc:  # pylint: disable=broad-except
            # Trace is best-effort — never propagate.
            logger.warning("TraceSink.write failed: %s (record=%r)", exc, record)

    def close(self) -> None:
        """Close the underlying file. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("TraceSink.close failed: %s", exc)
            self._fh = None


# Module-level singleton, initialized by reset_trace_sink() on server start.
_trace_sink: Optional[TraceSink] = None


def get_trace_sink() -> Optional[TraceSink]:
    """Return the global trace sink, or None if not initialized."""
    return _trace_sink


def reset_trace_sink(traces_dir: Optional[Path] = None) -> None:
    """Reset the singleton trace sink. Closes any prior sink before opening a new one.

    Call once on server start with the desired ``traces_dir``. Pass ``None``
    to disable tracing (closes any existing sink and leaves the singleton
    None).
    """
    global _trace_sink
    if _trace_sink is not None:
        _trace_sink.close()
    if traces_dir is None:
        _trace_sink = None
        return
    _trace_sink = TraceSink(traces_dir=Path(traces_dir))


def shutdown_trace_sink() -> None:
    """Close the singleton sink without replacing it. Call on server shutdown."""
    global _trace_sink
    if _trace_sink is not None:
        _trace_sink.close()
        _trace_sink = None


def build_chat_trace_record(
    *,
    request_id: str,
    model_id: str,
    adapter_id: Optional[str],
    prompt_tokens: int,
    completion_tokens: int,
    elapsed_seconds: float,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    seed: Optional[int],
) -> dict[str, Any]:
    """Build a chat-completion trace record. Centralizes shape so both the
    streaming and non-streaming paths produce identical records.

    Sampling params are wire-level (as received on the request body, after
    any server-side defaulting in pydantic). They reflect what the client
    sent, not the post-resolution effective values applied during sampling.
    """
    from ._version import __version__

    timestamp = datetime.now(tz=timezone.utc).isoformat(timespec="milliseconds")
    # ISO 8601 with 'Z' suffix for UTC clarity (Python uses '+00:00' otherwise).
    if timestamp.endswith("+00:00"):
        timestamp = timestamp[:-6] + "Z"
    return {
        "request_id": request_id,
        "timestamp": timestamp,
        "model_id": model_id,
        "adapter_id": adapter_id or "",
        "version": __version__,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "seed": seed,
    }
