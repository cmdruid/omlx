# SPDX-License-Identifier: Apache-2.0
"""Per-adapter health metrics persisted as a sidecar inside each adapter dir.

Layout: ``~/.omlx/adapters/<adapter_id>/.health.json``
Lifecycle:
- Created on first rescan/access with ``installed_at = now()``.
- Updated in-memory on load / request / validation events.
- Periodically flushed to disk (every 300s) and on shutdown.
- Auto-removed when the adapter directory is deleted (sidecar is inside).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, Optional

logger = logging.getLogger(__name__)

SIDECAR_FILENAME = ".health.json"
DEFAULT_FLUSH_INTERVAL_SECONDS = 300


def _now_iso() -> str:
    """ISO 8601 UTC timestamp with millisecond precision and Z suffix."""
    s = datetime.now(tz=timezone.utc).isoformat(timespec="milliseconds")
    return s[:-6] + "Z" if s.endswith("+00:00") else s


@dataclass
class AdapterHealth:
    """Health/usage metrics for a single LoRA adapter."""
    adapter_id: str
    installed_at: str
    last_loaded_at: Optional[str] = None
    last_request_at: Optional[str] = None
    load_count: int = 0
    compatible: Optional[bool] = None
    last_validated_at: Optional[str] = None
    last_load_error: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "AdapterHealth":
        data = json.loads(s)
        return cls(**data)


def load_health(adapter_dir: Path) -> Optional[AdapterHealth]:
    """Read .health.json from ``adapter_dir``. Returns None if missing.

    Returns None on parse error too — caller decides whether to overwrite
    with a fresh record or surface the error.
    """
    sidecar = Path(adapter_dir) / SIDECAR_FILENAME
    if not sidecar.exists():
        return None
    try:
        return AdapterHealth.from_json(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, TypeError, KeyError) as exc:
        logger.warning("Failed to parse %s: %s — treating as missing", sidecar, exc)
        return None


def save_health(adapter_dir: Path, health: AdapterHealth) -> None:
    """Atomically write .health.json into ``adapter_dir``.

    Uses tmp + rename so partial writes can't poison the sidecar.
    """
    sidecar = Path(adapter_dir) / SIDECAR_FILENAME
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
    tmp.write_text(health.to_json(), encoding="utf-8")
    tmp.replace(sidecar)


class AdapterHealthCache:
    """Thread-safe in-memory cache of AdapterHealth, flushed periodically.

    The cache key is the adapter directory path (Path), not adapter_id —
    different model bases may host adapters with the same id, and the dir
    is the unambiguous identifier.
    """

    def __init__(self, flush_interval_seconds: int = DEFAULT_FLUSH_INTERVAL_SECONDS) -> None:
        self._lock = Lock()
        self._cache: Dict[Path, AdapterHealth] = {}
        self._dirty: set[Path] = set()
        self._last_flush = time.time()
        self._flush_interval = flush_interval_seconds

    def get(self, adapter_dir: Path) -> Optional[AdapterHealth]:
        """Return the cached health, loading from disk on first access. None if missing."""
        adapter_dir = Path(adapter_dir)
        with self._lock:
            if adapter_dir in self._cache:
                return self._cache[adapter_dir]
        h = load_health(adapter_dir)
        if h is not None:
            with self._lock:
                self._cache[adapter_dir] = h
        return h

    def get_or_create(self, adapter_dir: Path, *, adapter_id: str) -> AdapterHealth:
        """Return cached health, creating fresh (with installed_at=now) if no
        sidecar exists. The new sidecar is persisted immediately."""
        existing = self.get(adapter_dir)
        if existing is not None:
            return existing
        fresh = AdapterHealth(adapter_id=adapter_id, installed_at=_now_iso())
        save_health(adapter_dir, fresh)
        with self._lock:
            self._cache[Path(adapter_dir)] = fresh
        return fresh

    def record_load_success(self, adapter_dir: Path) -> None:
        """Increment load_count and update last_loaded_at."""
        adapter_dir = Path(adapter_dir)
        with self._lock:
            h = self._cache.get(adapter_dir)
            if h is None:
                return
            self._cache[adapter_dir] = replace(
                h,
                last_loaded_at=_now_iso(),
                load_count=h.load_count + 1,
            )
            self._dirty.add(adapter_dir)
            self._maybe_flush_locked()

    def record_request_seen(self, adapter_dir: Path) -> None:
        """Update last_request_at."""
        adapter_dir = Path(adapter_dir)
        with self._lock:
            h = self._cache.get(adapter_dir)
            if h is None:
                return
            self._cache[adapter_dir] = replace(h, last_request_at=_now_iso())
            self._dirty.add(adapter_dir)
            self._maybe_flush_locked()

    def record_validation(self, adapter_dir: Path, *, compatible: bool, error: Optional[str]) -> None:
        """Set compatible / last_validated_at / last_load_error."""
        adapter_dir = Path(adapter_dir)
        with self._lock:
            h = self._cache.get(adapter_dir)
            if h is None:
                return
            self._cache[adapter_dir] = replace(
                h,
                compatible=compatible,
                last_validated_at=_now_iso(),
                last_load_error=error,
            )
            self._dirty.add(adapter_dir)
            self._maybe_flush_locked()

    def flush(self) -> None:
        """Persist all dirty cache entries."""
        with self._lock:
            dirty = list(self._dirty)
            self._dirty.clear()
            self._last_flush = time.time()
            snapshot = {p: self._cache[p] for p in dirty if p in self._cache}
        for adapter_dir, health in snapshot.items():
            try:
                save_health(adapter_dir, health)
            except OSError as exc:
                logger.warning("Failed to save .health.json at %s: %s", adapter_dir, exc)

    def _maybe_flush_locked(self) -> None:
        """Called holding self._lock. Flushes if interval elapsed."""
        now = time.time()
        if now - self._last_flush < self._flush_interval:
            return
        # Release-and-flush to avoid holding the lock during I/O.
        self._lock.release()
        try:
            self.flush()
        finally:
            self._lock.acquire()


# Module-level singleton.
_cache: Optional[AdapterHealthCache] = None


def get_adapter_health_cache() -> AdapterHealthCache:
    """Return the global cache, lazily constructing one on first access."""
    global _cache
    if _cache is None:
        _cache = AdapterHealthCache()
    return _cache


def shutdown_adapter_health_cache() -> None:
    """Flush and clear the singleton. Call on server shutdown."""
    global _cache
    if _cache is not None:
        _cache.flush()
        _cache = None
