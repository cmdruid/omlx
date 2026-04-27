# SPDX-License-Identifier: Apache-2.0
"""Tests for AdapterHealth — sidecar persistence + in-memory cache."""

from pathlib import Path

from omlx.adapter_health import (
    AdapterHealth,
    AdapterHealthCache,
    load_health,
    save_health,
)


def test_adapter_health_round_trips_through_json(tmp_path: Path):
    """A freshly-built AdapterHealth survives save → load."""
    h = AdapterHealth(
        adapter_id="rnd-001",
        installed_at="2026-04-26T10:00:00.000Z",
        last_loaded_at=None,
        last_request_at=None,
        load_count=0,
        compatible=None,
        last_validated_at=None,
        last_load_error=None,
    )
    save_health(tmp_path, h)
    loaded = load_health(tmp_path)
    assert loaded == h


def test_load_health_returns_none_when_sidecar_missing(tmp_path: Path):
    """Absent sidecar → None (caller decides to create fresh)."""
    assert load_health(tmp_path) is None


def test_save_health_writes_atomically(tmp_path: Path):
    """save_health uses tmp + rename so partial writes don't poison the sidecar."""
    h = AdapterHealth(
        adapter_id="rnd-001",
        installed_at="2026-04-26T10:00:00.000Z",
        last_loaded_at=None,
        last_request_at=None,
        load_count=0,
        compatible=None,
        last_validated_at=None,
        last_load_error=None,
    )
    save_health(tmp_path, h)
    sidecar = tmp_path / ".health.json"
    assert sidecar.exists()
    # No leftover .tmp file
    assert not list(tmp_path.glob("*.tmp"))


def test_cache_loads_on_first_access(tmp_path: Path):
    """AdapterHealthCache reads the sidecar lazily on .get()."""
    adapter_dir = tmp_path / "rnd-001"
    adapter_dir.mkdir()
    h = AdapterHealth(
        adapter_id="rnd-001",
        installed_at="2026-04-26T10:00:00.000Z",
        last_loaded_at=None,
        last_request_at=None,
        load_count=0,
        compatible=None,
        last_validated_at=None,
        last_load_error=None,
    )
    save_health(adapter_dir, h)
    cache = AdapterHealthCache()
    assert cache.get(adapter_dir) == h


def test_cache_creates_default_when_missing(tmp_path: Path):
    """get_or_create writes a fresh sidecar with installed_at=now() if absent."""
    adapter_dir = tmp_path / "rnd-002"
    adapter_dir.mkdir()
    cache = AdapterHealthCache()
    h = cache.get_or_create(adapter_dir, adapter_id="rnd-002")
    assert h.adapter_id == "rnd-002"
    assert h.installed_at  # set to now()
    # Sidecar persisted on get_or_create.
    assert (adapter_dir / ".health.json").exists()


def test_cache_records_load_event(tmp_path: Path):
    """record_load_success increments load_count and updates last_loaded_at."""
    adapter_dir = tmp_path / "rnd-003"
    adapter_dir.mkdir()
    cache = AdapterHealthCache()
    cache.get_or_create(adapter_dir, adapter_id="rnd-003")
    cache.record_load_success(adapter_dir)
    h = cache.get(adapter_dir)
    assert h.load_count == 1
    assert h.last_loaded_at is not None
    cache.record_load_success(adapter_dir)
    assert cache.get(adapter_dir).load_count == 2


def test_cache_records_request_event(tmp_path: Path):
    """record_request_seen updates last_request_at without bumping load_count."""
    adapter_dir = tmp_path / "rnd-004"
    adapter_dir.mkdir()
    cache = AdapterHealthCache()
    cache.get_or_create(adapter_dir, adapter_id="rnd-004")
    cache.record_request_seen(adapter_dir)
    h = cache.get(adapter_dir)
    assert h.last_request_at is not None
    assert h.load_count == 0  # request != load


def test_cache_records_validation(tmp_path: Path):
    """record_validation writes compatible + last_validated_at + last_load_error."""
    adapter_dir = tmp_path / "rnd-005"
    adapter_dir.mkdir()
    cache = AdapterHealthCache()
    cache.get_or_create(adapter_dir, adapter_id="rnd-005")
    cache.record_validation(adapter_dir, compatible=True, error=None)
    h = cache.get(adapter_dir)
    assert h.compatible is True
    assert h.last_validated_at is not None
    assert h.last_load_error is None
    cache.record_validation(adapter_dir, compatible=False, error="shape mismatch")
    h2 = cache.get(adapter_dir)
    assert h2.compatible is False
    assert h2.last_load_error == "shape mismatch"


def test_cache_flush_persists_dirty_entries(tmp_path: Path):
    """flush() writes any in-memory updates back to .health.json."""
    adapter_dir = tmp_path / "rnd-006"
    adapter_dir.mkdir()
    cache = AdapterHealthCache()
    cache.get_or_create(adapter_dir, adapter_id="rnd-006")
    cache.record_load_success(adapter_dir)
    cache.flush()

    # Re-read sidecar from disk and verify load_count is persisted.
    on_disk = load_health(adapter_dir)
    assert on_disk.load_count == 1


def test_health_fields_appear_in_get_status(tmp_path: Path):
    """engine_pool._health_fields_for surfaces fields populated via the cache.

    Direct-tests the helper used by engine_pool.get_status. Covers the
    rescan→load→request progression by simulating each event manually.
    """
    from omlx.adapter_health import (
        ensure_health_for_adapter,
        get_adapter_health_cache,
        shutdown_adapter_health_cache,
    )
    from omlx.engine_pool import _health_fields_for

    adapter_dir = tmp_path / "rnd-008"
    adapter_dir.mkdir()
    try:
        cache = get_adapter_health_cache()
        ensure_health_for_adapter(adapter_dir, adapter_id="rnd-008")
        cache.record_load_success(adapter_dir)
        cache.record_request_seen(adapter_dir)

        fields = _health_fields_for(str(adapter_dir))
        assert fields["installed_at"]
        assert fields["last_loaded_at"]
        assert fields["last_request_at"]
        assert fields["load_count"] == 1
        assert fields["compatible"] is None  # not validated yet
        assert fields["last_validated_at"] is None
        assert fields["last_load_error"] is None
    finally:
        shutdown_adapter_health_cache()


def test_health_fields_for_returns_empty_when_no_sidecar(tmp_path: Path):
    """_health_fields_for returns {} when no sidecar exists for the path."""
    from omlx.adapter_health import shutdown_adapter_health_cache
    from omlx.engine_pool import _health_fields_for

    try:
        shutdown_adapter_health_cache()  # ensure clean cache
        adapter_dir = tmp_path / "ghost"
        adapter_dir.mkdir()
        fields = _health_fields_for(str(adapter_dir))
        assert fields == {}
    finally:
        shutdown_adapter_health_cache()
