# SPDX-License-Identifier: Apache-2.0
"""Tests for EnginePool.swap_adapter primitive and AdapterSwapError."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from omlx.engine_pool import AdapterSwapError, EngineEntry


def _make_entry(model_id="test-model", adapters=None):
    """Build an EngineEntry-like mock for use with swap_adapter tests."""
    entry = MagicMock(spec=EngineEntry)
    entry.model_id = model_id
    entry.engine = None
    entry.available_adapters = adapters or []
    entry.swap_lock = asyncio.Lock()
    return entry


def test_adapter_swap_error_carries_type_and_message():
    """AdapterSwapError exposes error_type and message as attributes."""
    err = AdapterSwapError("adapter_not_found", "Adapter X not in available_adapters")
    assert err.error_type == "adapter_not_found"
    assert err.message == "Adapter X not in available_adapters"
    assert "Adapter X" in str(err)


async def test_swap_adapter_unknown_model_raises_model_not_found():
    """swap_adapter on a non-existent model raises AdapterSwapError(model_not_found)."""
    from omlx.engine_pool import EnginePool

    # Build a pool with no entries.
    pool = EnginePool.__new__(EnginePool)
    pool._entries = {}
    pool._settings_manager = None

    with pytest.raises(AdapterSwapError) as exc_info:
        await pool.swap_adapter("ghost-model", "rnd-001")
    assert exc_info.value.error_type == "model_not_found"


async def test_swap_adapter_unknown_adapter_raises_adapter_not_found():
    """swap_adapter with an adapter_id not in available_adapters raises adapter_not_found."""
    from omlx.engine_pool import EnginePool

    adapter = MagicMock()
    adapter.adapter_id = "rnd-001"
    entry = _make_entry(adapters=[adapter])

    pool = EnginePool.__new__(EnginePool)
    pool._entries = {"test-model": entry}
    pool._settings_manager = None

    with pytest.raises(AdapterSwapError) as exc_info:
        await pool.swap_adapter("test-model", "rnd-999")
    assert exc_info.value.error_type == "adapter_not_found"
    assert "rnd-999" in exc_info.value.message


async def test_swap_adapter_none_detaches_without_validation():
    """swap_adapter(model_id, None) detaches without checking available_adapters."""
    from omlx.engine_pool import EnginePool

    entry = _make_entry()  # no adapters, but None should still work
    entry.engine = None  # nothing currently loaded

    pool = EnginePool.__new__(EnginePool)
    pool._entries = {"test-model": entry}
    # Stub settings manager
    settings = MagicMock()
    settings.adapter_id = "rnd-001"
    sm = MagicMock()
    sm.get_settings = MagicMock(return_value=settings)
    sm.set_settings = MagicMock()
    pool._settings_manager = sm

    # Stub the load method
    pool._unload_engine = AsyncMock()
    pool._load_engine = AsyncMock()

    await pool.swap_adapter("test-model", None)

    # Settings updated to None
    assert settings.adapter_id is None
    sm.set_settings.assert_called_with("test-model", settings)
    # Engine unloaded if loaded — wasn't loaded, so no call to _unload_engine
    pool._unload_engine.assert_not_called()
    # Loaded with new (no) adapter
    pool._load_engine.assert_awaited_once_with("test-model")


async def test_swap_adapter_load_failure_wraps_in_adapter_swap_error():
    """When _load_engine raises, swap_adapter wraps the exception in AdapterSwapError."""
    from omlx.engine_pool import EnginePool

    adapter = MagicMock()
    adapter.adapter_id = "rnd-001"
    entry = _make_entry(adapters=[adapter])
    entry.engine = MagicMock()  # currently loaded

    pool = EnginePool.__new__(EnginePool)
    pool._entries = {"test-model": entry}

    settings = MagicMock()
    settings.adapter_id = None
    sm = MagicMock()
    sm.get_settings = MagicMock(return_value=settings)
    sm.set_settings = MagicMock()
    pool._settings_manager = sm

    pool._unload_engine = AsyncMock()
    pool._load_engine = AsyncMock(side_effect=RuntimeError("shape mismatch"))

    with pytest.raises(AdapterSwapError) as exc_info:
        await pool.swap_adapter("test-model", "rnd-001")

    assert exc_info.value.error_type == "adapter_load_failed"
    assert "shape mismatch" in exc_info.value.message


@pytest.mark.asyncio
async def test_swap_adapter_records_compatibility_false_on_load_failure(tmp_path):
    """A failed swap_adapter writes compatible=False to the health sidecar.

    Closes the architectural gap (Option B): the hard-block on subsequent
    selections fires automatically without requiring an explicit
    ?validate=true rescan.
    """
    from omlx.adapter_health import (
        get_adapter_health_cache,
        load_health,
        shutdown_adapter_health_cache,
    )
    from omlx.engine_pool import EnginePool

    adapter_dir = tmp_path / "rnd-broken"
    adapter_dir.mkdir()
    adapter = MagicMock()
    adapter.adapter_id = "rnd-broken"
    adapter.path = str(adapter_dir)
    entry = _make_entry(adapters=[adapter])
    entry.engine = MagicMock()  # currently loaded → triggers unload before load

    pool = EnginePool.__new__(EnginePool)
    pool._entries = {"test-model": entry}

    settings = MagicMock()
    settings.adapter_id = None
    sm = MagicMock()
    sm.get_settings = MagicMock(return_value=settings)
    sm.set_settings = MagicMock()
    pool._settings_manager = sm

    pool._unload_engine = AsyncMock()
    pool._load_engine = AsyncMock(side_effect=RuntimeError("shape mismatch in self_attn.q_proj"))

    try:
        with pytest.raises(AdapterSwapError) as exc_info:
            await pool.swap_adapter("test-model", "rnd-broken")
        assert exc_info.value.error_type == "adapter_load_failed"

        # Sidecar must reflect the failure on disk (record_adapter_load_failure
        # flushes the cache so consumers see it immediately).
        on_disk = load_health(adapter_dir)
        assert on_disk is not None
        assert on_disk.compatible is False
        assert "shape mismatch" in on_disk.last_load_error
        assert on_disk.last_validated_at is not None
    finally:
        shutdown_adapter_health_cache()


@pytest.mark.asyncio
async def test_swap_adapter_for_request_records_compatibility_false_on_load_failure(tmp_path):
    """A failed swap_adapter_for_request writes compatible=False too.

    Mirrors swap_adapter's behavior — the per-request override path also
    benefits from auto-arming the hard-block.
    """
    from omlx.adapter_health import (
        load_health,
        shutdown_adapter_health_cache,
    )
    from omlx.engine_pool import EnginePool
    from omlx.model_settings import ModelSettings

    adapter_dir = tmp_path / "rnd-broken"
    adapter_dir.mkdir()
    adapter = MagicMock()
    adapter.adapter_id = "rnd-broken"
    adapter.path = str(adapter_dir)
    entry = _make_entry(adapters=[adapter])
    entry.engine = None

    pool = EnginePool.__new__(EnginePool)
    pool._entries = {"test-model": entry}

    # ModelSettingsManager-shaped stub: has ._lock and ._settings dict.
    import threading
    sm = MagicMock()
    sm._lock = threading.Lock()
    sm._settings = {"test-model": ModelSettings()}
    pool._settings_manager = sm

    pool._unload_engine = AsyncMock()
    pool._load_engine = AsyncMock(side_effect=RuntimeError("OOM during adapter load"))

    try:
        with pytest.raises(AdapterSwapError) as exc_info:
            await pool.swap_adapter_for_request("test-model", "rnd-broken")
        assert exc_info.value.error_type == "adapter_load_failed"

        on_disk = load_health(adapter_dir)
        assert on_disk is not None
        assert on_disk.compatible is False
        assert "OOM during adapter load" in on_disk.last_load_error
    finally:
        shutdown_adapter_health_cache()


@pytest.mark.asyncio
async def test_swap_adapter_does_not_record_when_detaching_to_base(tmp_path):
    """adapter_id=None (detach) failures do NOT write to the sidecar.

    Detaching is a base-model load — there's no adapter to mark
    incompatible. We can't infer "incompatible" from a failure here.
    """
    from omlx.adapter_health import shutdown_adapter_health_cache
    from omlx.engine_pool import EnginePool

    entry = _make_entry()
    entry.engine = MagicMock()

    pool = EnginePool.__new__(EnginePool)
    pool._entries = {"test-model": entry}

    settings = MagicMock()
    settings.adapter_id = "rnd-001"
    sm = MagicMock()
    sm.get_settings = MagicMock(return_value=settings)
    sm.set_settings = MagicMock()
    pool._settings_manager = sm

    pool._unload_engine = AsyncMock()
    pool._load_engine = AsyncMock(side_effect=RuntimeError("base model OOM"))

    try:
        with pytest.raises(AdapterSwapError):
            await pool.swap_adapter("test-model", None)
        # No sidecar should have been touched (no adapter_id to mark).
        # The tmp_path itself is a directory; check no .health.json files exist.
        assert list(tmp_path.glob("**/.health.json")) == []
    finally:
        shutdown_adapter_health_cache()
