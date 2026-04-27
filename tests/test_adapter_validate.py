# SPDX-License-Identifier: Apache-2.0
"""Tests for ?validate=true on /admin/api/adapters/rescan and hard-block on compatible:false."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from omlx.adapter_health import (
    get_adapter_health_cache,
    shutdown_adapter_health_cache,
)
from omlx.engine_pool import AdapterSwapError


@pytest.mark.asyncio
async def test_validate_adapter_returns_true_on_clean_load(monkeypatch):
    """_validate_adapter returns (True, None) when swap_adapter_for_request succeeds."""
    from omlx.engine_pool import EnginePool

    pool = EnginePool.__new__(EnginePool)
    entry = MagicMock()
    entry.engine = None
    entry.available_adapters = []
    entry.swap_lock = asyncio.Lock()
    pool._entries = {"test-model": entry}
    pool._settings_manager = MagicMock()

    # Stub swap_adapter_for_request to succeed both times (validate + restore).
    pool.swap_adapter_for_request = AsyncMock()

    # Patch _loaded_adapter_id to return None (no adapter currently loaded).
    monkeypatch.setattr("omlx.engine_pool._loaded_adapter_id", lambda _e: None)

    adapter = MagicMock()
    adapter.adapter_id = "rnd-001"
    adapter.path = "/fake/rnd-001"

    ok, err = await pool._validate_adapter("test-model", adapter)
    assert ok is True
    assert err is None
    # Called twice: once to load the adapter under test, once to restore original.
    assert pool.swap_adapter_for_request.await_count == 2


@pytest.mark.asyncio
async def test_validate_adapter_returns_false_on_load_failure(monkeypatch):
    """_validate_adapter returns (False, <message>) when swap fails."""
    from omlx.engine_pool import EnginePool

    pool = EnginePool.__new__(EnginePool)
    entry = MagicMock()
    entry.engine = None
    entry.available_adapters = []
    entry.swap_lock = asyncio.Lock()
    pool._entries = {"test-model": entry}
    pool._settings_manager = MagicMock()

    pool.swap_adapter_for_request = AsyncMock(
        side_effect=[
            AdapterSwapError("adapter_load_failed", "shape mismatch in self_attn.q_proj"),
            None,  # restore call — succeeds
        ]
    )

    monkeypatch.setattr("omlx.engine_pool._loaded_adapter_id", lambda _e: None)

    adapter = MagicMock()
    adapter.adapter_id = "rnd-broken"
    adapter.path = "/fake/rnd-broken"

    ok, err = await pool._validate_adapter("test-model", adapter)
    assert ok is False
    assert "shape mismatch" in err


@pytest.mark.asyncio
async def test_rescan_with_validate_true_records_validation(monkeypatch, tmp_path):
    """rescan_adapters(validate=True) writes compatible field to .health.json."""
    from omlx.engine_pool import EnginePool

    pool = EnginePool.__new__(EnginePool)
    pool._lock = asyncio.Lock()

    # Build a fake entry with one adapter.
    adapter_dir = tmp_path / "rnd-001"
    adapter_dir.mkdir()
    adapter = MagicMock()
    adapter.adapter_id = "rnd-001"
    adapter.path = str(adapter_dir)

    entry = MagicMock()
    entry.engine = None
    entry.available_adapters = [adapter]
    entry.swap_lock = asyncio.Lock()

    pool._entries = {"test-model": entry}
    pool._settings_manager = None

    # Stub _validate_adapter to return success without touching the engine.
    pool._validate_adapter = AsyncMock(return_value=(True, None))

    # Stub discover_adapters to re-populate the entry's adapter list.
    # rescan_adapters calls available_adapters.clear() before scanning, so we must
    # re-add the adapter in the stub — otherwise the list stays empty and
    # ensure_health_for_adapter is never called.
    # The function is imported locally inside rescan_adapters, so patch the source module.
    def _stub_discover(root, entries):
        for e in entries.values():
            e.available_adapters.append(adapter)

    monkeypatch.setattr("omlx.model_discovery.discover_adapters", _stub_discover)

    # Redirect Path.home() so adapter_root points into tmp_path.
    omlx_adapters = tmp_path / ".omlx" / "adapters"
    omlx_adapters.mkdir(parents=True)

    original_home = Path.home
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    try:
        result = await pool.rescan_adapters(validate=True)

        # No manual cache.flush() — rescan_adapters flushes itself when
        # validate=True so consumers reading the sidecar immediately see the
        # outcome.
        sidecar = adapter_dir / ".health.json"
        assert sidecar.exists(), f"sidecar missing at {sidecar}"
        data = json.loads(sidecar.read_text())
        assert data["compatible"] is True
    finally:
        shutdown_adapter_health_cache()


@pytest.mark.asyncio
async def test_rescan_with_validate_true_records_failure(monkeypatch, tmp_path):
    """rescan_adapters(validate=True) writes compatible=False + last_load_error
    when _validate_adapter reports failure. End-to-end: cache → disk via the
    rescan-side flush."""
    from omlx.engine_pool import EnginePool

    pool = EnginePool.__new__(EnginePool)
    pool._lock = asyncio.Lock()

    adapter_dir = tmp_path / "rnd-broken"
    adapter_dir.mkdir()
    adapter = MagicMock()
    adapter.adapter_id = "rnd-broken"
    adapter.path = str(adapter_dir)

    entry = MagicMock()
    entry.engine = None
    entry.available_adapters = [adapter]
    entry.swap_lock = asyncio.Lock()

    pool._entries = {"test-model": entry}
    pool._settings_manager = None

    # Stub _validate_adapter to report incompatible.
    pool._validate_adapter = AsyncMock(return_value=(False, "shape mismatch in self_attn.q_proj"))

    def _stub_discover(root, entries):
        for e in entries.values():
            e.available_adapters.append(adapter)

    monkeypatch.setattr("omlx.model_discovery.discover_adapters", _stub_discover)

    omlx_adapters = tmp_path / ".omlx" / "adapters"
    omlx_adapters.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    try:
        await pool.rescan_adapters(validate=True)

        sidecar = adapter_dir / ".health.json"
        assert sidecar.exists(), f"sidecar missing at {sidecar}"
        data = json.loads(sidecar.read_text())
        assert data["compatible"] is False
        assert data["last_load_error"] == "shape mismatch in self_attn.q_proj"
        assert data["last_validated_at"] is not None
    finally:
        shutdown_adapter_health_cache()


@pytest.mark.asyncio
async def test_rescan_without_validate_does_not_change_compatibility(monkeypatch, tmp_path):
    """rescan_adapters(validate=False) (default) leaves compatible field
    untouched — does NOT trigger _validate_adapter or write compatibility."""
    from omlx.adapter_health import save_health, AdapterHealth, load_health
    from omlx.engine_pool import EnginePool

    pool = EnginePool.__new__(EnginePool)
    pool._lock = asyncio.Lock()

    adapter_dir = tmp_path / "rnd-005"
    adapter_dir.mkdir()
    adapter = MagicMock()
    adapter.adapter_id = "rnd-005"
    adapter.path = str(adapter_dir)

    # Pre-seed the sidecar with compatible=True so we can detect any unwanted change.
    pre = AdapterHealth(
        adapter_id="rnd-005",
        installed_at="2026-04-26T10:00:00.000Z",
        last_loaded_at=None,
        last_request_at=None,
        load_count=0,
        compatible=True,
        last_validated_at="2026-04-26T10:00:00.000Z",
        last_load_error=None,
    )
    save_health(adapter_dir, pre)

    entry = MagicMock()
    entry.engine = None
    entry.available_adapters = [adapter]
    entry.swap_lock = asyncio.Lock()

    pool._entries = {"test-model": entry}
    pool._settings_manager = None

    # _validate_adapter must NOT be called on the validate=False path.
    pool._validate_adapter = AsyncMock()

    def _stub_discover(root, entries):
        for e in entries.values():
            e.available_adapters.append(adapter)

    monkeypatch.setattr("omlx.model_discovery.discover_adapters", _stub_discover)

    omlx_adapters = tmp_path / ".omlx" / "adapters"
    omlx_adapters.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    try:
        await pool.rescan_adapters()  # default validate=False

        # _validate_adapter must not have been awaited.
        pool._validate_adapter.assert_not_awaited()

        # Sidecar's compatible field must still be True (untouched).
        on_disk = load_health(adapter_dir)
        assert on_disk is not None
        assert on_disk.compatible is True
        assert on_disk.last_load_error is None
    finally:
        shutdown_adapter_health_cache()


def test_load_adapter_endpoint_blocks_incompatible(monkeypatch, tmp_path):
    """POST /admin/api/models/<id>/adapter returns 400 for compatible:false."""
    from fastapi.testclient import TestClient

    from omlx.adapter_health import (
        get_adapter_health_cache,
        shutdown_adapter_health_cache,
    )
    from omlx.server import _server_state, app

    # Set up an adapter dir and pre-populate the health cache with compatible=False.
    adapter_dir = tmp_path / "rnd-broken"
    adapter_dir.mkdir()

    try:
        cache = get_adapter_health_cache()
        cache.get_or_create(adapter_dir, adapter_id="rnd-broken")
        cache.record_validation(adapter_dir, compatible=False, error="shape mismatch")

        # Build a fake pool entry that exposes the adapter.
        adapter = MagicMock()
        adapter.adapter_id = "rnd-broken"
        adapter.path = str(adapter_dir)
        entry = MagicMock()
        entry.available_adapters = [adapter]

        pool = MagicMock()
        pool.get_entry = MagicMock(return_value=entry)
        pool.swap_adapter = AsyncMock()  # must NOT be called

        sm_mock = MagicMock()
        settings_obj = MagicMock()
        settings_obj.adapter_id = None
        settings_obj.to_dict = MagicMock(return_value={})
        sm_mock.get_settings = MagicMock(return_value=settings_obj)
        sm_mock.set_settings = MagicMock()

        monkeypatch.setattr("omlx.admin.routes._get_engine_pool", lambda: pool)
        monkeypatch.setattr("omlx.admin.routes._get_settings_manager", lambda: sm_mock)

        original_key = _server_state.api_key
        _server_state.api_key = "test-key"
        try:
            with TestClient(app) as client:
                resp = client.post(
                    "/admin/api/models/test-model/adapter",
                    headers={"Authorization": "Bearer test-key"},
                    json={"adapter_id": "rnd-broken"},
                )
        finally:
            _server_state.api_key = original_key

        assert resp.status_code == 400
        body = resp.json()
        assert body["success"] is False
        err = body.get("error", {})
        assert err.get("type") == "adapter_incompatible"
        assert "shape mismatch" in err.get("message", "")
        # swap_adapter must NOT have been called — the hard-block fires first.
        pool.swap_adapter.assert_not_awaited()
    finally:
        shutdown_adapter_health_cache()
