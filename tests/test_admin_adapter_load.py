# SPDX-License-Identifier: Apache-2.0
"""Tests for POST /admin/api/models/<id>/adapter (synchronous adapter load)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from omlx.engine_pool import AdapterSwapError
from omlx.server import app, _server_state


def _stub_pool_with_adapter(adapter_id="rnd-001"):
    """Build a stub EnginePool exposing one model with one available adapter."""
    adapter = MagicMock()
    adapter.adapter_id = adapter_id
    entry = MagicMock()
    entry.model_id = "test-model"
    entry.engine = MagicMock()
    entry.available_adapters = [adapter]
    pool = MagicMock()
    pool.get_entry = MagicMock(return_value=entry)
    return pool, entry


def _stub_settings(adapter_id_value=None):
    """Build a stub settings manager that returns a mutable settings object."""
    settings = MagicMock()
    settings.adapter_id = adapter_id_value
    settings.to_dict = MagicMock(return_value={"adapter_id": settings.adapter_id})
    sm = MagicMock()
    sm.get_settings = MagicMock(return_value=settings)
    sm.set_settings = MagicMock()
    return sm, settings


def test_post_adapter_endpoint_returns_200_on_success(monkeypatch):
    """Successful swap returns 200 with success=true and matching loaded_adapter_id."""
    pool, _entry = _stub_pool_with_adapter("rnd-001")
    sm, settings = _stub_settings(adapter_id_value=None)
    pool.swap_adapter = AsyncMock()  # success — no raise

    monkeypatch.setattr("omlx.admin.routes._get_engine_pool", lambda: pool)
    monkeypatch.setattr("omlx.admin.routes._get_settings_manager", lambda: sm)

    original = _server_state.api_key
    _server_state.api_key = "test-key"
    try:
        with TestClient(app) as client:
            resp = client.post(
                "/admin/api/models/test-model/adapter",
                headers={"Authorization": "Bearer test-key"},
                json={"adapter_id": "rnd-001"},
            )
    finally:
        _server_state.api_key = original

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["model_id"] == "test-model"
    assert body["loaded_adapter_id"] == "rnd-001"
    pool.swap_adapter.assert_awaited_once_with("test-model", "rnd-001")


def test_post_adapter_endpoint_null_detaches(monkeypatch):
    """Body {adapter_id: null} detaches: success + loaded_adapter_id=None."""
    pool, _entry = _stub_pool_with_adapter()
    sm, settings = _stub_settings(adapter_id_value="rnd-001")
    pool.swap_adapter = AsyncMock()

    monkeypatch.setattr("omlx.admin.routes._get_engine_pool", lambda: pool)
    monkeypatch.setattr("omlx.admin.routes._get_settings_manager", lambda: sm)

    original = _server_state.api_key
    _server_state.api_key = "test-key"
    try:
        with TestClient(app) as client:
            resp = client.post(
                "/admin/api/models/test-model/adapter",
                headers={"Authorization": "Bearer test-key"},
                json={"adapter_id": None},
            )
    finally:
        _server_state.api_key = original

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["loaded_adapter_id"] is None
    pool.swap_adapter.assert_awaited_once_with("test-model", None)


def test_post_adapter_endpoint_unknown_id_returns_400(monkeypatch):
    """AdapterSwapError(adapter_not_found) → 400 with structured error."""
    pool, _entry = _stub_pool_with_adapter("rnd-001")
    sm, settings = _stub_settings()
    pool.swap_adapter = AsyncMock(
        side_effect=AdapterSwapError("adapter_not_found", "Adapter 'ghost' not in available_adapters")
    )

    monkeypatch.setattr("omlx.admin.routes._get_engine_pool", lambda: pool)
    monkeypatch.setattr("omlx.admin.routes._get_settings_manager", lambda: sm)

    original = _server_state.api_key
    _server_state.api_key = "test-key"
    try:
        with TestClient(app) as client:
            resp = client.post(
                "/admin/api/models/test-model/adapter",
                headers={"Authorization": "Bearer test-key"},
                json={"adapter_id": "ghost"},
            )
    finally:
        _server_state.api_key = original

    assert resp.status_code == 400
    body = resp.json()
    assert body["success"] is False
    err = body.get("error", {})
    assert err.get("type") == "adapter_not_found"
    assert "ghost" in err.get("message", "")


def test_post_adapter_endpoint_load_failure_reverts_settings(monkeypatch):
    """AdapterSwapError(adapter_load_failed) → 502 with reverted settings."""
    pool, _entry = _stub_pool_with_adapter("rnd-001")
    # settings.adapter_id starts at "rnd-baseline"
    sm, settings = _stub_settings(adapter_id_value="rnd-baseline")
    # The settings.adapter_id field will be mutated by swap_adapter; simulate that
    # by capturing the call sequence: get_settings returns the same object every time,
    # and swap_adapter (in real code) would have set it to "rnd-broken" before failing.
    # We simulate by having swap_adapter mutate it.
    async def _failing_swap(_model_id, adapter_id):
        # Simulate the settings mutation that happens inside swap_adapter
        settings.adapter_id = adapter_id
        raise AdapterSwapError("adapter_load_failed", "shape mismatch in self_attn.q_proj")

    pool.swap_adapter = _failing_swap

    monkeypatch.setattr("omlx.admin.routes._get_engine_pool", lambda: pool)
    monkeypatch.setattr("omlx.admin.routes._get_settings_manager", lambda: sm)

    original = _server_state.api_key
    _server_state.api_key = "test-key"
    try:
        with TestClient(app) as client:
            resp = client.post(
                "/admin/api/models/test-model/adapter",
                headers={"Authorization": "Bearer test-key"},
                json={"adapter_id": "rnd-broken"},
            )
    finally:
        _server_state.api_key = original

    assert resp.status_code == 502
    body = resp.json()
    assert body["success"] is False
    err = body.get("error", {})
    assert err.get("type") == "adapter_load_failed"
    assert "shape mismatch" in err.get("message", "")
    # Settings should have been reverted to the original "rnd-baseline" value.
    assert settings.adapter_id == "rnd-baseline"
    sm.set_settings.assert_called()  # called at least once for the revert


def test_post_adapter_endpoint_unknown_model_returns_404(monkeypatch):
    """A model id not in the pool returns 404 (entry is None)."""
    pool = MagicMock()
    pool.get_entry = MagicMock(return_value=None)
    sm, _ = _stub_settings()

    monkeypatch.setattr("omlx.admin.routes._get_engine_pool", lambda: pool)
    monkeypatch.setattr("omlx.admin.routes._get_settings_manager", lambda: sm)

    original = _server_state.api_key
    _server_state.api_key = "test-key"
    try:
        with TestClient(app) as client:
            resp = client.post(
                "/admin/api/models/ghost-model/adapter",
                headers={"Authorization": "Bearer test-key"},
                json={"adapter_id": "rnd-001"},
            )
    finally:
        _server_state.api_key = original

    assert resp.status_code == 404
