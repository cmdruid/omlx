# SPDX-License-Identifier: Apache-2.0
"""Tests for adapter_id validation in the settings PUT handler."""
from unittest.mock import AsyncMock, MagicMock
import pytest
from fastapi import HTTPException

import omlx.server  # noqa: F401
import omlx.admin.routes as admin_routes
from omlx.adapter_utils import AdapterInfo


def _setup_pool(pool):
    original = admin_routes._get_engine_pool
    admin_routes._get_engine_pool = lambda: pool
    return original


def _restore_pool(original):
    admin_routes._get_engine_pool = original


def _setup_settings_manager(mgr):
    original = admin_routes._get_settings_manager
    admin_routes._get_settings_manager = lambda: mgr
    return original


def _restore_settings_manager(original):
    admin_routes._get_settings_manager = original


def _setup_global_settings(gs):
    original = admin_routes._get_global_settings
    admin_routes._get_global_settings = lambda: gs
    return original


def _restore_global_settings(original):
    admin_routes._get_global_settings = original


def _setup_server_state(ss):
    original = admin_routes._get_server_state
    admin_routes._get_server_state = lambda: ss
    return original


def _restore_server_state(original):
    admin_routes._get_server_state = original


def _build_entry_with_adapters(model_id: str, adapter_ids: list[str]):
    """Create a MagicMock entry exposing available_adapters."""
    entry = MagicMock()
    entry.model_id = model_id
    entry.available_adapters = [
        AdapterInfo(adapter_id=aid, path=f"/p/{aid}", rank=16,
                    num_layers=16, fine_tune_type="lora")
        for aid in adapter_ids
    ]
    entry.is_pinned = False
    entry.is_loading = False
    entry.engine = None
    entry.engine_type = "batched"
    entry.model_type = "llm"
    entry.model_path = "/fake"
    return entry


@pytest.fixture
def mocked_admin():
    """Yield a tuple (pool, settings_manager) with standard mocks applied."""
    pool = MagicMock()
    settings_mgr = MagicMock()
    from omlx.model_settings import ModelSettings
    settings_mgr.get_settings = MagicMock(return_value=ModelSettings())
    settings_mgr.set_settings = MagicMock()
    global_settings = MagicMock()
    server_state = MagicMock()
    originals = (
        _setup_pool(pool),
        _setup_settings_manager(settings_mgr),
        _setup_global_settings(global_settings),
        _setup_server_state(server_state),
    )
    yield pool, settings_mgr
    _restore_pool(originals[0])
    _restore_settings_manager(originals[1])
    _restore_global_settings(originals[2])
    _restore_server_state(originals[3])


@pytest.mark.asyncio
async def test_valid_adapter_id_accepted(mocked_admin):
    pool, settings_mgr = mocked_admin
    entry = _build_entry_with_adapters("base", ["rnd-001", "rnd-002"])
    pool._entries = {"base": entry}
    pool.get_entry = MagicMock(return_value=entry)

    await admin_routes.update_model_settings(
        model_id="base",
        request=admin_routes.ModelSettingsRequest(adapter_id="rnd-001"),
        is_admin=True,
    )
    # settings_manager.set_settings was called with adapter_id="rnd-001"
    call_args = settings_mgr.set_settings.call_args
    saved_settings = call_args[0][1] if len(call_args[0]) > 1 else call_args.kwargs.get("settings")
    assert saved_settings.adapter_id == "rnd-001"


@pytest.mark.asyncio
async def test_invalid_adapter_id_rejected_with_400(mocked_admin):
    pool, _ = mocked_admin
    entry = _build_entry_with_adapters("base", ["rnd-001"])
    pool._entries = {"base": entry}
    pool.get_entry = MagicMock(return_value=entry)
    with pytest.raises(HTTPException) as exc:
        await admin_routes.update_model_settings(
            model_id="base",
            request=admin_routes.ModelSettingsRequest(adapter_id="does-not-exist"),
            is_admin=True,
        )
    assert exc.value.status_code == 400
    assert "does-not-exist" in exc.value.detail
    assert "rnd-001" in exc.value.detail  # lists valid ids


@pytest.mark.asyncio
async def test_null_adapter_id_clears_setting(mocked_admin):
    pool, settings_mgr = mocked_admin
    entry = _build_entry_with_adapters("base", ["rnd-001"])
    # Start with rnd-001 selected
    from omlx.model_settings import ModelSettings
    settings_mgr.get_settings.return_value = ModelSettings(adapter_id="rnd-001")
    pool._entries = {"base": entry}
    pool.get_entry = MagicMock(return_value=entry)

    await admin_routes.update_model_settings(
        model_id="base",
        request=admin_routes.ModelSettingsRequest(adapter_id=None),
        is_admin=True,
    )
    call_args = settings_mgr.set_settings.call_args
    saved_settings = call_args[0][1] if len(call_args[0]) > 1 else call_args.kwargs.get("settings")
    assert saved_settings.adapter_id is None


@pytest.mark.asyncio
async def test_empty_string_adapter_id_treated_as_null(mocked_admin):
    """Frontend sends '' for 'None (base only)'; handler must accept and clear."""
    pool, settings_mgr = mocked_admin
    entry = _build_entry_with_adapters("base", ["rnd-001"])
    pool._entries = {"base": entry}
    pool.get_entry = MagicMock(return_value=entry)
    await admin_routes.update_model_settings(
        model_id="base",
        request=admin_routes.ModelSettingsRequest(adapter_id=""),
        is_admin=True,
    )
    call_args = settings_mgr.set_settings.call_args
    saved_settings = call_args[0][1] if len(call_args[0]) > 1 else call_args.kwargs.get("settings")
    assert saved_settings.adapter_id is None


@pytest.mark.asyncio
async def test_model_with_no_available_adapters_rejects_any_adapter_id(mocked_admin):
    pool, _ = mocked_admin
    entry = _build_entry_with_adapters("base", [])  # empty list
    pool._entries = {"base": entry}
    pool.get_entry = MagicMock(return_value=entry)
    with pytest.raises(HTTPException) as exc:
        await admin_routes.update_model_settings(
            model_id="base",
            request=admin_routes.ModelSettingsRequest(adapter_id="anything"),
            is_admin=True,
        )
    assert exc.value.status_code == 400
    assert "anything" in exc.value.detail
    assert "none available" in exc.value.detail.lower() or "(none" in exc.value.detail


# =============================================================================
# B7: auto-unload engine when adapter_id changes on a loaded model
# =============================================================================


@pytest.mark.asyncio
async def test_adapter_id_change_on_loaded_model_unloads_engine(mocked_admin):
    pool, settings_mgr = mocked_admin
    entry = _build_entry_with_adapters("base", ["rnd-001", "rnd-002"])
    entry.engine = MagicMock()  # loaded
    pool._entries = {"base": entry}
    pool.get_entry = MagicMock(return_value=entry)
    pool._unload_engine = AsyncMock()

    from omlx.model_settings import ModelSettings
    settings_mgr.get_settings.return_value = ModelSettings(adapter_id="rnd-001")

    await admin_routes.update_model_settings(
        model_id="base",
        request=admin_routes.ModelSettingsRequest(adapter_id="rnd-002"),
        is_admin=True,
    )
    pool._unload_engine.assert_awaited_once_with("base")


@pytest.mark.asyncio
async def test_adapter_id_change_on_unloaded_model_does_not_call_unload(mocked_admin):
    pool, settings_mgr = mocked_admin
    entry = _build_entry_with_adapters("base", ["rnd-001", "rnd-002"])
    entry.engine = None  # unloaded
    pool._entries = {"base": entry}
    pool.get_entry = MagicMock(return_value=entry)
    pool._unload_engine = AsyncMock()

    from omlx.model_settings import ModelSettings
    settings_mgr.get_settings.return_value = ModelSettings(adapter_id="rnd-001")

    await admin_routes.update_model_settings(
        model_id="base",
        request=admin_routes.ModelSettingsRequest(adapter_id="rnd-002"),
        is_admin=True,
    )
    pool._unload_engine.assert_not_called()


@pytest.mark.asyncio
async def test_same_adapter_id_does_not_trigger_unload(mocked_admin):
    """Saving without changing adapter_id shouldn't unload even on a loaded model."""
    pool, settings_mgr = mocked_admin
    entry = _build_entry_with_adapters("base", ["rnd-001"])
    entry.engine = MagicMock()  # loaded
    pool._entries = {"base": entry}
    pool.get_entry = MagicMock(return_value=entry)
    pool._unload_engine = AsyncMock()

    from omlx.model_settings import ModelSettings
    settings_mgr.get_settings.return_value = ModelSettings(adapter_id="rnd-001")

    await admin_routes.update_model_settings(
        model_id="base",
        request=admin_routes.ModelSettingsRequest(adapter_id="rnd-001"),  # same
        is_admin=True,
    )
    pool._unload_engine.assert_not_called()


@pytest.mark.asyncio
async def test_unrelated_field_change_does_not_unload(mocked_admin):
    """If adapter_id wasn't in the payload at all, don't unload."""
    pool, settings_mgr = mocked_admin
    entry = _build_entry_with_adapters("base", ["rnd-001"])
    entry.engine = MagicMock()  # loaded
    pool._entries = {"base": entry}
    pool.get_entry = MagicMock(return_value=entry)
    pool._unload_engine = AsyncMock()

    from omlx.model_settings import ModelSettings
    settings_mgr.get_settings.return_value = ModelSettings(adapter_id="rnd-001")

    await admin_routes.update_model_settings(
        model_id="base",
        request=admin_routes.ModelSettingsRequest(temperature=0.5),  # unrelated
        is_admin=True,
    )
    pool._unload_engine.assert_not_called()


# =============================================================================
# Item 5: runtime rescan endpoint — admin-level tests
# =============================================================================


@pytest.mark.asyncio
async def test_rescan_route_returns_summary(mocked_admin):
    pool, _ = mocked_admin
    pool.rescan_adapters = AsyncMock(return_value={"attached": 1, "removed": 0, "total": 1})
    response = await admin_routes.rescan_adapters_route(is_admin=True)
    assert response == {"attached": 1, "removed": 0, "total": 1}
    pool.rescan_adapters.assert_awaited_once()


@pytest.mark.asyncio
async def test_rescan_route_503_when_no_pool():
    pool_original = admin_routes._get_engine_pool
    admin_routes._get_engine_pool = lambda: None
    try:
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            await admin_routes.rescan_adapters_route(is_admin=True)
        assert exc.value.status_code == 503
    finally:
        admin_routes._get_engine_pool = pool_original


def test_rescan_route_requires_admin():
    from inspect import signature
    sig = signature(admin_routes.rescan_adapters_route)
    is_admin_param = sig.parameters.get("is_admin")
    assert is_admin_param is not None
    assert "require_admin" in repr(is_admin_param.default)
