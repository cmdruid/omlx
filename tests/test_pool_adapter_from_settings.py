# SPDX-License-Identifier: Apache-2.0
"""Tests for pool resolving adapter_path from ModelSettings at load time."""
from unittest.mock import MagicMock

import pytest

from omlx.adapter_utils import AdapterInfo
from omlx.engine_pool import EngineEntry, EnginePool, _build_batched_engine_kwargs
from omlx.model_settings import ModelSettings


def _entry_with_adapters(adapters):
    e = MagicMock()
    e.model_id = "base"
    e.model_path = "/path/to/base"
    e.available_adapters = adapters
    return e


def test_no_settings_returns_bare_kwargs():
    entry = _entry_with_adapters([])
    kwargs = _build_batched_engine_kwargs(entry, None)
    assert kwargs == {"model_name": "/path/to/base"}


def test_settings_without_adapter_id_returns_bare_kwargs():
    entry = _entry_with_adapters([])
    kwargs = _build_batched_engine_kwargs(entry, ModelSettings())
    assert kwargs == {"model_name": "/path/to/base"}


def test_settings_with_adapter_id_adds_adapter_path():
    adapter = AdapterInfo(
        adapter_id="rnd-001", path="/path/to/rnd-001",
        rank=16, num_layers=16, fine_tune_type="lora",
    )
    entry = _entry_with_adapters([adapter])
    kwargs = _build_batched_engine_kwargs(entry, ModelSettings(adapter_id="rnd-001"))
    assert kwargs == {
        "model_name": "/path/to/base",
        "adapter_path": "/path/to/rnd-001",
    }


def test_unknown_adapter_id_logs_warning_and_loads_base_only(caplog):
    entry = _entry_with_adapters([])
    with caplog.at_level("WARNING"):
        kwargs = _build_batched_engine_kwargs(
            entry, ModelSettings(adapter_id="ghost-adapter")
        )
    assert kwargs == {"model_name": "/path/to/base"}
    assert any("ghost-adapter" in r.message for r in caplog.records)


def test_picks_correct_adapter_when_multiple_available():
    a1 = AdapterInfo(adapter_id="rnd-001", path="/p1", rank=16, num_layers=16, fine_tune_type="lora")
    a2 = AdapterInfo(adapter_id="rnd-002", path="/p2", rank=32, num_layers=16, fine_tune_type="lora")
    entry = _entry_with_adapters([a1, a2])
    kwargs = _build_batched_engine_kwargs(entry, ModelSettings(adapter_id="rnd-002"))
    assert kwargs["adapter_path"] == "/p2"


def test_engine_entry_available_adapters_exposed_through_pool_status():
    """Verify available_adapters round-trips through pool.get_status()."""
    pool = EnginePool(max_model_memory=None)
    ai = AdapterInfo(
        adapter_id="rnd-001", path="/p", rank=16, num_layers=16, fine_tune_type="lora"
    )
    entry = EngineEntry(
        model_id="base",
        model_path="/base",
        model_type="llm",
        engine_type="batched",
        estimated_size=1,
        available_adapters=[ai],
    )
    pool._entries["base"] = entry
    status = pool.get_status()
    base_status = next(
        (m for m in status.get("models", []) if m.get("id") == "base"), None
    )
    assert base_status is not None
    assert "available_adapters" in base_status
    assert len(base_status["available_adapters"]) == 1
    assert base_status["available_adapters"][0]["adapter_id"] == "rnd-001"
    assert base_status["available_adapters"][0]["rank"] == 16


def test_get_status_surfaces_loaded_adapter_id_when_engine_has_adapter():
    pool = EnginePool(max_model_memory=None)
    mock_engine = MagicMock()
    mock_engine._adapter_path = "/some/path/to/rnd-001-adapter"
    entry = EngineEntry(
        model_id="base",
        model_path="/base",
        model_type="llm",
        engine_type="batched",
        estimated_size=1,
    )
    entry.engine = mock_engine
    pool._entries["base"] = entry
    status = pool.get_status()
    base_status = next((m for m in status.get("models", []) if m.get("id") == "base"), None)
    assert base_status is not None
    assert base_status["loaded_adapter_id"] == "rnd-001-adapter"


def test_get_status_loaded_adapter_id_is_none_when_no_adapter():
    pool = EnginePool(max_model_memory=None)
    mock_engine = MagicMock()
    mock_engine._adapter_path = None
    entry = EngineEntry(
        model_id="base",
        model_path="/base",
        model_type="llm",
        engine_type="batched",
        estimated_size=1,
    )
    entry.engine = mock_engine
    pool._entries["base"] = entry
    status = pool.get_status()
    base_status = next((m for m in status.get("models", []) if m.get("id") == "base"), None)
    assert base_status is not None
    assert base_status["loaded_adapter_id"] is None


def test_get_status_loaded_adapter_id_is_none_when_engine_not_loaded():
    pool = EnginePool(max_model_memory=None)
    entry = EngineEntry(
        model_id="base",
        model_path="/base",
        model_type="llm",
        engine_type="batched",
        estimated_size=1,
    )
    entry.engine = None
    pool._entries["base"] = entry
    status = pool.get_status()
    base_status = next((m for m in status.get("models", []) if m.get("id") == "base"), None)
    assert base_status is not None
    assert base_status["loaded_adapter_id"] is None


# ---------------------------------------------------------------------------
# B5: adapter byte accounting in estimated_size
# ---------------------------------------------------------------------------

def test_adapter_path_bumps_entry_estimated_size(tmp_path):
    """Attaching an adapter bumps estimated_size by the adapter's on-disk size."""
    adapter_dir = tmp_path / "rnd-001"
    adapter_dir.mkdir()
    (adapter_dir / "adapters.safetensors").write_bytes(b"\x00" * 20_000_000)  # 20 MB

    ai = AdapterInfo(
        adapter_id="rnd-001", path=str(adapter_dir),
        rank=16, num_layers=16, fine_tune_type="lora",
    )
    entry = EngineEntry(
        model_id="base", model_path="/base",
        model_type="llm", engine_type="batched",
        estimated_size=1_000_000_000,
        available_adapters=[ai],
    )
    base_size = entry.estimated_size
    _build_batched_engine_kwargs(entry, ModelSettings(adapter_id="rnd-001"))
    assert entry.estimated_size == pytest.approx(base_size + 20_000_000, rel=0.05)


def test_repeated_adapter_kwargs_calls_are_idempotent(tmp_path):
    """Calling _build_batched_engine_kwargs twice for the same entry doesn't double-bump."""
    adapter_dir = tmp_path / "rnd-001"
    adapter_dir.mkdir()
    (adapter_dir / "adapters.safetensors").write_bytes(b"\x00" * 20_000_000)

    ai = AdapterInfo(
        adapter_id="rnd-001", path=str(adapter_dir),
        rank=16, num_layers=16, fine_tune_type="lora",
    )
    entry = EngineEntry(
        model_id="base", model_path="/base",
        model_type="llm", engine_type="batched",
        estimated_size=1_000_000_000,
        available_adapters=[ai],
    )
    _build_batched_engine_kwargs(entry, ModelSettings(adapter_id="rnd-001"))
    after_first = entry.estimated_size
    _build_batched_engine_kwargs(entry, ModelSettings(adapter_id="rnd-001"))
    after_second = entry.estimated_size
    assert after_first == after_second  # no drift on repeated calls


def test_switching_adapter_id_updates_size_bump(tmp_path):
    """Switching from adapter A to adapter B resets the bump to B's size, not the sum."""
    small = tmp_path / "small"
    small.mkdir()
    (small / "adapters.safetensors").write_bytes(b"\x00" * 5_000_000)
    big = tmp_path / "big"
    big.mkdir()
    (big / "adapters.safetensors").write_bytes(b"\x00" * 50_000_000)

    entry = EngineEntry(
        model_id="base", model_path="/base",
        model_type="llm", engine_type="batched",
        estimated_size=1_000_000_000,
        available_adapters=[
            AdapterInfo(adapter_id="small", path=str(small), rank=16, num_layers=16, fine_tune_type="lora"),
            AdapterInfo(adapter_id="big", path=str(big), rank=32, num_layers=16, fine_tune_type="lora"),
        ],
    )
    base_size = entry.estimated_size
    _build_batched_engine_kwargs(entry, ModelSettings(adapter_id="small"))
    assert entry.estimated_size == pytest.approx(base_size + 5_000_000, rel=0.05)
    _build_batched_engine_kwargs(entry, ModelSettings(adapter_id="big"))
    assert entry.estimated_size == pytest.approx(base_size + 50_000_000, rel=0.05)


def test_clearing_adapter_id_reverts_size_bump(tmp_path):
    """Calling with settings.adapter_id=None after a bump reverts to base size."""
    adapter_dir = tmp_path / "rnd-001"
    adapter_dir.mkdir()
    (adapter_dir / "adapters.safetensors").write_bytes(b"\x00" * 20_000_000)

    ai = AdapterInfo(
        adapter_id="rnd-001", path=str(adapter_dir),
        rank=16, num_layers=16, fine_tune_type="lora",
    )
    entry = EngineEntry(
        model_id="base", model_path="/base",
        model_type="llm", engine_type="batched",
        estimated_size=1_000_000_000,
        available_adapters=[ai],
    )
    base_size = entry.estimated_size
    _build_batched_engine_kwargs(entry, ModelSettings(adapter_id="rnd-001"))
    assert entry.estimated_size > base_size
    _build_batched_engine_kwargs(entry, ModelSettings(adapter_id=None))
    assert entry.estimated_size == base_size  # reverted to base
