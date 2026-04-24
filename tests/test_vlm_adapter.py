# SPDX-License-Identifier: Apache-2.0
"""Tests for VLM adapter plumbing (mirrors LLM adapter tests)."""
from unittest.mock import MagicMock

import pytest

from omlx.adapter_utils import AdapterInfo
from omlx.engine_pool import _build_vlm_engine_kwargs
from omlx.model_settings import ModelSettings


def _entry_with_adapters(adapters):
    e = MagicMock()
    e.model_id = "vlm-base"
    e.model_path = "/path/to/vlm-base"
    e.available_adapters = adapters
    e.estimated_size = 1_000_000_000
    e._adapter_size_bump = 0
    return e


def test_vlm_no_settings_returns_bare_kwargs():
    entry = _entry_with_adapters([])
    kwargs = _build_vlm_engine_kwargs(entry, None)
    assert kwargs == {"model_name": "/path/to/vlm-base"}


def test_vlm_settings_without_adapter_id_returns_bare_kwargs():
    entry = _entry_with_adapters([])
    kwargs = _build_vlm_engine_kwargs(entry, ModelSettings())
    assert kwargs == {"model_name": "/path/to/vlm-base"}


def test_vlm_settings_with_adapter_id_adds_adapter_path(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapters.safetensors").write_bytes(b"\x00" * 5_000_000)
    adapter = AdapterInfo(
        adapter_id="rnd-001",
        path=str(adapter_dir),
        rank=16,
        num_layers=16,
        fine_tune_type="lora",
    )
    entry = _entry_with_adapters([adapter])
    kwargs = _build_vlm_engine_kwargs(entry, ModelSettings(adapter_id="rnd-001"))
    assert kwargs == {
        "model_name": "/path/to/vlm-base",
        "adapter_path": str(adapter_dir),
    }


def test_vlm_unknown_adapter_id_logs_warning_and_loads_base_only(caplog):
    entry = _entry_with_adapters([])
    with caplog.at_level("WARNING"):
        kwargs = _build_vlm_engine_kwargs(entry, ModelSettings(adapter_id="ghost"))
    assert kwargs == {"model_name": "/path/to/vlm-base"}
    assert any("ghost" in r.message for r in caplog.records)


def test_vlm_adapter_path_bumps_entry_size(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapters.safetensors").write_bytes(b"\x00" * 10_000_000)
    adapter = AdapterInfo(
        adapter_id="rnd-001",
        path=str(adapter_dir),
        rank=16,
        num_layers=16,
        fine_tune_type="lora",
    )
    entry = _entry_with_adapters([adapter])
    base_size = entry.estimated_size
    _build_vlm_engine_kwargs(entry, ModelSettings(adapter_id="rnd-001"))
    assert entry.estimated_size == pytest.approx(base_size + 10_000_000, rel=0.05)


def test_vlm_engine_stores_adapter_path_from_constructor():
    from omlx.engine.vlm import VLMBatchedEngine

    engine = VLMBatchedEngine(model_name="vlm-base", adapter_path="/some/adapter")
    assert engine._adapter_path == "/some/adapter"
    assert engine._adapter_metadata is None  # resolved lazily in start()


def test_vlm_engine_defaults_adapter_path_to_none():
    from omlx.engine.vlm import VLMBatchedEngine

    engine = VLMBatchedEngine(model_name="vlm-base")
    assert engine._adapter_path is None
    assert engine._adapter_metadata is None
