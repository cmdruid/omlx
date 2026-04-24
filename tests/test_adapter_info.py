# SPDX-License-Identifier: Apache-2.0
# tests/test_adapter_info.py
"""Tests for AdapterInfo + DiscoveredModel.available_adapters scaffolding."""
import json
from pathlib import Path

import pytest

from omlx.adapter_utils import AdapterInfo, AdapterMetadata, resolve_adapter_metadata
from omlx.model_discovery import DiscoveredModel


def test_adapter_info_fields():
    ai = AdapterInfo(
        adapter_id="rnd-001",
        path="/abs/path",
        rank=16,
        num_layers=16,
        fine_tune_type="lora",
    )
    assert ai.adapter_id == "rnd-001"
    assert ai.path == "/abs/path"
    assert ai.rank == 16


def test_adapter_info_frozen():
    ai = AdapterInfo(adapter_id="a", path="/p", rank=16, num_layers=16, fine_tune_type="lora")
    with pytest.raises(Exception):  # FrozenInstanceError
        ai.rank = 32


def test_adapter_info_from_metadata(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "base",
        "fine_tune_type": "lora",
        "num_layers": 16,
        "lora_parameters": {"rank": 16, "scale": 2.0, "dropout": 0.0, "keys": []},
    }))
    md = resolve_adapter_metadata(tmp_path)
    ai = AdapterInfo.from_metadata("rnd-001", str(tmp_path), md)
    assert ai.adapter_id == "rnd-001"
    assert ai.path == str(tmp_path)
    assert ai.rank == md.rank
    assert ai.num_layers == md.num_layers
    assert ai.fine_tune_type == md.fine_tune_type


def test_discovered_model_available_adapters_defaults_to_empty_list():
    dm = DiscoveredModel(
        model_id="base",
        model_path="/p",
        model_type="llm",
        engine_type="batched",
        estimated_size=1,
    )
    assert dm.available_adapters == []


def test_discovered_model_available_adapters_accepts_list():
    ai = AdapterInfo(adapter_id="a", path="/p", rank=16, num_layers=16, fine_tune_type="lora")
    dm = DiscoveredModel(
        model_id="base",
        model_path="/p",
        model_type="llm",
        engine_type="batched",
        estimated_size=1,
        available_adapters=[ai],
    )
    assert len(dm.available_adapters) == 1
    assert dm.available_adapters[0].adapter_id == "a"


def test_discovered_model_available_adapters_instances_are_independent():
    """default_factory should give each instance its own list."""
    dm1 = DiscoveredModel(model_id="a", model_path="/p", model_type="llm", engine_type="batched", estimated_size=1)
    dm2 = DiscoveredModel(model_id="b", model_path="/p", model_type="llm", engine_type="batched", estimated_size=1)
    assert dm1.available_adapters is not dm2.available_adapters
