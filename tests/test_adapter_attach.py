# SPDX-License-Identifier: Apache-2.0
"""Tests for two-pass adapter attach in discover_models."""
import json
from pathlib import Path

import pytest

from omlx.model_discovery import discover_models


def _plant_base(path: Path, name: str):
    d = path / name
    d.mkdir()
    (d / "config.json").write_text('{"model_type": "mixtral"}')
    # estimate_model_size requires at least one weight file
    (d / "model.safetensors").write_bytes(b"\x00" * 1024)
    return d


def _plant_adapter(path: Path, name: str, base_ref: str):
    d = path / name
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": base_ref,
        "fine_tune_type": "lora",
        "num_layers": 16,
        "lora_parameters": {
            "rank": 16, "scale": 2.0, "dropout": 0.0,
            "keys": ["self_attn.q_proj"],
        },
    }))
    (d / "adapters.safetensors").write_bytes(b"")
    return d


def test_adapter_attaches_to_base_by_model_id(tmp_path):
    _plant_base(tmp_path, "gpt-oss-20b")
    _plant_adapter(tmp_path, "rnd-001-adapter", base_ref="gpt-oss-20b")

    result = discover_models(tmp_path)

    # Adapter does NOT appear as a top-level discovered model
    assert "rnd-001-adapter" not in result
    # Base is discovered with the adapter attached
    assert "gpt-oss-20b" in result
    base = result["gpt-oss-20b"]
    assert len(base.available_adapters) == 1
    assert base.available_adapters[0].adapter_id == "rnd-001-adapter"
    assert base.available_adapters[0].rank == 16


def test_adapter_attaches_by_hf_basename(tmp_path):
    """Adapter config references 'mlx-community/gpt-oss-20b' but base is under 'gpt-oss-20b' locally."""
    _plant_base(tmp_path, "gpt-oss-20b-MXFP4-Q8")
    _plant_adapter(
        tmp_path,
        "rnd-001-adapter",
        base_ref="mlx-community/gpt-oss-20b-MXFP4-Q8",
    )

    result = discover_models(tmp_path)

    base = result["gpt-oss-20b-MXFP4-Q8"]
    assert len(base.available_adapters) == 1
    assert base.available_adapters[0].adapter_id == "rnd-001-adapter"


def test_multiple_adapters_attach_to_same_base(tmp_path):
    _plant_base(tmp_path, "base")
    _plant_adapter(tmp_path, "rnd-001", base_ref="base")
    _plant_adapter(tmp_path, "rnd-002", base_ref="base")

    result = discover_models(tmp_path)
    base = result["base"]
    adapter_ids = {a.adapter_id for a in base.available_adapters}
    assert adapter_ids == {"rnd-001", "rnd-002"}


def test_adapter_with_unknown_base_is_skipped_with_warning(tmp_path, caplog):
    _plant_adapter(tmp_path, "orphan", base_ref="nonexistent-base")

    with caplog.at_level("WARNING"):
        result = discover_models(tmp_path)

    # Orphan adapter not registered anywhere
    assert "orphan" not in result
    # And not attached (there's no base to attach to)
    # A warning was logged naming the missing base
    assert any("nonexistent-base" in r.message for r in caplog.records)


def test_adapter_with_malformed_config_is_skipped_with_warning(tmp_path, caplog):
    _plant_base(tmp_path, "base")
    bad = tmp_path / "bad-adapter"
    bad.mkdir()
    (bad / "adapter_config.json").write_text("not json")

    with caplog.at_level("WARNING"):
        result = discover_models(tmp_path)

    assert "bad-adapter" not in result
    assert result["base"].available_adapters == []
    # A warning was logged
    assert any("bad-adapter" in r.message for r in caplog.records)


def test_discovery_count_does_not_include_adapters(tmp_path, caplog):
    """Adapters should not inflate the discovered-models count."""
    _plant_base(tmp_path, "base")
    _plant_adapter(tmp_path, "adapter-1", base_ref="base")

    result = discover_models(tmp_path)
    # Only 1 discovered MODEL (the base); the adapter is attached, not listed.
    assert len(result) == 1
