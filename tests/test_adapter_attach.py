# SPDX-License-Identifier: Apache-2.0
"""Tests for two-root adapter discovery with omlx.json sidecar."""
import json
from pathlib import Path

import pytest

from omlx.adapter_utils import resolve_omlx_sidecar
from omlx.model_discovery import (
    discover_models,
    discover_adapters,
)


# --- fixtures ---


def _plant_sidecar(dir: Path, supported: list[str]):
    (dir / "omlx.json").write_text(json.dumps({"supported_models": supported}))


def _plant_adapter_config(dir: Path, rank: int = 16, keys: list[str] | None = None):
    (dir / "adapter_config.json").write_text(json.dumps({
        "fine_tune_type": "lora",
        "num_layers": 16,
        "lora_parameters": {
            "rank": rank, "scale": 2.0, "dropout": 0.0,
            "keys": keys or ["self_attn.q_proj"],
        },
    }))
    (dir / "adapters.safetensors").write_bytes(b"")


def _plant_base(dir: Path, name: str):
    d = dir / name
    d.mkdir()
    (d / "config.json").write_text('{"model_type": "mixtral"}')
    (d / "model.safetensors").write_bytes(b"\x00" * 100_000)
    return d


def _plant_adapter(adapter_root: Path, name: str, supported: list[str] | None, rank: int = 16):
    d = adapter_root / name
    d.mkdir()
    _plant_adapter_config(d, rank=rank)
    if supported is not None:
        _plant_sidecar(d, supported)
    return d


# --- sidecar parsing ---


def test_sidecar_reads_supported_models(tmp_path):
    d = tmp_path / "adapter"
    d.mkdir()
    _plant_sidecar(d, ["gpt-oss-20b"])
    sc = resolve_omlx_sidecar(d)
    assert sc.supported_models == ("gpt-oss-20b",)


def test_sidecar_missing_raises_filenotfound(tmp_path):
    d = tmp_path / "adapter"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_omlx_sidecar(d)


def test_sidecar_without_supported_models_raises_valueerror(tmp_path):
    d = tmp_path / "adapter"
    d.mkdir()
    (d / "omlx.json").write_text("{}")
    with pytest.raises(ValueError, match="supported_models"):
        resolve_omlx_sidecar(d)


def test_sidecar_with_malformed_supported_models_raises_valueerror(tmp_path):
    d = tmp_path / "adapter"
    d.mkdir()
    (d / "omlx.json").write_text(json.dumps({"supported_models": "not-a-list"}))
    with pytest.raises(ValueError):
        resolve_omlx_sidecar(d)


# --- two-root discovery + attach ---


def test_adapter_attaches_to_single_supported_base(tmp_path):
    models_root = tmp_path / "models"
    models_root.mkdir()
    _plant_base(models_root, "gpt-oss-20b")

    adapter_root = tmp_path / "adapters"
    adapter_root.mkdir()
    _plant_adapter(adapter_root, "rnd-001", supported=["gpt-oss-20b"])

    models = discover_models(models_root)
    discover_adapters(adapter_root, models)

    assert "rnd-001" not in models  # adapter is NOT a top-level model
    assert len(models["gpt-oss-20b"].available_adapters) == 1
    assert models["gpt-oss-20b"].available_adapters[0].adapter_id == "rnd-001"
    assert models["gpt-oss-20b"].available_adapters[0].rank == 16


def test_adapter_attaches_to_multiple_supported_bases(tmp_path):
    """A single adapter listed as supporting two bases attaches to both."""
    models_root = tmp_path / "models"
    models_root.mkdir()
    _plant_base(models_root, "gpt-oss-20b-Q4")
    _plant_base(models_root, "gpt-oss-20b-Q8")

    adapter_root = tmp_path / "adapters"
    adapter_root.mkdir()
    _plant_adapter(
        adapter_root, "rnd-001",
        supported=["gpt-oss-20b-Q4", "gpt-oss-20b-Q8"],
    )

    models = discover_models(models_root)
    discover_adapters(adapter_root, models)

    assert "rnd-001" in {a.adapter_id for a in models["gpt-oss-20b-Q4"].available_adapters}
    assert "rnd-001" in {a.adapter_id for a in models["gpt-oss-20b-Q8"].available_adapters}


def test_adapter_with_hf_style_basename_matches(tmp_path):
    """supported_models: ['mlx-community/gpt-oss-20b-MXFP4-Q8'] matches local 'gpt-oss-20b-MXFP4-Q8'."""
    models_root = tmp_path / "models"
    models_root.mkdir()
    _plant_base(models_root, "gpt-oss-20b-MXFP4-Q8")

    adapter_root = tmp_path / "adapters"
    adapter_root.mkdir()
    _plant_adapter(adapter_root, "rnd-001",
                   supported=["mlx-community/gpt-oss-20b-MXFP4-Q8"])

    models = discover_models(models_root)
    discover_adapters(adapter_root, models)

    assert len(models["gpt-oss-20b-MXFP4-Q8"].available_adapters) == 1


def test_adapter_with_unknown_base_is_skipped_with_warning(tmp_path, caplog):
    models_root = tmp_path / "models"
    models_root.mkdir()
    _plant_base(models_root, "base")

    adapter_root = tmp_path / "adapters"
    adapter_root.mkdir()
    _plant_adapter(adapter_root, "orphan", supported=["nonexistent-base"])

    models = discover_models(models_root)
    with caplog.at_level("WARNING"):
        discover_adapters(adapter_root, models)

    assert models["base"].available_adapters == []
    assert any("orphan" in r.message for r in caplog.records)


def test_adapter_without_sidecar_is_skipped_with_warning(tmp_path, caplog):
    models_root = tmp_path / "models"
    models_root.mkdir()
    _plant_base(models_root, "base")

    adapter_root = tmp_path / "adapters"
    adapter_root.mkdir()
    d = adapter_root / "no-sidecar"
    d.mkdir()
    _plant_adapter_config(d)  # config but no omlx.json

    models = discover_models(models_root)
    with caplog.at_level("WARNING"):
        discover_adapters(adapter_root, models)

    assert models["base"].available_adapters == []
    assert any("no-sidecar" in r.message for r in caplog.records)


def test_adapter_with_malformed_adapter_config_is_skipped(tmp_path, caplog):
    models_root = tmp_path / "models"
    models_root.mkdir()
    _plant_base(models_root, "base")

    adapter_root = tmp_path / "adapters"
    adapter_root.mkdir()
    d = adapter_root / "bad"
    d.mkdir()
    (d / "adapter_config.json").write_text("not json")
    _plant_sidecar(d, ["base"])

    models = discover_models(models_root)
    with caplog.at_level("WARNING"):
        discover_adapters(adapter_root, models)

    assert models["base"].available_adapters == []
    assert any("bad" in r.message for r in caplog.records)


def test_adapter_root_missing_is_no_op(tmp_path):
    """If ~/.omlx/adapters/ doesn't exist, discover_adapters returns without error."""
    models_root = tmp_path / "models"
    models_root.mkdir()
    _plant_base(models_root, "base")
    models = discover_models(models_root)
    # adapters/ does not exist
    discover_adapters(tmp_path / "adapters", models)  # should not raise
    assert models["base"].available_adapters == []


def test_empty_adapter_root_is_no_op(tmp_path):
    models_root = tmp_path / "models"
    models_root.mkdir()
    _plant_base(models_root, "base")
    adapter_root = tmp_path / "adapters"
    adapter_root.mkdir()
    models = discover_models(models_root)
    discover_adapters(adapter_root, models)
    assert models["base"].available_adapters == []


def test_old_layout_adapter_under_models_root_is_not_discovered(tmp_path):
    """Pre-refactor adapters under ~/.omlx/models/<name>/ are silently ignored (hard cut)."""
    models_root = tmp_path / "models"
    models_root.mkdir()
    _plant_base(models_root, "base")
    # Adapter placed in the OLD (wrong) location
    old_style = models_root / "rnd-001-adapter"
    old_style.mkdir()
    _plant_adapter_config(old_style)
    _plant_sidecar(old_style, ["base"])  # even with sidecar, should not be picked up

    models = discover_models(models_root)

    # rnd-001-adapter should NOT be in models (no config.json → not a base)
    assert "rnd-001-adapter" not in models
    # No adapter root was scanned, so no adapters attached
    assert models["base"].available_adapters == []
