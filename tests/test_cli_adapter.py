# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx adapter list / show CLI commands."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _plant_base(models_root: Path, name: str):
    d = models_root / name
    d.mkdir()
    (d / "config.json").write_text('{"model_type": "mixtral"}')
    (d / "model.safetensors").write_bytes(b"\x00" * 100_000)
    return d


def _plant_adapter(adapter_root: Path, name: str, supported: list[str], rank: int = 16):
    d = adapter_root / name
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps({
        "fine_tune_type": "lora",
        "num_layers": 16,
        "lora_parameters": {
            "rank": rank, "scale": 2.0, "dropout": 0.0,
            "keys": ["self_attn.q_proj", "self_attn.k_proj"],
        },
    }))
    (d / "adapters.safetensors").write_bytes(b"")
    (d / "omlx.json").write_text(json.dumps({"supported_models": supported}))
    return d


def _make_fake_omlx_home(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / ".omlx"
    models = root / "models"
    adapters = root / "adapters"
    models.mkdir(parents=True)
    adapters.mkdir()
    return models, adapters


def test_adapter_list_empty(tmp_path, capsys):
    from omlx.cli import adapter_list_command
    models_dir, _ = _make_fake_omlx_home(tmp_path)
    args = MagicMock(models_dir=str(models_dir))
    rc = adapter_list_command(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "No adapters discovered" in out


def test_adapter_list_shows_discovered_adapter(tmp_path, capsys):
    from omlx.cli import adapter_list_command
    models_dir, adapters_dir = _make_fake_omlx_home(tmp_path)
    _plant_base(models_dir, "gpt-oss-20b")
    _plant_adapter(adapters_dir, "rnd-001", supported=["gpt-oss-20b"])

    args = MagicMock(models_dir=str(models_dir))
    rc = adapter_list_command(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "rnd-001" in out
    assert "gpt-oss-20b" in out
    assert "lora" in out


def test_adapter_show_existing(tmp_path, capsys):
    from omlx.cli import adapter_show_command
    models_dir, adapters_dir = _make_fake_omlx_home(tmp_path)
    _plant_base(models_dir, "gpt-oss-20b")
    _plant_adapter(adapters_dir, "rnd-001", supported=["gpt-oss-20b"])

    args = MagicMock(models_dir=str(models_dir), adapter_id="rnd-001")
    rc = adapter_show_command(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "id:" in out and "rnd-001" in out
    assert "rank:" in out and "16" in out
    assert "supported_models:" in out
    assert "gpt-oss-20b" in out
    assert "attaches_to:" in out
    assert "locally discovered" in out


def test_adapter_show_nonexistent_returns_error(tmp_path, capsys):
    from omlx.cli import adapter_show_command
    models_dir, _ = _make_fake_omlx_home(tmp_path)
    args = MagicMock(models_dir=str(models_dir), adapter_id="ghost")
    rc = adapter_show_command(args)
    assert rc == 1
    err = capsys.readouterr().err
    assert "ghost" in err or "not found" in err


def test_adapter_show_with_orphaned_sidecar(tmp_path, capsys):
    """Adapter exists but its supported_models lists a base that isn't discovered."""
    from omlx.cli import adapter_show_command
    models_dir, adapters_dir = _make_fake_omlx_home(tmp_path)
    # No base planted
    _plant_adapter(adapters_dir, "orphan", supported=["nonexistent-base"])

    args = MagicMock(models_dir=str(models_dir), adapter_id="orphan")
    rc = adapter_show_command(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "orphan" in out
    assert "nonexistent-base" in out  # listed under supported_models
    assert "(none" in out or "does not match" in out  # listed under attaches_to
