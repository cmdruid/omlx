# SPDX-License-Identifier: Apache-2.0
"""Tests for adapter_path threading through BatchedEngine."""
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omlx.engine.batched import BatchedEngine


def _make_adapter_dir(tmp_path: Path) -> Path:
    """Minimal adapter dir usable by resolve_adapter_metadata."""
    d = tmp_path / "adapter"
    d.mkdir()
    (d / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "base",
                "fine_tune_type": "lora",
                "num_layers": 16,
                "lora_parameters": {
                    "rank": 16,
                    "scale": 2.0,
                    "dropout": 0.0,
                    "keys": [],
                },
            }
        )
    )
    (d / "adapters.safetensors").write_bytes(b"")
    return d


# ---------------------------------------------------------------------------
# Constructor tests — no async, no mocking needed
# ---------------------------------------------------------------------------


def test_engine_stores_adapter_path_from_constructor():
    engine = BatchedEngine(model_name="base", adapter_path="/some/adapter")
    assert engine._adapter_path == "/some/adapter"
    assert engine._adapter_metadata is None  # not resolved until start()


def test_engine_defaults_adapter_path_to_none():
    engine = BatchedEngine(model_name="base")
    assert engine._adapter_path is None
    assert engine._adapter_metadata is None


def test_engine_adapter_path_keyword_only_by_convention():
    """adapter_path must be passable as a keyword arg (the only sane way)."""
    engine = BatchedEngine(model_name="base", adapter_path="/path")
    assert engine._adapter_path == "/path"


# ---------------------------------------------------------------------------
# start() tests — heavily mocked to avoid real MLX / Metal calls
# ---------------------------------------------------------------------------


def _make_fake_engine_core():
    """Build a minimal fake AsyncEngineCore that satisfies start()."""
    fake_inner_engine = MagicMock()
    fake_inner_engine.start = AsyncMock()
    fake_inner_engine.scheduler = MagicMock()

    fake_engine_core = MagicMock()
    fake_engine_core.engine = fake_inner_engine
    return fake_engine_core


@pytest.mark.asyncio
async def test_engine_passes_adapter_path_to_load(tmp_path):
    adapter_dir = _make_adapter_dir(tmp_path)
    engine = BatchedEngine(model_name="base", adapter_path=str(adapter_dir))

    fake_model = MagicMock()
    fake_tokenizer = MagicMock()
    fake_engine_core = _make_fake_engine_core()

    # `load` and other heavy deps are imported inline inside start(), so we
    # patch them at their source modules (not as omlx.engine.batched attributes).
    with (
        patch("mlx_lm.load", return_value=(fake_model, fake_tokenizer)) as mock_load,
        patch(
            "omlx.utils.model_loading.apply_post_load_transforms",
            side_effect=lambda m, *a, **k: m,
        ),
        patch("omlx.engine_core.get_mlx_executor", return_value=None),
        patch(
            "omlx.engine_core.AsyncEngineCore",
            return_value=fake_engine_core,
        ),
        patch("omlx.engine_core.EngineConfig"),
        patch("omlx.scheduler.SchedulerConfig"),
        patch("omlx.utils.tokenizer.get_tokenizer_config", return_value={}),
    ):
        await engine.start()

    mock_load.assert_called_once()
    kwargs = mock_load.call_args.kwargs
    assert kwargs.get("adapter_path") == str(adapter_dir)
    # Metadata should now be populated from adapter_config.json
    assert engine._adapter_metadata is not None
    assert engine._adapter_metadata.rank == 16
    assert engine._adapter_metadata.base_ref == "base"


@pytest.mark.asyncio
async def test_engine_omits_adapter_path_when_none():
    engine = BatchedEngine(model_name="base")

    fake_model = MagicMock()
    fake_tokenizer = MagicMock()
    fake_engine_core = _make_fake_engine_core()

    with (
        patch("mlx_lm.load", return_value=(fake_model, fake_tokenizer)) as mock_load,
        patch(
            "omlx.utils.model_loading.apply_post_load_transforms",
            side_effect=lambda m, *a, **k: m,
        ),
        patch("omlx.engine_core.get_mlx_executor", return_value=None),
        patch(
            "omlx.engine_core.AsyncEngineCore",
            return_value=fake_engine_core,
        ),
        patch("omlx.engine_core.EngineConfig"),
        patch("omlx.scheduler.SchedulerConfig"),
        patch("omlx.utils.tokenizer.get_tokenizer_config", return_value={}),
    ):
        await engine.start()

    mock_load.assert_called_once()
    kwargs = mock_load.call_args.kwargs
    # adapter_path=None is acceptable; mlx_lm.load treats it as no adapter
    assert kwargs.get("adapter_path") is None
    # Metadata stays None when no adapter path was given
    assert engine._adapter_metadata is None


@pytest.mark.asyncio
async def test_engine_metadata_not_set_before_start(tmp_path):
    """Metadata stays None until start() is called, even if adapter_path is set."""
    adapter_dir = _make_adapter_dir(tmp_path)
    engine = BatchedEngine(model_name="base", adapter_path=str(adapter_dir))

    # No start() call — metadata is still None
    assert engine._adapter_metadata is None
    assert engine._adapter_path == str(adapter_dir)
