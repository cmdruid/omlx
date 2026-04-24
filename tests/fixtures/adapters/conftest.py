"""Pytest fixtures for LoRA adapter test scenarios.

Each fixture yields a freshly-built adapter directory under tmp_path with a valid
adapter_config.json and an empty adapters.safetensors placeholder. Tests that need
to exercise load_adapters() should monkey-patch it — the safetensors file is empty
and not a real adapter payload.
"""
import json
from pathlib import Path

import pytest

_RANK16_CONFIG = {
    "base_model_name_or_path": "mlx-community/gpt-oss-20b-MXFP4-Q8",
    "fine_tune_type": "lora",
    "num_layers": 16,
    "lora_parameters": {
        "rank": 16,
        "scale": 2.0,
        "dropout": 0.0,
        "keys": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        ],
    },
}

_RANK32_CONFIG = {**_RANK16_CONFIG, "lora_parameters": {**_RANK16_CONFIG["lora_parameters"], "rank": 32}}


def _build_adapter(dest: Path, config: dict) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "adapter_config.json").write_text(json.dumps(config, indent=2))
    (dest / "adapters.safetensors").write_bytes(b"")
    return dest


@pytest.fixture
def rank16_adapter_path(tmp_path: Path) -> Path:
    return _build_adapter(tmp_path / "rank16_qkvo", _RANK16_CONFIG)


@pytest.fixture
def rank32_adapter_path(tmp_path: Path) -> Path:
    return _build_adapter(tmp_path / "rank32_qkvo", _RANK32_CONFIG)
