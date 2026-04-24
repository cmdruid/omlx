# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.adapter_utils."""
import json
from pathlib import Path

import pytest

from omlx.adapter_utils import (
    AdapterMetadata,
    resolve_adapter_metadata,
)


# --- resolve_adapter_metadata ---


def test_resolve_reads_base_model_ref(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "mlx-community/gpt-oss-20b-MXFP4-Q8",
        "fine_tune_type": "lora",
        "num_layers": 16,
        "lora_parameters": {"rank": 16, "scale": 2.0, "dropout": 0.0, "keys": []},
    }))
    md = resolve_adapter_metadata(tmp_path)
    assert md.base_ref == "mlx-community/gpt-oss-20b-MXFP4-Q8"
    assert md.rank == 16
    assert md.num_layers == 16
    assert md.fine_tune_type == "lora"


def test_resolve_raises_on_missing_config(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_adapter_metadata(tmp_path)


def test_resolve_returns_empty_base_ref_when_absent(tmp_path):
    """base_model_name_or_path is optional; model matching now uses omlx.json sidecar."""
    (tmp_path / "adapter_config.json").write_text("{}")
    md = resolve_adapter_metadata(tmp_path)
    assert md.base_ref == ""


def test_resolve_rejects_full_fine_tune(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "foo",
        "fine_tune_type": "full",
    }))
    with pytest.raises(ValueError, match="full fine-tune not supported"):
        resolve_adapter_metadata(tmp_path)
