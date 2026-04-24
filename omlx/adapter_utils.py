# SPDX-License-Identifier: Apache-2.0
"""Utilities for LoRA/PEFT adapter configuration handling.

Public types:
- ``AdapterMetadata`` — full normalized adapter config read from adapter_config.json.
- ``AdapterInfo`` — lightweight UI/API-facing summary attached to a base model's
  ``DiscoveredModel.available_adapters`` list.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AdapterMetadata:
    """Resolved adapter config, normalized for oMLX consumption."""
    base_ref: str               # raw base_model_name_or_path (HF repo string or local path)
    fine_tune_type: str         # "lora" or "dora"; "full" rejected
    rank: int
    scale: float
    dropout: float
    num_layers: int
    keys: tuple[str, ...]       # target module names, frozen for hashability
    raw: dict[str, Any]         # full adapter_config.json for downstream use


@dataclass(frozen=True)
class AdapterInfo:
    """Summary of a LoRA adapter available for a given base model.

    Distinct from `AdapterMetadata` — this is the UI/API-facing shape
    attached to a `DiscoveredModel.available_adapters`. It captures
    just what the admin dashboard needs to display and what the engine
    pool needs to look up a path from a settings id.
    """
    adapter_id: str          # dir name, e.g. "rnd-001-adapter"
    path: str                # absolute path to the adapter directory
    rank: int
    num_layers: int
    fine_tune_type: str      # "lora" or "dora"

    @classmethod
    def from_metadata(cls, adapter_id: str, path: str, md: "AdapterMetadata") -> "AdapterInfo":
        return cls(
            adapter_id=adapter_id,
            path=path,
            rank=md.rank,
            num_layers=md.num_layers,
            fine_tune_type=md.fine_tune_type,
        )


def resolve_adapter_metadata(adapter_dir: Path) -> AdapterMetadata:
    """Read adapter_config.json from adapter_dir and return normalized metadata.

    Raises:
        FileNotFoundError: if adapter_config.json is missing.
        ValueError: if base_model_name_or_path is missing, or fine_tune_type is "full".
    """
    config_path = Path(adapter_dir) / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found at {config_path}")
    with config_path.open() as f:
        raw = json.load(f)
    # Accept both PEFT-style key and mlx-lm training-config key ("model")
    base_ref = raw.get("base_model_name_or_path") or raw.get("model")
    if not base_ref:
        raise ValueError(
            f"adapter_config.json at {config_path} missing base_model_name_or_path"
        )
    fine_tune_type = raw.get("fine_tune_type", "lora")
    if fine_tune_type == "full":
        raise ValueError(
            f"adapter at {adapter_dir} has fine_tune_type=full; "
            "full fine-tune not supported as an overlay"
        )
    lora_params = raw.get("lora_parameters", {})
    return AdapterMetadata(
        base_ref=base_ref,
        fine_tune_type=fine_tune_type,
        rank=int(lora_params.get("rank", 0)),
        scale=float(lora_params.get("scale", 1.0)),
        dropout=float(lora_params.get("dropout", 0.0)),
        num_layers=int(raw.get("num_layers", 0)),
        keys=tuple(lora_params.get("keys", [])),
        raw=raw,
    )
