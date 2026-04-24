# SPDX-License-Identifier: Apache-2.0
"""Tests for adapter_id field on ModelSettings + ModelSettingsRequest."""
from dataclasses import asdict

from omlx.model_settings import ModelSettings


def test_model_settings_adapter_id_defaults_none():
    s = ModelSettings()
    assert s.adapter_id is None


def test_model_settings_adapter_id_accepts_string():
    s = ModelSettings(adapter_id="rnd-001-adapter")
    assert s.adapter_id == "rnd-001-adapter"


def test_model_settings_adapter_id_roundtrips_through_asdict():
    s = ModelSettings(adapter_id="rnd-001-adapter")
    d = asdict(s)
    assert d["adapter_id"] == "rnd-001-adapter"
    # Round-trip via dict → dataclass
    s2 = ModelSettings(**d)
    assert s2.adapter_id == "rnd-001-adapter"


def test_model_settings_request_accepts_adapter_id():
    from omlx.admin.routes import ModelSettingsRequest
    req = ModelSettingsRequest(adapter_id="rnd-001-adapter")
    assert req.adapter_id == "rnd-001-adapter"


def test_model_settings_request_adapter_id_defaults_none():
    from omlx.admin.routes import ModelSettingsRequest
    req = ModelSettingsRequest()
    assert req.adapter_id is None
