# SPDX-License-Identifier: Apache-2.0
"""Schema-level tests for ChatCompletionRequest.adapter_id (B8 — schema only).

Behavior wiring (the actual swap in the route handler) lands in B9.
"""
from omlx.api.openai_models import ChatCompletionRequest


def _base_kwargs():
    return {
        "model": "x",
        "messages": [{"role": "user", "content": "hi"}],
    }


def test_request_without_adapter_id_field_has_default_none_and_unset():
    """Absent adapter_id: field defaults to None AND not in model_fields_set."""
    req = ChatCompletionRequest(**_base_kwargs())
    assert req.adapter_id is None
    assert "adapter_id" not in req.model_fields_set


def test_request_with_explicit_null_adapter_id_is_in_model_fields_set():
    """Explicit null: field is None AND in model_fields_set."""
    req = ChatCompletionRequest(**_base_kwargs(), adapter_id=None)
    assert req.adapter_id is None
    assert "adapter_id" in req.model_fields_set


def test_request_with_adapter_id_string_round_trips():
    """String value round-trips through pydantic."""
    req = ChatCompletionRequest(**_base_kwargs(), adapter_id="rnd-005")
    assert req.adapter_id == "rnd-005"
    assert "adapter_id" in req.model_fields_set


def test_request_adapter_id_field_is_optional_in_serialization():
    """Default-None field is excluded by exclude_none and exclude_unset."""
    req = ChatCompletionRequest(**_base_kwargs())
    dumped = req.model_dump(exclude_none=True)
    assert "adapter_id" not in dumped
    dumped_unset = req.model_dump(exclude_unset=True)
    assert "adapter_id" not in dumped_unset


def test_request_adapter_id_explicit_null_in_dump_when_not_excluded():
    """Explicit null appears in model_dump unless excluded."""
    req = ChatCompletionRequest(**_base_kwargs(), adapter_id=None)
    dumped = req.model_dump()
    assert "adapter_id" in dumped
    assert dumped["adapter_id"] is None
    # exclude_unset=False is default; explicit null is "set" so it appears.
    dumped_unset = req.model_dump(exclude_unset=True)
    assert "adapter_id" in dumped_unset


def test_request_validates_adapter_id_must_be_string_or_none():
    """Non-string, non-null adapter_id is rejected by pydantic validation."""
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ChatCompletionRequest(**_base_kwargs(), adapter_id=123)
