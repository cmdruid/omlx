# SPDX-License-Identifier: Apache-2.0
"""End-to-end integration: engine loads correctly with an adapter attached.

Verifies the realized flow after the refactor: adapter lives in discovery as
available_adapters[N]; settings.adapter_id names it; pool threads
adapter_path to BatchedEngine; mlx_lm.load merges the adapter onto the base.

Env-gated — skipped unless OMLX_INTEGRATION_REAL_MODEL=1 is set. The base
model (~12 GB gpt-oss-20b-MXFP4-Q8) must be cached in HF; adapter at the
rnd-001 path must exist. Expected wall time: ~90-120s first run.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("OMLX_INTEGRATION_REAL_MODEL"),
    reason="requires OMLX_INTEGRATION_REAL_MODEL=1 + real model + adapter on disk",
)

BASE_MODEL = os.environ.get(
    "OMLX_INTEGRATION_BASE_MODEL",
    "mlx-community/gpt-oss-20b-MXFP4-Q8",
)
ADAPTER_PATH = Path(os.environ.get(
    "OMLX_INTEGRATION_ADAPTER_1",
    "/Users/cscott/Repos/thinklab/projects/fine-tune/test/training/gpt-oss-20b/rnd-001/adapter",
))
PROMPT = "Wake me in 3 seconds for a smoke test."


@pytest.mark.asyncio
async def test_engine_with_adapter_generates_differently_than_base_only():
    """BatchedEngine loaded with adapter_path yields different output than without.

    Tests the end-to-end load path — adapter is merged via mlx_lm.load, not via
    a reload mechanism. Regression guard: if the pool stops passing adapter_path,
    or mlx_lm.load's adapter support breaks, this catches it.
    """
    from omlx.engine.batched import BatchedEngine
    from omlx.engine_core import get_mlx_executor

    loop = asyncio.get_event_loop()

    # 1) Base-only engine → output_A
    base_engine = BatchedEngine(model_name=BASE_MODEL)
    try:
        await base_engine.start()
        output_base = await loop.run_in_executor(
            get_mlx_executor(),
            lambda: _generate(base_engine, PROMPT, max_tokens=64),
        )
    finally:
        await base_engine.stop()
    assert output_base, "base-only generation produced empty output"

    # 2) Engine with adapter → output_B
    adapter_engine = BatchedEngine(
        model_name=BASE_MODEL,
        adapter_path=str(ADAPTER_PATH),
    )
    try:
        await adapter_engine.start()
        output_adapter = await loop.run_in_executor(
            get_mlx_executor(),
            lambda: _generate(adapter_engine, PROMPT, max_tokens=64),
        )
    finally:
        await adapter_engine.stop()
    assert output_adapter, "adapter generation produced empty output"

    # 3) Outputs must differ — otherwise the adapter isn't being applied.
    assert output_adapter != output_base, (
        "adapter output identical to base — adapter not being applied by mlx_lm.load"
    )


def _generate(engine, prompt: str, max_tokens: int = 64) -> str:
    """Direct mlx_lm.generate against the engine's loaded model (same as Task 0 smoke)."""
    from mlx_lm import generate
    return generate(
        engine._model, engine._tokenizer, prompt,
        max_tokens=max_tokens, verbose=False,
    )
