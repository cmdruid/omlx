# SPDX-License-Identifier: Apache-2.0
"""Benchmark: measure the overhead of loading + serving with a LoRA adapter.

Gated behind OMLX_BENCHMARK=1 (skipped by default). Not a strict pass/fail test —
it records numbers. Loose assertions flag catastrophic regressions, but precise
results are expected to shift with hardware. Paste the stdout into an upstream PR
to back the "load-time adapter merge has negligible hot-path cost" claim.

What's measured:
  - Cold load time (first engine instance, adapter absent / present)
  - Peak GPU memory after load (mx.get_peak_memory())
  - Time-to-first-token on a fixed prompt (averaged over N runs)
  - Generation throughput: tokens/sec for a 64-token completion

Expected wall time: ~3-5 minutes on M5 with gpt-oss-20b-MXFP4-Q8 in warm HF cache.
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from statistics import mean, stdev

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("OMLX_BENCHMARK"),
    reason="requires OMLX_BENCHMARK=1 + real model + adapter on disk",
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
TTFT_RUNS = 5  # average over this many runs
THROUGHPUT_TOKENS = 64


@pytest.mark.asyncio
async def test_benchmark_adapter_load_and_inference_overhead(capsys):
    """Measure base-only vs with-adapter load time, TTFT, throughput, peak memory."""
    from omlx.engine.batched import BatchedEngine
    from omlx.engine_core import get_mlx_executor
    import mlx.core as mx

    loop = asyncio.get_event_loop()
    results: dict[str, dict] = {}

    cases = [
        ("base_only", None),
        ("with_adapter", str(ADAPTER_PATH)),
    ]

    for label, adapter_path in cases:
        # Reset peak memory between cases for clean measurement.
        try:
            mx.reset_peak_memory()
        except AttributeError:
            pass  # older mlx versions; peak will be monotonic across the test

        kwargs = {"model_name": BASE_MODEL}
        if adapter_path is not None:
            kwargs["adapter_path"] = adapter_path

        # --- Cold load ---
        engine = BatchedEngine(**kwargs)
        t0 = time.perf_counter()
        await engine.start()
        load_ms = (time.perf_counter() - t0) * 1000.0
        peak_gb = mx.get_peak_memory() / 1e9

        try:
            # --- TTFT (time to first token) over N runs ---
            ttft_samples_ms: list[float] = []
            for _ in range(TTFT_RUNS):
                t0 = time.perf_counter()
                out = await loop.run_in_executor(
                    get_mlx_executor(),
                    lambda: _generate(engine, PROMPT, max_tokens=1),
                )
                ttft_samples_ms.append((time.perf_counter() - t0) * 1000.0)

            # --- Throughput: 64-token completion ---
            t0 = time.perf_counter()
            out = await loop.run_in_executor(
                get_mlx_executor(),
                lambda: _generate(engine, PROMPT, max_tokens=THROUGHPUT_TOKENS),
            )
            gen_s = time.perf_counter() - t0
            # out may be shorter than THROUGHPUT_TOKENS if eos; use the target
            # for throughput comparison (both cases pay the same ceiling).
            tps = THROUGHPUT_TOKENS / gen_s if gen_s > 0 else 0.0

            results[label] = {
                "load_ms": load_ms,
                "peak_gb": peak_gb,
                "ttft_mean_ms": mean(ttft_samples_ms),
                "ttft_stdev_ms": stdev(ttft_samples_ms) if len(ttft_samples_ms) > 1 else 0.0,
                "throughput_tps": tps,
                "sample_output": (out or "")[:80],
            }
        finally:
            await engine.stop()

    # --- Pretty-print ---
    lines = [
        "",
        "=" * 70,
        f"Adapter overhead benchmark — base={BASE_MODEL}",
        f"  adapter={ADAPTER_PATH.name}, prompt={PROMPT!r}",
        "=" * 70,
        f"{'metric':<22} {'base_only':>14} {'with_adapter':>14} {'delta':>12}",
        "-" * 70,
    ]
    b = results["base_only"]
    a = results["with_adapter"]

    def row(key: str, unit: str, fmt: str = "{:.1f}"):
        bv, av = b[key], a[key]
        delta_pct = (av - bv) / bv * 100 if bv else 0
        lines.append(
            f"{key:<22} "
            f"{fmt.format(bv) + ' ' + unit:>14} "
            f"{fmt.format(av) + ' ' + unit:>14} "
            f"{delta_pct:+.1f} %".rjust(12)
        )

    row("load_ms", "ms")
    row("peak_gb", "GB", "{:.2f}")
    row("ttft_mean_ms", "ms")
    row("throughput_tps", "tok/s", "{:.1f}")
    lines.append("-" * 70)
    lines.append(f"base_only sample:    {b['sample_output']!r}")
    lines.append(f"with_adapter sample: {a['sample_output']!r}")
    lines.append("=" * 70)

    report = "\n".join(lines)
    print(report)
    with capsys.disabled():
        # Also write to stderr so pytest's -s captures it even with output
        # redirection.
        import sys
        sys.stderr.write(report + "\n")

    # --- Loose regression guards ---
    # Load: with-adapter should not be more than 2x base. Rank-16 adapter is
    # ~20 MB on a 12 GB base; we expect <5% overhead in practice. 2x is
    # paranoid.
    assert a["load_ms"] < b["load_ms"] * 2.0, (
        f"with-adapter load time {a['load_ms']:.0f}ms much higher than "
        f"base-only {b['load_ms']:.0f}ms — investigate"
    )
    # TTFT: should be within 30% (sampling noise + warmup); loose bound.
    assert a["ttft_mean_ms"] < b["ttft_mean_ms"] * 1.5, (
        f"with-adapter TTFT {a['ttft_mean_ms']:.0f}ms much higher than "
        f"base-only {b['ttft_mean_ms']:.0f}ms"
    )
    # Throughput: should be within 30%.
    assert a["throughput_tps"] > b["throughput_tps"] * 0.7, (
        f"with-adapter throughput {a['throughput_tps']:.1f} tok/s well below "
        f"base-only {b['throughput_tps']:.1f} tok/s"
    )


def _generate(engine, prompt: str, max_tokens: int = 64) -> str:
    """Direct mlx_lm.generate against the engine's loaded model."""
    from mlx_lm import generate
    return generate(
        engine._model, engine._tokenizer, prompt,
        max_tokens=max_tokens, verbose=False,
    )
