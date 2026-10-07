"""Reproducible benchmark runner for SUPAHMLX baseline."""

from __future__ import annotations

import platform
from time import perf_counter

from supahmlx.backends.mlx_compare import detect_mlx
from supahmlx.benchmark.models import BenchmarkRecord
from supahmlx.runtime.engine import RuntimeModel


def run_benchmark(
    runtime: RuntimeModel,
    model_path: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    compare_mlx: bool,
) -> BenchmarkRecord:
    """Run a deterministic benchmark and return structured metrics."""
    prompt_tokens = runtime.tokenizer.encode(prompt)

    ttft_started = perf_counter()
    first_chunk = next(
        runtime.stream_generate(
            prompt,
            max_tokens=1,
            temperature=temperature,
            top_p=top_p,
            seed=0,
        )
    )
    ttft_s = perf_counter() - ttft_started

    full_started = perf_counter()
    result = runtime.generate(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=0,
    )
    full_elapsed = perf_counter() - full_started

    decode_tps = result.generated_tokens / full_elapsed if full_elapsed else 0.0
    prefill_tps = len(prompt_tokens) / ttft_s if ttft_s else 0.0

    mlx_status = None
    if compare_mlx:
        status = detect_mlx()
        mlx_status = status.reason if not status.available else f"available:{status.version}"

    _ = first_chunk  # consumed TTFT probe chunk

    return BenchmarkRecord(
        model_id="toy-reference",
        model_path=model_path,
        backend=runtime.backend.name,
        mode=runtime.mode.name,
        hardware=f"{platform.system()}-{platform.machine()}",
        prompt_tokens=len(prompt_tokens),
        context_tokens=len(prompt_tokens),
        generation_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        prefill_tokens_per_s=prefill_tps,
        decode_tokens_per_s=decode_tps,
        ttft_s=ttft_s,
        total_latency_s=result.latency_s,
        generated_tokens=result.generated_tokens,
        peak_memory_mb=None,
        quality_metric_name=None,
        quality_metric_value=None,
        mlx_status=mlx_status,
    )
