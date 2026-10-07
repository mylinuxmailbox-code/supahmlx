"""Benchmark data models and serialization helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class BenchmarkRecord:
    model_id: str
    model_path: str
    backend: str
    mode: str
    hardware: str
    prompt_tokens: int
    context_tokens: int
    generation_tokens: int
    temperature: float
    top_p: float
    prefill_tokens_per_s: float
    decode_tokens_per_s: float
    ttft_s: float
    total_latency_s: float
    generated_tokens: int
    peak_memory_mb: float | None
    quality_metric_name: str | None
    quality_metric_value: float | None
    mlx_status: str | None = None
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)
