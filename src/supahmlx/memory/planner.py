"""Memory planning placeholders."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryPlan:
    policy: str


def plan_memory(mode_name: str) -> MemoryPlan:
    """Return a mode-linked memory plan for future paged/quantized KV cache strategies."""
    return MemoryPlan(policy=mode_name)
