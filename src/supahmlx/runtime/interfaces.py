"""Core protocols for runtime extensibility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from supahmlx.config import ModeConfig


@dataclass(frozen=True)
class ExecutionPlan:
    backend_name: str
    mode: ModeConfig
    device: str


class ModelBackend(Protocol):
    """Backend protocol used by RuntimeModel."""

    name: str

    def generate_token_logits(self, context_tokens: list[int]) -> list[float]:
        """Return logits for the next token."""


class PrecisionStrategy(Protocol):
    """Precision policy hook."""

    def describe(self) -> str:
        """Return human-readable strategy name."""


class MemoryPlanner(Protocol):
    """Memory planning hook."""

    def describe(self) -> str:
        """Return planner summary."""


class GraphOptimizer(Protocol):
    """Graph optimization hook."""

    def optimize(self) -> None:
        """Optimize loaded graph."""


class RuntimeProfiler(Protocol):
    """Profiling hook for generation phases."""

    def start(self, phase: str) -> None:
        """Mark start time for a phase."""

    def stop(self, phase: str) -> float:
        """Stop phase and return elapsed seconds."""
