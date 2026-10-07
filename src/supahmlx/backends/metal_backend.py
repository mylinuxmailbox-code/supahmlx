"""Optional Metal backend integration points."""

from __future__ import annotations

import platform
from pathlib import Path

from supahmlx.errors import BackendUnavailableError


def metal_kernel_source_path() -> Path:
    """Return path to baseline Metal kernels."""
    return Path(__file__).resolve().parent.parent / "metal" / "kernels" / "baseline_kernels.metal"


def ensure_metal_runtime_available() -> None:
    """Validate minimal platform prerequisites for future Metal backend."""
    if platform.system() != "Darwin":
        raise BackendUnavailableError(
            "Metal backend is only available on macOS. "
            "Use device='cpu' or device='auto' for the reference backend."
        )


class MetalBackendPlaceholder:
    """Placeholder backend to anchor future Apple GPU implementation work."""

    name = "metal-placeholder"

    def __init__(self) -> None:
        ensure_metal_runtime_available()

    def generate_token_logits(self, context_tokens: list[int]) -> list[float]:
        raise BackendUnavailableError(
            "Metal backend kernels are placeholders in Phase 1. "
            "Use reference backend for correctness baselines and benchmarking scaffolding."
        )
