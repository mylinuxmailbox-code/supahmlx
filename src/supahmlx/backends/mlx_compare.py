"""Optional MLX comparison backend detection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MlxAvailability:
    available: bool
    reason: str
    version: str | None = None


def detect_mlx() -> MlxAvailability:
    """Detect whether MLX is importable for benchmark comparison."""
    try:
        import mlx  # type: ignore
    except Exception:  # pragma: no cover - environment dependent
        return MlxAvailability(
            available=False,
            reason=(
                "MLX is not installed. Install optional extras with "
                "`pip install supahmlx[mlx]` to enable --compare-mlx."
            ),
        )
    version = getattr(mlx, "__version__", "unknown")
    return MlxAvailability(available=True, reason="MLX available", version=version)
