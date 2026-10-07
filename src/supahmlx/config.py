"""Runtime mode policies for quality/speed tradeoffs."""

from __future__ import annotations

from dataclasses import dataclass

from supahmlx.errors import ConfigurationError


@dataclass(frozen=True)
class ModeConfig:
    """Validated runtime policy.

    These values are policy defaults and not benchmark claims.
    """

    name: str
    temperature_default: float
    top_p_default: float
    top_k_default: int
    use_speculative: bool
    kv_cache_policy: str
    precision_policy: str


_MODES: dict[str, ModeConfig] = {
    "quality": ModeConfig(
        name="quality",
        temperature_default=0.7,
        top_p_default=0.95,
        top_k_default=50,
        use_speculative=False,
        kv_cache_policy="full_precision",
        precision_policy="conservative",
    ),
    "balanced": ModeConfig(
        name="balanced",
        temperature_default=0.8,
        top_p_default=0.92,
        top_k_default=40,
        use_speculative=True,
        kv_cache_policy="mixed",
        precision_policy="balanced",
    ),
    "turbo": ModeConfig(
        name="turbo",
        temperature_default=0.9,
        top_p_default=0.9,
        top_k_default=30,
        use_speculative=True,
        kv_cache_policy="aggressive",
        precision_policy="throughput",
    ),
    "insane": ModeConfig(
        name="insane",
        temperature_default=1.0,
        top_p_default=0.85,
        top_k_default=20,
        use_speculative=True,
        kv_cache_policy="experimental",
        precision_policy="max_throughput",
    ),
}


def parse_mode(mode: str | ModeConfig) -> ModeConfig:
    """Return a validated mode configuration."""
    if isinstance(mode, ModeConfig):
        if mode.name not in _MODES:
            raise ConfigurationError(f"Unknown custom mode name: {mode.name}")
        return mode
    normalized = mode.strip().lower()
    if normalized not in _MODES:
        available = ", ".join(sorted(_MODES))
        raise ConfigurationError(f"Unknown mode '{mode}'. Available modes: {available}.")
    return _MODES[normalized]


def available_modes() -> tuple[str, ...]:
    """Return mode names in stable order."""
    return tuple(_MODES)
