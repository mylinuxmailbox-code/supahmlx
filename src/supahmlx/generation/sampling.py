"""Numerically stable sampling utilities."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from supahmlx.errors import ConfigurationError


@dataclass
class SamplingConfig:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    greedy: bool = False


class Sampler:
    """Supports greedy and seeded stochastic token selection."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    @staticmethod
    def _softmax(logits: list[float]) -> list[float]:
        if not logits:
            raise ConfigurationError("Cannot sample from empty logits")
        max_logit = max(logits)
        exps = [math.exp(v - max_logit) for v in logits]
        total = sum(exps)
        return [v / total for v in exps]

    @staticmethod
    def _apply_top_k(probs: list[float], top_k: int | None) -> list[float]:
        if top_k is None or top_k <= 0 or top_k >= len(probs):
            return probs
        ranked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        allowed = set(ranked[:top_k])
        filtered = [p if i in allowed else 0.0 for i, p in enumerate(probs)]
        denom = sum(filtered)
        return [p / denom if denom else 0.0 for p in filtered]

    @staticmethod
    def _apply_top_p(probs: list[float], top_p: float) -> list[float]:
        if top_p >= 1.0:
            return probs
        if not 0 < top_p <= 1:
            raise ConfigurationError("top_p must be in (0, 1].")
        ranked = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
        cumulative = 0.0
        keep: set[int] = set()
        for idx, prob in ranked:
            keep.add(idx)
            cumulative += prob
            if cumulative >= top_p:
                break
        filtered = [p if i in keep else 0.0 for i, p in enumerate(probs)]
        denom = sum(filtered)
        return [p / denom if denom else 0.0 for p in filtered]

    def sample(self, logits: list[float], config: SamplingConfig) -> int:
        if config.greedy or config.temperature == 0:
            return max(range(len(logits)), key=logits.__getitem__)
        if config.temperature < 0:
            raise ConfigurationError("temperature must be >= 0")
        scaled = [v / config.temperature for v in logits]
        probs = self._softmax(scaled)
        probs = self._apply_top_k(probs, config.top_k)
        probs = self._apply_top_p(probs, config.top_p)
        r = self._rng.random()
        cumulative = 0.0
        for idx, prob in enumerate(probs):
            cumulative += prob
            if r <= cumulative:
                return idx
        return len(probs) - 1
