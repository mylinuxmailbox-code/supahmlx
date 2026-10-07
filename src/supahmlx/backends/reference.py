"""Deterministic reference backend used for tests and baseline correctness."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToyReferenceModel:
    """Tiny model metadata for deterministic logits."""

    vocab_size: int = 128


class ReferenceBackend:
    """Toy backend that deterministically maps context -> next-token logits."""

    name = "reference-cpu"

    def __init__(self, model: ToyReferenceModel) -> None:
        self.model = model

    def generate_token_logits(self, context_tokens: list[int]) -> list[float]:
        seed = 17
        for tok in context_tokens[-16:]:
            seed = (seed * 31 + tok) % 2_147_483_647
        logits = []
        for i in range(self.model.vocab_size):
            val = ((seed ^ (i * 1_103_515_245)) & 0xFFFF) / 65535.0
            logits.append((val * 2.0) - 1.0)
        return logits
