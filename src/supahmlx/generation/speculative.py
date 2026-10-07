"""Speculative decoding interfaces and conservative reference implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SpeculativeStats:
    drafted_tokens: int
    accepted_tokens: int


class SpeculativeDecoder(Protocol):
    """Contract for speculative decoding strategies."""

    def step(self, prompt_tokens: list[int], max_tokens: int) -> tuple[list[int], SpeculativeStats]:
        """Generate continuation tokens and report speculative stats."""


class ConservativeSpeculativeDecoder:
    """Reference implementation that never overclaims speculative speedups.

    It emits tokens using the target path only and reports zero drafted tokens.
    """

    def __init__(self, target_step) -> None:
        self._target_step = target_step

    def step(self, prompt_tokens: list[int], max_tokens: int) -> tuple[list[int], SpeculativeStats]:
        generated = self._target_step(prompt_tokens, max_tokens)
        return generated, SpeculativeStats(drafted_tokens=0, accepted_tokens=len(generated))
