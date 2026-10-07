"""Tokenizer interface."""

from __future__ import annotations

from typing import Protocol


class Tokenizer(Protocol):
    """Dependency-light tokenizer contract for reference workflows."""

    def encode(self, text: str) -> list[int]:
        """Encode text into token ids."""

    def decode(self, token_ids: list[int]) -> str:
        """Decode token ids into text."""
