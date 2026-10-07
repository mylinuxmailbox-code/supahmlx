"""Simple whitespace tokenizer used in tests and examples."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SimpleTokenizer:
    """A tiny tokenizer with stable IDs assigned on first sight of tokens."""

    vocab: dict[str, int] = field(default_factory=lambda: {"<unk>": 0})

    def encode(self, text: str) -> list[int]:
        pieces = text.split()
        token_ids: list[int] = []
        for piece in pieces:
            token_ids.append(self.vocab.setdefault(piece, len(self.vocab)))
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        inv = {idx: token for token, idx in self.vocab.items()}
        return " ".join(inv.get(token_id, "<unk>") for token_id in token_ids)
