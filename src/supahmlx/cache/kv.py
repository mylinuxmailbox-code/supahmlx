"""Reference KV cache abstraction with validation and stats."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from supahmlx.errors import ConfigurationError

FloatTensor4D = Sequence[Sequence[Sequence[Sequence[float]]]]


@dataclass(frozen=True)
class KVCacheStats:
    current_seq_len: int
    max_seq_len: int
    utilization: float


class KVCache:
    """Shape-validating KV cache for deterministic reference backends.

    Expected shape for append keys/values: [layers][tokens][heads][head_dim]
    """

    def __init__(self, layers: int, heads: int, head_dim: int, max_seq_len: int) -> None:
        if min(layers, heads, head_dim, max_seq_len) <= 0:
            raise ConfigurationError("KV cache dimensions must be positive integers.")
        self.layers = layers
        self.heads = heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self._keys: list[list[list[list[float]]]] = [[] for _ in range(layers)]
        self._values: list[list[list[list[float]]]] = [[] for _ in range(layers)]

    @property
    def seq_len(self) -> int:
        return len(self._keys[0]) if self._keys else 0

    def _validate_tensor(self, tensor: FloatTensor4D, name: str) -> int:
        if len(tensor) != self.layers:
            raise ConfigurationError(f"{name} expected {self.layers} layers, got {len(tensor)}.")
        token_count: int | None = None
        for layer_idx, layer_tokens in enumerate(tensor):
            if token_count is None:
                token_count = len(layer_tokens)
            elif len(layer_tokens) != token_count:
                raise ConfigurationError(f"{name} layer token count mismatch at layer {layer_idx}.")
            for token_heads in layer_tokens:
                if len(token_heads) != self.heads:
                    raise ConfigurationError(f"{name} expected {self.heads} heads per token.")
                for head_vec in token_heads:
                    if len(head_vec) != self.head_dim:
                        raise ConfigurationError(f"{name} expected head_dim={self.head_dim}.")
        return token_count or 0

    def append(self, keys: FloatTensor4D, values: FloatTensor4D) -> None:
        key_tokens = self._validate_tensor(keys, "keys")
        value_tokens = self._validate_tensor(values, "values")
        if key_tokens != value_tokens:
            raise ConfigurationError("keys and values must have same token count.")
        if self.seq_len + key_tokens > self.max_seq_len:
            raise ConfigurationError(
                f"KV cache context limit exceeded: {self.seq_len + key_tokens}>{self.max_seq_len}."
            )
        for layer_idx in range(self.layers):
            self._keys[layer_idx].extend(list(keys[layer_idx]))
            self._values[layer_idx].extend(list(values[layer_idx]))

    def read(self, start: int = 0, end: int | None = None) -> tuple[FloatTensor4D, FloatTensor4D]:
        if start < 0:
            raise ConfigurationError("start must be >= 0")
        stop = self.seq_len if end is None else end
        if stop < start or stop > self.seq_len:
            raise ConfigurationError("Invalid read range for KV cache.")
        keys = [layer[start:stop] for layer in self._keys]
        values = [layer[start:stop] for layer in self._values]
        return keys, values

    def reset(self) -> None:
        self._keys = [[] for _ in range(self.layers)]
        self._values = [[] for _ in range(self.layers)]

    def stats(self) -> KVCacheStats:
        utilization = self.seq_len / self.max_seq_len
        return KVCacheStats(
            current_seq_len=self.seq_len,
            max_seq_len=self.max_seq_len,
            utilization=utilization,
        )
