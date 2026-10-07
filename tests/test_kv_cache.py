import pytest

from supahmlx.cache.kv import KVCache
from supahmlx.errors import ConfigurationError


def _tensor(value: float):
    return [[[[value, value], [value, value]]]]


def test_kv_cache_append_read_reset_stats():
    cache = KVCache(layers=1, heads=2, head_dim=2, max_seq_len=2)
    cache.append(_tensor(1.0), _tensor(2.0))
    keys, values = cache.read(0, 1)
    assert keys[0][0][0] == [1.0, 1.0]
    assert values[0][0][1] == [2.0, 2.0]
    assert cache.stats().current_seq_len == 1
    cache.reset()
    assert cache.stats().current_seq_len == 0


def test_kv_cache_shape_validation():
    cache = KVCache(layers=1, heads=2, head_dim=2, max_seq_len=2)
    bad = [[[[1.0, 1.0]]]]
    with pytest.raises(ConfigurationError):
        cache.append(bad, bad)
