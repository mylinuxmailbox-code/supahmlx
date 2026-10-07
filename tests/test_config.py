import pytest

from supahmlx.config import available_modes, parse_mode
from supahmlx.errors import ConfigurationError


def test_modes_available():
    assert available_modes() == ("quality", "balanced", "turbo", "insane")


def test_invalid_mode_raises():
    with pytest.raises(ConfigurationError):
        parse_mode("fastest")
