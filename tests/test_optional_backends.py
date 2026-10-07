import platform

import pytest

from supahmlx.backends.metal_backend import ensure_metal_runtime_available
from supahmlx.backends.mlx_compare import detect_mlx
from supahmlx.errors import BackendUnavailableError


def test_mlx_detection_no_crash():
    status = detect_mlx()
    assert isinstance(status.available, bool)
    assert isinstance(status.reason, str)


def test_metal_backend_optional_behavior():
    if platform.system() == "Darwin":
        ensure_metal_runtime_available()
    else:
        with pytest.raises(BackendUnavailableError):
            ensure_metal_runtime_available()
