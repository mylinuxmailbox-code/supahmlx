"""Public loader entrypoint."""

from __future__ import annotations

from pathlib import Path

from supahmlx.backends.metal_backend import MetalBackendPlaceholder, ensure_metal_runtime_available
from supahmlx.backends.reference import ReferenceBackend
from supahmlx.config import ModeConfig, parse_mode
from supahmlx.errors import BackendUnavailableError
from supahmlx.model.loading import load_reference_model
from supahmlx.runtime.engine import RuntimeModel
from supahmlx.runtime.interfaces import ExecutionPlan
from supahmlx.tokenizer.simple import SimpleTokenizer


def load(
    model_path: str | Path, mode: str | ModeConfig = "balanced", device: str = "auto"
) -> RuntimeModel:
    """Load a SUPAHMLX runtime object.

    Phase 1 returns a deterministic reference runtime for toy model markers.
    """
    mode_cfg = parse_mode(mode)
    model = load_reference_model(model_path)
    normalized_device = device.lower()
    if normalized_device in {"auto", "cpu", "reference"}:
        backend = ReferenceBackend(model)
        resolved_device = "cpu"
    elif normalized_device in {"metal", "gpu", "apple-gpu"}:
        ensure_metal_runtime_available()
        backend = MetalBackendPlaceholder()
        resolved_device = "metal"
    else:
        raise BackendUnavailableError(
            f"Unknown device '{device}'. Supported devices: auto, cpu, reference, metal, gpu."
        )
    tokenizer = SimpleTokenizer()
    plan = ExecutionPlan(backend_name=backend.name, mode=mode_cfg, device=resolved_device)
    return RuntimeModel(backend=backend, tokenizer=tokenizer, mode=mode_cfg, plan=plan)
