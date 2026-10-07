"""Model-loading helpers with explicit unsupported-format errors."""

from __future__ import annotations

from pathlib import Path

from supahmlx.backends.reference import ToyReferenceModel
from supahmlx.errors import UnsupportedModelFormatError

SUPPORTED_REFERENCE_SUFFIXES = {".toy", ".json"}


def load_reference_model(model_path: str | Path) -> ToyReferenceModel:
    """Load a tiny reference model marker file.

    Phase 1 supports a toy reference format to establish architecture and correctness.
    """
    path = Path(model_path)
    if path.suffix.lower() not in SUPPORTED_REFERENCE_SUFFIXES:
        raise UnsupportedModelFormatError(
            f"Unsupported model format for '{path}'. Phase 1 currently supports only "
            f"toy reference files {sorted(SUPPORTED_REFERENCE_SUFFIXES)}. "
            "Next step: add loaders for GGUF/MLX model formats under supahmlx.model."
        )
    if not path.exists():
        raise UnsupportedModelFormatError(
            f"Model path '{path}' does not exist. Provide a .toy or .json toy model marker "
            "for the reference backend."
        )
    return ToyReferenceModel()
