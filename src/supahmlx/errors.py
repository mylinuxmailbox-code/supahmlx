"""Project-specific exception types."""


class SupahMLXError(Exception):
    """Base exception for SUPAHMLX."""


class ConfigurationError(SupahMLXError):
    """Invalid user configuration."""


class UnsupportedModelFormatError(SupahMLXError):
    """Raised when the input model format is unsupported."""


class BackendUnavailableError(SupahMLXError):
    """Raised when a requested backend is unavailable."""
