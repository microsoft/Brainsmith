"""
Custom exceptions for the DSE V3 Design Space Constructor.
"""


class BrainsmithError(Exception):
    """Base exception for all Brainsmith errors."""
    pass


class BlueprintParseError(BrainsmithError):
    """Raised when there's an error parsing a blueprint file."""
    pass


class ValidationError(BrainsmithError):
    """Raised when design space validation fails."""
    def __init__(self, message: str, errors: list = None, warnings: list = None):
        self.errors = errors or []
        self.warnings = warnings or []
        super().__init__(message)


class ConfigurationError(BrainsmithError):
    """Raised when there's an error in the configuration."""
    pass


class PluginNotFoundError(BlueprintParseError):
    """Raised when a referenced plugin doesn't exist in the registry."""
    pass