"""
Plugin System Exceptions

Custom exceptions for the BrainSmith plugin system.
"""


class PluginError(Exception):
    """Base exception for plugin system errors."""
    pass


class PluginNotFoundError(PluginError):
    """Raised when a requested plugin cannot be found."""
    pass


class PluginRegistrationError(PluginError):
    """Raised when plugin registration fails."""
    pass


class PluginValidationError(PluginError):
    """Raised when plugin validation fails."""
    pass


class PluginDependencyError(PluginError):
    """Raised when plugin dependencies cannot be satisfied."""
    pass