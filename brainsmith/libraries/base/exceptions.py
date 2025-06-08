"""
Exception classes for the library system.
"""


class LibraryError(Exception):
    """Base exception for library-related errors."""
    pass


class LibraryNotFoundError(LibraryError):
    """Raised when a requested library is not found."""
    pass


class LibraryInitializationError(LibraryError):
    """Raised when library initialization fails."""
    pass


class LibraryConfigurationError(LibraryError):
    """Raised when library configuration is invalid."""
    pass


class LibraryOperationError(LibraryError):
    """Raised when a library operation fails."""
    pass


class ComponentNotFoundError(LibraryError):
    """Raised when a library component is not found."""
    pass


class ParameterValidationError(LibraryError):
    """Raised when parameter validation fails."""
    pass