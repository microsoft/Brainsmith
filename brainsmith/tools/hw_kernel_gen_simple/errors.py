"""
Simple error definitions for HWKG.

Provides clean error hierarchy without complex error handling frameworks.
"""


class HWKGError(Exception):
    """Base exception for all HWKG errors."""
    pass


class RTLParsingError(HWKGError):
    """Error during RTL file parsing."""
    pass


class CompilerDataError(HWKGError):
    """Error loading or processing compiler data."""
    pass


class TemplateError(HWKGError):
    """Error during template rendering."""
    pass


class GenerationError(HWKGError):
    """Error during file generation."""
    pass


class ConfigurationError(HWKGError):
    """Error in configuration or command line arguments."""
    pass