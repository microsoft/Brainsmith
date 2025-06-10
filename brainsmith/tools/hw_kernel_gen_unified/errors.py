"""
Error handling for unified HWKG.

Based on hw_kernel_gen_simple error handling with structured hierarchy
following HWKG Axiom 7: Hierarchical Error Handling.
"""


class HWKGError(Exception):
    """Base exception for all HWKG unified errors."""
    pass


class RTLParsingError(HWKGError):
    """Error during RTL file parsing."""
    pass


class CompilerDataError(HWKGError):
    """Error loading or processing compiler data."""
    pass


class TemplateError(HWKGError):
    """Error during template processing."""
    pass


class GenerationError(HWKGError):
    """Error during code generation."""
    pass


class ConfigurationError(HWKGError):
    """Error in configuration or CLI arguments."""
    pass


class BDimProcessingError(HWKGError):
    """Error during advanced BDIM pragma processing."""
    pass