"""
Error handling for Kernel Integrator.

Based on hw_kernel_gen_simple error handling with structured hierarchy
following KI Axiom 7: Hierarchical Error Handling.
"""


class KIError(Exception):
    """Base exception for all KI errors."""
    pass


class RTLParsingError(KIError):
    """Error during RTL file parsing."""
    pass


class CompilerDataError(KIError):
    """Error loading or processing compiler data."""
    pass


class TemplateError(KIError):
    """Error during template processing."""
    pass


class GenerationError(KIError):
    """Error during code generation."""
    pass


class ConfigurationError(KIError):
    """Error in configuration or CLI arguments."""
    pass
