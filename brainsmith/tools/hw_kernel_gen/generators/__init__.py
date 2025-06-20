"""
Generator system for HWKG template rendering.

This module provides the generator system that replaces hardcoded template
handling with discoverable, modular generators.
"""

from .base import GeneratorBase
from .manager import GeneratorManager

# Import all generators for auto-discovery
from .hw_custom_op_generator import HWCustomOpGenerator
from .rtl_wrapper_generator import RTLWrapperGenerator
from .rtl_backend_generator import RTLBackendGenerator

__all__ = [
    "GeneratorBase",
    "GeneratorManager",
    "HWCustomOpGenerator",
    "RTLWrapperGenerator", 
    "RTLBackendGenerator",
]