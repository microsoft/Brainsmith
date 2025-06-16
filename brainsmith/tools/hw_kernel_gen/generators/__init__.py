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
from .test_suite_generator import TestSuiteGenerator

__all__ = [
    "GeneratorBase",
    "GeneratorManager",
    "HWCustomOpGenerator",
    "RTLWrapperGenerator", 
    "TestSuiteGenerator",
]