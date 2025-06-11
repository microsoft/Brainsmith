"""
Pragma Processor (Unified HWKG alias).

This module provides an alias to the pragma converter,
making it available through the unified HWKG interface for
backward compatibility and unified access.
"""

# Direct alias to the actual implementation
from ...dataflow.rtl_integration.pragma_converter import (
    PragmaToStrategyConverter,
    PragmaConversionResult,
    create_pragma_converter
)

__all__ = [
    'PragmaToStrategyConverter',
    'PragmaConversionResult',
    'create_pragma_converter'
]