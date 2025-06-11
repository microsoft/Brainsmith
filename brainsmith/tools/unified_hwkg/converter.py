"""
RTL to DataflowModel Converter (Unified HWKG alias).

This module provides an alias to the RTL integration converter,
making it available through the unified HWKG interface for
backward compatibility and unified access.
"""

# Direct alias to the actual implementation
from ...dataflow.rtl_integration.rtl_converter import (
    RTLDataflowConverter,
    ConversionResult,
    create_rtl_dataflow_converter
)

__all__ = [
    'RTLDataflowConverter',
    'ConversionResult', 
    'create_rtl_dataflow_converter'
]