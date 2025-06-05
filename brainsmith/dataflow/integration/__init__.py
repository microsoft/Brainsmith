"""
Integration module for Interface-Wise Dataflow Modeling Framework.

This module provides integration components that bridge the dataflow framework
with existing Brainsmith infrastructure, including RTL Parser and HW Kernel Generator.
"""

from .rtl_conversion import RTLInterfaceConverter, validate_conversion_result, ConversionValidationError

__all__ = [
    'RTLInterfaceConverter',
    'validate_conversion_result', 
    'ConversionValidationError'
]