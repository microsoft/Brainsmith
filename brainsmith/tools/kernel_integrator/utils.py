############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Utility functions for Kernel Integrator.

This module contains common utility functions used throughout the Kernel
Integrator to reduce code duplication and provide consistent behavior.
"""

import re


def pascal_case(name: str) -> str:
    """
    Convert snake_case or kebab-case to PascalCase.
    
    Args:
        name: String to convert (e.g., "my_module_name" or "my-module-name")
        
    Returns:
        PascalCase string (e.g., "MyModuleName")
        
    Examples:
        >>> pascal_case("thresholding_axi")
        "ThresholdingAxi"
        >>> pascal_case("matrix-multiply")
        "MatrixMultiply"
        >>> pascal_case("my_custom_op")
        "MyCustomOp"
    """
    # Replace hyphens with underscores
    name = name.replace('-', '_')
    
    # Split on underscores and capitalize each part
    parts = name.split('_')
    return ''.join(word.capitalize() for word in parts if word)


def snake_case(name: str) -> str:
    """
    Convert PascalCase or kebab-case to snake_case.
    
    Args:
        name: String to convert (e.g., "MyModuleName" or "my-module-name")
        
    Returns:
        snake_case string (e.g., "my_module_name")
        
    Examples:
        >>> snake_case("ThresholdingAxi")
        "thresholding_axi"
        >>> snake_case("MatrixMultiply")
        "matrix_multiply"
    """
    # Replace hyphens with underscores
    name = name.replace('-', '_')
    
    # Insert underscores before capitals and convert to lowercase
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


# Module exports
__all__ = [
    "pascal_case",
    "snake_case",
]