############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""QONNX datatype integration for kernel modeling.

This module establishes QONNX datatypes as the standard type system across
both hw_kernel_gen and core/dataflow/core, eliminating duplicate type definitions.
"""

from typing import Union, List
from qonnx.core.datatype import DataType as QONNXDataType, BaseDataType

# Re-export QONNX types as the standard
DataType = QONNXDataType
BaseDataType = BaseDataType

# Import constraint types from hw_kernel_gen for reuse
from brainsmith.tools.hw_kernel_gen.data import (
    DatatypeConstraintGroup,
    validate_datatype_against_constraints
)

# Common type aliases for convenience
INT8 = DataType["INT8"]
INT16 = DataType["INT16"]
INT32 = DataType["INT32"]
UINT8 = DataType["UINT8"]
UINT16 = DataType["UINT16"]
UINT32 = DataType["UINT32"]
BINARY = DataType["BINARY"]
BIPOLAR = DataType["BIPOLAR"]
TERNARY = DataType["TERNARY"]


def create_simple_datatype(name: str, bits: int, signed: bool = True) -> BaseDataType:
    """Create QONNX datatype from simple parameters.
    
    This helper provides a compatibility layer for code that previously
    used the custom DataType class from core/dataflow/core/types.py.
    
    Args:
        name: Type name (e.g., "INT", "UINT", "BIPOLAR", "BINARY")
        bits: Bit width
        signed: Whether the type is signed (ignored for special types)
        
    Returns:
        QONNX BaseDataType instance
        
    Raises:
        ValueError: If the datatype specification is invalid
    """
    name = name.upper()
    
    # Handle special types
    if name == "BIPOLAR":
        return DataType["BIPOLAR"]
    elif name == "BINARY":
        return DataType["BINARY"]
    elif name == "TERNARY":
        return DataType["TERNARY"]
    
    # Handle INT/UINT types
    if name in ["INT", "UINT"]:
        prefix = "INT" if (name == "INT" or signed) else "UINT"
        dtype_str = f"{prefix}{bits}"
        if dtype_str in DataType:
            return DataType[dtype_str]
        else:
            raise ValueError(f"QONNX does not support {dtype_str}")
    
    # Handle floating point types
    if name in ["FP16", "FP32", "FP64", "BFLOAT16", "FLOAT16", "FLOAT32", "FLOAT64"]:
        # Normalize names
        if name == "FLOAT16":
            name = "FP16"
        elif name == "FLOAT32":
            name = "FP32"
        elif name == "FLOAT64":
            name = "FP64"
            
        if name in DataType:
            return DataType[name]
        else:
            raise ValueError(f"Unknown floating point type: {name}")
    
    raise ValueError(f"Unknown datatype: {name}")


def datatype_from_string(dtype_str: str) -> BaseDataType:
    """Parse QONNX datatype from string representation.
    
    Args:
        dtype_str: String like "INT8", "UINT16", "BIPOLAR", etc.
        
    Returns:
        QONNX BaseDataType instance
        
    Raises:
        ValueError: If the string doesn't represent a valid QONNX type
    """
    dtype_str = dtype_str.upper().strip()
    
    try:
        return DataType[dtype_str]
    except KeyError:
        raise ValueError(f"Unknown QONNX datatype: {dtype_str}")


# Module exports
__all__ = [
    # QONNX types (now the standard)
    "DataType",
    "BaseDataType",
    
    # Constraint types
    "DatatypeConstraintGroup",
    "validate_datatype_against_constraints",
    
    # Helper functions
    "create_simple_datatype",
    "datatype_from_string",
    
    # Common type shortcuts
    "INT8", "INT16", "INT32",
    "UINT8", "UINT16", "UINT32",
    "BINARY", "BIPOLAR", "TERNARY",
]