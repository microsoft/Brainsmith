############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Basic types for kernel modeling"""

from typing import Tuple, Union, List, Optional
from dataclasses import dataclass
from enum import Enum
import math
import functools
import re


# Type aliases
Shape = Tuple[int, ...]
RaggedShape = Union[Shape, List[Shape]]


class InterfaceDirection(Enum):
    """Direction of data flow for an interface"""
    INPUT = "input"
    OUTPUT = "output"
    
    @classmethod
    def from_string(cls, direction: str) -> "InterfaceDirection":
        """Create from string representation"""
        direction = direction.lower()
        for member in cls:
            if member.value == direction:
                return member
        raise ValueError(f"Unknown interface direction: {direction}")


@dataclass(frozen=True)
class DataType:
    """Data type with bit width"""
    name: str  # "INT8", "INT16", "FP16", etc.
    bits: int
    signed: bool = True
    
    def __post_init__(self):
        """Validate data type"""
        if self.bits <= 0:
            raise ValueError(f"Bit width must be positive, got {self.bits}")
        if not self.name:
            raise ValueError("Data type name cannot be empty")
    
    @classmethod
    def from_string(cls, dtype_str: str) -> "DataType":
        """Parse from FINN-style dtype string
        
        Examples:
            "INT8" -> DataType("INT8", 8, signed=True)
            "UINT16" -> DataType("UINT16", 16, signed=False)
            "BIPOLAR" -> DataType("BIPOLAR", 1, signed=True)
            "BINARY" -> DataType("BINARY", 1, signed=False)
        """
        dtype_str = dtype_str.upper().strip()
        
        # Handle special cases
        if dtype_str == "BIPOLAR":
            return cls("BIPOLAR", 1, signed=True)
        elif dtype_str == "BINARY":
            return cls("BINARY", 1, signed=False)
        
        # Parse INT/UINT types
        match = re.match(r'^(U?)INT(\d+)$', dtype_str)
        if match:
            signed = match.group(1) != 'U'
            bits = int(match.group(2))
            return cls(dtype_str, bits, signed=signed)
        
        # Parse FP types
        if dtype_str in ["FP16", "FP32", "FP64", "BFLOAT16"]:
            bits_map = {"FP16": 16, "FP32": 32, "FP64": 64, "BFLOAT16": 16}
            return cls(dtype_str, bits_map[dtype_str], signed=True)
        
        raise ValueError(f"Unknown data type format: {dtype_str}")
    
    def to_finn_string(self) -> str:
        """Convert to FINN-compatible string representation"""
        return self.name
    
    def __str__(self) -> str:
        return self.name


# Utility functions for shape manipulation
def prod(shape: Shape) -> int:
    """Compute product of shape dimensions"""
    return functools.reduce(lambda a, b: a * b, shape, 1)


def shape_to_string(shape: Shape) -> str:
    """Convert shape to string representation"""
    return f"({','.join(map(str, shape))})"


def parse_shape(shape_str: str) -> Shape:
    """Parse shape from string
    
    Examples:
        "(32,64)" -> (32, 64)
        "32,64" -> (32, 64)
        "(32)" -> (32,)
    """
    # Remove parentheses and whitespace
    shape_str = shape_str.strip().strip("()")
    
    if not shape_str:
        return tuple()
    
    # Split by comma and convert to integers
    parts = [int(x.strip()) for x in shape_str.split(",")]
    return tuple(parts)


def shapes_compatible(shape1: Shape, shape2: Shape) -> bool:
    """Check if two shapes are compatible for operations"""
    if len(shape1) != len(shape2):
        return False
    
    for d1, d2 in zip(shape1, shape2):
        if d1 != d2 and d1 != 1 and d2 != 1:  # Allow broadcasting
            return False
    
    return True


def broadcast_shapes(shape1: Shape, shape2: Shape) -> Shape:
    """Compute broadcast shape of two shapes"""
    if not shapes_compatible(shape1, shape2):
        raise ValueError(f"Shapes {shape1} and {shape2} are not compatible for broadcasting")
    
    return tuple(max(d1, d2) for d1, d2 in zip(shape1, shape2))


def flatten_shape(shape: Shape) -> int:
    """Get total number of elements in shape"""
    return prod(shape)


def reshape_compatible(old_shape: Shape, new_shape: Shape) -> bool:
    """Check if reshape is valid"""
    return prod(old_shape) == prod(new_shape)


def tile_shape(tensor_shape: Shape, block_shape: Shape) -> Shape:
    """Compute number of blocks needed to tile tensor
    
    Returns shape where each dimension is ceil(tensor_dim / block_dim)
    """
    if len(tensor_shape) != len(block_shape):
        raise ValueError(f"Shape dimensions must match: {len(tensor_shape)} != {len(block_shape)}")
    
    return tuple(
        math.ceil(t / b) for t, b in zip(tensor_shape, block_shape)
    )


def is_valid_tiling(tensor_shape: Shape, block_shape: Shape) -> bool:
    """Check if block shape evenly tiles tensor shape"""
    if len(tensor_shape) != len(block_shape):
        return False
    
    for t, b in zip(tensor_shape, block_shape):
        if b > t or b <= 0:
            return False
    
    return True


# Common data types
INT8 = DataType("INT8", 8, signed=True)
INT16 = DataType("INT16", 16, signed=True)
INT32 = DataType("INT32", 32, signed=True)
UINT8 = DataType("UINT8", 8, signed=False)
UINT16 = DataType("UINT16", 16, signed=False)
UINT32 = DataType("UINT32", 32, signed=False)
BINARY = DataType("BINARY", 1, signed=False)
BIPOLAR = DataType("BIPOLAR", 1, signed=True)


@dataclass
class SDIMParameterInfo:
    """Information about SDIM parameters for an interface"""
    interface_name: str
    total_dimensions: int
    free_dimensions: List[int]
    constrained_dimensions: Dict[int, str]  # dim -> constraint type
    block_dims: Shape