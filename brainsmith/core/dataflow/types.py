############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Basic types for kernel modeling"""

from typing import Tuple, Union, List, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import math
import functools
import re


# Type aliases
Shape = Tuple[int, ...]

# New unified shape expression types for kernel integrator integration
ShapeExpr = Union[int, str]  # Single dimension: 784 or "N"
ShapeSpec = List[ShapeExpr]  # Complete shape: [1, 784] or ["N", 768]

# === Enums ===

class ShapeHierarchy(Enum):
    """Shape hierarchy level for dimension constraints and relationships.

    Specifies which level of the shape hierarchy to validate or manipulate:
    - STREAM: stream_shape (streaming parallelism, elements per cycle)
    - BLOCK: block_shape (block tiling dimensions)
    - TENSOR: tensor_shape (full logical dimensions)
    """
    STREAM = "stream"
    BLOCK = "block"
    TENSOR = "tensor"


class ProtocolType(Enum):
    """Supported hardware protocols for kernel interfaces."""
    AXI_STREAM = "axi_stream"
    AXI_LITE = "axi_lite"
    CONTROL = "control"


class Direction(Enum):
    """Direction of ports."""
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"


class InterfaceType(Enum):
    """Fundamental interface types for all kernels."""
    INPUT = "input"      # AXI-Stream input for activation data
    OUTPUT = "output"    # AXI-Stream output for result data
    WEIGHT = "weight"    # AXI-Stream input for weight/parameter data
    CONFIG = "config"    # AXI-Lite for runtime configuration
    CONTROL = "control"  # Global control signals (clk, rst, etc.)
    UNKNOWN = "unknown"  # Unknown interface type

    @property
    def protocol(self) -> ProtocolType:
        """Get the hardware protocol for this interface type"""
        protocol_map = {
            InterfaceType.INPUT: ProtocolType.AXI_STREAM,
            InterfaceType.OUTPUT: ProtocolType.AXI_STREAM,
            InterfaceType.WEIGHT: ProtocolType.AXI_STREAM,
            InterfaceType.CONFIG: ProtocolType.AXI_LITE,
            InterfaceType.CONTROL: ProtocolType.CONTROL,
            InterfaceType.UNKNOWN: None,
        }
        return protocol_map[self]

    @property
    def direction(self) -> Direction:
        """Get the expected direction for this interface type"""
        direction_map = {
            InterfaceType.INPUT: Direction.INPUT,
            InterfaceType.WEIGHT: Direction.INPUT,
            InterfaceType.OUTPUT: Direction.OUTPUT,
            InterfaceType.CONFIG: Direction.INOUT,
            InterfaceType.CONTROL: Direction.INPUT,
            InterfaceType.UNKNOWN: None,
        }
        return direction_map[self]

# === Utility Functions ===

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


# Common data types are now imported from qonnx_types module


@dataclass
class SDIMParameterInfo:
    """Information about SDIM parameters for an interface"""
    interface_name: str
    total_dimensions: int
    free_dimensions: List[int]
    constrained_dimensions: Dict[int, str]  # dim -> constraint type
    block_shape: Shape