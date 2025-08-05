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
RaggedShape = Union[Shape, List[Shape]]

# New unified shape expression types for kernel integrator integration
ShapeExpr = Union[int, str]  # Single dimension: 784 or "N"
ShapeSpec = List[ShapeExpr]  # Complete shape: [1, 784] or ["N", 768]


# Fundamental interface types (moved from kernel_integrator)
class InterfaceType(Enum):
    """Fundamental interface types for all kernels."""
    INPUT = "input"      # AXI-Stream input for activation data
    OUTPUT = "output"    # AXI-Stream output for result data
    WEIGHT = "weight"    # AXI-Stream input for weight/parameter data
    CONFIG = "config"    # AXI-Lite for runtime configuration
    CONTROL = "control"  # Global control signals (clk, rst, etc.)
    UNKNOWN = "unknown"  # Unrecognized interfaces
    
    @property
    def protocol(self) -> str:
        """Get the hardware protocol for this interface type"""
        protocol_map = {
            InterfaceType.INPUT: "axi_stream",
            InterfaceType.OUTPUT: "axi_stream",
            InterfaceType.WEIGHT: "axi_stream",
            InterfaceType.CONFIG: "axi_lite",
            InterfaceType.CONTROL: "global_control",
            InterfaceType.UNKNOWN: "unknown"
        }
        return protocol_map[self]
    
    @property
    def is_dataflow(self) -> bool:
        """Check if this interface participates in dataflow"""
        return self in [InterfaceType.INPUT, InterfaceType.OUTPUT, InterfaceType.WEIGHT]
    
    @property
    def is_axi_stream(self) -> bool:
        """Check if this interface uses AXI-Stream protocol"""
        return self.protocol == "axi_stream"
    
    @property
    def is_axi_lite(self) -> bool:
        """Check if this interface uses AXI-Lite protocol"""
        return self.protocol == "axi_lite"
    
    @property
    def is_configuration(self) -> bool:
        """Check if this interface is for configuration"""
        return self in [InterfaceType.CONFIG, InterfaceType.CONTROL]
    
    @property
    def direction(self) -> str:
        """Get the expected direction for this interface type"""
        direction_map = {
            InterfaceType.INPUT: "input",
            InterfaceType.WEIGHT: "input",
            InterfaceType.OUTPUT: "output",
            InterfaceType.CONFIG: "bidirectional",
            InterfaceType.CONTROL: "input",
            InterfaceType.UNKNOWN: "unknown"
        }
        return direction_map[self]


# Note: InterfaceDirection and DataType have been moved to separate modules
# Use InputInterface/OutputInterface instead of direction-based interfaces
# Use QONNX DataType from qonnx_types module


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


# Common data types are now imported from qonnx_types module


@dataclass
class SDIMParameterInfo:
    """Information about SDIM parameters for an interface"""
    interface_name: str
    total_dimensions: int
    free_dimensions: List[int]
    constrained_dimensions: Dict[int, str]  # dim -> constraint type
    block_dims: Shape