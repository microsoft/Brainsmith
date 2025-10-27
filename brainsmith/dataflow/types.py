############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Basic types for kernel modeling"""

from typing import Tuple, Union, List, Dict, Any, Callable, Optional, TYPE_CHECKING
from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
import functools


# Type aliases
Shape = Tuple[int, ...]

# New unified shape expression types for kernel integrator integration
ShapeExpr = Union[int, str]  # Single dimension: 784 or "N"
ShapeSpec = List[ShapeExpr]  # Complete shape: [1, 784] or ["N", 768]

# Sentinel for "copy full dimension" in tiling specs
class _FullDimType:
    """Singleton sentinel for FULL_DIM constant.

    Used in tiling specs to indicate 'copy full dimension from reference'.
    Analogous to ":" in NumPy slicing - takes the dimension as-is.

    Example:
        block_tiling=[FULL_DIM, FULL_DIM]  # Use complete tensor dimensions
    """
    __slots__ = ()  # No instance dict, true singleton

    def __repr__(self):
        return "FULL_DIM"

FULL_DIM = _FullDimType()


# Sentinel for "copy full shape" in tiling specs
class _FullShapeType:
    """Singleton sentinel for FULL_SHAPE constant.

    Used in tiling specs to indicate 'expand to full rank with FULL_DIM'.
    Unlike FULL_DIM (which copies a single dimension), FULL_SHAPE expands
    to match the complete shape hierarchy.

    Usage: block_tiling=FULL_SHAPE (not [FULL_SHAPE])

    Examples:
        block_tiling=FULL_SHAPE  # For 4D tensor → [FULL_DIM, FULL_DIM, FULL_DIM, FULL_DIM]
        stream_tiling=FULL_SHAPE  # Copies resolved block_shape
    """
    __slots__ = ()  # No instance dict, true singleton

    def __repr__(self):
        return "FULL_SHAPE"

FULL_SHAPE = _FullShapeType()


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


# === Dimension and Datatype Specifications ===

# Simplified dimension specification (union type replaces ABC hierarchy)
# Supported formats:
#   int:                  Literal dimension (1 only allowed)
#   str:                  Parameter name ("SIMD", "PE")
#   tuple[str, int]:      Shorthand for deriving from interface, uses context hierarchy
#                         ("input", -1) → derive from input's BLOCK/STREAM based on context
#   tuple[str, int, ShapeHierarchy]: Shorthand with explicit hierarchy override
#                         ("input", -1, STREAM) → always derive from input's STREAM hierarchy
#   Callable:             Custom computation function
#   FULL_DIM:             Copy full dimension from reference shape
# Note: For rank-agnostic specs, use FULL_SHAPE (not a DimSpec, but a TilingSpec alternative)
#       block_tiling=FULL_SHAPE expands to [FULL_DIM, FULL_DIM, ...] matching tensor rank
# Example: block_tiling=[1, 1, ("input", -1)] means dims [1, 1, <last BLOCK dim of input>]
DimSpec = Union[
    int,                                               # Literal (1 only)
    str,                                               # Parameter name
    Tuple[str, int],                                   # Derive from interface (context hierarchy)
    Tuple[str, int, 'ShapeHierarchy'],                # Derive with explicit hierarchy
    Callable[[Dict[str, Any], Callable, Any, Optional[str]], int],  # Custom computation (unified signature)
    type(FULL_DIM),                                    # Copy full dimension
]

# NEW: Datatype specification (union type replaces ABC hierarchy)
# Supported formats:
#   DataType:             Fixed datatype (e.g., DataType["INT8"])
#   str:                  Derive from interface name (shorthand: "input" means copy from input)
#   VALUE_OPTIMIZED:      Optimize from actual tensor values (static inputs only)
#   Callable:             Custom datatype computation function
# Example: "input" means copy datatype from input interface
if TYPE_CHECKING:
    from qonnx.core.datatype import BaseDataType

    DatatypeSpec = Union[
        BaseDataType,                                               # Fixed datatype
        str,                                                        # Interface name (derive from)
        Callable[[Dict, Callable, Any, str], BaseDataType],        # Custom computation
        type(lambda: None),                                         # Sentinel type (VALUE_OPTIMIZED)
    ]


# Sentinel for value-optimized datatype derivation
class _ValueOptimizedType:
    """Sentinel type for VALUE_OPTIMIZED constant."""
    def __repr__(self):
        return "VALUE_OPTIMIZED"
    __str__ = __repr__

VALUE_OPTIMIZED = _ValueOptimizedType()


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

