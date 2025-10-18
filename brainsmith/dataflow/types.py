############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Basic types for kernel modeling"""

from typing import Tuple, Union, List, Dict, Any, Callable
from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
import functools


# Type aliases
Shape = Tuple[int, ...]

# New unified shape expression types for kernel integrator integration
ShapeExpr = Union[int, str]  # Single dimension: 784 or "N"
ShapeSpec = List[ShapeExpr]  # Complete shape: [1, 784] or ["N", 768]

# Sentinel value for tiling specs: "copy full dimension from reference shape"
# Used in block_tiling and stream_tiling to indicate that a dimension should
# be taken as-is from the parent shape (analogous to ":" in NumPy slicing).
# Example: block_tiling=[FULL_DIM] means "use the full tensor dimension"
class _FullDimType:
    """Sentinel type for FULL_DIM constant."""
    def __repr__(self):
        return "FULL_DIM"
    __str__ = __repr__

FULL_DIM = _FullDimType()


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


# === Base Classes ===

@dataclass(frozen=True)
class DimensionSource(ABC):
    """Base class for dimension derivation strategies.

    Subclass to add new dimension derivation patterns.
    All subclasses must be frozen dataclasses for immutability and hashability.

    The resolve() method is called during model building to compute the
    dimension value from available interfaces and parameters.

    Concrete implementations (DerivedDim, ScaledDim, etc.) are in dimension_sources.py
    """

    @abstractmethod
    def resolve(self, interfaces: Dict[str, Any], param_getter: Callable[[str], Any]) -> int:
        """Compute dimension value from interfaces and parameters.

        Args:
            interfaces: Dict mapping interface name -> InterfaceModel
            param_getter: Function to retrieve nodeattr values (e.g., get_nodeattr)

        Returns:
            Resolved dimension value (positive integer)

        Raises:
            ValueError: If dimension cannot be resolved or is invalid
        """
        pass


# Template dimension specification (for schemas)
# Supports both static values (str/int) and dynamic derivation (DimensionSource subclasses)
DimSpec = Union[str, int, DimensionSource]


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

