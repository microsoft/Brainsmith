############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Dimension derivation patterns for kernel schemas.

This module provides an extensible system for deriving dimension values from
other interfaces. All patterns inherit from the DimensionSource ABC.

Common Patterns:
    - DerivedDim: Copy dimension from another interface
    - ScaledDim: Scale dimension by constant factor
    - SumDims: Sum dimensions from multiple sources
    - MaxDim: Maximum dimension across sources
    - ComputedDim: Custom computation (escape hatch)

Example:
    >>> # Simple copy
    >>> DerivedDim("input", -1)  # Copy input's last dimension

    >>> # Scale by factor
    >>> ScaledDim("input", 0, 2.0)  # Double input's first dimension

    >>> # Sum from multiple sources (for Concat)
    >>> SumDims([("input0", -1), ("input1", -1), ("input2", -1)])
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple
import math


@dataclass(frozen=True)
class DimensionSource(ABC):
    """Base class for dimension derivation strategies.

    Subclass to add new dimension derivation patterns.
    All subclasses must be frozen dataclasses for immutability and hashability.

    The resolve() method is called during model building to compute the
    dimension value from available interfaces and parameters.
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


@dataclass(frozen=True)
class DerivedDim(DimensionSource):
    """Copy dimension from another interface.

    Most common pattern for shape-preserving operations. Used when output
    dimensions should exactly match input dimensions.

    Examples:
        LayerNorm output stream = input stream:
        >>> DerivedDim("input", -1)

        Copy specific dimension at specific hierarchy:
        >>> from brainsmith.dataflow.types import ShapeHierarchy
        >>> DerivedDim("input", 0, hierarchy=ShapeHierarchy.TENSOR)
    """
    source_interface: str
    source_dim: int  # Supports negative indexing (-1 for last dimension)
    hierarchy: Optional['ShapeHierarchy'] = None  # Defaults to STREAM

    def __post_init__(self):
        """Set default hierarchy to STREAM if not specified."""
        if self.hierarchy is None:
            # Import here to avoid circular dependency
            from .types import ShapeHierarchy
            object.__setattr__(self, 'hierarchy', ShapeHierarchy.STREAM)

    def resolve(self, interfaces: Dict[str, Any], param_getter: Callable) -> int:
        """Copy dimension from source interface."""
        if self.source_interface not in interfaces:
            raise ValueError(f"Source interface '{self.source_interface}' not found")

        source = interfaces[self.source_interface]

        try:
            shape = source.get_shape(self.hierarchy)
        except AttributeError:
            raise ValueError(
                f"Source interface '{self.source_interface}' does not have "
                f"get_shape() method"
            )

        # Support negative indexing
        dim_index = self.source_dim
        if dim_index < 0:
            dim_index = len(shape) + dim_index

        if not (0 <= dim_index < len(shape)):
            raise ValueError(
                f"Dimension index {self.source_dim} out of range for "
                f"shape {shape} (interface '{self.source_interface}' "
                f"at {self.hierarchy.value} level)"
            )

        value = shape[dim_index]
        if value is None:
            raise ValueError(
                f"Dimension '{self.source_interface}'.{self.hierarchy.value}[{self.source_dim}] "
                f"is not yet resolved"
            )

        return value


@dataclass(frozen=True)
class ScaledDim(DimensionSource):
    """Scale dimension by constant factor.

    Common for upsampling/downsampling operations where output dimensions
    are a fixed multiple or fraction of input dimensions.

    Examples:
        Upsample 2x:
        >>> ScaledDim("input", -1, 2.0)

        Downsample 2x (stride=2):
        >>> ScaledDim("input", -1, 0.5)

        Pooling with stride=4:
        >>> ScaledDim("input", 1, 0.25)
    """
    source_interface: str
    source_dim: int
    scale_factor: float
    hierarchy: Optional['ShapeHierarchy'] = None  # Defaults to STREAM

    def __post_init__(self):
        """Set default hierarchy to STREAM if not specified."""
        if self.hierarchy is None:
            from .types import ShapeHierarchy
            object.__setattr__(self, 'hierarchy', ShapeHierarchy.STREAM)

    def resolve(self, interfaces: Dict[str, Any], param_getter: Callable) -> int:
        """Scale dimension by factor."""
        if self.source_interface not in interfaces:
            raise ValueError(f"Source interface '{self.source_interface}' not found")

        source = interfaces[self.source_interface]
        shape = source.get_shape(self.hierarchy)

        # Support negative indexing
        dim_index = self.source_dim
        if dim_index < 0:
            dim_index = len(shape) + dim_index

        if not (0 <= dim_index < len(shape)):
            raise ValueError(
                f"Dimension index {self.source_dim} out of range for "
                f"shape {shape} (interface '{self.source_interface}')"
            )

        base_value = shape[dim_index]
        if base_value is None:
            raise ValueError(
                f"Source dimension '{self.source_interface}'.{self.hierarchy.value}[{self.source_dim}] "
                f"is not yet resolved"
            )

        if self.scale_factor <= 0:
            raise ValueError(f"scale_factor must be positive, got {self.scale_factor}")

        # For scale_down, check divisibility
        if self.scale_factor < 1.0:
            divisor = 1.0 / self.scale_factor
            # Divisor must be an integer (or very close)
            int_divisor = round(divisor)
            if not math.isclose(divisor, int_divisor, abs_tol=1e-9):
                raise ValueError(
                    f"scale_factor {self.scale_factor} is not 1/n for integer n "
                    f"(divisor = {divisor})"
                )

            # Source must divide evenly
            if base_value % int_divisor != 0:
                raise ValueError(
                    f"Source dimension {base_value} not evenly divisible by {int_divisor} "
                    f"(scale_factor = {self.scale_factor} = 1/{int_divisor})"
                )

            return base_value // int_divisor
        else:
            # For scale_up, compute and verify exactness
            raw_value = base_value * self.scale_factor
            result = round(raw_value)
            if not math.isclose(raw_value, result, abs_tol=1e-9):
                raise ValueError(
                    f"Scaled dimension {base_value} * {self.scale_factor} "
                    f"= {raw_value} is not an exact integer"
                )

            if result <= 0:
                raise ValueError(
                    f"Scaled dimension must be positive, got {result}"
                )

            return result


@dataclass(frozen=True)
class SumDims(DimensionSource):
    """Sum dimensions from multiple interfaces.

    Common for concatenation operations where output size is the sum of
    input sizes along one dimension.

    Example:
        Concat output channels = sum of all input channels:
        >>> SumDims([("input0", -1), ("input1", -1), ("input2", -1)])

        Custom hierarchy:
        >>> from brainsmith.dataflow.types import ShapeHierarchy
        >>> SumDims([("input0", -1), ("input1", -1)], hierarchy=ShapeHierarchy.BLOCK)
    """
    sources: Tuple[Tuple[str, int], ...]  # Immutable list of (interface, dim_index)
    hierarchy: Optional['ShapeHierarchy'] = None  # Defaults to STREAM

    def __post_init__(self):
        """Set default hierarchy to STREAM if not specified."""
        if self.hierarchy is None:
            from .types import ShapeHierarchy
            object.__setattr__(self, 'hierarchy', ShapeHierarchy.STREAM)

    def resolve(self, interfaces: Dict[str, Any], param_getter: Callable) -> int:
        """Sum dimensions from all sources."""
        if not self.sources:
            raise ValueError("SumDims requires at least one source")

        total = 0
        for interface_name, dim_index in self.sources:
            if interface_name not in interfaces:
                raise ValueError(f"Source interface '{interface_name}' not found")

            source = interfaces[interface_name]
            shape = source.get_shape(self.hierarchy)

            # Support negative indexing
            idx = dim_index if dim_index >= 0 else len(shape) + dim_index

            if not (0 <= idx < len(shape)):
                raise ValueError(
                    f"Dimension index {dim_index} out of range for "
                    f"shape {shape} (interface '{interface_name}' "
                    f"at {self.hierarchy.value} level)"
                )

            value = shape[idx]
            if value is None:
                raise ValueError(
                    f"Dimension '{interface_name}'.{self.hierarchy.value}[{dim_index}] "
                    f"is not yet resolved"
                )

            total += value

        return total


@dataclass(frozen=True)
class MaxDim(DimensionSource):
    """Maximum dimension across multiple interfaces.

    Useful for broadcast compatibility and padding operations where output
    size must accommodate the largest input.

    Examples:
        Broadcast-compatible output = max of input dimensions:
        >>> MaxDim([("input0", -1), ("input1", -1)])

        Padding to maximum spatial size:
        >>> MaxDim([("input0", 1), ("input1", 1), ("input2", 1)])
    """
    sources: Tuple[Tuple[str, int], ...]
    hierarchy: Optional['ShapeHierarchy'] = None  # Defaults to STREAM

    def __post_init__(self):
        """Set default hierarchy to STREAM if not specified."""
        if self.hierarchy is None:
            from .types import ShapeHierarchy
            object.__setattr__(self, 'hierarchy', ShapeHierarchy.STREAM)

    def resolve(self, interfaces: Dict[str, Any], param_getter: Callable) -> int:
        """Return maximum dimension across all sources."""
        if not self.sources:
            raise ValueError("MaxDim requires at least one source")

        max_value = 0
        for interface_name, dim_index in self.sources:
            if interface_name not in interfaces:
                raise ValueError(f"Source interface '{interface_name}' not found")

            source = interfaces[interface_name]
            shape = source.get_shape(self.hierarchy)

            # Support negative indexing
            idx = dim_index if dim_index >= 0 else len(shape) + dim_index

            if not (0 <= idx < len(shape)):
                raise ValueError(
                    f"Dimension index {dim_index} out of range for "
                    f"shape {shape} (interface '{interface_name}')"
                )

            value = shape[idx]
            if value is None:
                raise ValueError(
                    f"Dimension '{interface_name}'.{self.hierarchy.value}[{dim_index}] "
                    f"is not yet resolved"
                )

            max_value = max(max_value, value)

        return max_value


@dataclass(frozen=True)
class ComputedDim(DimensionSource):
    """Custom dimension computation.

    Escape hatch for complex dimension logic that doesn't fit built-in patterns.
    Provide clear documentation via the description field.

    Examples:
        Convolution output size:
        >>> def compute_conv_out(ifs, pg):
        ...     h = ifs["input"].tensor_shape[1]
        ...     k = pg("kernel_size")
        ...     s = pg("stride")
        ...     p = pg("padding")
        ...     return (h + 2*p - k) // s + 1
        >>> ComputedDim(compute_conv_out, "Conv output height")

        Complex multi-source logic:
        >>> def custom_logic(interfaces, param_getter):
        ...     # Your custom computation here
        ...     return computed_value
        >>> ComputedDim(custom_logic, "Custom dimension logic")
    """
    compute_fn: Callable[[Dict[str, Any], Callable], int]
    description: str = ""  # Strongly encouraged for documentation

    def resolve(self, interfaces: Dict[str, Any], param_getter: Callable) -> int:
        """Call custom function to compute dimension."""
        try:
            result = self.compute_fn(interfaces, param_getter)
        except Exception as e:
            desc = f" ({self.description})" if self.description else ""
            raise ValueError(
                f"ComputedDim{desc} function raised exception: {e}"
            )

        if not isinstance(result, int):
            desc = f" ({self.description})" if self.description else ""
            raise ValueError(
                f"ComputedDim{desc} function must return int, "
                f"got {type(result).__name__}"
            )

        if result <= 0:
            desc = f" ({self.description})" if self.description else ""
            raise ValueError(
                f"ComputedDim{desc} function returned invalid value {result} "
                f"(must be positive)"
            )

        return result

    def __repr__(self):
        """Show description in repr for better error messages."""
        if self.description:
            return f"ComputedDim({self.description})"
        return "ComputedDim(custom)"


__all__ = [
    'DimensionSource',
    'DerivedDim',
    'ScaledDim',
    'SumDims',
    'MaxDim',
    'ComputedDim',
]
