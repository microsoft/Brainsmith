############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Cross-interface relationship validation for kernel schemas.

This module provides an extensible system for validating invariants across
multiple interfaces. All patterns inherit from the InterfaceRelationship ABC.

Common Patterns:
    - DatatypesEqual: All interfaces must have identical datatypes
    - DimensionsEqual: Specified dimensions must match across interfaces
    - CustomRelationship: Custom validation logic (escape hatch)

Example:
    >>> # ElementwiseAdd: Both inputs must have same datatype
    >>> DatatypesEqual(("input0", "input1"))

    >>> # Concat: All inputs must have same spatial dimensions
    >>> DimensionsEqual(
    ...     ("input0", "input1", "input2"),
    ...     dim_index=slice(0, -1),
    ...     hierarchy=ShapeHierarchy.TENSOR
    ... )

    >>> # MatMul: Matrix multiplication compatibility
    >>> def check_matmul_dims(model, param_getter):
    ...     input_shape = model.get_interface("input").tensor_shape
    ...     weight_shape = model.get_interface("weight").tensor_shape
    ...     if input_shape[-1] != weight_shape[0]:
    ...         return f"MatMul dimension mismatch: {input_shape[-1]} != {weight_shape[0]}"
    ...     return None
    >>> CustomRelationship(check_matmul_dims, "MatMul dimension compatibility")
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Union
from qonnx.core.datatype import DataType


@dataclass(frozen=True)
class InterfaceRelationship(ABC):
    """Base class for cross-interface validation relationships.

    Subclass to add new validation patterns.
    All subclasses must be frozen dataclasses for immutability and hashability.

    The check() method is called during model building to validate invariants
    across multiple interfaces. Returns None if valid, error string if invalid.
    """

    @abstractmethod
    def check(self, kernel_model: Any, param_getter: Callable[[str], Any]) -> Optional[str]:
        """Validate relationship across interfaces.

        Args:
            kernel_model: KernelModel instance with all interfaces resolved
            param_getter: Function to retrieve nodeattr values

        Returns:
            None if validation passes, error message string if validation fails

        Note:
            Should NOT raise exceptions. Return error strings for better
            composability and error aggregation.
        """
        pass


@dataclass(frozen=True)
class DatatypesEqual(InterfaceRelationship):
    """Validate all specified interfaces have identical datatypes.

    Common for operations requiring datatype consistency across inputs.
    Checks that all interfaces have exactly the same QONNX DataType.

    Examples:
        ElementwiseAdd: Both inputs must have identical datatypes:
        >>> DatatypesEqual(("input0", "input1"))

        Multi-input operation:
        >>> DatatypesEqual(("input0", "input1", "input2"))

        Mixed inputs and outputs (rare):
        >>> DatatypesEqual(("input", "output"))
    """
    interface_names: Tuple[str, ...]

    def check(self, kernel_model: Any, param_getter: Callable) -> Optional[str]:
        """Validate all interfaces have identical datatypes."""
        if len(self.interface_names) < 2:
            return "DatatypesEqual requires at least 2 interfaces"

        # Collect all datatypes
        datatypes = {}
        for name in self.interface_names:
            try:
                interface = kernel_model.get_interface(name)
            except (AttributeError, KeyError):
                return f"Interface '{name}' not found in kernel model"

            try:
                datatypes[name] = interface.datatype
            except AttributeError:
                return f"Interface '{name}' does not have datatype attribute"

        # Check all match the first
        first_name = self.interface_names[0]
        first_dt = datatypes[first_name]

        for name in self.interface_names[1:]:
            if datatypes[name] != first_dt:
                return (
                    f"Datatype mismatch: '{first_name}' has {first_dt.name}, "
                    f"but '{name}' has {datatypes[name].name}"
                )

        return None


@dataclass(frozen=True)
class DimensionsEqual(InterfaceRelationship):
    """Validate specified dimensions match across interfaces.

    Flexible dimension indexing supports:
    - Single dimension index (int): All interfaces checked at same position
    - Per-interface indices (tuple[int, ...]): Different position per interface
    - Slice: Multiple dimensions must match (e.g., spatial dims)
    - None: Full shape equality

    Examples:
        ElementwiseAdd: Full tensor shape equality:
        >>> DimensionsEqual(
        ...     ("input0", "input1"),
        ...     dim_index=None,
        ...     hierarchy=ShapeHierarchy.TENSOR
        ... )

        Concat: Spatial dimensions must match (all but last):
        >>> DimensionsEqual(
        ...     ("input0", "input1", "input2"),
        ...     dim_index=slice(0, -1),
        ...     hierarchy=ShapeHierarchy.TENSOR
        ... )

        MatMul: input[-1] must equal weight[0]:
        >>> DimensionsEqual(
        ...     ("input", "weight"),
        ...     dim_index=(-1, 0),
        ...     hierarchy=ShapeHierarchy.TENSOR
        ... )

        Broadcast compatibility (last dimension):
        >>> DimensionsEqual(
        ...     ("input0", "input1"),
        ...     dim_index=-1,
        ...     hierarchy=ShapeHierarchy.STREAM
        ... )
    """
    interface_names: Tuple[str, ...]
    dim_index: Union[None, int, Tuple[int, ...], slice]
    hierarchy: Optional['ShapeHierarchy'] = None

    def __post_init__(self):
        """Set default hierarchy to TENSOR if not specified."""
        if self.hierarchy is None:
            from .types import ShapeHierarchy
            object.__setattr__(self, 'hierarchy', ShapeHierarchy.TENSOR)

        # Validate dim_index format
        if isinstance(self.dim_index, tuple):
            if len(self.dim_index) != len(self.interface_names):
                raise ValueError(
                    f"When dim_index is tuple, length must match interface_names. "
                    f"Got {len(self.dim_index)} indices for {len(self.interface_names)} interfaces"
                )

    def check(self, kernel_model: Any, param_getter: Callable) -> Optional[str]:
        """Validate dimensions match across interfaces."""
        if len(self.interface_names) < 2:
            return "DimensionsEqual requires at least 2 interfaces"

        # Collect shapes
        shapes = {}
        for name in self.interface_names:
            try:
                interface = kernel_model.get_interface(name)
            except (AttributeError, KeyError):
                return f"Interface '{name}' not found in kernel model"

            try:
                shapes[name] = interface.get_shape(self.hierarchy)
            except AttributeError:
                return f"Interface '{name}' does not have get_shape() method"

        # Extract dimensions to compare
        dimensions = {}

        if self.dim_index is None:
            # Full shape equality
            dimensions = shapes
            comparison_type = "full shape"

        elif isinstance(self.dim_index, int):
            # Same index for all interfaces
            for name in self.interface_names:
                shape = shapes[name]
                idx = self.dim_index if self.dim_index >= 0 else len(shape) + self.dim_index

                if not (0 <= idx < len(shape)):
                    return (
                        f"Dimension index {self.dim_index} out of range for "
                        f"interface '{name}' with shape {shape}"
                    )

                dimensions[name] = (shape[idx],)  # Wrap in tuple for uniform comparison
            comparison_type = f"dimension at index {self.dim_index}"

        elif isinstance(self.dim_index, tuple):
            # Per-interface indices
            for name, idx in zip(self.interface_names, self.dim_index):
                shape = shapes[name]
                resolved_idx = idx if idx >= 0 else len(shape) + idx

                if not (0 <= resolved_idx < len(shape)):
                    return (
                        f"Dimension index {idx} out of range for "
                        f"interface '{name}' with shape {shape}"
                    )

                dimensions[name] = (shape[resolved_idx],)
            comparison_type = f"dimensions at indices {self.dim_index}"

        elif isinstance(self.dim_index, slice):
            # Slice of dimensions
            for name in self.interface_names:
                shape = shapes[name]
                dimensions[name] = shape[self.dim_index]

            comparison_type = f"dimensions {self.dim_index}"

        else:
            return f"Invalid dim_index type: {type(self.dim_index).__name__}"

        # Compare all against first
        first_name = self.interface_names[0]
        first_dims = dimensions[first_name]

        for name in self.interface_names[1:]:
            if dimensions[name] != first_dims:
                return (
                    f"Dimension mismatch ({comparison_type}): "
                    f"'{first_name}' has {first_dims}, but '{name}' has {dimensions[name]}"
                )

        return None


@dataclass(frozen=True)
class CustomRelationship(InterfaceRelationship):
    """Custom cross-interface validation logic.

    Escape hatch for complex validation that doesn't fit built-in patterns.
    Provide clear documentation via the description field.

    The check function should:
    - Return None if validation passes
    - Return error string if validation fails
    - NOT raise exceptions (use return values instead)

    Examples:
        MatMul dimension compatibility:
        >>> def check_matmul(model, pg):
        ...     input_shape = model.get_interface("input").tensor_shape
        ...     weight_shape = model.get_interface("weight").tensor_shape
        ...     if input_shape[-1] != weight_shape[0]:
        ...         return f"MatMul incompatible: K={input_shape[-1]} vs K={weight_shape[0]}"
        ...     return None
        >>> CustomRelationship(check_matmul, "MatMul dimension compatibility")

        Range compatibility check:
        >>> def check_ranges(model, pg):
        ...     input_dt = model.get_interface("input").datatype
        ...     weight_dt = model.get_interface("weight").datatype
        ...     if input_dt.min() * weight_dt.min() < -2**15:
        ...         return f"Product range overflow: {input_dt.min()} * {weight_dt.min()}"
        ...     return None
        >>> CustomRelationship(check_ranges, "MatMul range overflow check")

        Multiple constraint validation:
        >>> def check_concat(model, pg):
        ...     # Check spatial dims match
        ...     # Check channel compatibility
        ...     # Check datatype compatibility
        ...     # Return first error found, or None
        ...     return None
        >>> CustomRelationship(check_concat, "Concat compatibility")
    """
    check_fn: Callable[[Any, Callable], Optional[str]]
    description: str = ""  # Strongly encouraged for documentation

    def check(self, kernel_model: Any, param_getter: Callable) -> Optional[str]:
        """Call custom validation function."""
        try:
            result = self.check_fn(kernel_model, param_getter)
        except Exception as e:
            desc = f" ({self.description})" if self.description else ""
            return f"CustomRelationship{desc} raised exception: {e}"

        if result is not None and not isinstance(result, str):
            desc = f" ({self.description})" if self.description else ""
            return (
                f"CustomRelationship{desc} must return None or str, "
                f"got {type(result).__name__}"
            )

        return result

    def __repr__(self):
        """Show description in repr for better error messages."""
        if self.description:
            return f"CustomRelationship({self.description})"
        return "CustomRelationship(custom)"


__all__ = [
    'InterfaceRelationship',
    'DatatypesEqual',
    'DimensionsEqual',
    'CustomRelationship',
]
