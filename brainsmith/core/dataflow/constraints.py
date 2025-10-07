############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Interface constraint system for dataflow modeling.

InterfaceConstraints validate properties of single interfaces (datatype, dimensions).
They are applied during interface model creation in the kernel build process.

Shape Hierarchy:
All dimension constraints support targeting different shape levels via the shape_hierarchy
parameter (defaults to ShapeHierarchy.STREAM):
- ShapeHierarchy.STREAM: Validate stream_shape (streaming parallelism)
- ShapeHierarchy.BLOCK: Validate block_shape (block tiling dimensions)
- ShapeHierarchy.TENSOR: Validate tensor_shape (full logical dimensions)

For cross-interface relationships, see relationships.py.
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Union
from abc import ABC, abstractmethod

from .types import ShapeHierarchy


# =============================================================================
# InterfaceConstraint Hierarchy (Single Interface Validation)
# =============================================================================

@dataclass(frozen=True)
class InterfaceConstraint(ABC):
    """Constraint scoped to a single interface.

    Validates properties of a single interface (datatype, dimensions).
    Applied during interface model creation.
    """

    interface_name: str

    @abstractmethod
    def check(
        self,
        interface_model: Any,  # InputModel or OutputModel
        nodeattr_getter: Callable[[str], Any]
    ) -> Optional[str]:
        """Check constraint on interface model.

        Args:
            interface_model: InputModel or OutputModel to validate
            nodeattr_getter: Function to resolve nodeattr names (e.g., self.get_nodeattr)

        Returns:
            None if constraint is satisfied
            Error message string if constraint is violated
        """
        pass

    @abstractmethod
    def describe(self) -> str:
        pass


# =============================================================================
# Concrete Constraint Classes
# =============================================================================

@dataclass(frozen=True)
class DatatypeConstraint(InterfaceConstraint):
    """Datatype must match base type and bit width constraints.

    Examples:
        DatatypeConstraint("input", "INT", 4, 8)    # INT4-INT8
        DatatypeConstraint("output", "UINT", 8, 16) # UINT8, UINT16
        DatatypeConstraint("data", "ANY", 8, 32)    # Any type, 8-32 bits
    """

    base_type: str  # "INT", "UINT", "FIXED", "FLOAT", "BIPOLAR", "TERNARY", "BINARY", "ANY"
    min_width: int  # Minimum bit width (inclusive)
    max_width: int  # Maximum bit width (inclusive)

    def __post_init__(self):
        """Validate constraint parameters."""
        if self.min_width <= 0:
            raise ValueError(f"min_width must be positive, got {self.min_width}")
        if self.max_width < self.min_width:
            raise ValueError(f"max_width ({self.max_width}) must be >= min_width ({self.min_width})")

        valid_base_types = ["INT", "UINT", "FIXED", "FLOAT", "BIPOLAR", "TERNARY", "BINARY", "ANY"]
        if self.base_type not in valid_base_types:
            raise ValueError(f"Invalid base_type '{self.base_type}'. Must be one of {valid_base_types}")

    def check(self, interface_model: Any, nodeattr_getter: Callable[[str], Any]) -> Optional[str]:
        """Validate datatype against constraints."""
        datatype = interface_model.datatype
        bitwidth = datatype.bitwidth()

        # Check bitwidth range
        if not (self.min_width <= bitwidth <= self.max_width):
            return f"Datatype {datatype.get_canonical_name()} bitwidth {bitwidth} not in range [{self.min_width}, {self.max_width}]"

        # Special case: ANY matches any type (only bitwidth matters)
        if self.base_type == "ANY":
            return None

        # Check base type using validation mapping
        canonical_name = datatype.get_canonical_name()

        type_validators = {
            "INT": lambda: canonical_name.startswith("INT") and datatype.signed(),
            "UINT": lambda: canonical_name.startswith("UINT") or canonical_name == "BINARY",
            "FIXED": lambda: canonical_name.startswith("FIXED<"),
            "FLOAT": lambda: canonical_name.startswith("FLOAT"),
            "BIPOLAR": lambda: canonical_name == "BIPOLAR",
            "TERNARY": lambda: canonical_name == "TERNARY",
            "BINARY": lambda: canonical_name == "BINARY",
        }

        validator = type_validators.get(self.base_type)
        if validator and not validator():
            return f"Datatype {canonical_name} does not match {self.base_type} constraint"

        return None

    def describe(self) -> str:
        return f"{self.interface_name}.datatype: {self.base_type}[{self.min_width}..{self.max_width}]"


@dataclass(frozen=True)
class DimensionConstraint(InterfaceConstraint, ABC):
    """Base class for constraints on dimension(s) of a single interface.

    Validates dimensions at any shape hierarchy level (stream/block/tensor).
    Supports nodeattr references for constraint values.

    Dimension Index Semantics:
    - int: Single dimension index (e.g., 1 for dimension 1)
    - List[int]: Multiple dimension indices (e.g., [0, 1, 2] for first three dimensions)
    - None: Total size (product of all dimensions)

    Subclasses must define shape_hierarchy field (defaults to ShapeHierarchy.STREAM).
    """

    dim_index: Union[int, List[int], None]

    @abstractmethod
    def _get_constraint_value(self) -> Union[int, str, float]:
        """Return the constraint value (divisor, min_value, etc.)."""
        pass

    @abstractmethod
    def _validate_dimension(
        self,
        dim_value: int,
        constraint_value: Union[int, float],
        dim_desc: str
    ) -> Optional[str]:
        """Validate dimension value against constraint. Returns error message or None."""
        pass

    def check(self, interface_model: Any, nodeattr_getter: Callable[[str], Any]) -> Optional[str]:
        """Check constraint on interface model. Returns error message or None."""
        # Handle multiple dimensions
        if isinstance(self.dim_index, list):
            return self._check_multiple_dimensions(interface_model, nodeattr_getter)

        # Single dimension or total size (original logic)
        # Extract dimension value and descriptor
        dim_value, dim_desc, error = self._extract_dimension(interface_model)
        if error:
            return error

        # Resolve constraint value (handles nodeattr lookup)
        constraint_value, error = self._resolve_value(
            self._get_constraint_value(),
            nodeattr_getter
        )
        if error:
            return error

        # Delegate to subclass validation
        return self._validate_dimension(dim_value, constraint_value, dim_desc)

    def _extract_dimension(
        self,
        interface_model: Any
    ) -> tuple[Optional[int], Optional[str], Optional[str]]:
        """Extract dimension value and descriptor. Returns (dim_value, dim_desc, error_msg)."""
        target_shape = interface_model.get_shape(self.shape_hierarchy)
        shape_name = self.shape_hierarchy.value

        if self.dim_index is None:
            # Total size (product of all dimensions)
            dim_value = math.prod(target_shape)
            dim_desc = f"{shape_name}_total"
        else:
            # Specific dimension index
            if self.dim_index >= len(target_shape):
                error = (f"{shape_name} dimension index {self.dim_index} "
                        f"out of range for shape {target_shape}")
                return None, None, error
            dim_value = target_shape[self.dim_index]
            dim_desc = f"{shape_name}[{self.dim_index}]"

        return dim_value, dim_desc, None

    def _resolve_value(
        self,
        value: Union[int, str, float],
        nodeattr_getter: Callable[[str], Any]
    ) -> tuple[Optional[Union[int, float]], Optional[str]]:
        """Resolve literal value or nodeattr reference. Returns (resolved_value, error_msg)."""
        if isinstance(value, str):
            # Nodeattr reference - look it up
            try:
                resolved = nodeattr_getter(value)
                return resolved, None
            except (AttributeError, KeyError):
                return None, f"Nodeattr '{value}' not found"
        else:
            # Literal value - use directly
            return value, None

    def _check_multiple_dimensions(
        self,
        interface_model: Any,
        nodeattr_getter: Callable[[str], Any]
    ) -> Optional[str]:
        """Check constraint across multiple dimensions. Returns aggregated error or None."""
        target_shape = interface_model.get_shape(self.shape_hierarchy)
        shape_name = self.shape_hierarchy.value

        # Resolve constraint value once (shared across all dimensions)
        constraint_value, error = self._resolve_value(
            self._get_constraint_value(),
            nodeattr_getter
        )
        if error:
            return error

        # Check each dimension
        failures = []
        for idx in self.dim_index:
            # Validate index is in range
            if idx >= len(target_shape):
                failures.append(f"{shape_name}[{idx}] out of range for shape {target_shape}")
                continue

            # Get dimension value and descriptor
            dim_value = target_shape[idx]
            dim_desc = f"{shape_name}[{idx}]"

            # Validate this dimension
            error = self._validate_dimension(dim_value, constraint_value, dim_desc)
            if error:
                failures.append(error)

        # Return aggregated error or None
        if failures:
            return "; ".join(failures)
        return None

    def _make_dim_descriptor(self) -> str:
        """Build human-readable dimension descriptor."""
        shape_name = self.shape_hierarchy.value

        if self.dim_index is None:
            return f"{shape_name}_total"
        elif isinstance(self.dim_index, list):
            # Multiple dimensions
            indices = ",".join(str(i) for i in self.dim_index)
            return f"{shape_name}[{indices}]"
        else:
            # Single dimension
            return f"{shape_name}[{self.dim_index}]"


@dataclass(frozen=True)
class DimensionDivisible(DimensionConstraint):
    """Dimension(s) must be divisible by a value.

    Validates: shape[dim_index] % divisor == 0

    Examples:
        DimensionDivisible("input", 1, 8)       # stream[1] % 8 == 0
        DimensionDivisible("input", 1, "SIMD")  # stream[1] % SIMD (nodeattr)
        DimensionDivisible("input", [0, 1], 8)  # stream[0,1] all % 8 == 0
    """

    divisor: Union[int, str]
    shape_hierarchy: ShapeHierarchy = ShapeHierarchy.STREAM

    def _get_constraint_value(self) -> Union[int, str]:
        return self.divisor

    def _validate_dimension(
        self,
        dim_value: int,
        constraint_value: int,
        dim_desc: str
    ) -> Optional[str]:
        """Validate divisibility constraint."""
        if dim_value % constraint_value != 0:
            return f"{dim_desc} ({dim_value}) not divisible by {constraint_value}"
        return None

    def describe(self) -> str:
        dim_str = self._make_dim_descriptor()
        return f"{self.interface_name}.{dim_str} % {self.divisor} == 0"


@dataclass(frozen=True)
class DimensionMinValue(DimensionConstraint):
    """Dimension(s) must be >= minimum value.

    Validates: shape[dim_index] >= min_value

    Examples:
        DimensionMinValue("input", 0, 1)            # stream[0] >= 1
        DimensionMinValue("output", 1, "MIN_SIZE")  # stream[1] >= MIN_SIZE (nodeattr)
        DimensionMinValue("input", [0, 1], 128)     # stream[0,1] all >= 128
    """

    min_value: Union[int, str]
    shape_hierarchy: ShapeHierarchy = ShapeHierarchy.STREAM

    def _get_constraint_value(self) -> Union[int, str]:
        return self.min_value

    def _validate_dimension(
        self,
        dim_value: int,
        constraint_value: Union[int, float],
        dim_desc: str
    ) -> Optional[str]:
        """Validate minimum value constraint."""
        if dim_value < constraint_value:
            return f"{dim_desc} ({dim_value}) must be >= {constraint_value}"
        return None

    def describe(self) -> str:
        dim_str = self._make_dim_descriptor()
        return f"{self.interface_name}.{dim_str} >= {self.min_value}"


@dataclass(frozen=True)
class DimensionMaxValue(DimensionConstraint):
    """Dimension(s) must be <= maximum value.

    Validates: shape[dim_index] <= max_value

    Examples:
        DimensionMaxValue("input", 1, 64)             # stream[1] <= 64
        DimensionMaxValue("output", 0, "MAX_SIZE")    # stream[0] <= MAX_SIZE (nodeattr)
        DimensionMaxValue("input", [0, 2, 3], 4096)   # stream[0,2,3] all <= 4096
    """

    max_value: Union[int, str]
    shape_hierarchy: ShapeHierarchy = ShapeHierarchy.STREAM

    def _get_constraint_value(self) -> Union[int, str]:
        return self.max_value

    def _validate_dimension(
        self,
        dim_value: int,
        constraint_value: Union[int, float],
        dim_desc: str
    ) -> Optional[str]:
        """Validate maximum value constraint."""
        if dim_value > constraint_value:
            return f"{dim_desc} ({dim_value}) must be <= {constraint_value}"
        return None

    def describe(self) -> str:
        dim_str = self._make_dim_descriptor()
        return f"{self.interface_name}.{dim_str} <= {self.max_value}"


__all__ = [
    # Base classes
    "InterfaceConstraint",
    "DimensionConstraint",
    # Concrete constraints
    "DatatypeConstraint",
    "DimensionDivisible",
    "DimensionMinValue",
    "DimensionMaxValue",
]
