############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Constraint system for dataflow modeling.

Two independent hierarchies:
- InterfaceConstraint: Validates single interface (datatype, dimensions)
- InterfaceRelationship: Expresses relationships between interfaces (can be generative or validative)

Shape Hierarchy:
All dimension constraints and relationships support targeting different shape levels via
the shape_hierarchy parameter (defaults to ShapeHierarchy.STREAM):
- ShapeHierarchy.STREAM: Validate/relate stream_shape (streaming parallelism)
- ShapeHierarchy.BLOCK: Validate/relate block_shape (block tiling dimensions)
- ShapeHierarchy.TENSOR: Validate/relate tensor_shape (full logical dimensions)

InterfaceRelationships can be:
- Generative: Set unset output dimensions during KernelModel construction (stream_shape only)
- Validative: Check dimension relationships after resolution (all shape hierarchy levels)

Note: Only stream_shape can have unset (None) dimensions during construction, so generative
behavior only applies when shape_hierarchy == ShapeHierarchy.STREAM.
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union
from abc import ABC, abstractmethod
from qonnx.core.datatype import BaseDataType

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
        """Human-readable description of the constraint."""
        pass


# =============================================================================
# InterfaceRelationship Hierarchy (Cross-Interface Relationships)
# =============================================================================

@dataclass(frozen=True)
class InterfaceRelationship(ABC):
    """Relationship between multiple interfaces (cross-interface scope).

    Can be generative (sets unset dimensions) or validative (checks dimensions).
    Relationships are applied during KernelModel construction.
    """

    @abstractmethod
    def check(self, interfaces: Dict[str, Any]) -> Optional[str]:
        """Validate relationship across all interfaces.

        Called after all dimensions have been resolved.

        Args:
            interfaces: Dict mapping interface names to InterfaceModel objects

        Returns:
            None if relationship is satisfied
            Error message string if relationship is violated
        """
        pass

    def resolve(
        self,
        interfaces: Dict[str, Any],
        mutable_outputs: List[Any]
    ) -> bool:
        """Try to resolve unset output dimensions using this relationship.

        Called iteratively during KernelModel construction to set unset dimensions.

        Args:
            interfaces: Dict of interface name → InterfaceModel (updated in place)
            mutable_outputs: List of OutputModels (updated in place)

        Returns:
            True if any dimensions were resolved, False otherwise
        """
        return False  # Default: non-generative relationship

    @abstractmethod
    def describe(self) -> str:
        """Human-readable description of the relationship."""
        pass


# =============================================================================
# Interface Constraints (Single Interface Scope)
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

        # Check base type
        canonical_name = datatype.get_canonical_name()

        if self.base_type == "INT" and not (canonical_name.startswith("INT") and datatype.signed()):
            return f"Datatype {canonical_name} does not match INT constraint"
        elif self.base_type == "UINT" and not (canonical_name.startswith("UINT") or canonical_name == "BINARY"):
            return f"Datatype {canonical_name} does not match UINT constraint"
        elif self.base_type == "FIXED" and not canonical_name.startswith("FIXED<"):
            return f"Datatype {canonical_name} does not match FIXED constraint"
        elif self.base_type == "FLOAT" and not canonical_name.startswith("FLOAT"):
            return f"Datatype {canonical_name} does not match FLOAT constraint"
        elif self.base_type == "BIPOLAR" and canonical_name != "BIPOLAR":
            return f"Datatype {canonical_name} does not match BIPOLAR constraint"
        elif self.base_type == "TERNARY" and canonical_name != "TERNARY":
            return f"Datatype {canonical_name} does not match TERNARY constraint"
        elif self.base_type == "BINARY" and canonical_name != "BINARY":
            return f"Datatype {canonical_name} does not match BINARY constraint"

        return None

    def describe(self) -> str:
        """Human-readable description."""
        return f"{self.interface_name}.datatype: {self.base_type}[{self.min_width}..{self.max_width}]"


@dataclass(frozen=True)
class DimensionDivisible(InterfaceConstraint):
    """Dimension must be divisible by a value.

    Validates: shape[dim_index] % divisor == 0

    The shape_hierarchy parameter determines which shape to validate:
    - ShapeHierarchy.STREAM: stream_shape (default)
    - ShapeHierarchy.BLOCK: block_shape
    - ShapeHierarchy.TENSOR: tensor_shape

    Examples:
        DimensionDivisible("input", 1, 8)       # stream[1] % 8 == 0
        DimensionDivisible("input", 1, "SIMD")  # stream[1] % SIMD == 0
        DimensionDivisible("output", None, 16)  # total_elements % 16 == 0
        DimensionDivisible("input", 2, 8, ShapeHierarchy.BLOCK)  # block[2] % 8 == 0
    """

    dim_index: Optional[int]  # None = total size (product of all dims)
    divisor: Union[int, str]  # Literal value or nodeattr name
    shape_hierarchy: ShapeHierarchy = ShapeHierarchy.STREAM

    def check(self, interface_model: Any, nodeattr_getter: Callable[[str], Any]) -> Optional[str]:
        """Validate divisibility constraint on hierarchy level dimensions."""
        # Select target shape based on shape_hierarchy
        target_shape = interface_model.get_shape(self.shape_hierarchy)
        shape_name = self.shape_hierarchy.value

        # Get dimension value
        if self.dim_index is None:
            dim_value = math.prod(target_shape)
            dim_desc = f"{shape_name}_total"
        else:
            if self.dim_index >= len(target_shape):
                return f"{shape_name.capitalize()} dimension index {self.dim_index} out of range for {shape_name} shape {target_shape}"
            dim_value = target_shape[self.dim_index]
            dim_desc = f"{shape_name}[{self.dim_index}]"

        # Resolve divisor (literal or nodeattr)
        if isinstance(self.divisor, str):
            try:
                divisor_value = nodeattr_getter(self.divisor)
            except (AttributeError, KeyError):
                return f"Nodeattr '{self.divisor}' not found"
        else:
            divisor_value = self.divisor

        # Validate divisibility
        if dim_value % divisor_value != 0:
            return f"{dim_desc} ({dim_value}) not divisible by {divisor_value}"

        return None

    def describe(self) -> str:
        """Human-readable description."""
        shape_name = self.shape_hierarchy.value
        dim_str = f"{shape_name}_total" if self.dim_index is None else f"{shape_name}[{self.dim_index}]"
        return f"{self.interface_name}.{dim_str} % {self.divisor} == 0"


@dataclass(frozen=True)
class DimensionMinValue(InterfaceConstraint):
    """Dimension must be >= minimum value.

    Validates: shape[dim_index] >= min_value

    The shape_hierarchy parameter determines which shape to validate:
    - ShapeHierarchy.STREAM: stream_shape (default)
    - ShapeHierarchy.BLOCK: block_shape
    - ShapeHierarchy.TENSOR: tensor_shape

    Examples:
        DimensionMinValue("input", 0, 1)        # stream[0] >= 1
        DimensionMinValue("output", 1, "MIN_SIZE")  # stream[1] >= MIN_SIZE
        DimensionMinValue("input", 0, 128, ShapeHierarchy.TENSOR)  # tensor[0] >= 128
    """

    dim_index: Optional[int]  # None = total size
    min_value: Union[int, str]  # Literal value or nodeattr name
    shape_hierarchy: ShapeHierarchy = ShapeHierarchy.STREAM

    def check(self, interface_model: Any, nodeattr_getter: Callable[[str], Any]) -> Optional[str]:
        """Validate minimum value constraint on hierarchy level dimensions."""
        # Select target shape based on shape_hierarchy
        target_shape = interface_model.get_shape(self.shape_hierarchy)
        shape_name = self.shape_hierarchy.value

        # Get dimension value
        if self.dim_index is None:
            dim_value = math.prod(target_shape)
            dim_desc = f"{shape_name}_total"
        else:
            if self.dim_index >= len(target_shape):
                return f"{shape_name.capitalize()} dimension index {self.dim_index} out of range for {shape_name} shape {target_shape}"
            dim_value = target_shape[self.dim_index]
            dim_desc = f"{shape_name}[{self.dim_index}]"

        # Resolve min_value (literal or nodeattr)
        if isinstance(self.min_value, str):
            try:
                min_val = nodeattr_getter(self.min_value)
            except (AttributeError, KeyError):
                return f"Nodeattr '{self.min_value}' not found"
        else:
            min_val = self.min_value

        # Validate minimum
        if dim_value < min_val:
            return f"{dim_desc} ({dim_value}) must be >= {min_val}"

        return None

    def describe(self) -> str:
        """Human-readable description."""
        shape_name = self.shape_hierarchy.value
        dim_str = f"{shape_name}_total" if self.dim_index is None else f"{shape_name}[{self.dim_index}]"
        return f"{self.interface_name}.{dim_str} >= {self.min_value}"


@dataclass(frozen=True)
class DimensionMaxValue(InterfaceConstraint):
    """Dimension must be <= maximum value.

    Validates: shape[dim_index] <= max_value

    The shape_hierarchy parameter determines which shape to validate:
    - ShapeHierarchy.STREAM: stream_shape (default)
    - ShapeHierarchy.BLOCK: block_shape
    - ShapeHierarchy.TENSOR: tensor_shape

    Examples:
        DimensionMaxValue("input", 1, 64)       # stream[1] <= 64
        DimensionMaxValue("output", 0, "MAX_SIZE")  # stream[0] <= MAX_SIZE
        DimensionMaxValue("input", 1, 4096, ShapeHierarchy.TENSOR)  # tensor[1] <= 4096
    """

    dim_index: Optional[int]  # None = total size
    max_value: Union[int, str]  # Literal value or nodeattr name
    shape_hierarchy: ShapeHierarchy = ShapeHierarchy.STREAM

    def check(self, interface_model: Any, nodeattr_getter: Callable[[str], Any]) -> Optional[str]:
        """Validate maximum value constraint on hierarchy level dimensions."""
        # Select target shape based on shape_hierarchy
        target_shape = interface_model.get_shape(self.shape_hierarchy)
        shape_name = self.shape_hierarchy.value

        # Get dimension value
        if self.dim_index is None:
            dim_value = math.prod(target_shape)
            dim_desc = f"{shape_name}_total"
        else:
            if self.dim_index >= len(target_shape):
                return f"{shape_name.capitalize()} dimension index {self.dim_index} out of range for {shape_name} shape {target_shape}"
            dim_value = target_shape[self.dim_index]
            dim_desc = f"{shape_name}[{self.dim_index}]"

        # Resolve max_value (literal or nodeattr)
        if isinstance(self.max_value, str):
            try:
                max_val = nodeattr_getter(self.max_value)
            except (AttributeError, KeyError):
                return f"Nodeattr '{self.max_value}' not found"
        else:
            max_val = self.max_value

        # Validate maximum
        if dim_value > max_val:
            return f"{dim_desc} ({dim_value}) must be <= {max_val}"

        return None

    def describe(self) -> str:
        """Human-readable description."""
        shape_name = self.shape_hierarchy.value
        dim_str = f"{shape_name}_total" if self.dim_index is None else f"{shape_name}[{self.dim_index}]"
        return f"{self.interface_name}.{dim_str} <= {self.max_value}"


# =============================================================================
# Interface Relationships (Cross-Interface Scope)
# =============================================================================

@dataclass(frozen=True)
class DimensionEquality(InterfaceRelationship):
    """Two dimensions must be equal.

    Generative: If target dimension is unset, sets target = source.
    Validative: If both set, validates equality.

    The shape_hierarchy parameter determines which shape to compare:
    - ShapeHierarchy.STREAM: stream_shape (default)
    - ShapeHierarchy.BLOCK: block_shape
    - ShapeHierarchy.TENSOR: tensor_shape

    Examples:
        DimensionEquality("input", 0, "output", 0)  # output.stream[0] := input.stream[0]
        DimensionEquality("in1", None, "in2", None) # in2.stream_total := in1.stream_total
        DimensionEquality("in", 1, "out", 1, ShapeHierarchy.BLOCK)  # out.block[1] := in.block[1]
    """

    source_interface: str
    source_dim: Optional[int]  # None = total size
    target_interface: str
    target_dim: Optional[int]  # None = total size
    shape_hierarchy: ShapeHierarchy = ShapeHierarchy.STREAM

    def resolve(
        self,
        interfaces: Dict[str, Any],
        mutable_outputs: List[Any]
    ) -> bool:
        """Set target dimension from source if unset.

        Note: Only stream_shape can have unset (None) dimensions during construction.
        block_shape and tensor_shape are always fully resolved, so generative behavior
        only applies when shape_hierarchy == ShapeHierarchy.STREAM.
        """
        # Only stream_shape can be generative
        if self.shape_hierarchy != ShapeHierarchy.STREAM:
            return False

        source = interfaces.get(self.source_interface)
        target = interfaces.get(self.target_interface)

        if not source or not target:
            return False

        # Get source shape based on shape_target
        source_shape = source.stream_shape

        # Get source value
        if self.source_dim is None:
            source_value = math.prod(source_shape)
        else:
            if self.source_dim >= len(source_shape):
                return False
            source_value = source_shape[self.source_dim]

        if source_value is None:
            return False  # Source not resolved yet

        # Get target shape
        target_shape = target.stream_shape

        # Get target value
        if self.target_dim is None:
            # Can't set total size directly
            return False
        else:
            if self.target_dim >= len(target_shape):
                return False
            target_value = target_shape[self.target_dim]

        if target_value is not None:
            return False  # Target already set

        # GENERATIVE: Set target = source
        # Import here to avoid circular dependency
        from .models import OutputModel

        new_stream = list(target_shape)
        new_stream[self.target_dim] = source_value

        new_target = OutputModel(
            name=target.name,
            tensor_shape=target.tensor_shape,
            block_shape=target.block_shape,
            stream_shape=tuple(new_stream),
            datatype=target.datatype
        )

        # Update in mutable_outputs list
        for i, out in enumerate(mutable_outputs):
            if out.name == target.name:
                mutable_outputs[i] = new_target
                break

        # Update interfaces dict
        interfaces[target.name] = new_target
        return True

    def check(self, interfaces: Dict[str, Any]) -> Optional[str]:
        """Validate equality between dimensions."""
        source = interfaces.get(self.source_interface)
        target = interfaces.get(self.target_interface)

        if source is None or target is None:
            return None  # Interfaces not available (optional interfaces)

        # Get shapes based on shape_hierarchy
        source_shape = source.get_shape(self.shape_hierarchy)
        target_shape = target.get_shape(self.shape_hierarchy)
        shape_name = self.shape_hierarchy.value

        # Get dimension values
        if self.source_dim is None:
            source_value = math.prod(source_shape)
            source_desc = f"{self.source_interface}.{shape_name}_total"
        else:
            if self.source_dim >= len(source_shape):
                return f"{self.source_interface}.{shape_name}[{self.source_dim}] out of range"
            source_value = source_shape[self.source_dim]
            source_desc = f"{self.source_interface}.{shape_name}[{self.source_dim}]"

        if self.target_dim is None:
            target_value = math.prod(target_shape)
            target_desc = f"{self.target_interface}.{shape_name}_total"
        else:
            if self.target_dim >= len(target_shape):
                return f"{self.target_interface}.{shape_name}[{self.target_dim}] out of range"
            target_value = target_shape[self.target_dim]
            target_desc = f"{self.target_interface}.{shape_name}[{self.target_dim}]"

        # Validate equality
        if source_value != target_value:
            return f"{source_desc} ({source_value}) must equal {target_desc} ({target_value})"

        return None

    def describe(self) -> str:
        """Human-readable description."""
        shape_name = self.shape_hierarchy.value
        source_desc = (f"{self.source_interface}.{shape_name}[{self.source_dim}]"
                      if self.source_dim is not None
                      else f"{self.source_interface}.{shape_name}_total")
        target_desc = (f"{self.target_interface}.{shape_name}[{self.target_dim}]"
                      if self.target_dim is not None
                      else f"{self.target_interface}.{shape_name}_total")
        return f"{source_desc} == {target_desc}"


@dataclass(frozen=True)
class DimensionDivisibleBy(InterfaceRelationship):
    """Target dimension must be divisible by source dimension.

    Validative only: Cannot infer target from divisibility constraint alone.

    The shape_hierarchy parameter determines which shape to compare:
    - ShapeHierarchy.STREAM: stream_shape (default)
    - ShapeHierarchy.BLOCK: block_shape
    - ShapeHierarchy.TENSOR: tensor_shape

    Validates: target.shape[target_dim] % source.shape[source_dim] == 0

    Examples:
        DimensionDivisibleBy("output", 1, "input", 1)  # output.stream[1] % input.stream[1] == 0
        DimensionDivisibleBy("out", 0, "in", 0, ShapeHierarchy.BLOCK)  # out.block[0] % in.block[0] == 0
    """

    target_interface: str
    target_dim: Optional[int]  # None = total size
    source_interface: str
    source_dim: Optional[int]  # None = total size
    shape_hierarchy: ShapeHierarchy = ShapeHierarchy.STREAM

    # No resolve() method - not generative (can't infer from divisibility alone)

    def check(self, interfaces: Dict[str, Any]) -> Optional[str]:
        """Validate divisibility between dimensions."""
        source = interfaces.get(self.source_interface)
        target = interfaces.get(self.target_interface)

        if source is None or target is None:
            return None  # Interfaces not available

        # Get shapes based on shape_hierarchy
        source_shape = source.get_shape(self.shape_hierarchy)
        target_shape = target.get_shape(self.shape_hierarchy)
        shape_name = self.shape_hierarchy.value

        # Get dimension values
        if self.source_dim is None:
            source_value = math.prod(source_shape)
            source_desc = f"{self.source_interface}.{shape_name}_total"
        else:
            if self.source_dim >= len(source_shape):
                return f"{self.source_interface}.{shape_name}[{self.source_dim}] out of range"
            source_value = source_shape[self.source_dim]
            source_desc = f"{self.source_interface}.{shape_name}[{self.source_dim}]"

        if self.target_dim is None:
            target_value = math.prod(target_shape)
            target_desc = f"{self.target_interface}.{shape_name}_total"
        else:
            if self.target_dim >= len(target_shape):
                return f"{self.target_interface}.{shape_name}[{self.target_dim}] out of range"
            target_value = target_shape[self.target_dim]
            target_desc = f"{self.target_interface}.{shape_name}[{self.target_dim}]"

        # Validate divisibility
        if target_value % source_value != 0:
            return f"{target_desc} ({target_value}) must be divisible by {source_desc} ({source_value})"

        return None

    def describe(self) -> str:
        """Human-readable description."""
        shape_name = self.shape_hierarchy.value
        source_desc = (f"{self.source_interface}.{shape_name}[{self.source_dim}]"
                      if self.source_dim is not None
                      else f"{self.source_interface}.{shape_name}_total")
        target_desc = (f"{self.target_interface}.{shape_name}[{self.target_dim}]"
                      if self.target_dim is not None
                      else f"{self.target_interface}.{shape_name}_total")
        return f"{target_desc} % {source_desc} == 0"


@dataclass(frozen=True)
class DimensionScaled(InterfaceRelationship):
    """Target dimension must equal source dimension times a scale factor.

    Generative: If target dimension is unset, sets target = source * scale.
    Validative: If both set, validates relationship.

    The shape_hierarchy parameter determines which shape to compare:
    - ShapeHierarchy.STREAM: stream_shape (default)
    - ShapeHierarchy.BLOCK: block_shape
    - ShapeHierarchy.TENSOR: tensor_shape

    Note: Generative behavior only applies when shape_hierarchy == ShapeHierarchy.STREAM,
    as only stream_shape can have unset dimensions during construction.

    Examples:
        DimensionScaled("input", 0, "output", 0, 2)  # output.stream[0] := input.stream[0] * 2
        DimensionScaled("in", 1, "out", 1, 0.5, ShapeHierarchy.BLOCK)  # out.block[1] == in.block[1] * 0.5
    """

    source_interface: str
    source_dim: Optional[int]  # None = total size
    target_interface: str
    target_dim: Optional[int]  # None = total size
    scale_factor: Union[int, float]
    shape_hierarchy: ShapeHierarchy = ShapeHierarchy.STREAM

    def resolve(
        self,
        interfaces: Dict[str, Any],
        mutable_outputs: List[Any]
    ) -> bool:
        """Set target dimension from scaled source if unset.

        Note: Only stream_shape can have unset (None) dimensions during construction.
        block_shape and tensor_shape are always fully resolved, so generative behavior
        only applies when shape_hierarchy == ShapeHierarchy.STREAM.
        """
        # Only stream_shape can be generative
        if self.shape_hierarchy != ShapeHierarchy.STREAM:
            return False

        source = interfaces.get(self.source_interface)
        target = interfaces.get(self.target_interface)

        if not source or not target:
            return False

        # Get source shape based on shape_target
        source_shape = source.stream_shape

        # Get source value
        if self.source_dim is None:
            source_value = math.prod(source_shape)
        else:
            if self.source_dim >= len(source_shape):
                return False
            source_value = source_shape[self.source_dim]

        if source_value is None:
            return False  # Source not resolved yet

        # Get target shape
        target_shape = target.stream_shape

        # Get target value
        if self.target_dim is None:
            # Can't set total size directly
            return False
        else:
            if self.target_dim >= len(target_shape):
                return False
            target_value = target_shape[self.target_dim]

        if target_value is not None:
            return False  # Target already set

        # GENERATIVE: Set target = source * scale
        from .models import OutputModel

        expected = int(source_value * self.scale_factor)
        new_stream = list(target_shape)
        new_stream[self.target_dim] = expected

        new_target = OutputModel(
            name=target.name,
            tensor_shape=target.tensor_shape,
            block_shape=target.block_shape,
            stream_shape=tuple(new_stream),
            datatype=target.datatype
        )

        # Update in mutable_outputs list
        for i, out in enumerate(mutable_outputs):
            if out.name == target.name:
                mutable_outputs[i] = new_target
                break

        # Update interfaces dict
        interfaces[target.name] = new_target
        return True

    def check(self, interfaces: Dict[str, Any]) -> Optional[str]:
        """Validate scaled equality between dimensions."""
        source = interfaces.get(self.source_interface)
        target = interfaces.get(self.target_interface)

        if source is None or target is None:
            return None  # Interfaces not available

        # Get shapes based on shape_hierarchy
        source_shape = source.get_shape(self.shape_hierarchy)
        target_shape = target.get_shape(self.shape_hierarchy)
        shape_name = self.shape_hierarchy.value

        # Get dimension values
        if self.source_dim is None:
            source_value = math.prod(source_shape)
            source_desc = f"{self.source_interface}.{shape_name}_total"
        else:
            if self.source_dim >= len(source_shape):
                return f"{self.source_interface}.{shape_name}[{self.source_dim}] out of range"
            source_value = source_shape[self.source_dim]
            source_desc = f"{self.source_interface}.{shape_name}[{self.source_dim}]"

        if self.target_dim is None:
            target_value = math.prod(target_shape)
            target_desc = f"{self.target_interface}.{shape_name}_total"
        else:
            if self.target_dim >= len(target_shape):
                return f"{self.target_interface}.{shape_name}[{self.target_dim}] out of range"
            target_value = target_shape[self.target_dim]
            target_desc = f"{self.target_interface}.{shape_name}[{self.target_dim}]"

        # Validate scaled equality
        expected = source_value * self.scale_factor
        if target_value != expected:
            return f"{target_desc} ({target_value}) must equal {source_desc} ({source_value}) * {self.scale_factor} = {expected}"

        return None

    def describe(self) -> str:
        """Human-readable description."""
        shape_name = self.shape_hierarchy.value
        source_desc = (f"{self.source_interface}.{shape_name}[{self.source_dim}]"
                      if self.source_dim is not None
                      else f"{self.source_interface}.{shape_name}_total")
        target_desc = (f"{self.target_interface}.{shape_name}[{self.target_dim}]"
                      if self.target_dim is not None
                      else f"{self.target_interface}.{shape_name}_total")
        return f"{target_desc} == {source_desc} * {self.scale_factor}"


__all__ = [
    # Base classes
    "InterfaceConstraint",
    "InterfaceRelationship",
    # Interface constraints
    "DatatypeConstraint",
    "DimensionDivisible",
    "DimensionMinValue",
    "DimensionMaxValue",
    # Interface relationships
    "DimensionEquality",
    "DimensionDivisibleBy",
    "DimensionScaled",
]
