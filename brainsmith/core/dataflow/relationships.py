############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Interface relationship system for dataflow modeling.

InterfaceRelationships express constraints between multiple interfaces (cross-interface scope).
They can be generative (set unset dimensions) or validative (check dimension relationships).

Shape Hierarchy:
All relationships support targeting different shape levels via the shape_hierarchy parameter
(defaults to ShapeHierarchy.STREAM):
- ShapeHierarchy.STREAM: Relate stream_shape (streaming parallelism)
- ShapeHierarchy.BLOCK: Relate block_shape (block tiling dimensions)
- ShapeHierarchy.TENSOR: Relate tensor_shape (full logical dimensions)

Generative vs. Validative:
- Generative: Set unset output dimensions during KernelModel construction (stream_shape only)
- Validative: Check dimension relationships after resolution (all shape hierarchy levels)

Note: Only stream_shape can have unset (None) dimensions during construction, so generative
behavior only applies when shape_hierarchy == ShapeHierarchy.STREAM.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod

from .types import ShapeHierarchy


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
# Concrete Relationship Classes
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
    # Base class
    "InterfaceRelationship",
    # Concrete relationships
    "DimensionEquality",
    "DimensionDivisibleBy",
    "DimensionScaled",
]
