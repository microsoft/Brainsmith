############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Atomic dimension constraints for dataflow modeling.

This module provides constraint types that validate individual dimensions
or relationships between dimensions across interfaces.

## Constraint Categories

### Atomic Constraints (Single Dimension)
- **DivisibleConstraint**: Dimension divisible by value
- **MinValueConstraint**: Dimension >= minimum value
- **MaxValueConstraint**: Dimension <= maximum value

### Cross-Interface Constraints
- **EqualityConstraint**: Two dimensions must be equal
- **DivisibleByDimensionConstraint**: Dimension divisible by another dimension
- **ScaledEqualityConstraint**: Dimension equals another * scale factor

## Design Principles

**Composition over Monoliths:**
    Range constraints via Min + Max composition:

        # Range: 8 <= input[0] <= 1024
        schema.add_dimension_constraint(MinValueConstraint("input", 0, 8))
        schema.add_dimension_constraint(MaxValueConstraint("input", 0, 1024))

**DRY (Don't Repeat Yourself):**
    All constraints use shared helper functions for:
    - Interface lookup (_get_interface)
    - Dimension extraction (_get_dimension_value)
    - Parameter resolution (_resolve_parameter)

**Simplicity:**
    Each constraint class focuses on ONE validation rule.
    Complex validations are composed from simple constraints.

## Examples

### Basic Usage

    from brainsmith.core.dataflow import (
        DivisibleConstraint, MinValueConstraint,
        equal_dimension, divisible_dimension
    )

    # Atomic constraint: SIMD must divide input dimension
    input_schema.add_dimension_constraint(
        DivisibleConstraint("input", 0, "SIMD")
    )

    # Range via composition
    input_schema.add_dimension_constraint(MinValueConstraint("input", 1, 64))
    input_schema.add_dimension_constraint(MaxValueConstraint("input", 1, 1024))

    # Cross-interface: matrix columns == vector length
    schema.relationships.append(
        equal_dimension("matrix", "vector", 1, 0)
    )

### Power-of-2 Validation

For power-of-2 checks, use a simple bit operation in custom validation:

    def is_power_of_two(n):
        return n > 0 and (n & (n - 1)) == 0

    if not is_power_of_two(dimension):
        raise ValueError(f"{dimension} must be a power of 2")
"""

from dataclasses import dataclass
from typing import Dict, Any, Union, Optional, Tuple, List, Callable
from abc import ABC, abstractmethod
import math

from .relationships import ValidationResult, ConstraintViolation


# ===========================================================================
# Helper Functions (DRY)
# ===========================================================================

def _get_interface(
    context: Dict[str, Any],
    name: str
) -> Tuple[Any, Optional[ConstraintViolation]]:
    """Get interface from context with error handling.

    Returns:
        (interface, None) on success
        (None, violation) on error
    """
    if name not in context:
        violation = ConstraintViolation(
            constraint_type="constraint_evaluation",
            message=f"Interface '{name}' not found in context",
            severity="error"
        )
        return None, violation
    return context[name], None


def _get_dimension_value(
    interface: Any,
    dim_index: Optional[int],
    interface_name: str
) -> Tuple[Optional[int], Optional[str], Optional[ConstraintViolation]]:
    """Extract dimension value and description.

    Returns:
        (value, description, None) on success
        (None, None, violation) on error
    """
    if dim_index is None:
        from .types import prod
        return prod(interface.tensor_shape), "total", None

    if dim_index >= len(interface.tensor_shape):
        violation = ConstraintViolation(
            constraint_type="constraint_evaluation",
            message=f"Dimension index {dim_index} out of range for {interface_name}",
            severity="error"
        )
        return None, None, violation

    return interface.tensor_shape[dim_index], f"dim[{dim_index}]", None


def _resolve_parameter(
    param: Union[int, str, float],
    context: Dict[str, Any],
    param_type: str = "parameter"
) -> Tuple[Optional[Any], Optional[ConstraintViolation]]:
    """Resolve parameter from context or return literal.

    Returns:
        (value, None) on success
        (None, violation) on error
    """
    if isinstance(param, str):
        if param not in context:
            violation = ConstraintViolation(
                constraint_type="constraint_evaluation",
                message=f"{param_type} '{param}' not found in context",
                severity="error"
            )
            return None, violation
        return context[param], None
    return param, None


def _get_two_dimension_values(
    context: Dict[str, Any],
    source_interface: str,
    source_dim: Optional[int],
    target_interface: str,
    target_dim: Optional[int]
) -> Tuple[Optional[int], Optional[str], Optional[int], Optional[str], List[ConstraintViolation]]:
    """Get values from two interfaces.

    Returns:
        (source_value, source_desc, target_value, target_desc, []) on success
        (None, None, None, None, [violations]) on error
    """
    violations = []

    # Get source
    source_intf, err = _get_interface(context, source_interface)
    if err:
        return None, None, None, None, [err]

    source_value, source_desc, err = _get_dimension_value(source_intf, source_dim, source_interface)
    if err:
        return None, None, None, None, [err]

    # Get target
    target_intf, err = _get_interface(context, target_interface)
    if err:
        return None, None, None, None, [err]

    target_value, target_desc, err = _get_dimension_value(target_intf, target_dim, target_interface)
    if err:
        return None, None, None, None, [err]

    return source_value, source_desc, target_value, target_desc, []


# ===========================================================================
# Base Class
# ===========================================================================

@dataclass(frozen=True)
class DimensionConstraint(ABC):
    """Base class for dimension constraints.

    Constraints validate dimensions against rules. They can reference:
    - Literal values (e.g., divisor=8)
    - Node attributes (e.g., divisor="SIMD")
    - Other interface dimensions (for cross-interface constraints)

    Two validation methods:
    - check_interface(): For atomic constraints (single interface)
    - check_relationship(): For cross-interface constraints (multiple interfaces)
    """

    @abstractmethod
    def validate_with_context(self, context: Dict[str, Any]) -> ValidationResult:
        """Validate constraint given a context of values.

        DEPRECATED: Use check_interface() or check_relationship() instead.

        Args:
            context: Dictionary containing:
                - Node attributes (e.g., {"SIMD": 16})
                - Interface models (e.g., {"input": InputModel(...)})

        Returns:
            ValidationResult with any violations
        """
        pass

    @abstractmethod
    def check_interface(
        self,
        interface_name: str,
        interface_model: Any,  # InputModel or OutputModel
        nodeattr_getter: Callable[[str], Any]
    ) -> Optional[str]:
        """Check constraint against a single interface.

        For atomic constraints: Validate the dimension (typically stream_shape)
        For cross-interface constraints: Return None (not applicable)

        Args:
            interface_name: Name of interface being validated
            interface_model: Full interface model (InputModel or OutputModel) with all shape levels
            nodeattr_getter: Function to resolve nodeattr names (e.g., self.get_nodeattr)

        Returns:
            None if constraint is valid or not applicable to this interface
            Error message string if constraint is violated

        Note:
            Constraints typically validate stream_shape (streaming parallelism dimensions).
            For FPGA kernels: stream_shape defines elements processed per cycle.
        """
        pass

    @abstractmethod
    def check_relationship(
        self,
        interfaces: Dict[str, Any]
    ) -> Optional[str]:
        """Check constraint across multiple interfaces.

        For atomic constraints: Return None (not applicable)
        For cross-interface constraints: Validate the relationship

        Args:
            interfaces: Dict mapping interface names to InterfaceModel objects

        Returns:
            None if relationship is valid or not applicable
            Error message string if relationship is violated
        """
        pass

    @abstractmethod
    def describe(self) -> str:
        """Human-readable description of the constraint."""
        pass


# ===========================================================================
# Atomic Constraints (Single Dimension)
# ===========================================================================

@dataclass(frozen=True)
class DivisibleConstraint(DimensionConstraint):
    """Dimension must be evenly divisible by a value.

    Examples:
        DivisibleConstraint("input", 0, 8)  # input[0] % 8 == 0
        DivisibleConstraint("input", 0, "SIMD")  # input[0] % SIMD == 0
        DivisibleConstraint("input", None, 64)  # total elements % 64 == 0
    """

    interface_name: str
    dim_index: Optional[int]  # None = total size (product of all dims)
    divisor: Union[int, str]  # Literal or nodeattr name

    def validate_with_context(self, context: Dict[str, Any]) -> ValidationResult:
        """Validate divisibility constraint."""
        # Get interface
        interface, err = _get_interface(context, self.interface_name)
        if err:
            return ValidationResult(violations=[err])

        # Get dimension value
        dim_value, dim_desc, err = _get_dimension_value(interface, self.dim_index, self.interface_name)
        if err:
            return ValidationResult(violations=[err])

        # Get divisor
        divisor_value, err = _resolve_parameter(self.divisor, context, "Divisor")
        if err:
            return ValidationResult(violations=[err])

        # Validate
        if dim_value % divisor_value != 0:
            violation = ConstraintViolation(
                constraint_type="divisibility",
                message=f"{self.interface_name}.{dim_desc} must be divisible by {divisor_value}",
                expected=f"multiple of {divisor_value}",
                actual=dim_value,
                severity="error",
                details={"remainder": dim_value % divisor_value}
            )
            return ValidationResult(violations=[violation])

        return ValidationResult()

    def check_interface(
        self,
        interface_name: str,
        interface_model: Any,
        nodeattr_getter: Callable[[str], Any]
    ) -> Optional[str]:
        """Check divisibility constraint on interface stream dimensions."""
        # Only validate this interface
        if interface_name != self.interface_name:
            return None

        # Use stream_shape for validation (streaming parallelism dimensions)
        # InputModel has stream_shape, OutputModel uses block_shape as stream equivalent
        if hasattr(interface_model, 'stream_shape'):
            stream_shape = interface_model.stream_shape
        else:
            # OutputModel: use block_shape as stream dimensions
            stream_shape = interface_model.block_shape

        # Get dimension value
        if self.dim_index is None:
            dim_value = math.prod(stream_shape)
            dim_desc = "stream_total"
        else:
            if self.dim_index >= len(stream_shape):
                return f"Stream dimension index {self.dim_index} out of range for stream shape {stream_shape}"
            dim_value = stream_shape[self.dim_index]
            dim_desc = f"stream[{self.dim_index}]"

        # Resolve divisor (literal or nodeattr)
        if isinstance(self.divisor, str):
            try:
                divisor_value = nodeattr_getter(self.divisor)
            except (AttributeError, KeyError) as e:
                return f"Nodeattr '{self.divisor}' not found: {e}"
        else:
            divisor_value = self.divisor

        # Validate divisibility
        if dim_value % divisor_value != 0:
            return f"{dim_desc} ({dim_value}) must be divisible by {divisor_value}"

        return None

    def check_relationship(self, interfaces: Dict[str, Any]) -> Optional[str]:
        """Not a cross-interface constraint."""
        return None

    def describe(self) -> str:
        """Human-readable description."""
        dim_str = "total" if self.dim_index is None else f"dim[{self.dim_index}]"
        return f"{self.interface_name}.{dim_str} % {self.divisor} == 0"


@dataclass(frozen=True)
class MinValueConstraint(DimensionConstraint):
    """Dimension must be >= minimum value.

    Examples:
        MinValueConstraint("input", 0, 1)  # input[0] >= 1
        MinValueConstraint("output", 1, "MIN_CHANNELS")  # output[1] >= MIN_CHANNELS
    """

    interface_name: str
    dim_index: Optional[int]
    min_value: Union[int, str]

    def validate_with_context(self, context: Dict[str, Any]) -> ValidationResult:
        """Validate minimum value constraint."""
        # Get interface
        interface, err = _get_interface(context, self.interface_name)
        if err:
            return ValidationResult(violations=[err])

        # Get dimension value
        dim_value, dim_desc, err = _get_dimension_value(interface, self.dim_index, self.interface_name)
        if err:
            return ValidationResult(violations=[err])

        # Get min value
        min_val, err = _resolve_parameter(self.min_value, context, "Min value")
        if err:
            return ValidationResult(violations=[err])

        # Validate
        if dim_value < min_val:
            violation = ConstraintViolation(
                constraint_type="min_value",
                message=f"{self.interface_name}.{dim_desc} must be >= {min_val}",
                expected=f">= {min_val}",
                actual=dim_value,
                severity="error"
            )
            return ValidationResult(violations=[violation])

        return ValidationResult()

    def check_interface(
        self,
        interface_name: str,
        interface_model: Any,
        nodeattr_getter: Callable[[str], Any]
    ) -> Optional[str]:
        """Check minimum value constraint on interface stream dimensions."""
        if interface_name != self.interface_name:
            return None

        # Use stream_shape for validation
        if hasattr(interface_model, 'stream_shape'):
            stream_shape = interface_model.stream_shape
        else:
            stream_shape = interface_model.block_shape

        # Get dimension value
        if self.dim_index is None:
            dim_value = math.prod(stream_shape)
            dim_desc = "stream_total"
        else:
            if self.dim_index >= len(stream_shape):
                return f"Stream dimension index {self.dim_index} out of range for stream shape {stream_shape}"
            dim_value = stream_shape[self.dim_index]
            dim_desc = f"stream[{self.dim_index}]"

        # Resolve min value
        if isinstance(self.min_value, str):
            try:
                min_val = nodeattr_getter(self.min_value)
            except (AttributeError, KeyError) as e:
                return f"Nodeattr '{self.min_value}' not found: {e}"
        else:
            min_val = self.min_value

        # Validate
        if dim_value < min_val:
            return f"{dim_desc} ({dim_value}) must be >= {min_val}"

        return None

    def check_relationship(self, interfaces: Dict[str, Any]) -> Optional[str]:
        """Not a cross-interface constraint."""
        return None

    def describe(self) -> str:
        """Human-readable description."""
        dim_str = "total" if self.dim_index is None else f"dim[{self.dim_index}]"
        return f"{self.interface_name}.{dim_str} >= {self.min_value}"


@dataclass(frozen=True)
class MaxValueConstraint(DimensionConstraint):
    """Dimension must be <= maximum value.

    Examples:
        MaxValueConstraint("input", 0, 1024)  # input[0] <= 1024
        MaxValueConstraint("output", 1, "MAX_CHANNELS")  # output[1] <= MAX_CHANNELS
    """

    interface_name: str
    dim_index: Optional[int]
    max_value: Union[int, str]

    def validate_with_context(self, context: Dict[str, Any]) -> ValidationResult:
        """Validate maximum value constraint."""
        # Get interface
        interface, err = _get_interface(context, self.interface_name)
        if err:
            return ValidationResult(violations=[err])

        # Get dimension value
        dim_value, dim_desc, err = _get_dimension_value(interface, self.dim_index, self.interface_name)
        if err:
            return ValidationResult(violations=[err])

        # Get max value
        max_val, err = _resolve_parameter(self.max_value, context, "Max value")
        if err:
            return ValidationResult(violations=[err])

        # Validate
        if dim_value > max_val:
            violation = ConstraintViolation(
                constraint_type="max_value",
                message=f"{self.interface_name}.{dim_desc} must be <= {max_val}",
                expected=f"<= {max_val}",
                actual=dim_value,
                severity="error"
            )
            return ValidationResult(violations=[violation])

        return ValidationResult()

    def check_interface(
        self,
        interface_name: str,
        interface_model: Any,
        nodeattr_getter: Callable[[str], Any]
    ) -> Optional[str]:
        """Check maximum value constraint on interface stream dimensions."""
        if interface_name != self.interface_name:
            return None

        # Use stream_shape for validation
        if hasattr(interface_model, 'stream_shape'):
            stream_shape = interface_model.stream_shape
        else:
            stream_shape = interface_model.block_shape

        # Get dimension value
        if self.dim_index is None:
            dim_value = math.prod(stream_shape)
            dim_desc = "stream_total"
        else:
            if self.dim_index >= len(stream_shape):
                return f"Stream dimension index {self.dim_index} out of range for stream shape {stream_shape}"
            dim_value = stream_shape[self.dim_index]
            dim_desc = f"stream[{self.dim_index}]"

        # Resolve max value
        if isinstance(self.max_value, str):
            try:
                max_val = nodeattr_getter(self.max_value)
            except (AttributeError, KeyError) as e:
                return f"Nodeattr '{self.max_value}' not found: {e}"
        else:
            max_val = self.max_value

        # Validate
        if dim_value > max_val:
            return f"{dim_desc} ({dim_value}) must be <= {max_val}"

        return None

    def check_relationship(self, interfaces: Dict[str, Any]) -> Optional[str]:
        """Not a cross-interface constraint."""
        return None

    def describe(self) -> str:
        """Human-readable description."""
        dim_str = "total" if self.dim_index is None else f"dim[{self.dim_index}]"
        return f"{self.interface_name}.{dim_str} <= {self.max_value}"


# ===========================================================================
# Cross-Interface Constraints
# ===========================================================================

@dataclass(frozen=True)
class BinaryDimensionConstraint(DimensionConstraint, ABC):
    """Base for constraints comparing two dimensions.

    Subclasses only need to implement _validate_relationship().
    All dimension extraction and error handling is done here.
    """

    source_interface: str
    source_dim: Optional[int]
    target_interface: str
    target_dim: Optional[int]

    def validate_with_context(self, context: Dict[str, Any]) -> ValidationResult:
        """Extract both values and delegate to subclass."""
        # Get both dimension values
        source_value, source_desc, target_value, target_desc, errors = _get_two_dimension_values(
            context, self.source_interface, self.source_dim, self.target_interface, self.target_dim
        )

        if errors:
            return ValidationResult(violations=errors)

        # Delegate to subclass for specific validation
        violation = self._validate_relationship(
            source_value, source_desc, target_value, target_desc
        )

        if violation:
            return ValidationResult(violations=[violation])
        return ValidationResult()

    def check_interface(
        self,
        interface_name: str,
        interface_model: Any,
        nodeattr_getter: Callable[[str], Any]
    ) -> Optional[str]:
        """Cross-interface constraints cannot be validated with single interface."""
        return None

    def check_relationship(self, interfaces: Dict[str, Any]) -> Optional[str]:
        """Validate relationship between two interfaces (using stream dimensions)."""
        # Get source interface
        source = interfaces.get(self.source_interface)
        target = interfaces.get(self.target_interface)

        if source is None or target is None:
            return None  # Interfaces not available (optional?)

        # Use stream_shape for validation (streaming parallelism)
        source_stream = source.stream_shape if hasattr(source, 'stream_shape') else source.block_shape
        target_stream = target.stream_shape if hasattr(target, 'stream_shape') else target.block_shape

        # Get dimension values from stream shapes
        if self.source_dim is None:
            source_value = math.prod(source_stream)
            source_desc = "stream_total"
        else:
            if self.source_dim >= len(source_stream):
                return f"{self.source_interface}: stream dimension index {self.source_dim} out of range"
            source_value = source_stream[self.source_dim]
            source_desc = f"stream[{self.source_dim}]"

        if self.target_dim is None:
            target_value = math.prod(target_stream)
            target_desc = "stream_total"
        else:
            if self.target_dim >= len(target_stream):
                return f"{self.target_interface}: stream dimension index {self.target_dim} out of range"
            target_value = target_stream[self.target_dim]
            target_desc = f"stream[{self.target_dim}]"

        # Delegate to subclass
        violation = self._validate_relationship(
            source_value, source_desc, target_value, target_desc
        )

        if violation:
            return violation.message
        return None

    @abstractmethod
    def _validate_relationship(
        self,
        source_value: int,
        source_desc: str,
        target_value: int,
        target_desc: str
    ) -> Optional[ConstraintViolation]:
        """Validate the specific relationship.

        Returns:
            ConstraintViolation if invalid, None if valid
        """
        pass


@dataclass(frozen=True)
class EqualityConstraint(BinaryDimensionConstraint):
    """Two dimensions must be equal.

    Examples:
        EqualityConstraint("input", 0, "output", 0)  # input[0] == output[0]
        EqualityConstraint("matrix", 1, "vector", 0)  # matrix[1] == vector[0]
        EqualityConstraint("input", None, "output", None)  # total sizes equal
    """

    def _validate_relationship(
        self, source_value, source_desc, target_value, target_desc
    ) -> Optional[ConstraintViolation]:
        if source_value != target_value:
            return ConstraintViolation(
                constraint_type="dimension_equality",
                message=f"{self.source_interface}.{source_desc} must equal {self.target_interface}.{target_desc}",
                expected=source_value,
                actual=target_value,
                severity="error"
            )
        return None

    def describe(self) -> str:
        """Human-readable description."""
        source_desc = "total" if self.source_dim is None else f"dim[{self.source_dim}]"
        target_desc = "total" if self.target_dim is None else f"dim[{self.target_dim}]"
        return f"{self.source_interface}.{source_desc} == {self.target_interface}.{target_desc}"


@dataclass(frozen=True)
class DivisibleByDimensionConstraint(BinaryDimensionConstraint):
    """Dimension must be divisible by another dimension.

    Examples:
        DivisibleByDimensionConstraint("output", 0, "input", 0)  # output[0] % input[0] == 0
        DivisibleByDimensionConstraint("block", None, "stream", None)  # block_total % stream_total == 0

    Note: source is dividend, target is divisor (source % target == 0)
    """

    def _validate_relationship(
        self, source_value, source_desc, target_value, target_desc
    ) -> Optional[ConstraintViolation]:
        # Note: source is dividend, target is divisor
        dividend = source_value
        divisor = target_value

        if divisor == 0:
            return ConstraintViolation(
                constraint_type="constraint_evaluation",
                message=f"Divisor {self.target_interface}.{target_desc} cannot be zero",
                severity="error"
            )

        if dividend % divisor != 0:
            return ConstraintViolation(
                constraint_type="divisibility",
                message=f"{self.source_interface}.{source_desc} must be divisible by {self.target_interface}.{target_desc}",
                expected=f"multiple of {divisor}",
                actual=dividend,
                severity="error",
                details={"remainder": dividend % divisor}
            )
        return None

    def describe(self) -> str:
        """Human-readable description."""
        source_desc = "total" if self.source_dim is None else f"dim[{self.source_dim}]"
        target_desc = "total" if self.target_dim is None else f"dim[{self.target_dim}]"
        return f"{self.source_interface}.{source_desc} % {self.target_interface}.{target_desc} == 0"


@dataclass(frozen=True)
class ScaledEqualityConstraint(BinaryDimensionConstraint):
    """Dimension must equal another dimension * scale_factor.

    Examples:
        ScaledEqualityConstraint("output", 0, "input", 0, 2)  # output[0] == input[0] * 2
        ScaledEqualityConstraint("y", 1, "x", 1, "SCALE")  # y[1] == x[1] * SCALE
    """

    scale_factor: Union[int, float, str] = 1

    def validate_with_context(self, context: Dict[str, Any]) -> ValidationResult:
        """Validate scaled equality constraint."""
        # Get both dimension values via parent
        source_value, source_desc, target_value, target_desc, errors = _get_two_dimension_values(
            context, self.source_interface, self.source_dim, self.target_interface, self.target_dim
        )

        if errors:
            return ValidationResult(violations=errors)

        # Resolve scale factor
        scale, err = _resolve_parameter(self.scale_factor, context, "Scale factor")
        if err:
            return ValidationResult(violations=[err])

        # Validate with scale
        violation = self._validate_with_scale(
            source_value, source_desc, target_value, target_desc, scale
        )

        if violation:
            return ValidationResult(violations=[violation])
        return ValidationResult()

    def _validate_relationship(
        self, source_value, source_desc, target_value, target_desc
    ) -> Optional[ConstraintViolation]:
        """Validate scaled equality relationship."""
        return self._validate_with_scale(
            source_value, source_desc, target_value, target_desc, self.scale_factor
        )

    def _validate_with_scale(
        self, source_value, source_desc, target_value, target_desc, scale
    ) -> Optional[ConstraintViolation]:
        expected = source_value * scale
        if target_value != expected:
            return ConstraintViolation(
                constraint_type="scaled_equality",
                message=f"{self.target_interface}.{target_desc} must equal {self.source_interface}.{source_desc} * {scale}",
                expected=expected,
                actual=target_value,
                severity="error"
            )
        return None

    def describe(self) -> str:
        """Human-readable description."""
        source_desc = "total" if self.source_dim is None else f"dim[{self.source_dim}]"
        target_desc = "total" if self.target_dim is None else f"dim[{self.target_dim}]"
        return f"{self.target_interface}.{target_desc} == {self.source_interface}.{source_desc} * {self.scale_factor}"


__all__ = [
    "DimensionConstraint",
    # Atomic constraints
    "DivisibleConstraint",
    "MinValueConstraint",
    "MaxValueConstraint",
    # Cross-interface constraints
    "EqualityConstraint",
    "DivisibleByDimensionConstraint",
    "ScaledEqualityConstraint",
]
