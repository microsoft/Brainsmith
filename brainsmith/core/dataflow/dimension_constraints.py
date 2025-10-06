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

Two categories of constraints:
1. Atomic constraints: Validate a single dimension against a rule
2. Cross-interface constraints: Validate relationships between dimensions

All constraints are immutable and composable.
"""

from dataclasses import dataclass
from typing import Dict, Any, Union, Optional
from abc import ABC, abstractmethod
import math

from .relationships import ValidationResult, ConstraintViolation


@dataclass(frozen=True)
class DimensionConstraint(ABC):
    """Base class for dimension constraints.

    Constraints validate dimensions against rules. They can reference:
    - Literal values (e.g., divisor=8)
    - Node attributes (e.g., divisor="SIMD")
    - Other interface dimensions (for cross-interface constraints)
    """

    @abstractmethod
    def validate_with_context(self, context: Dict[str, Any]) -> ValidationResult:
        """Validate constraint given a context of values.

        Args:
            context: Dictionary containing:
                - Node attributes (e.g., {"SIMD": 16})
                - Interface models (e.g., {"input": InputModel(...)})

        Returns:
            ValidationResult with any violations
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
        violations = []

        # Get interface model
        if self.interface_name not in context:
            violations.append(ConstraintViolation(
                constraint_type="constraint_evaluation",
                message=f"Interface '{self.interface_name}' not found in context",
                severity="error"
            ))
            return ValidationResult(violations=violations)

        interface = context[self.interface_name]

        # Get dimension value
        if self.dim_index is None:
            # Total size
            from .types import prod
            dim_value = prod(interface.tensor_shape)
            dim_desc = "total_size"
        else:
            # Specific dimension
            if self.dim_index >= len(interface.tensor_shape):
                violations.append(ConstraintViolation(
                    constraint_type="constraint_evaluation",
                    message=f"Dimension index {self.dim_index} out of range for {self.interface_name}",
                    severity="error"
                ))
                return ValidationResult(violations=violations)
            dim_value = interface.tensor_shape[self.dim_index]
            dim_desc = f"dim[{self.dim_index}]"

        # Get divisor value
        if isinstance(self.divisor, str):
            if self.divisor not in context:
                violations.append(ConstraintViolation(
                    constraint_type="constraint_evaluation",
                    message=f"Divisor parameter '{self.divisor}' not found in context",
                    severity="error"
                ))
                return ValidationResult(violations=violations)
            divisor_value = context[self.divisor]
        else:
            divisor_value = self.divisor

        # Validate divisibility
        if dim_value % divisor_value != 0:
            violations.append(ConstraintViolation(
                constraint_type="divisibility",
                message=f"{self.interface_name}.{dim_desc} must be divisible by {divisor_value}",
                expected=f"multiple of {divisor_value}",
                actual=dim_value,
                severity="error",
                details={"remainder": dim_value % divisor_value}
            ))

        return ValidationResult(violations=violations)

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
        violations = []

        # Get interface model
        if self.interface_name not in context:
            violations.append(ConstraintViolation(
                constraint_type="constraint_evaluation",
                message=f"Interface '{self.interface_name}' not found in context",
                severity="error"
            ))
            return ValidationResult(violations=violations)

        interface = context[self.interface_name]

        # Get dimension value
        if self.dim_index is None:
            from .types import prod
            dim_value = prod(interface.tensor_shape)
            dim_desc = "total_size"
        else:
            if self.dim_index >= len(interface.tensor_shape):
                violations.append(ConstraintViolation(
                    constraint_type="constraint_evaluation",
                    message=f"Dimension index {self.dim_index} out of range for {self.interface_name}",
                    severity="error"
                ))
                return ValidationResult(violations=violations)
            dim_value = interface.tensor_shape[self.dim_index]
            dim_desc = f"dim[{self.dim_index}]"

        # Get min value
        if isinstance(self.min_value, str):
            if self.min_value not in context:
                violations.append(ConstraintViolation(
                    constraint_type="constraint_evaluation",
                    message=f"Min value parameter '{self.min_value}' not found in context",
                    severity="error"
                ))
                return ValidationResult(violations=violations)
            min_val = context[self.min_value]
        else:
            min_val = self.min_value

        # Validate minimum
        if dim_value < min_val:
            violations.append(ConstraintViolation(
                constraint_type="min_value",
                message=f"{self.interface_name}.{dim_desc} must be >= {min_val}",
                expected=f">= {min_val}",
                actual=dim_value,
                severity="error"
            ))

        return ValidationResult(violations=violations)

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
        violations = []

        # Get interface model
        if self.interface_name not in context:
            violations.append(ConstraintViolation(
                constraint_type="constraint_evaluation",
                message=f"Interface '{self.interface_name}' not found in context",
                severity="error"
            ))
            return ValidationResult(violations=violations)

        interface = context[self.interface_name]

        # Get dimension value
        if self.dim_index is None:
            from .types import prod
            dim_value = prod(interface.tensor_shape)
            dim_desc = "total_size"
        else:
            if self.dim_index >= len(interface.tensor_shape):
                violations.append(ConstraintViolation(
                    constraint_type="constraint_evaluation",
                    message=f"Dimension index {self.dim_index} out of range for {self.interface_name}",
                    severity="error"
                ))
                return ValidationResult(violations=violations)
            dim_value = interface.tensor_shape[self.dim_index]
            dim_desc = f"dim[{self.dim_index}]"

        # Get max value
        if isinstance(self.max_value, str):
            if self.max_value not in context:
                violations.append(ConstraintViolation(
                    constraint_type="constraint_evaluation",
                    message=f"Max value parameter '{self.max_value}' not found in context",
                    severity="error"
                ))
                return ValidationResult(violations=violations)
            max_val = context[self.max_value]
        else:
            max_val = self.max_value

        # Validate maximum
        if dim_value > max_val:
            violations.append(ConstraintViolation(
                constraint_type="max_value",
                message=f"{self.interface_name}.{dim_desc} must be <= {max_val}",
                expected=f"<= {max_val}",
                actual=dim_value,
                severity="error"
            ))

        return ValidationResult(violations=violations)

    def describe(self) -> str:
        """Human-readable description."""
        dim_str = "total" if self.dim_index is None else f"dim[{self.dim_index}]"
        return f"{self.interface_name}.{dim_str} <= {self.max_value}"


@dataclass(frozen=True)
class RangeConstraint(DimensionConstraint):
    """Dimension must be in [min, max] range.

    Examples:
        RangeConstraint("input", 0, 8, 1024)  # 8 <= input[0] <= 1024
        RangeConstraint("input", 1, "MIN_CH", "MAX_CH")  # MIN_CH <= input[1] <= MAX_CH
    """

    interface_name: str
    dim_index: Optional[int]
    min_value: Union[int, str]
    max_value: Union[int, str]

    def validate_with_context(self, context: Dict[str, Any]) -> ValidationResult:
        """Validate range constraint."""
        violations = []

        # Get interface model
        if self.interface_name not in context:
            violations.append(ConstraintViolation(
                constraint_type="constraint_evaluation",
                message=f"Interface '{self.interface_name}' not found in context",
                severity="error"
            ))
            return ValidationResult(violations=violations)

        interface = context[self.interface_name]

        # Get dimension value
        if self.dim_index is None:
            from .types import prod
            dim_value = prod(interface.tensor_shape)
            dim_desc = "total_size"
        else:
            if self.dim_index >= len(interface.tensor_shape):
                violations.append(ConstraintViolation(
                    constraint_type="constraint_evaluation",
                    message=f"Dimension index {self.dim_index} out of range for {self.interface_name}",
                    severity="error"
                ))
                return ValidationResult(violations=violations)
            dim_value = interface.tensor_shape[self.dim_index]
            dim_desc = f"dim[{self.dim_index}]"

        # Get min/max values
        if isinstance(self.min_value, str):
            if self.min_value not in context:
                violations.append(ConstraintViolation(
                    constraint_type="constraint_evaluation",
                    message=f"Min value parameter '{self.min_value}' not found in context",
                    severity="error"
                ))
                return ValidationResult(violations=violations)
            min_val = context[self.min_value]
        else:
            min_val = self.min_value

        if isinstance(self.max_value, str):
            if self.max_value not in context:
                violations.append(ConstraintViolation(
                    constraint_type="constraint_evaluation",
                    message=f"Max value parameter '{self.max_value}' not found in context",
                    severity="error"
                ))
                return ValidationResult(violations=violations)
            max_val = context[self.max_value]
        else:
            max_val = self.max_value

        # Validate range
        if dim_value < min_val or dim_value > max_val:
            violations.append(ConstraintViolation(
                constraint_type="range",
                message=f"{self.interface_name}.{dim_desc} must be in [{min_val}, {max_val}]",
                expected=f"[{min_val}, {max_val}]",
                actual=dim_value,
                severity="error"
            ))

        return ValidationResult(violations=violations)

    def describe(self) -> str:
        """Human-readable description."""
        dim_str = "total" if self.dim_index is None else f"dim[{self.dim_index}]"
        return f"{self.min_value} <= {self.interface_name}.{dim_str} <= {self.max_value}"


@dataclass(frozen=True)
class PowerOfTwoConstraint(DimensionConstraint):
    """Dimension must be a power of 2.

    Examples:
        PowerOfTwoConstraint("input", 0)  # input[0] must be 2^n
        PowerOfTwoConstraint("buffer", None)  # total size must be 2^n
    """

    interface_name: str
    dim_index: Optional[int]

    def validate_with_context(self, context: Dict[str, Any]) -> ValidationResult:
        """Validate power of 2 constraint."""
        violations = []

        # Get interface model
        if self.interface_name not in context:
            violations.append(ConstraintViolation(
                constraint_type="constraint_evaluation",
                message=f"Interface '{self.interface_name}' not found in context",
                severity="error"
            ))
            return ValidationResult(violations=violations)

        interface = context[self.interface_name]

        # Get dimension value
        if self.dim_index is None:
            from .types import prod
            dim_value = prod(interface.tensor_shape)
            dim_desc = "total_size"
        else:
            if self.dim_index >= len(interface.tensor_shape):
                violations.append(ConstraintViolation(
                    constraint_type="constraint_evaluation",
                    message=f"Dimension index {self.dim_index} out of range for {self.interface_name}",
                    severity="error"
                ))
                return ValidationResult(violations=violations)
            dim_value = interface.tensor_shape[self.dim_index]
            dim_desc = f"dim[{self.dim_index}]"

        # Validate power of 2
        if dim_value <= 0 or (dim_value & (dim_value - 1)) != 0:
            violations.append(ConstraintViolation(
                constraint_type="power_of_two",
                message=f"{self.interface_name}.{dim_desc} must be a power of 2",
                expected="2^n",
                actual=dim_value,
                severity="error"
            ))

        return ValidationResult(violations=violations)

    def describe(self) -> str:
        """Human-readable description."""
        dim_str = "total" if self.dim_index is None else f"dim[{self.dim_index}]"
        return f"{self.interface_name}.{dim_str} == 2^n"


# ===========================================================================
# Cross-Interface Constraints
# ===========================================================================

@dataclass(frozen=True)
class EqualityConstraint(DimensionConstraint):
    """Two dimensions must be equal.

    Examples:
        EqualityConstraint("input", 0, "output", 0)  # input[0] == output[0]
        EqualityConstraint("matrix", 1, "vector", 0)  # matrix[1] == vector[0]
        EqualityConstraint("input", None, "output", None)  # total sizes equal
    """

    source_interface: str
    source_dim: Optional[int]
    target_interface: str
    target_dim: Optional[int]

    def validate_with_context(self, context: Dict[str, Any]) -> ValidationResult:
        """Validate equality constraint."""
        violations = []

        # Get source interface
        if self.source_interface not in context:
            violations.append(ConstraintViolation(
                constraint_type="constraint_evaluation",
                message=f"Source interface '{self.source_interface}' not found in context",
                severity="error"
            ))
            return ValidationResult(violations=violations)

        # Get target interface
        if self.target_interface not in context:
            violations.append(ConstraintViolation(
                constraint_type="constraint_evaluation",
                message=f"Target interface '{self.target_interface}' not found in context",
                severity="error"
            ))
            return ValidationResult(violations=violations)

        source_intf = context[self.source_interface]
        target_intf = context[self.target_interface]

        # Get source dimension value
        if self.source_dim is None:
            from .types import prod
            source_value = prod(source_intf.tensor_shape)
            source_desc = "total"
        else:
            if self.source_dim >= len(source_intf.tensor_shape):
                violations.append(ConstraintViolation(
                    constraint_type="constraint_evaluation",
                    message=f"Source dimension {self.source_dim} out of range",
                    severity="error"
                ))
                return ValidationResult(violations=violations)
            source_value = source_intf.tensor_shape[self.source_dim]
            source_desc = f"dim[{self.source_dim}]"

        # Get target dimension value
        if self.target_dim is None:
            from .types import prod
            target_value = prod(target_intf.tensor_shape)
            target_desc = "total"
        else:
            if self.target_dim >= len(target_intf.tensor_shape):
                violations.append(ConstraintViolation(
                    constraint_type="constraint_evaluation",
                    message=f"Target dimension {self.target_dim} out of range",
                    severity="error"
                ))
                return ValidationResult(violations=violations)
            target_value = target_intf.tensor_shape[self.target_dim]
            target_desc = f"dim[{self.target_dim}]"

        # Validate equality
        if source_value != target_value:
            violations.append(ConstraintViolation(
                constraint_type="dimension_equality",
                message=f"{self.source_interface}.{source_desc} must equal {self.target_interface}.{target_desc}",
                expected=source_value,
                actual=target_value,
                severity="error"
            ))

        return ValidationResult(violations=violations)

    def describe(self) -> str:
        """Human-readable description."""
        source_desc = "total" if self.source_dim is None else f"dim[{self.source_dim}]"
        target_desc = "total" if self.target_dim is None else f"dim[{self.target_dim}]"
        return f"{self.source_interface}.{source_desc} == {self.target_interface}.{target_desc}"


@dataclass(frozen=True)
class DivisibleByDimensionConstraint(DimensionConstraint):
    """Dimension must be divisible by another dimension.

    Examples:
        DivisibleByDimensionConstraint("output", 0, "input", 0)  # output[0] % input[0] == 0
        DivisibleByDimensionConstraint("block", None, "stream", None)  # block_total % stream_total == 0
    """

    interface_name: str
    dim_index: Optional[int]
    divisor_interface: str
    divisor_dim: Optional[int]

    def validate_with_context(self, context: Dict[str, Any]) -> ValidationResult:
        """Validate divisibility by dimension constraint."""
        violations = []

        # Get interface
        if self.interface_name not in context:
            violations.append(ConstraintViolation(
                constraint_type="constraint_evaluation",
                message=f"Interface '{self.interface_name}' not found in context",
                severity="error"
            ))
            return ValidationResult(violations=violations)

        # Get divisor interface
        if self.divisor_interface not in context:
            violations.append(ConstraintViolation(
                constraint_type="constraint_evaluation",
                message=f"Divisor interface '{self.divisor_interface}' not found in context",
                severity="error"
            ))
            return ValidationResult(violations=violations)

        interface = context[self.interface_name]
        divisor_intf = context[self.divisor_interface]

        # Get dimension value
        if self.dim_index is None:
            from .types import prod
            dim_value = prod(interface.tensor_shape)
            dim_desc = "total"
        else:
            if self.dim_index >= len(interface.tensor_shape):
                violations.append(ConstraintViolation(
                    constraint_type="constraint_evaluation",
                    message=f"Dimension {self.dim_index} out of range",
                    severity="error"
                ))
                return ValidationResult(violations=violations)
            dim_value = interface.tensor_shape[self.dim_index]
            dim_desc = f"dim[{self.dim_index}]"

        # Get divisor value
        if self.divisor_dim is None:
            from .types import prod
            divisor_value = prod(divisor_intf.tensor_shape)
            divisor_desc = "total"
        else:
            if self.divisor_dim >= len(divisor_intf.tensor_shape):
                violations.append(ConstraintViolation(
                    constraint_type="constraint_evaluation",
                    message=f"Divisor dimension {self.divisor_dim} out of range",
                    severity="error"
                ))
                return ValidationResult(violations=violations)
            divisor_value = divisor_intf.tensor_shape[self.divisor_dim]
            divisor_desc = f"dim[{self.divisor_dim}]"

        # Validate divisibility
        if divisor_value == 0:
            violations.append(ConstraintViolation(
                constraint_type="constraint_evaluation",
                message=f"Divisor {self.divisor_interface}.{divisor_desc} cannot be zero",
                severity="error"
            ))
        elif dim_value % divisor_value != 0:
            violations.append(ConstraintViolation(
                constraint_type="divisibility",
                message=f"{self.interface_name}.{dim_desc} must be divisible by {self.divisor_interface}.{divisor_desc}",
                expected=f"multiple of {divisor_value}",
                actual=dim_value,
                severity="error",
                details={"remainder": dim_value % divisor_value}
            ))

        return ValidationResult(violations=violations)

    def describe(self) -> str:
        """Human-readable description."""
        dim_desc = "total" if self.dim_index is None else f"dim[{self.dim_index}]"
        divisor_desc = "total" if self.divisor_dim is None else f"dim[{self.divisor_dim}]"
        return f"{self.interface_name}.{dim_desc} % {self.divisor_interface}.{divisor_desc} == 0"


@dataclass(frozen=True)
class ScaledEqualityConstraint(DimensionConstraint):
    """Dimension must equal another dimension * scale_factor.

    Examples:
        ScaledEqualityConstraint("output", 0, "input", 0, 2)  # output[0] == input[0] * 2
        ScaledEqualityConstraint("y", 1, "x", 1, "SCALE")  # y[1] == x[1] * SCALE
    """

    target_interface: str
    target_dim: Optional[int]
    source_interface: str
    source_dim: Optional[int]
    scale_factor: Union[int, float, str]

    def validate_with_context(self, context: Dict[str, Any]) -> ValidationResult:
        """Validate scaled equality constraint."""
        violations = []

        # Get source interface
        if self.source_interface not in context:
            violations.append(ConstraintViolation(
                constraint_type="constraint_evaluation",
                message=f"Source interface '{self.source_interface}' not found in context",
                severity="error"
            ))
            return ValidationResult(violations=violations)

        # Get target interface
        if self.target_interface not in context:
            violations.append(ConstraintViolation(
                constraint_type="constraint_evaluation",
                message=f"Target interface '{self.target_interface}' not found in context",
                severity="error"
            ))
            return ValidationResult(violations=violations)

        source_intf = context[self.source_interface]
        target_intf = context[self.target_interface]

        # Get source value
        if self.source_dim is None:
            from .types import prod
            source_value = prod(source_intf.tensor_shape)
            source_desc = "total"
        else:
            if self.source_dim >= len(source_intf.tensor_shape):
                violations.append(ConstraintViolation(
                    constraint_type="constraint_evaluation",
                    message=f"Source dimension {self.source_dim} out of range",
                    severity="error"
                ))
                return ValidationResult(violations=violations)
            source_value = source_intf.tensor_shape[self.source_dim]
            source_desc = f"dim[{self.source_dim}]"

        # Get target value
        if self.target_dim is None:
            from .types import prod
            target_value = prod(target_intf.tensor_shape)
            target_desc = "total"
        else:
            if self.target_dim >= len(target_intf.tensor_shape):
                violations.append(ConstraintViolation(
                    constraint_type="constraint_evaluation",
                    message=f"Target dimension {self.target_dim} out of range",
                    severity="error"
                ))
                return ValidationResult(violations=violations)
            target_value = target_intf.tensor_shape[self.target_dim]
            target_desc = f"dim[{self.target_dim}]"

        # Get scale factor
        if isinstance(self.scale_factor, str):
            if self.scale_factor not in context:
                violations.append(ConstraintViolation(
                    constraint_type="constraint_evaluation",
                    message=f"Scale factor parameter '{self.scale_factor}' not found in context",
                    severity="error"
                ))
                return ValidationResult(violations=violations)
            scale = context[self.scale_factor]
        else:
            scale = self.scale_factor

        # Validate scaled equality
        expected_value = source_value * scale
        if target_value != expected_value:
            violations.append(ConstraintViolation(
                constraint_type="scaled_equality",
                message=f"{self.target_interface}.{target_desc} must equal {self.source_interface}.{source_desc} * {scale}",
                expected=expected_value,
                actual=target_value,
                severity="error"
            ))

        return ValidationResult(violations=violations)

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
    "RangeConstraint",
    "PowerOfTwoConstraint",
    # Cross-interface constraints
    "EqualityConstraint",
    "DivisibleByDimensionConstraint",
    "ScaledEqualityConstraint",
]
