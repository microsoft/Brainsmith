############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Unified constraint system for dataflow modeling.

Constraints are pure predicates that validate properties across different
contexts (ONNX inference, kernel build-time). They work with the ValidationContext
protocol to provide context-agnostic validation.

Architecture:
    - Constraint(ABC): Base class with check(ctx) → Optional[str]
    - ValidationContext: Protocol for accessing tensor/interface properties
    - Concrete constraints: Datatype, shape, ONNX-specific validations

Example usage:
    # Define constraints once
    constraints = [
        DatatypeInteger(("input0", "input1")),
        ShapesEqual(("input0", "input1")),
        IsDynamic(("input0", "input1")),
    ]

    # Apply on ONNX (inference-time)
    onnx_ctx = OnnxValidationContext(node, model)
    for c in constraints:
        if c.check(onnx_ctx):
            return False  # Constraint violated

    # Apply on kernel (build-time)
    kernel_ctx = KernelValidationContext(kernel_model, get_nodeattr)
    for c in constraints:
        error = c.check(kernel_ctx)
        if error:
            raise ValueError(error)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

from qonnx.core.datatype import DataType

from .validation import ValidationContext, ShapeHierarchy

logger = logging.getLogger(__name__)


# =============================================================================
# Base Constraint
# =============================================================================

@dataclass(frozen=True)
class Constraint(ABC):
    """A validation rule that can be checked in any context.

    Constraints are pure predicates - they describe what must be true.
    They can be checked on ONNX nodes (inference) or kernel models (build).

    Subclasses must implement:
    - check(ctx: ValidationContext) → Optional[str]
    - describe() → str
    """

    @abstractmethod
    def check(self, ctx: ValidationContext) -> Optional[str]:
        """Check constraint in given context.

        Args:
            ctx: ValidationContext (ONNX or Kernel)

        Returns:
            None if satisfied, error message string if violated
        """
        pass

    @abstractmethod
    def describe(self) -> str:
        """Human-readable description of constraint."""
        pass


# =============================================================================
# Datatype Constraints
# =============================================================================

@dataclass(frozen=True)
class DatatypeInteger(Constraint):
    """Specified interfaces must have integer datatypes.

    Example:
        DatatypeInteger(("input0", "input1"))
    """

    interfaces: tuple[str, ...]

    def check(self, ctx: ValidationContext) -> Optional[str]:
        """Validate all interfaces are integer types."""
        for name in self.interfaces:
            try:
                dt = ctx.get_datatype(name)
            except KeyError as e:
                return f"Interface '{name}' not found: {e}"

            if not dt.is_integer():
                return f"Interface '{name}' has non-integer datatype {dt.name}"

        return None

    def describe(self) -> str:
        return f"{self.interfaces} must be integer datatypes"


@dataclass(frozen=True)
class DatatypeFloat(Constraint):
    """Specified interfaces must have floating-point datatypes.

    Example:
        DatatypeFloat(("input0", "input1"))
    """

    interfaces: tuple[str, ...]

    def check(self, ctx: ValidationContext) -> Optional[str]:
        """Validate all interfaces are float types."""
        for name in self.interfaces:
            try:
                dt = ctx.get_datatype(name)
            except KeyError as e:
                return f"Interface '{name}' not found: {e}"

            canonical = dt.get_canonical_name()
            if not canonical.startswith("FLOAT"):
                return f"Interface '{name}' has non-float datatype {dt.name}"

        return None

    def describe(self) -> str:
        return f"{self.interfaces} must be float datatypes"


@dataclass(frozen=True)
class DatatypeInRange(Constraint):
    """Interface datatype must match base type and bit width range.

    Examples:
        DatatypeInRange("input", "INT", 4, 8)    # INT4-INT8
        DatatypeInRange("output", "UINT", 8, 16) # UINT8, UINT16
        DatatypeInRange("data", "ANY", 8, 32)    # Any type, 8-32 bits
    """

    interface: str
    base_type: str  # "INT", "UINT", "FLOAT", "FIXED", "BIPOLAR", "TERNARY", "BINARY", "ANY"
    min_bits: int
    max_bits: int

    def __post_init__(self):
        """Validate constraint parameters."""
        if self.min_bits <= 0:
            raise ValueError(f"min_bits must be positive, got {self.min_bits}")
        if self.max_bits < self.min_bits:
            raise ValueError(f"max_bits ({self.max_bits}) must be >= min_bits ({self.min_bits})")

        valid_types = ["INT", "UINT", "FLOAT", "FIXED", "BIPOLAR", "TERNARY", "BINARY", "ANY"]
        if self.base_type not in valid_types:
            raise ValueError(f"Invalid base_type '{self.base_type}'. Must be one of {valid_types}")

    def check(self, ctx: ValidationContext) -> Optional[str]:
        """Validate datatype matches base type and bit width range."""
        try:
            dt = ctx.get_datatype(self.interface)
        except KeyError as e:
            return f"Interface '{self.interface}' not found: {e}"

        bitwidth = dt.bitwidth()
        canonical = dt.get_canonical_name()

        # Check bitwidth range
        if not (self.min_bits <= bitwidth <= self.max_bits):
            return (
                f"{self.interface} bitwidth {bitwidth} not in range "
                f"[{self.min_bits}, {self.max_bits}]"
            )

        # Special case: ANY matches any type (only bitwidth matters)
        if self.base_type == "ANY":
            return None

        # Type validation
        type_validators = {
            "INT": lambda: canonical.startswith("INT") and dt.signed(),
            "UINT": lambda: canonical.startswith("UINT") or canonical == "BINARY",
            "FLOAT": lambda: canonical.startswith("FLOAT"),
            "FIXED": lambda: canonical.startswith("FIXED<"),
            "BIPOLAR": lambda: canonical == "BIPOLAR",
            "TERNARY": lambda: canonical == "TERNARY",
            "BINARY": lambda: canonical == "BINARY",
        }

        validator = type_validators.get(self.base_type)
        if validator and not validator():
            return f"{self.interface} type {canonical} does not match {self.base_type}"

        return None

    def describe(self) -> str:
        return f"{self.interface} ∈ {self.base_type}[{self.min_bits}..{self.max_bits}]"


@dataclass(frozen=True)
class DatatypesEqual(Constraint):
    """All specified interfaces must have identical datatypes.

    Example:
        DatatypesEqual(("input0", "input1", "output"))
    """

    interfaces: tuple[str, ...]

    def check(self, ctx: ValidationContext) -> Optional[str]:
        """Validate all interfaces have identical datatypes."""
        if len(self.interfaces) < 2:
            return "DatatypesEqual requires at least 2 interfaces"

        # Collect all datatypes
        datatypes = {}
        for name in self.interfaces:
            try:
                datatypes[name] = ctx.get_datatype(name)
            except KeyError as e:
                return f"Interface '{name}' not found: {e}"

        # Check all match the first
        first_name = self.interfaces[0]
        first_dt = datatypes[first_name]

        for name in self.interfaces[1:]:
            if datatypes[name] != first_dt:
                return (
                    f"Datatype mismatch: '{first_name}' has {first_dt.name}, "
                    f"but '{name}' has {datatypes[name].name}"
                )

        return None

    def describe(self) -> str:
        return f"{self.interfaces} must have equal datatypes"


# =============================================================================
# Shape Constraints
# =============================================================================

@dataclass(frozen=True)
class ShapesEqual(Constraint):
    """Specified interfaces must have identical shapes.

    Examples:
        # Full tensor shape equality
        ShapesEqual(("input0", "input1"), hierarchy=ShapeHierarchy.TENSOR)

        # Stream shape equality
        ShapesEqual(("input0", "input1"), hierarchy=ShapeHierarchy.STREAM)

        # Partial shape equality (spatial dims only)
        ShapesEqual(("input0", "input1"), dim_slice=slice(0, -1))
    """

    interfaces: tuple[str, ...]
    hierarchy: ShapeHierarchy = ShapeHierarchy.TENSOR
    dim_slice: Optional[slice] = None

    def check(self, ctx: ValidationContext) -> Optional[str]:
        """Validate all interfaces have identical shapes."""
        if len(self.interfaces) < 2:
            return "ShapesEqual requires at least 2 interfaces"

        # Collect shapes
        shapes = {}
        for name in self.interfaces:
            try:
                shape = ctx.get_shape(name, self.hierarchy)
                # Apply slice if specified
                if self.dim_slice:
                    shape = shape[self.dim_slice]
                shapes[name] = shape
            except KeyError as e:
                return f"Interface '{name}' not found: {e}"

        # Check all match the first
        first_name = self.interfaces[0]
        first_shape = shapes[first_name]

        for name in self.interfaces[1:]:
            if shapes[name] != first_shape:
                return (
                    f"Shape mismatch ({self.hierarchy.value}): "
                    f"'{first_name}' has {first_shape}, but '{name}' has {shapes[name]}"
                )

        return None

    def describe(self) -> str:
        dim_desc = f"[{self.dim_slice}]" if self.dim_slice else ""
        return f"{self.interfaces}{dim_desc} must have equal shapes ({self.hierarchy.value})"


@dataclass(frozen=True)
class DimensionDivisible(Constraint):
    """Interface dimension must be divisible by value.

    Examples:
        DimensionDivisible("input", 1, 8)       # stream[1] % 8 == 0
        DimensionDivisible("input", -1, "PE")   # stream[-1] % PE (param)
    """

    interface: str
    dim_index: int
    divisor: Union[int, str]  # int literal or param name
    hierarchy: ShapeHierarchy = ShapeHierarchy.STREAM

    def check(self, ctx: ValidationContext) -> Optional[str]:
        """Validate dimension is divisible by divisor."""
        try:
            shape = ctx.get_shape(self.interface, self.hierarchy)
        except KeyError as e:
            return f"Interface '{self.interface}' not found: {e}"

        # Handle negative indices
        dim_idx = self.dim_index if self.dim_index >= 0 else len(shape) + self.dim_index

        if not (0 <= dim_idx < len(shape)):
            return f"{self.interface} dimension {self.dim_index} out of range for shape {shape}"

        dim_value = shape[dim_idx]

        # Resolve divisor (literal or param reference)
        if isinstance(self.divisor, str):
            try:
                divisor = ctx.get_param(self.divisor)
            except (RuntimeError, KeyError) as e:
                # ONNX context doesn't support params - skip check (graceful degradation)
                return None
        else:
            divisor = self.divisor

        if dim_value % divisor != 0:
            return (
                f"{self.interface}.{self.hierarchy.value}[{self.dim_index}] = {dim_value} "
                f"not divisible by {divisor}"
            )

        return None

    def describe(self) -> str:
        return f"{self.interface}.{self.hierarchy.value}[{self.dim_index}] % {self.divisor} == 0"


@dataclass(frozen=True)
class DimensionInRange(Constraint):
    """Interface dimension must be within range [min, max].

    Examples:
        DimensionInRange("input", 0, 1, 1024)       # tensor[0] ∈ [1, 1024]
        DimensionInRange("input", -1, "MIN", "MAX") # tensor[-1] ∈ [MIN, MAX] (params)
    """

    interface: str
    dim_index: int
    min_value: Union[int, str]
    max_value: Union[int, str]
    hierarchy: ShapeHierarchy = ShapeHierarchy.STREAM

    def check(self, ctx: ValidationContext) -> Optional[str]:
        """Validate dimension is within range."""
        try:
            shape = ctx.get_shape(self.interface, self.hierarchy)
        except KeyError as e:
            return f"Interface '{self.interface}' not found: {e}"

        # Handle negative indices
        dim_idx = self.dim_index if self.dim_index >= 0 else len(shape) + self.dim_index

        if not (0 <= dim_idx < len(shape)):
            return f"{self.interface} dimension {self.dim_index} out of range for shape {shape}"

        dim_value = shape[dim_idx]

        # Resolve min/max (literals or param references)
        try:
            min_val = ctx.get_param(self.min_value) if isinstance(self.min_value, str) else self.min_value
            max_val = ctx.get_param(self.max_value) if isinstance(self.max_value, str) else self.max_value
        except (RuntimeError, KeyError):
            # ONNX context doesn't support params - skip check
            return None

        if not (min_val <= dim_value <= max_val):
            return (
                f"{self.interface}.{self.hierarchy.value}[{self.dim_index}] = {dim_value} "
                f"not in range [{min_val}, {max_val}]"
            )

        return None

    def describe(self) -> str:
        return (
            f"{self.interface}.{self.hierarchy.value}[{self.dim_index}] "
            f"∈ [{self.min_value}, {self.max_value}]"
        )


@dataclass(frozen=True)
class DimensionEquals(Constraint):
    """Interface dimension must equal specific value.

    Examples:
        DimensionEquals("input", 0, 1)      # tensor[0] == 1 (batch size)
        DimensionEquals("input", -1, "PE")  # tensor[-1] == PE (param)
    """

    interface: str
    dim_index: int
    value: Union[int, str]
    hierarchy: ShapeHierarchy = ShapeHierarchy.STREAM

    def check(self, ctx: ValidationContext) -> Optional[str]:
        """Validate dimension equals value."""
        try:
            shape = ctx.get_shape(self.interface, self.hierarchy)
        except KeyError as e:
            return f"Interface '{self.interface}' not found: {e}"

        # Handle negative indices
        dim_idx = self.dim_index if self.dim_index >= 0 else len(shape) + self.dim_index

        if not (0 <= dim_idx < len(shape)):
            return f"{self.interface} dimension {self.dim_index} out of range for shape {shape}"

        dim_value = shape[dim_idx]

        # Resolve value (literal or param reference)
        try:
            expected = ctx.get_param(self.value) if isinstance(self.value, str) else self.value
        except (RuntimeError, KeyError):
            # ONNX context doesn't support params - skip check
            return None

        if dim_value != expected:
            return (
                f"{self.interface}.{self.hierarchy.value}[{self.dim_index}] = {dim_value}, "
                f"expected {expected}"
            )

        return None

    def describe(self) -> str:
        return f"{self.interface}.{self.hierarchy.value}[{self.dim_index}] == {self.value}"


# =============================================================================
# ONNX-Specific Constraints (gracefully degrade on kernel context)
# =============================================================================

@dataclass(frozen=True)
class IsDynamic(Constraint):
    """Interfaces must be dynamic (no initializer).

    Only meaningful for ONNX contexts. Always passes on kernel contexts.

    Example:
        IsDynamic(("input0", "input1"))
    """

    interfaces: tuple[str, ...]

    def check(self, ctx: ValidationContext) -> Optional[str]:
        """Validate all interfaces are dynamic."""
        for name in self.interfaces:
            if not ctx.is_dynamic(name):
                return f"Interface '{name}' must be dynamic (has initializer)"
        return None

    def describe(self) -> str:
        return f"{self.interfaces} must be dynamic (no initializers)"


@dataclass(frozen=True)
class IsStatic(Constraint):
    """Interfaces must be static (have initializer).

    Only meaningful for ONNX contexts. Always passes on kernel contexts.

    Example:
        IsStatic(("weight",))
    """

    interfaces: tuple[str, ...]

    def check(self, ctx: ValidationContext) -> Optional[str]:
        """Validate all interfaces are static."""
        for name in self.interfaces:
            if ctx.is_dynamic(name):
                return f"Interface '{name}' must be static (needs initializer)"
        return None

    def describe(self) -> str:
        return f"{self.interfaces} must be static (have initializers)"


@dataclass(frozen=True)
class HasLayout(Constraint):
    """Interface must have specified layout.

    Only meaningful for ONNX contexts. Always passes on kernel contexts.

    Example:
        HasLayout("input", "NHWC")
    """

    interface: str
    layout: str  # "NHWC" or "NCHW"

    def check(self, ctx: ValidationContext) -> Optional[str]:
        """Validate interface has expected layout."""
        actual_layout = ctx.get_layout(self.interface)

        if actual_layout is None:
            # Kernel context - layout not tracked (graceful degradation)
            return None

        # Import here to avoid circular dependency
        import qonnx.core.data_layout as DataLayout
        expected_layout = getattr(DataLayout, self.layout, None)

        if actual_layout != expected_layout:
            return f"{self.interface} has layout {actual_layout}, expected {self.layout}"

        return None

    def describe(self) -> str:
        return f"{self.interface} must have layout {self.layout}"


# =============================================================================
# Node Attribute Constraint
# =============================================================================

@dataclass(frozen=True)
class NodeAttributeEquals(Constraint):
    """Validate ONNX node attribute equals expected value(s).

    Only applicable during ONNX inference validation (gracefully skips in kernel context).
    Useful for validating ONNX node configuration before hardware conversion.

    Args:
        attribute_name: ONNX node attribute to check
        expected_values: Single value or list of acceptable values

    Example:
        # Single value
        NodeAttributeEquals("axis", -1)

        # Multiple acceptable values
        NodeAttributeEquals("axis", [-1, 3])

        # None is a valid expected value (attribute not set)
        NodeAttributeEquals("axis", [None, -1])
    """

    attribute_name: str
    expected_values: Any

    def __post_init__(self):
        """Normalize expected_values to list for consistent checking."""
        # Convert single value to list for uniform checking
        if not isinstance(self.expected_values, (list, tuple)):
            object.__setattr__(self, 'expected_values', [self.expected_values])

    def check(self, ctx: ValidationContext) -> Optional[str]:
        """Check if node attribute matches expected value(s).

        Gracefully skips in kernel context (returns None).
        """
        # Use sentinel to distinguish "attribute not found" from "attribute is None"
        _SENTINEL = object()

        try:
            actual_value = ctx.get_node_attribute(self.attribute_name, default=_SENTINEL)
        except RuntimeError:
            # Kernel context - node attributes not available
            # This constraint only applies during ONNX inference, so skip gracefully
            return None

        # If we got the sentinel, attribute doesn't exist
        if actual_value is _SENTINEL:
            # Check if None is an acceptable value
            if None in self.expected_values:
                return None
            return (f"Node attribute '{self.attribute_name}' not found, "
                   f"expected one of {self.expected_values}")

        # Check if actual value matches any expected value
        if actual_value not in self.expected_values:
            return (f"Node attribute '{self.attribute_name}' is {actual_value}, "
                   f"expected one of {self.expected_values}")

        return None

    def describe(self) -> str:
        if len(self.expected_values) == 1:
            return f"Node attribute '{self.attribute_name}' must equal {self.expected_values[0]}"
        return f"Node attribute '{self.attribute_name}' must be one of {self.expected_values}"


# =============================================================================
# Custom Constraint
# =============================================================================

@dataclass(frozen=True)
class Custom(Constraint):
    """Custom validation logic.

    The check function receives ValidationContext and returns Optional[str].

    Example:
        def check_matmul_compat(ctx):
            input_shape = ctx.get_shape("input")
            weight_shape = ctx.get_shape("weight")
            if input_shape[-1] != weight_shape[0]:
                return f"MatMul incompatible: {input_shape[-1]} vs {weight_shape[0]}"
            return None

        Custom(check_matmul_compat, "MatMul dimension compatibility")
    """

    check_fn: Callable[[ValidationContext], Optional[str]]
    description: str

    def check(self, ctx: ValidationContext) -> Optional[str]:
        """Call custom validation function."""
        try:
            return self.check_fn(ctx)
        except Exception as e:
            return f"Custom constraint '{self.description}' failed: {e}"

    def describe(self) -> str:
        return self.description


__all__ = [
    # Base class
    'Constraint',
    # Datatype constraints
    'DatatypeInteger',
    'DatatypeFloat',
    'DatatypeInRange',
    'DatatypesEqual',
    # Shape constraints
    'ShapesEqual',
    'DimensionDivisible',
    'DimensionInRange',
    'DimensionEquals',
    # ONNX-specific constraints
    'IsDynamic',
    'IsStatic',
    'HasLayout',
    'NodeAttributeEquals',
    # Custom constraint
    'Custom',
]
