############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Unified constraint system for dataflow modeling.

Constraints are pure predicates that validate properties across different
kernel construction phases. They use duck typing to work with any validation
context that provides the required methods (get_datatype, get_shape, etc.).

Architecture:
    - Constraint(ABC): Base class with check(ctx) → Optional[str]
    - DesignSpaceValidationContext: Structural validation (tensor/block shapes, datatypes)
    - ConfigurationValidationContext: Parametric validation (stream shapes, params)
    - Concrete constraints: Datatype, shape, ONNX-specific validations

Example usage:
    # Define constraints once
    constraints = [
        DatatypeInteger(("input0", "input1")),
        ShapesEqual(("input0", "input1")),
        IsDynamic(("input0", "input1")),
    ]

    # Apply during design space build (structural constraints)
    ctx = DesignSpaceValidationContext(inputs, outputs, internal_datatypes, param_getter)
    for c in constraints:
        error = c.check(ctx)
        if error:
            raise ValueError(error)

    # Apply during configuration (optimization constraints)
    ctx = ConfigurationValidationContext(configured_model, params)
    for c in constraints:
        error = c.check(ctx)
        if error:
            raise ValueError(error)
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union, Protocol, runtime_checkable

from qonnx.core.datatype import DataType

from .validation import ShapeHierarchy

logger = logging.getLogger(__name__)


# =============================================================================
# Base Constraint
# =============================================================================

@runtime_checkable
class Constraint(Protocol):
    """Validation rule for kernel constraints.

    Pure predicate that validates kernel properties during construction.
    Uses duck typing to work with any validation context providing required methods.

    Required methods:
    - check(ctx) → Optional[str]
    - describe() → str
    """

    def check(self, ctx) -> Optional[str]:
        """Check constraint in given context.

        Args:
            ctx: Validation context (DesignSpaceValidationContext or ConfigurationValidationContext)

        Returns:
            None if satisfied, error message string if violated
        """
        ...

    def describe(self) -> str:
        """Human-readable description of constraint."""
        ...

    @property
    def evaluation_phase(self) -> str:
        """When to evaluate this constraint during kernel construction.

        Returns:
            'structural' - Evaluated once during design space construction (Phase 1)
                          Constraints that determine backend compatibility
                          (tensor shapes, block shapes, datatypes, etc.)

            'optimization' - Evaluated per-configuration during configure() (Phase 2)
                            Constraints that bound optimization space
                            (stream shapes, parallelization parameters, etc.)

        Default implementation uses heuristic:
        - Constraints with hierarchy == STREAM are optimization constraints
        - All other constraints are structural

        Subclasses can override this property for explicit classification.

        Examples:
            DatatypeInteger: 'structural' (no hierarchy, datatype determines compatibility)
            ShapesEqual(hierarchy=TENSOR): 'structural' (tensor shape determines compatibility)
            ShapesEqual(hierarchy=BLOCK): 'structural' (block shape determines compatibility)
            ShapesEqual(hierarchy=STREAM): 'optimization' (stream shape bounds optimization)
            DimensionDivisible(hierarchy=STREAM): 'optimization' (stream dim bounds optimization)
        """
        # Heuristic: stream-level shape constraints are optimization constraints
        if hasattr(self, 'hierarchy'):
            if self.hierarchy == ShapeHierarchy.STREAM:
                return 'optimization'

        # All other constraints are structural by default
        # (datatype, layout, node attribute, custom, etc.)
        return 'structural'


# =============================================================================
# Datatype Constraints
# =============================================================================

@dataclass(frozen=True)
class DatatypeInteger:
    """Specified interfaces must have integer datatypes.

    Example:
        DatatypeInteger(("input0", "input1"))
    """

    interfaces: tuple[str, ...]

    def check(self, ctx) -> Optional[str]:
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

    @property
    def evaluation_phase(self) -> str:
        return 'structural'


@dataclass(frozen=True)
class DatatypeFloat:
    """Specified interfaces must have floating-point datatypes.

    Example:
        DatatypeFloat(("input0", "input1"))
    """

    interfaces: tuple[str, ...]

    def check(self, ctx) -> Optional[str]:
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

    @property
    def evaluation_phase(self) -> str:
        return 'structural'


@dataclass(frozen=True)
class DatatypeInRange:
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

    def check(self, ctx) -> Optional[str]:
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

    @property
    def evaluation_phase(self) -> str:
        return 'structural'


@dataclass(frozen=True)
class DatatypesEqual:
    """All specified interfaces must have identical datatypes.

    Example:
        DatatypesEqual(("input0", "input1", "output"))
    """

    interfaces: tuple[str, ...]

    def check(self, ctx) -> Optional[str]:
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

    @property
    def evaluation_phase(self) -> str:
        return 'structural'


# =============================================================================
# Shape Constraints
# =============================================================================

@dataclass(frozen=True)
class ShapesEqual:
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

    def check(self, ctx) -> Optional[str]:
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

    @property
    def evaluation_phase(self) -> str:
        return 'optimization' if self.hierarchy == ShapeHierarchy.STREAM else 'structural'


@dataclass(frozen=True)
class DimensionDivisible:
    """Interface dimension must be divisible by value.

    Examples:
        DimensionDivisible("input", 1, 8)       # stream[1] % 8 == 0
        DimensionDivisible("input", -1, "PE")   # stream[-1] % PE (param)
    """

    interface: str
    dim_index: int
    divisor: Union[int, str]  # int literal or param name
    hierarchy: ShapeHierarchy = ShapeHierarchy.STREAM

    def check(self, ctx) -> Optional[str]:
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

    @property
    def evaluation_phase(self) -> str:
        return 'optimization' if self.hierarchy == ShapeHierarchy.STREAM else 'structural'


@dataclass(frozen=True)
class DimensionInRange:
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

    def check(self, ctx) -> Optional[str]:
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

    @property
    def evaluation_phase(self) -> str:
        return 'optimization' if self.hierarchy == ShapeHierarchy.STREAM else 'structural'


@dataclass(frozen=True)
class DimensionEquals:
    """Interface dimension must equal specific value.

    Examples:
        DimensionEquals("input", 0, 1)      # tensor[0] == 1 (batch size)
        DimensionEquals("input", -1, "PE")  # tensor[-1] == PE (param)
    """

    interface: str
    dim_index: int
    value: Union[int, str]
    hierarchy: ShapeHierarchy = ShapeHierarchy.STREAM

    def check(self, ctx) -> Optional[str]:
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

    @property
    def evaluation_phase(self) -> str:
        return 'optimization' if self.hierarchy == ShapeHierarchy.STREAM else 'structural'


@dataclass(frozen=True)
class TensorDimMatches:
    """Validate specific dimension of tensor matches one of allowed values.

    Checks shape[dim_index] against specifications at TENSOR hierarchy level.
    Use this for validating tensor structure (e.g., channel dimension must
    match specific value or reference).

    For BLOCK/STREAM hierarchy validation, use DimensionEquals or CustomConstraint.

    Examples:
        # Channel dimension must be 1 or match input channels
        TensorDimMatches("parameters", -1, [1, ("input", -1)])

        # Batch dimension must be 1, 2, or 4
        TensorDimMatches("input", 0, [1, 2, 4])

    Args:
        interface: Name of interface to validate
        dim_index: Which dimension to check (supports negative indexing)
        allowed: List of allowed values (int literals or (interface, dim_index) tuples)
    """

    interface: str
    dim_index: int
    allowed: tuple[Union[int, str, tuple[str, int]], ...]

    def __init__(self, interface: str, dim_index: int, allowed: list):
        """Initialize with list that gets converted to tuple for frozen dataclass."""
        object.__setattr__(self, 'interface', interface)
        object.__setattr__(self, 'dim_index', dim_index)
        object.__setattr__(self, 'allowed', tuple(allowed))

    def check(self, ctx) -> Optional[str]:
        """Validate dimension matches one of allowed values."""
        try:
            shape = ctx.get_shape(self.interface, ShapeHierarchy.TENSOR)
        except KeyError as e:
            return f"Interface '{self.interface}' not found: {e}"

        # Handle negative indices
        dim_idx = self.dim_index if self.dim_index >= 0 else len(shape) + self.dim_index

        if not (0 <= dim_idx < len(shape)):
            return f"{self.interface} dimension {self.dim_index} out of range for shape {shape}"

        actual_value = shape[dim_idx]

        # Check against all allowed values
        for spec in self.allowed:
            if isinstance(spec, int):
                if actual_value == spec:
                    return None  # Match found
            elif isinstance(spec, str):
                # Parameter reference
                try:
                    expected = ctx.get_param(spec)
                    if actual_value == expected:
                        return None
                except (RuntimeError, KeyError):
                    continue  # Skip this spec if param not available
            elif isinstance(spec, tuple):
                # (interface, dim_index) reference
                ref_interface, ref_dim_idx = spec
                try:
                    ref_shape = ctx.get_shape(ref_interface, ShapeHierarchy.TENSOR)
                    ref_dim_idx_resolved = ref_dim_idx if ref_dim_idx >= 0 else len(ref_shape) + ref_dim_idx
                    if 0 <= ref_dim_idx_resolved < len(ref_shape):
                        expected = ref_shape[ref_dim_idx_resolved]
                        if actual_value == expected:
                            return None
                except KeyError:
                    continue  # Skip this spec if interface not found
            else:
                return f"Invalid spec type {type(spec).__name__} in allowed list"

        # No match found
        allowed_desc = ", ".join(str(s) for s in self.allowed)
        return (
            f"{self.interface}.tensor[{self.dim_index}] = {actual_value}, "
            f"expected one of: {allowed_desc}"
        )

    def describe(self) -> str:
        allowed_desc = ", ".join(str(s) for s in self.allowed)
        return f"{self.interface}.tensor[{self.dim_index}] ∈ {{{allowed_desc}}}"

    @property
    def evaluation_phase(self) -> str:
        return 'structural'


@dataclass(frozen=True)
class TensorSizeMatches:
    """Validate total element count matches one of allowed values.

    Checks np.prod(shape) against specifications at TENSOR hierarchy level.
    Use this for validating broadcast compatibility (e.g., parameters can be
    scalar or per-channel).

    Does NOT validate specific shape structure - any shape with matching
    element count is valid. For dimension-specific validation, use TensorDimMatches.

    For BLOCK/STREAM hierarchy validation, use CustomConstraint.

    Examples:
        # Parameters must be scalar (1) or per-channel
        TensorSizeMatches("parameters", [1, ("input", -1)])

        # Weight matrix must have 64*64 or 128*128 elements
        TensorSizeMatches("weights", [4096, 16384])

    Args:
        interface: Name of interface to validate
        allowed: List of allowed sizes (int literals, param names, or (interface, dim_index) tuples)
    """

    interface: str
    allowed: tuple[Union[int, str, tuple[str, int]], ...]

    def __init__(self, interface: str, allowed: list):
        """Initialize with list that gets converted to tuple for frozen dataclass."""
        object.__setattr__(self, 'interface', interface)
        object.__setattr__(self, 'allowed', tuple(allowed))

    def check(self, ctx) -> Optional[str]:
        """Validate total element count matches one of allowed values."""
        import numpy as np

        try:
            shape = ctx.get_shape(self.interface, ShapeHierarchy.TENSOR)
        except KeyError as e:
            return f"Interface '{self.interface}' not found: {e}"

        actual_size = int(np.prod(shape))

        # Check against all allowed values
        for spec in self.allowed:
            if isinstance(spec, int):
                if actual_size == spec:
                    return None  # Match found
            elif isinstance(spec, str):
                # Parameter reference
                try:
                    expected = ctx.get_param(spec)
                    if actual_size == expected:
                        return None
                except (RuntimeError, KeyError):
                    continue  # Skip this spec if param not available
            elif isinstance(spec, tuple):
                # (interface, dim_index) reference
                ref_interface, ref_dim_idx = spec
                try:
                    ref_shape = ctx.get_shape(ref_interface, ShapeHierarchy.TENSOR)
                    ref_dim_idx_resolved = ref_dim_idx if ref_dim_idx >= 0 else len(ref_shape) + ref_dim_idx
                    if 0 <= ref_dim_idx_resolved < len(ref_shape):
                        expected = ref_shape[ref_dim_idx_resolved]
                        if actual_size == expected:
                            return None
                except KeyError:
                    continue  # Skip this spec if interface not found
            else:
                return f"Invalid spec type {type(spec).__name__} in allowed list"

        # No match found
        allowed_desc = ", ".join(str(s) for s in self.allowed)
        return (
            f"{self.interface} tensor size {actual_size} (shape {shape}), "
            f"expected one of: {allowed_desc}"
        )

    def describe(self) -> str:
        allowed_desc = ", ".join(str(s) for s in self.allowed)
        return f"np.prod({self.interface}.tensor) ∈ {{{allowed_desc}}}"

    @property
    def evaluation_phase(self) -> str:
        return 'structural'


# =============================================================================
# ONNX-Specific Constraints (gracefully degrade on kernel context)
# =============================================================================

@dataclass(frozen=True)
class IsDynamic:
    """Interfaces must be dynamic (no initializer).

    Only meaningful for ONNX contexts. Always passes on kernel contexts.

    Examples:
        IsDynamic("input")                # Single interface (ergonomic)
        IsDynamic(("input0", "input1"))   # Multiple interfaces
    """

    interfaces: tuple[str, ...]

    def __init__(self, interfaces: Union[str, tuple[str, ...]]):
        """Initialize with automatic normalization of string to tuple.

        Args:
            interfaces: Single interface name (string) or tuple of interface names
        """
        if isinstance(interfaces, str):
            interfaces = (interfaces,)
        object.__setattr__(self, 'interfaces', interfaces)

    def check(self, ctx) -> Optional[str]:
        """Validate all interfaces are dynamic."""
        for name in self.interfaces:
            if not ctx.is_dynamic(name):
                return f"Interface '{name}' must be dynamic (has initializer)"
        return None

    def describe(self) -> str:
        return f"{self.interfaces} must be dynamic (no initializers)"

    @property
    def evaluation_phase(self) -> str:
        return 'structural'


@dataclass(frozen=True)
class IsStatic:
    """Interfaces must be static (have initializer).

    Only meaningful for ONNX contexts. Always passes on kernel contexts.

    Examples:
        IsStatic("weights")               # Single interface (ergonomic)
        IsStatic(("weights", "biases"))   # Multiple interfaces
    """

    interfaces: tuple[str, ...]

    def __init__(self, interfaces: Union[str, tuple[str, ...]]):
        """Initialize with automatic normalization of string to tuple.

        Args:
            interfaces: Single interface name (string) or tuple of interface names
        """
        if isinstance(interfaces, str):
            interfaces = (interfaces,)
        object.__setattr__(self, 'interfaces', interfaces)

    def check(self, ctx) -> Optional[str]:
        """Validate all interfaces are static."""
        for name in self.interfaces:
            if ctx.is_dynamic(name):
                return f"Interface '{name}' must be static (needs initializer)"
        return None

    def describe(self) -> str:
        return f"{self.interfaces} must be static (have initializers)"

    @property
    def evaluation_phase(self) -> str:
        return 'structural'


@dataclass(frozen=True)
class HasLayout:
    """Interface must have specified layout.

    Only meaningful for ONNX contexts. Always passes on kernel contexts.

    Example:
        HasLayout("input", "NHWC")
    """

    interface: str
    layout: str  # "NHWC" or "NCHW"

    def check(self, ctx) -> Optional[str]:
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

    @property
    def evaluation_phase(self) -> str:
        return 'structural'


# =============================================================================
# Node Attribute Constraint
# =============================================================================

@dataclass(frozen=True)
class NodeAttributeEquals:
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

    def check(self, ctx) -> Optional[str]:
        """Check if node attribute matches expected value(s).

        Gracefully skips in kernel context (returns None).
        """
        _SENTINEL = object()  # Distinguishes "not found" from "value is None"

        try:
            actual_value = ctx.get_param(self.attribute_name)
        except (RuntimeError, KeyError):
            # RuntimeError: no param_getter (kernel context)
            # KeyError: param not found in ONNX context
            actual_value = _SENTINEL

        if actual_value is _SENTINEL:
            if None in self.expected_values:
                return None  # None is acceptable
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

    @property
    def evaluation_phase(self) -> str:
        return 'structural'


@dataclass(frozen=True)
class AttrCompare:
    """Validate ONNX node attribute against a comparison operator.

    Only applicable during ONNX inference validation (gracefully skips in kernel context).
    Supports standard comparison operators for numeric attributes.

    Args:
        attribute_name: ONNX node attribute to check
        operator: Comparison operator ('==', '!=', '<', '<=', '>', '>=')
        value: Value to compare against (int, float, or param name string)

    Examples:
        # Epsilon must be positive
        AttrCompare("epsilon", ">", 0)

        # Axis must be -1
        AttrCompare("axis", "==", -1)

        # Temperature must be at most 1.0
        AttrCompare("temperature", "<=", 1.0)

        # Compare against parameter
        AttrCompare("channels", ">=", "MIN_CHANNELS")

        # Not equal check
        AttrCompare("mode", "!=", 0)
    """

    attribute_name: str
    operator: str  # '==', '!=', '<', '<=', '>', '>='
    value: Union[int, float, str]  # literal or param name

    # Valid operators
    _OPERATORS = {
        '==': lambda a, b: a == b,
        '!=': lambda a, b: a != b,
        '<': lambda a, b: a < b,
        '<=': lambda a, b: a <= b,
        '>': lambda a, b: a > b,
        '>=': lambda a, b: a >= b,
    }

    def __post_init__(self):
        """Validate operator."""
        if self.operator not in self._OPERATORS:
            raise ValueError(
                f"Invalid operator '{self.operator}'. "
                f"Must be one of {list(self._OPERATORS.keys())}"
            )

    def check(self, ctx) -> Optional[str]:
        """Check if node attribute satisfies comparison.

        Gracefully skips in kernel context (returns None).
        """
        _SENTINEL = object()

        try:
            actual_value = ctx.get_param(self.attribute_name)
        except (RuntimeError, KeyError):
            # RuntimeError: no param_getter (kernel context)
            # KeyError: param not found in ONNX context
            actual_value = _SENTINEL

        if actual_value is _SENTINEL:
            return f"Node attribute '{self.attribute_name}' not found"

        # Resolve comparison value (literal or param reference)
        try:
            compare_value = (
                ctx.get_param(self.value)
                if isinstance(self.value, str)
                else self.value
            )
        except (RuntimeError, KeyError):
            # Param not available - skip check
            return None

        # Type compatibility check for numeric comparisons
        if not isinstance(actual_value, (int, float)) or not isinstance(compare_value, (int, float)):
            return (
                f"Node attribute '{self.attribute_name}' comparison requires numeric types, "
                f"got {type(actual_value).__name__} {self.operator} {type(compare_value).__name__}"
            )

        # Perform comparison
        comparison_fn = self._OPERATORS[self.operator]
        if not comparison_fn(actual_value, compare_value):
            return (
                f"Node attribute '{self.attribute_name}' = {actual_value} "
                f"fails constraint {self.operator} {compare_value}"
            )

        return None

    def describe(self) -> str:
        return f"Node attribute '{self.attribute_name}' {self.operator} {self.value}"

    @property
    def evaluation_phase(self) -> str:
        return 'structural'


# =============================================================================
# CustomConstraint Constraint
# =============================================================================

@dataclass(frozen=True)
class CustomConstraint:
    """CustomConstraint validation logic.

    The check function receives a validation context and returns Optional[str].

    Example:
        def check_matmul_compat(ctx):
            input_shape = ctx.get_shape("input")
            weight_shape = ctx.get_shape("weight")
            if input_shape[-1] != weight_shape[0]:
                return f"MatMul incompatible: {input_shape[-1]} vs {weight_shape[0]}"
            return None

        CustomConstraint(check_matmul_compat, "MatMul dimension compatibility")
    """

    check_fn: Callable[[Any], Optional[str]]
    description: str

    def check(self, ctx) -> Optional[str]:
        """Call custom validation function."""
        try:
            return self.check_fn(ctx)
        except Exception as e:
            return f"Custom constraint '{self.description}' failed: {e}"

    def describe(self) -> str:
        return self.description

    @property
    def evaluation_phase(self) -> str:
        return 'structural'


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
    'TensorDimMatches',
    'TensorSizeMatches',
    # ONNX-specific constraints
    'IsDynamic',
    'IsStatic',
    'HasLayout',
    'NodeAttributeEquals',
    'AttrCompare',
    # Custom constraint
    'CustomConstraint',
]
