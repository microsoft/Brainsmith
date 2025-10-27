############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Consolidated schema definitions for dataflow kernels.

Schemas define STRUCTURE, not STORAGE. They specify:
- Interface definitions (inputs, outputs)
- Tiling templates (block_tiling, stream_tiling)
- Validation rules (constraints, relationships)

Schemas are stateless - they never define what gets stored in nodeattrs.
Storage decisions are implementation details of operator classes.

Key classes:
- InterfaceSchema: Base class for input/output interfaces
- InputSchema: Schema for input interfaces
- OutputSchema: Schema for output interfaces
- KernelSchema: Complete kernel definition with inputs and outputs
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union, Dict, Any, Callable, Set, TYPE_CHECKING
from abc import ABC, abstractmethod

from qonnx.core.datatype import BaseDataType
from .types import FULL_DIM, FULL_SHAPE

if TYPE_CHECKING:
    from .constraints import Constraint
    from .builder import BuildContext

logger = logging.getLogger(__name__)

# Type aliases for better clarity
TilingSpec = Union[
    Sequence[Union[int, str, type(FULL_DIM), Callable]],  # List of DimSpecs
    type(FULL_SHAPE),                                      # Bare sentinel (rank-agnostic)
]


# ============================================================================
# DSE Dimension Type
# ============================================================================

@dataclass(frozen=True)
class DSEDimension:
    """Explorable dimension in design space.

    Represents resource allocation or implementation choices that can be
    explored during DSE (ram_style, res_type, mem_mode, etc.).

    Does NOT include tiling dimensions (PE, SIMD) - those are auto-extracted
    from stream_tiling templates with valid values computed from factoring.

    Attributes:
        name: Dimension name (e.g., "ram_style", "res_type")
        values: Valid values for this dimension
            - Set: Explicit values like {1, 2, 4, 8} or {"distributed", "block"}
            - Callable: Computed from BuildContext (for context-dependent values)
        default: Default value (None = auto-select smallest/first from values)

    Examples:
        >>> # Explicit integer values
        >>> DSEDimension("mem_depth", {128, 256, 512, 1024}, default=256)

        >>> # Explicit string values
        >>> DSEDimension("ram_style", {"distributed", "block"}, default="distributed")

        >>> # Auto-default (will use min for numeric, alphabetical first for string)
        >>> DSEDimension("res_type", {"lut", "dsp"})
    """
    name: str
    values: Union[
        Set[Union[int, str]],
        Callable[['BuildContext'], Set[Union[int, str]]]
    ]
    default: Optional[Union[int, str]] = None


# ============================================================================
# Helper Functions
# ============================================================================

def _extract_tiling_params(block_tiling: Optional[TilingSpec], stream_tiling: Optional[TilingSpec]) -> List[str]:
    """Extract unique string parameters from tiling specs."""
    params = set()
    for spec in (block_tiling, stream_tiling):
        if spec is None:
            continue
        # Skip FULL_SHAPE sentinel (has no parameters - expands to FULL_DIM)
        if spec is FULL_SHAPE:
            continue
        # Extract string parameters from list-based specs
        params.update(dim for dim in spec if isinstance(dim, str))
    return list(params)


# ============================================================================
# SCHEMA TYPES (Unified System)
# ============================================================================


@dataclass(frozen=True)
class InputSchema:
    """Self-contained input specification with embedded requirements.

    Complete schema for an input interface including both structure (tiling)
    and transformation requirements (layout).

    Attributes:
        name: Interface name (e.g., "input", "input0")
        block_tiling: Block tiling specification (e.g., [FULL_DIM, FULL_DIM])
        stream_tiling: Stream tiling specification (e.g., ["SIMD"], [1, 1, 1, "PE"])
        datatype: Datatype spec (None to use from ONNX, or DatatypeSpec union type to derive/optimize)
        required_layout: DECLARATIVE layout requirement (e.g., "NHWC", "NCHW")
                        Documents what layout the kernel expects. Actual enforcement
                        is handled by the normalize_dataflow_layouts preprocessing
                        step which must run before kernel inference.
                        None means no layout requirement.
    """

    # Identity
    name: str

    # Structure
    block_tiling: Optional[TilingSpec] = None
    stream_tiling: Optional[TilingSpec] = None
    datatype: Optional[Any] = None  # DatatypeSpec union type

    # Transformation requirements (NEW - embedded in interface)
    required_layout: Optional[str] = None

    def __post_init__(self):
        """Validate interface requirements."""
        if self.required_layout and self.required_layout not in {"NCHW", "NHWC"}:
            raise ValueError(
                f"Invalid required_layout '{self.required_layout}' for input '{self.name}'. "
                f"Must be 'NCHW' or 'NHWC'."
            )

    @property
    def tiling_attrs(self) -> List[str]:
        """Extract unique template parameter names from tiling specs."""
        return _extract_tiling_params(self.block_tiling, self.stream_tiling)


@dataclass(frozen=True)
class OutputSchema:
    """Self-contained output specification with embedded requirements.

    Complete schema for an output interface including both structure (tiling),
    datatype derivation, and transformation requirements (layout).

    Attributes:
        name: Interface name (e.g., "output", "output0")
        block_tiling: Block tiling specification
        stream_tiling: Stream tiling specification
        datatype: Datatype spec (None to use from ONNX, or DatatypeSpec union type to derive)
        required_layout: DECLARATIVE output layout requirement (e.g., "NHWC")
                        Documents what layout the kernel produces. Most kernels
                        preserve input layout (NHWC in, NHWC out). Actual enforcement
                        is handled by the normalize_dataflow_layouts preprocessing step.
        preserves_input_layout: Whether this output preserves the layout of the first input
                               (default True, common for element-wise operations)
    """

    # Identity
    name: str

    # Structure
    block_tiling: Optional[TilingSpec] = None
    stream_tiling: Optional[TilingSpec] = None
    datatype: Optional[Any] = None  # DatatypeSpec union type

    # Transformation requirements (NEW)
    required_layout: Optional[str] = None
    preserves_input_layout: bool = True  # Most kernels preserve layout

    def __post_init__(self):
        """Validate interface requirements."""
        if self.required_layout and self.required_layout not in {"NCHW", "NHWC"}:
            raise ValueError(
                f"Invalid required_layout '{self.required_layout}' for output '{self.name}'. "
                f"Must be 'NCHW' or 'NHWC'."
            )

    @property
    def tiling_attrs(self) -> List[str]:
        """Extract unique template parameter names from tiling specs."""
        return _extract_tiling_params(self.block_tiling, self.stream_tiling)


@dataclass
class KernelSchema:
    """Complete kernel specification - structure and validation.

    Unified schema that combines interface definitions, validation constraints,
    and parallelization parameters in one place.

    Defines kernel STRUCTURE:
    - Input/output interfaces with tiling templates and layout requirements
    - Unified validation constraints (datatype, shape, cross-interface)
    - Internal datatype derivation patterns
    - Kernel-specific parameters (algorithm, hardware, features)
    - Transformation metadata (attribute_mapping)
    - DSE dimensions (explorable parameters: tiling + resource)

    Does NOT define STORAGE:
    - Shapes are extracted from ModelWrapper or computed from templates
    - Only datatypes and user parameters persist in nodeattrs
    - KernelOp handles storage implementation

    Does NOT define BEHAVIOR:
    - Execution logic is in KernelOp.execute_node()
    - Resource estimation is in KernelOp methods

    The constraints field uses the unified Constraint system for all validation:
    - Single-interface constraints (datatype, dimension ranges)
    - Cross-interface constraints (shape equality, datatype equality)
    - ONNX-specific constraints (dynamic/static inputs, layouts)
    - Custom validation logic
    """

    # ============= IDENTITY =============
    name: str

    # ============= STRUCTURE =============
    inputs: List[InputSchema] = field(default_factory=list)
    outputs: List[OutputSchema] = field(default_factory=list)
    internal_datatypes: Dict[str, Any] = field(default_factory=dict)  # DatatypeSpec union type
    kernel_params: Dict[str, tuple] = field(default_factory=dict)

    # ============= DSE DIMENSIONS =============
    dse_dimensions: Dict[str, DSEDimension] = field(default_factory=dict)
    """Explorable resource/implementation dimensions (ram_style, res_type, etc.).

    Tiling dimensions (PE, SIMD) NOT declared here - auto-extracted from
    stream_tiling templates with defaults computed from factoring.

    Example: {"ram_style": DSEDimension("ram_style", {"distributed", "block"}, "distributed")}
    """

    # ============= VALIDATION =============
    constraints: List['Constraint'] = field(default_factory=list)

    # ============= TRANSFORMATION =============
    attribute_mapping: Dict[str, str] = field(default_factory=dict)
    """Map ONNX attributes to kernel parameters.

    Example: {"epsilon": "epsilon", "axis": "normalized_axis"}
    """

    def __post_init__(self):
        """Validate schema structure and transformation consistency."""
        self.validate()

        # Validate transformation fields
        self._validate_transformation_fields()

    def validate(self) -> None:
        """Validate the schema structure."""

        # Check unique input names
        input_names = [inp.name for inp in self.inputs]
        if len(input_names) != len(set(input_names)):
            raise ValueError(f"Duplicate input names in kernel '{self.name}'")

        # Check unique output names
        output_names = [out.name for out in self.outputs]
        if len(output_names) != len(set(output_names)):
            raise ValueError(f"Duplicate output names in kernel '{self.name}'")

        # Check input/output names don't conflict (globally unique)
        input_name_set = set(input_names)
        output_name_set = set(output_names)
        conflicts = input_name_set & output_name_set
        if conflicts:
            raise ValueError(
                f"Interface names must be unique across inputs and outputs in kernel '{self.name}'. "
                f"Duplicate names: {', '.join(sorted(conflicts))}"
            )

        # Check internal datatype names don't conflict with interfaces
        all_interface_names = input_name_set | output_name_set
        for internal_name in self.internal_datatypes.keys():
            if internal_name in all_interface_names:
                raise ValueError(
                    f"Internal datatype '{internal_name}' conflicts with interface name in kernel '{self.name}'"
                )

    def _validate_transformation_fields(self) -> None:
        """Validate transformation-related fields are consistent."""
        # Validate attribute_mapping references kernel_params
        for hw_param in self.attribute_mapping.values():
            if hw_param not in self.kernel_params:
                raise ValueError(
                    f"attribute_mapping maps to '{hw_param}' but it's not in kernel_params. "
                    f"Available params: {list(self.kernel_params.keys())}"
                )

        # Validate dse_dimensions have unique names
        for dim_name in self.dse_dimensions.keys():
            if dim_name in self.kernel_params:
                raise ValueError(
                    f"DSE dimension '{dim_name}' conflicts with kernel_param. "
                    f"DSE dimensions must have unique names."
                )

    def build_nodeattr_registry(self) -> Dict[str, tuple]:
        """Build nodeattr registry from schema definition.

        Schemas define STRUCTURE, not STORAGE. Generates persistence layer
        from structural schema, returning only attributes that need persistence:
        - Datatypes (for interfaces and internals)
        - Tiling parameters (SIMD, PE, etc.) - auto-extracted from stream_tiling
        - DSE dimensions (ram_style, res_type, etc.) - from dse_dimensions
        - Kernel-specific parameters (epsilon, algorithm, etc.) - from kernel_params

        Shapes are NEVER stored in nodeattrs. They are either:
        - Tensor shapes: extracted from ModelWrapper (ONNX graph)
        - Block/stream shapes: computed from schema templates

        Returns:
            Dict mapping nodeattr name to (type, required, default_value)
            Format: {"attrName": ("i"|"s"|"f", True|False, default)}
        """
        attrs = {}

        # Datatypes
        for i in range(len(self.inputs)):
            attrs[f"input{i}Datatype"] = ("s", False, "")

        for i in range(len(self.outputs)):
            attrs[f"output{i}Datatype"] = ("s", False, "")

        for internal_name in self.internal_datatypes.keys():
            attrs[f"{internal_name}Datatype"] = ("s", False, "")

        # Tiling parameters (PE, SIMD, etc.) - auto-extracted
        template_params = self._extract_template_params()
        for param in template_params:
            attrs[param] = ("i", False, 1)  # Default 1, will be computed from factoring

        # DSE dimensions (resource parameters)
        for dim_name, dim_spec in self.dse_dimensions.items():
            # Determine type from values
            if callable(dim_spec.values):
                # Callable - assume int for now (can't inspect without context)
                attrs[dim_name] = ("i", False, dim_spec.default if dim_spec.default is not None else 1)
            else:
                # Set - check first value type
                first_val = next(iter(dim_spec.values))
                if isinstance(first_val, int):
                    default = dim_spec.default if dim_spec.default is not None else min(dim_spec.values)
                    attrs[dim_name] = ("i", False, default)
                else:
                    default = dim_spec.default if dim_spec.default is not None else sorted(dim_spec.values)[0]
                    attrs[dim_name] = ("s", False, default)

        # Kernel-specific parameters (structural)
        attrs.update(self.kernel_params)

        return attrs

    def _extract_template_params(self) -> set:
        """Extract unique template parameter names from tiling specs.

        Returns:
            Set of template parameter names (strings found in tiling specs)
        """
        params = set()
        for interface in self.inputs + self.outputs:
            params.update(interface.tiling_attrs)
        return params

    def get_structural_nodeattrs(self) -> set:
        """Get nodeattrs that affect design space (rebuild if changed).

        Structural nodeattrs are those whose changes require rebuilding
        the entire KernelDesignSpace (not just reconfiguration).

        These include:
        - All datatypes (input, output, internal): Affect internal datatype
          derivation (e.g., accumulator width depends on input datatype)
        - Parameters in block_tiling (rare): Affect block shape computation

        Returns:
            Set of structural nodeattr names

        Example:
            >>> schema.get_structural_nodeattrs()
            {'input0Datatype', 'output0Datatype', 'accumulatorDatatype'}
        """
        structural = set()

        # All datatypes are structural
        for i in range(len(self.inputs)):
            structural.add(f"input{i}Datatype")
        for i in range(len(self.outputs)):
            structural.add(f"output{i}Datatype")
        for internal_name in self.internal_datatypes.keys():
            structural.add(f"{internal_name}Datatype")

        # Parameters in block_tiling are structural (rare)
        for inp in self.inputs:
            if inp.block_tiling and inp.block_tiling is not FULL_SHAPE:
                for elem in inp.block_tiling:
                    if isinstance(elem, str):
                        structural.add(elem)
        for out in self.outputs:
            if out.block_tiling and out.block_tiling is not FULL_SHAPE:
                for elem in out.block_tiling:
                    if isinstance(elem, str):
                        structural.add(elem)

        return structural

    def get_optimization_nodeattrs(self) -> set:
        """Get nodeattrs that affect optimization (re-explore if changed).

        Optimization nodeattrs are those whose changes only require
        re-exploring the design space (trying different stream shapes),
        not rebuilding the entire design space.

        These include:
        - Parallelization parameters (SIMD, PE, MW, MH, etc.): Appear in
          stream_tiling templates and determine stream shapes during DSE

        Returns:
            Set of optimization nodeattr names

        Example:
            >>> schema.get_optimization_nodeattrs()
            {'SIMD', 'PE'}
        """
        # Parameters in stream_tiling affect optimization
        return self._extract_template_params()
