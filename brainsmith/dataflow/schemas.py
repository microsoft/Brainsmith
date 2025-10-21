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
from typing import List, Optional, Sequence, Union, Dict, Any, Callable, TYPE_CHECKING
from abc import ABC, abstractmethod

from qonnx.core.datatype import BaseDataType
from .dimension_sources import DimensionSource
from .datatype_sources import DatatypeSource
from .types import FULL_DIM

if TYPE_CHECKING:
    from .constraints import Constraint

logger = logging.getLogger(__name__)

# Type aliases for better clarity
TilingSpec = Sequence[Union[int, str, type(FULL_DIM), DimensionSource]]


# ============================================================================
# SCHEMA TYPES (Unified System)
# ============================================================================


@dataclass(frozen=True)
class InputSchema:
    """Self-contained input specification with embedded requirements.

    Complete schema for an input interface including both structure (tiling)
    and transformation requirements (layout, constraints).

    Attributes:
        name: Interface name (e.g., "input", "input0")
        block_tiling: Block tiling specification (e.g., [FULL_DIM, FULL_DIM])
        stream_tiling: Stream tiling specification (e.g., ["SIMD"], [1, 1, 1, "PE"])
        required_layout: Required data layout for transformation (e.g., "NHWC", "NCHW")
                        This is part of what the interface IS, not how we create it.
                        None means no layout requirement.
        constraints: Interface-level validation constraints (e.g., IsDynamic(), DatatypeInteger())
                    These are scoped to this specific interface.
    """

    # Identity
    name: str

    # Structure
    block_tiling: Optional[TilingSpec] = None
    stream_tiling: Optional[TilingSpec] = None

    # Transformation requirements (NEW - embedded in interface)
    required_layout: Optional[str] = None

    # Interface-level constraints (NEW - scoped to this interface)
    constraints: List['Constraint'] = field(default_factory=list)

    def __post_init__(self):
        """Validate interface requirements."""
        # Validate layout if specified
        if self.required_layout and self.required_layout not in {"NCHW", "NHWC"}:
            raise ValueError(
                f"Invalid required_layout '{self.required_layout}' for input '{self.name}'. "
                f"Must be 'NCHW' or 'NHWC'."
            )

    @property
    def tiling_attrs(self) -> List[str]:
        """Extract unique template parameter names from tiling specs."""
        params = set()
        for spec in (self.block_tiling, self.stream_tiling):
            if spec:
                params.update(dim for dim in spec if isinstance(dim, str))
        return list(params)


@dataclass(frozen=True)
class OutputSchema:
    """Self-contained output specification with embedded requirements.

    Complete schema for an output interface including both structure (tiling),
    datatype derivation, and transformation requirements (layout).

    Attributes:
        name: Interface name (e.g., "output", "output0")
        block_tiling: Block tiling specification
        stream_tiling: Stream tiling specification
        datatype: Datatype source (None to use from ONNX, or DatatypeSource to derive)
        required_layout: Required output layout (e.g., "NHWC")
        preserves_input_layout: Whether this output preserves the layout of the first input
                               (default True, common for element-wise operations)
    """

    # Identity
    name: str

    # Structure
    block_tiling: Optional[TilingSpec] = None
    stream_tiling: Optional[TilingSpec] = None
    datatype: Optional[DatatypeSource] = None

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
        params = set()
        for spec in (self.block_tiling, self.stream_tiling):
            if spec:
                params.update(dim for dim in spec if isinstance(dim, str))
        return list(params)


@dataclass
class KernelSchema:
    """Complete kernel specification - structure, validation, and transformation.

    Unified schema that combines interface definitions, validation constraints,
    and transformation requirements in one place.

    Defines kernel STRUCTURE:
    - Input/output interfaces with tiling templates and layout requirements
    - Unified validation constraints (datatype, shape, cross-interface)
    - Internal datatype derivation patterns
    - Kernel-specific parameters (algorithm, hardware, features)

    Defines kernel TRANSFORMATION:
    - source_ops: Which ONNX ops can be transformed to this kernel
    - attribute_mapping: Map ONNX attributes to kernel parameters
    - initial_parallelization: Initial DSE entry point

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
    domain: str = "brainsmith.kernels"

    # ============= STRUCTURE =============
    inputs: List[InputSchema] = field(default_factory=list)
    outputs: List[OutputSchema] = field(default_factory=list)
    internal_datatypes: Dict[str, DatatypeSource] = field(default_factory=dict)
    kernel_params: Dict[str, tuple] = field(default_factory=dict)

    # ============= VALIDATION =============
    constraints: List['Constraint'] = field(default_factory=list)

    # ============= TRANSFORMATION =============
    source_ops: List[str] = field(default_factory=list)
    """ONNX op types that can be transformed to this kernel.

    Empty list means no ONNX transformation support.
    Example: ["FuncLayerNorm", "LayerNormalization"]
    """

    attribute_mapping: Dict[str, str] = field(default_factory=dict)
    """Map ONNX attributes to kernel parameters.

    Example: {"epsilon": "epsilon", "axis": "normalized_axis"}
    """

    initial_parallelization: Dict[str, int] = field(default_factory=lambda: {"SIMD": 1})
    """Initial parallelization parameters for DSE entry point.

    Example: {"SIMD": 1, "PE": 1}
    """

    def __post_init__(self):
        """Validate schema structure and transformation consistency."""
        self.validate()

        # Validate transformation fields if present
        if self.source_ops:
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

        # Validate initial_parallelization values
        for param, value in self.initial_parallelization.items():
            if not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"initial_parallelization['{param}'] must be positive integer, got {value}"
                )

    def can_transform(self, node, model) -> bool:
        """Check if this schema can transform the given ONNX node.

        Pure validation - no side effects.

        Args:
            node: ONNX NodeProto to validate
            model: ModelWrapper for graph context

        Returns:
            True if transformation possible, False otherwise
        """
        # Import here to avoid circular dependency
        from onnx import NodeProto
        from qonnx.core.modelwrapper import ModelWrapper

        # Check op type
        if not self.source_ops or node.op_type not in self.source_ops:
            return False

        # Check interface counts match
        if len(node.input) != len(self.inputs):
            logger.debug(
                f"{self.name}: Input count mismatch. "
                f"Schema expects {len(self.inputs)}, node has {len(node.input)}"
            )
            return False

        if len(node.output) != len(self.outputs):
            logger.debug(
                f"{self.name}: Output count mismatch. "
                f"Schema expects {len(self.outputs)}, node has {len(node.output)}"
            )
            return False

        # Check global constraints
        from .validation import OnnxValidationContext
        ctx = OnnxValidationContext(node=node, model=model, schema=self)
        for constraint in self.constraints:
            error = constraint.check(ctx)
            if error:
                logger.debug(f"{self.name}: {error}")
                return False

        return True

    def get_nodeattr_types(self) -> Dict[str, tuple]:
        """Generate nodeattr registry from schema.

        Schemas define STRUCTURE, not STORAGE. Returns only attributes
        that need persistence:
        - Datatypes (for interfaces and internals)
        - User parameters (SIMD, PE, etc.)
        - Kernel-specific parameters (epsilon, algorithm, etc.)

        Shapes are NEVER stored in nodeattrs. They are either:
        - Tensor shapes: extracted from ModelWrapper (ONNX graph)
        - Block/stream shapes: computed from schema templates

        Returns:
            Dict mapping nodeattr name to (type, required, default_value)
            Format: {"attrName": ("i"|"s"|"f", True|False, default)}
        """
        attrs = {}

        # ================================================================
        # 1. Interface Datatypes (datatypes only, NO shapes)
        # ================================================================

        for i in range(len(self.inputs)):
            attrs[f"input{i}Datatype"] = ("s", False, "")

        for i in range(len(self.outputs)):
            attrs[f"output{i}Datatype"] = ("s", False, "")

        # ================================================================
        # 2. Internal Datatypes
        # ================================================================

        for internal_name in self.internal_datatypes.keys():
            attr_name = f"{internal_name}Datatype"
            attrs[attr_name] = ("s", False, "")

        # ================================================================
        # 3. User Parameters (template params from tiling specs)
        # ================================================================

        template_params = self._extract_template_params()
        for param in template_params:
            attrs[param] = ("i", True, 1)

        # ================================================================
        # 4. Kernel-Specific Parameters (algorithm, hardware, features)
        # ================================================================

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
            if inp.block_tiling:
                for elem in inp.block_tiling:
                    if isinstance(elem, str):
                        structural.add(elem)
        for out in self.outputs:
            if out.block_tiling:
                for elem in out.block_tiling:
                    if isinstance(elem, str):
                        structural.add(elem)

        return structural

    def get_parametric_nodeattrs(self) -> set:
        """Get nodeattrs that affect configuration (reconfigure if changed).

        Parametric nodeattrs are those whose changes only require
        reconfiguration (selecting a different stream shape), not
        rebuilding the entire design space.

        These include:
        - Parallelization parameters (SIMD, PE, MW, MH, etc.): Appear in
          stream_tiling templates and determine stream shapes

        Returns:
            Set of parametric nodeattr names

        Example:
            >>> schema.get_parametric_nodeattrs()
            {'SIMD', 'PE'}
        """
        # Parameters in stream_tiling are parametric
        return self._extract_template_params()
