############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""
Immutable kernel models for the dataflow system.

These models represent runtime instances built by KernelModelBuilder from:
- ModelWrapper context (tensor shapes, datatypes from ONNX graph)
- KernelSchema (structural definition, tiling templates)
- Nodeattrs (user parameters like SIMD, PE)

Key principles:
- All models are immutable (frozen dataclasses)
- Builder constructs fully resolved models (no post-processing)
- Models are cached, rebuilt on context/parameter changes
- Shapes are NEVER stored in nodeattrs (extracted from context)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union, Any, Set, Sequence, TYPE_CHECKING
from abc import ABC
import math

from .types import Shape, ShapeHierarchy, prod
from qonnx.core.datatype import BaseDataType

if TYPE_CHECKING:
    from .dimension_sources import DimensionSource
    from .constraints import Constraint

# Type alias for tiling specifications (avoid circular import)
TilingSpec = Sequence[Union[int, str, Any]]  # Any covers FULL_DIM and DimensionSource

logger = logging.getLogger(__name__)


# =============================================================================
# Design Space Exploration Model Types
# =============================================================================

@dataclass(frozen=True)
class InterfaceDesignSpace:
    """Interface definition within kernel design space.

    Contains properties that remain constant across all parallelization
    configurations during design space exploration (DSE). The stream_tiling
    template is preserved for later resolution with specific parameters.

    This model is built once from ONNX context and reused across all
    configurations explored during DSE.

    Attributes:
        name: Interface name (e.g., "input", "output")
        tensor_shape: Full logical shape from ONNX graph
        block_shape: Resolved from block_tiling template
        stream_tiling: Template for stream shape (resolved during configure())
            Examples: ["SIMD"], [DerivedDim("input", -1)]
        datatype: Resolved datatype (INT8, FLOAT32, etc.)
        is_weight: True if backed by ONNX initializer (inputs only)
    """
    name: str
    tensor_shape: Shape
    block_shape: Shape
    stream_tiling: TilingSpec  # Preserved from schema - NOT resolved
    datatype: BaseDataType
    is_weight: bool = False


@dataclass(frozen=True)
class InterfaceConfiguration:
    """Interface with specific parallelization configuration.

    References parent InterfaceDesignSpace to avoid duplication.
    Only stores the stream_shape which varies across configurations.

    This flyweight pattern ensures minimal memory overhead when exploring
    multiple configurations while maintaining immutability.

    Attributes:
        design_space: Reference to parent design space interface (shared)
        stream_shape: Resolved from parallelization parameters
            Example: (64,) for SIMD=64
    """
    design_space: InterfaceDesignSpace
    stream_shape: Shape

    # Convenience properties that delegate to design space
    @property
    def name(self) -> str:
        """Interface name from design space."""
        return self.design_space.name

    @property
    def tensor_shape(self) -> Shape:
        """Tensor shape from design space."""
        return self.design_space.tensor_shape

    @property
    def block_shape(self) -> Shape:
        """Block shape from design space."""
        return self.design_space.block_shape

    @property
    def datatype(self) -> BaseDataType:
        """Datatype from design space."""
        return self.design_space.datatype

    @property
    def is_weight(self) -> bool:
        """Weight flag from design space."""
        return self.design_space.is_weight

    def get_shape(self, hierarchy: ShapeHierarchy) -> Shape:
        """Get shape at specified hierarchy level.

        Args:
            hierarchy: Which level of the shape hierarchy to retrieve

        Returns:
            Shape at the specified level

        Raises:
            ValueError: If hierarchy is invalid
        """
        if hierarchy == ShapeHierarchy.STREAM:
            return self.stream_shape
        elif hierarchy == ShapeHierarchy.BLOCK:
            return self.design_space.block_shape
        elif hierarchy == ShapeHierarchy.TENSOR:
            return self.design_space.tensor_shape
        else:
            raise ValueError(f"Invalid hierarchy: {hierarchy}")

    # Computed properties (for compatibility with existing code)
    @property
    def tensor_folding_factor(self) -> int:
        """Number of blocks to cover full tensor (tensor_shape / block_shape)."""
        num_blocks = 1
        for t, b in zip(self.tensor_shape, self.block_shape):
            num_blocks *= math.ceil(t / b)
        return num_blocks

    @property
    def block_folding_factor(self) -> int:
        """Cycles to stream one block (block_shape / stream_shape)."""
        cycles = 1
        for b, s in zip(self.block_shape, self.stream_shape):
            cycles *= math.ceil(b / s)
        return cycles

    @property
    def streaming_bandwidth(self) -> int:
        """Elements streamed per cycle."""
        return prod(self.stream_shape)

    @property
    def stream_width_bits(self) -> int:
        """Stream width in bits."""
        return self.streaming_bandwidth * self.datatype.bitwidth()


# =============================================================================
# Design Space Exploration Kernel Models
# =============================================================================

@dataclass(frozen=True)
class KernelDesignSpace:
    """Defines kernel design space for parallelization exploration.

    Built once from ONNX context using KernelModelBuilder.build().
    Acts as factory for KernelConfiguration instances via configure() method.

    This model contains all properties that remain constant during Design Space
    Exploration (DSE) plus valid ranges for parallelization parameters (SIMD, PE, etc).

    Construction flow:
    1. build() creates design space once (expensive ~10ms)
    2. configure() creates configurations many times (fast <1ms each)

    Attributes:
        name: Kernel name (e.g., "LayerNorm", "Softmax")
        inputs: Design space interfaces for inputs (dict: name → interface)
            Stream shapes unresolved. Ordered by declaration (Python 3.7+).
        outputs: Design space interfaces for outputs (dict: name → interface)
            Stream shapes unresolved. Ordered by declaration (Python 3.7+).
        internal_datatypes: Resolved internal datatypes (accumulator, etc.)
        parametric_constraints: Constraints validated per configuration
            Examples: stream-level DimensionDivisible, ShapesEqual(STREAM)
        parallelization_params: Valid divisor sets for each parameter
            Example: {"SIMD": {1, 2, 3, 4, 6, 8, ..., 768}, "PE": {1, 2, 4, 8}}

        Note: Structural constraints (e.g., DatatypeInteger, tensor/block shapes)
        are validated during build() but not stored - they're checked once and then
        discarded since they never need re-validation.
    """
    name: str
    inputs: Dict[str, InterfaceDesignSpace]
    outputs: Dict[str, InterfaceDesignSpace]
    internal_datatypes: Dict[str, BaseDataType]
    parametric_constraints: List['Constraint']
    parallelization_params: Dict[str, Set[int]]

    @property
    def input_list(self) -> List[InterfaceDesignSpace]:
        """Inputs in declaration order (for ONNX positional mapping).

        Returns inputs as list preserving dict insertion order (Python 3.7+).
        Useful when mapping to ONNX node.input[i] positions.
        """
        return list(self.inputs.values())

    @property
    def output_list(self) -> List[InterfaceDesignSpace]:
        """Outputs in declaration order (for ONNX positional mapping).

        Returns outputs as list preserving dict insertion order (Python 3.7+).
        Useful when mapping to ONNX node.output[i] positions.
        """
        return list(self.outputs.values())

    def configure(self, params: Dict[str, int]) -> 'KernelConfiguration':
        """Select specific configuration from design space.

        Process:
        1. Validate params are in valid ranges
        2. Resolve stream shapes for inputs (using params)
        3. Resolve stream shapes for outputs (using params + inputs via DerivedDim)
        4. Build InterfaceConfiguration instances
        5. Create KernelConfiguration
        6. Validate parametric constraints
        7. Return KernelConfiguration

        Args:
            params: Parallelization parameters, e.g., {"SIMD": 64, "PE": 1}

        Returns:
            KernelConfiguration with fully resolved stream shapes

        Raises:
            ValueError: If params invalid or missing required parameters
            ValidationError: If parametric constraints fail

        Performance: Target <1ms for typical kernel
        """
        # Import here to avoid circular dependency
        from .template_resolution import resolve_template
        from .validation import ValidationError

        # 1. Validate params
        for param_name, value in params.items():
            if param_name not in self.parallelization_params:
                raise ValueError(
                    f"Unknown parallelization parameter: '{param_name}'. "
                    f"Known parameters: {list(self.parallelization_params.keys())}"
                )
            if value not in self.parallelization_params[param_name]:
                valid = sorted(self.parallelization_params[param_name])
                raise ValueError(
                    f"Invalid value for {param_name}={value}. "
                    f"Must be divisor of block dimension. Valid values: {valid}"
                )

        # Check all required params are provided
        for param_name in self.parallelization_params.keys():
            if param_name not in params:
                raise ValueError(f"Missing required parameter: '{param_name}'")

        # 2. Resolve stream shapes for inputs
        configured_inputs = {}  # Dict[str, InterfaceConfiguration]
        interface_lookup = {}  # For DerivedDim resolution

        for ds_inp in self.inputs.values():
            if ds_inp.stream_tiling is None:
                # No stream tiling specified - use block shape
                stream_shape = ds_inp.block_shape
            else:
                stream_shape = resolve_template(
                    ds_inp.stream_tiling,
                    ds_inp.block_shape,  # reference shape for stream tiling
                    lambda name: params[name],
                    interface_lookup,  # Empty for inputs (no derivation yet)
                )

            cfg_inp = InterfaceConfiguration(
                design_space=ds_inp,
                stream_shape=stream_shape,
            )
            configured_inputs[ds_inp.name] = cfg_inp  # Store in dict by name
            interface_lookup[ds_inp.name] = cfg_inp

        # 3. Resolve stream shapes for outputs (can reference inputs via DerivedDim)
        configured_outputs = {}  # Dict[str, InterfaceConfiguration]

        for ds_out in self.outputs.values():
            if ds_out.stream_tiling is None:
                # No stream tiling specified - use block shape
                stream_shape = ds_out.block_shape
            else:
                stream_shape = resolve_template(
                    ds_out.stream_tiling,
                    ds_out.block_shape,  # reference shape for stream tiling
                    lambda name: params[name],
                    interface_lookup,  # Now includes inputs
                )

            cfg_out = InterfaceConfiguration(
                design_space=ds_out,
                stream_shape=stream_shape,
            )
            configured_outputs[ds_out.name] = cfg_out  # Store in dict by name
            interface_lookup[ds_out.name] = cfg_out

        # 4. Create configuration
        configuration = KernelConfiguration(
            design_space=self,
            inputs=configured_inputs,
            outputs=configured_outputs,
            params=params,
        )

        # 5. Validate parametric constraints
        # Import ConfigurationValidationContext here to avoid circular dependency
        from .validation import ConfigurationValidationContext

        ctx = ConfigurationValidationContext(configuration, params)

        for constraint in self.parametric_constraints:
            error = constraint.validate(ctx)
            if error:
                raise ValidationError(
                    f"Parametric constraint failed for configuration {params}: {error}"
                )

        return configuration


@dataclass(frozen=True)
class KernelConfiguration:
    """Specific kernel configuration with resolved parallelization.

    Holds reference to parent KernelDesignSpace to avoid duplicating
    data that doesn't vary across configurations (flyweight pattern).

    Created by KernelDesignSpace.configure() with specific parallelization
    parameters. Multiple KernelConfiguration instances can share the same
    KernelDesignSpace during DSE.

    Attributes:
        design_space: Reference to parent design space (shared)
        inputs: Configuration interfaces with resolved stream shapes (dict: name → interface)
            Ordered by declaration (Python 3.7+).
        outputs: Configuration interfaces with resolved stream shapes (dict: name → interface)
            Ordered by declaration (Python 3.7+).
        params: Current parallelization parameters
            Example: {"SIMD": 64, "PE": 1}
    """
    design_space: KernelDesignSpace
    inputs: Dict[str, InterfaceConfiguration]
    outputs: Dict[str, InterfaceConfiguration]
    params: Dict[str, int]

    @property
    def input_list(self) -> List[InterfaceConfiguration]:
        """Inputs in declaration order (for ONNX positional mapping)."""
        return list(self.inputs.values())

    @property
    def output_list(self) -> List[InterfaceConfiguration]:
        """Outputs in declaration order (for ONNX positional mapping)."""
        return list(self.outputs.values())

    # Convenience properties that delegate to design space
    @property
    def name(self) -> str:
        """Kernel name from design space."""
        return self.design_space.name

    @property
    def internal_datatypes(self) -> Dict[str, BaseDataType]:
        """Internal datatypes from design space."""
        return self.design_space.internal_datatypes

    # Computed properties for compatibility with existing code
    @property
    def initiation_interval(self) -> int:
        """Kernel initiation interval in cycles."""
        if not self.inputs:
            return 1
        # InterfaceConfiguration has tensor_folding_factor and block_folding_factor
        return max(inp.tensor_folding_factor * inp.block_folding_factor
                   for inp in self.inputs.values())

    @property
    def max_block_folding_factor(self) -> int:
        """Maximum block folding factor across all inputs."""
        if not self.inputs:
            return 1
        return max(inp.block_folding_factor for inp in self.inputs.values())

    @property
    def max_tensor_folding_factor(self) -> int:
        """Maximum tensor folding factor across all inputs."""
        if not self.inputs:
            return 1
        return max(inp.tensor_folding_factor for inp in self.inputs.values())

    @property
    def total_output_values(self) -> int:
        """Total output values across all outputs."""
        return sum(prod(out.tensor_shape) for out in self.outputs.values())

    def output_stream_width_bits(self, output_idx: int = 0) -> int:
        """Stream width in bits for output.

        Returns the actual stream width based on the output's stream_shape.
        """
        output = self.output_list[output_idx]
        return output.stream_width_bits

    def output_stream_shape(self, output_idx: int = 0) -> Shape:
        """Stream shape for output.

        Returns the output's stream_shape attribute (resolved during configure).
        """
        return self.output_list[output_idx].stream_shape
