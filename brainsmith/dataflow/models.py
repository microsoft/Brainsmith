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


@dataclass(frozen=True)
class InternalDatatypeModel:
    """Minimal interface model for deriving from internal datatypes."""
    datatype: BaseDataType


# =============================================================================
# Two-Phase Model Types (Invariant + Configured)
# =============================================================================

@dataclass(frozen=True)
class InvariantInterfaceModel:
    """Interface properties that don't vary during DSE (Design Space Exploration).

    Contains all resolved properties except stream_shape, which is computed from
    parallelization parameters during configuration.

    This model is built once per kernel instance and reused across all
    parallelization configurations. It preserves the stream_tiling template
    for later resolution via InvariantKernelModel.configure().

    Attributes:
        name: Interface name (e.g., "input", "output")
        tensor_shape: Full logical shape from ONNX graph
        block_shape: Resolved from block_tiling template
        stream_tiling: UNRESOLVED template for stream shape
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
class ConfiguredInterfaceModel:
    """Lightweight interface model with resolved stream shape.

    References parent InvariantInterfaceModel to avoid duplication.
    Only stores the variant property (stream_shape) that changes during DSE.

    This flyweight pattern ensures minimal memory overhead per configuration
    while maintaining immutability guarantees.

    Attributes:
        invariant: Reference to parent invariant model (shared across configs)
        stream_shape: Resolved from parallelization parameters
            Example: (64,) for SIMD=64
    """
    invariant: InvariantInterfaceModel
    stream_shape: Shape

    # Convenience properties that delegate to invariant
    @property
    def name(self) -> str:
        """Interface name from invariant model."""
        return self.invariant.name

    @property
    def tensor_shape(self) -> Shape:
        """Tensor shape from invariant model."""
        return self.invariant.tensor_shape

    @property
    def block_shape(self) -> Shape:
        """Block shape from invariant model."""
        return self.invariant.block_shape

    @property
    def datatype(self) -> BaseDataType:
        """Datatype from invariant model."""
        return self.invariant.datatype

    @property
    def is_weight(self) -> bool:
        """Weight flag from invariant model."""
        return self.invariant.is_weight

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
            return self.invariant.block_shape
        elif hierarchy == ShapeHierarchy.TENSOR:
            return self.invariant.tensor_shape
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


@dataclass(frozen=True)
class InterfaceModel(ABC):
    """Base class for input/output interface models.

    Provides common fields and methods for all interface model types.

    Shape hierarchy:
    - tensor_shape: Full logical tensor dimensions
    - block_shape: Block tiling dimensions
    - stream_shape: Streaming dimensions per cycle
    """

    name: str
    tensor_shape: Shape
    block_shape: Shape
    stream_shape: Shape
    datatype: BaseDataType

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

    def get_shape(self, hierarchy: ShapeHierarchy) -> Tuple[Optional[int], ...]:
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
            return self.block_shape
        elif hierarchy == ShapeHierarchy.TENSOR:
            return self.tensor_shape
        else:
            raise ValueError(f"Invalid hierarchy: {hierarchy}")


@dataclass(frozen=True)
class InputModel(InterfaceModel):
    """Immutable input interface model.

    Represents a concrete input with resolved dimensions and datatypes.
    """

    is_weight: bool = False

    @property
    def initiation_interval(self) -> int:
        """Cycles to stream entire tensor."""
        return self.tensor_folding_factor * self.block_folding_factor


@dataclass(frozen=True)
class OutputModel(InterfaceModel):
    """Immutable output interface model.

    Represents a concrete output with fully resolved dimensions and datatypes.
    All dimensions are resolved by KernelModelBuilder before construction.
    """
    pass


@dataclass(frozen=True)
class KernelModel:
    """Immutable kernel model representing a configured kernel instance.

    This model is a runtime snapshot built by KernelModelBuilder from:
    - ModelWrapper context (tensor shapes, datatypes from ONNX graph)
    - KernelSchema (structural definition, tiling templates)
    - Nodeattrs (user parameters like SIMD, PE)

    All dimensions are fully resolved by the builder before model construction.
    Models are cached and rebuilt when:
    - User parameters change (via set_nodeattr())
    - Context updates (new call to get_kernel_model(ctx))
    """

    # Kernel identity
    name: str

    # Interface models (fully resolved)
    inputs: Tuple[InputModel, ...]
    outputs: Tuple[OutputModel, ...]

    # Performance characteristics
    latency_cycles: Tuple[int, int] = (1, 1)
    pipeline_depth: int = 1
    clock_freq_mhz: float = 100.0

    def get_input(self, name: str) -> Optional[InputModel]:
        """Get input model by name."""
        for inp in self.inputs:
            if inp.name == name:
                return inp
        return None

    def get_output(self, name: str) -> Optional[OutputModel]:
        """Get output model by name."""
        for out in self.outputs:
            if out.name == name:
                return out
        return None

    def get_interface(self, name: str) -> Optional[Union[InputModel, OutputModel]]:
        """Get interface model by name (searches inputs then outputs)."""
        # Try inputs first
        interface = self.get_input(name)
        if interface is not None:
            return interface
        # Try outputs
        return self.get_output(name)

    @property
    def initiation_interval(self) -> int:
        """Kernel initiation interval in cycles."""
        if not self.inputs:
            return 1
        return max(inp.initiation_interval for inp in self.inputs)

    @property
    def max_block_folding_factor(self) -> int:
        """Maximum block folding factor across all inputs."""
        if not self.inputs:
            return 1
        return max(inp.block_folding_factor for inp in self.inputs)

    @property
    def max_tensor_folding_factor(self) -> int:
        """Maximum tensor folding factor across all inputs."""
        if not self.inputs:
            return 1
        return max(inp.tensor_folding_factor for inp in self.inputs)

    @property
    def total_output_values(self) -> int:
        """Total output values across all outputs."""
        return sum(prod(out.tensor_shape) for out in self.outputs)

    def output_stream_width_bits(self, output_idx: int = 0) -> int:
        """Stream width in bits for output.

        Returns the actual stream width based on the output's stream_shape,
        not derived from input block folding factors.
        """
        output = self.outputs[output_idx]
        return output.stream_width_bits

    def output_stream_shape(self, output_idx: int = 0) -> Shape:
        """Stream shape for output.

        Returns the OutputModel's stream_shape attribute (resolved by builder).
        """
        return self.outputs[output_idx].stream_shape

    @property
    def throughput_fps(self) -> float:
        """Throughput in inferences per second."""
        cycles_per_inf = self.initiation_interval
        clock_hz = self.clock_freq_mhz * 1e6
        return clock_hz / cycles_per_inf


# =============================================================================
# Two-Phase Kernel Models (Invariant + Configured)
# =============================================================================

@dataclass(frozen=True)
class InvariantKernelModel:
    """Kernel model with invariant properties resolved, stream shapes deferred.

    Built once per kernel instance using KernelModelBuilder.build_invariant().
    Acts as factory for ConfiguredKernelModel instances via configure() method.

    This model contains all properties that don't vary during Design Space
    Exploration (DSE), plus pre-computed valid parallelization ranges.

    Two-phase construction:
    1. build_invariant() creates this model once (expensive ~10ms)
    2. configure() creates ConfiguredKernelModel many times (fast <1ms)

    Attributes:
        name: Kernel name (e.g., "LayerNorm", "Softmax")
        inputs: Invariant input interface models (stream shapes unresolved)
        outputs: Invariant output interface models (stream shapes unresolved)
        internal_datatypes: Resolved internal datatypes (accumulator, etc.)
        invariant_constraints: Constraints validated once at build time
            Examples: IsDynamic, DatatypeInteger, tensor/block-level shapes
        variant_constraints: Constraints validated per configuration
            Examples: stream-level DimensionDivisible
        parallelization_params: Valid divisor sets for each parameter
            Example: {"SIMD": {1, 2, 3, 4, 6, 8, ..., 768}, "PE": {1, 2, 4, 8}}
    """
    name: str
    inputs: Tuple[InvariantInterfaceModel, ...]
    outputs: Tuple[InvariantInterfaceModel, ...]
    internal_datatypes: Dict[str, BaseDataType]
    invariant_constraints: List['Constraint']
    variant_constraints: List['Constraint']
    parallelization_params: Dict[str, Set[int]]

    def configure(self, params: Dict[str, int]) -> 'ConfiguredKernelModel':
        """Resolve stream shapes and create configured model.

        Process:
        1. Validate params are in valid ranges
        2. Resolve stream shapes for inputs (can reference params)
        3. Resolve stream shapes for outputs (can reference params + inputs via DerivedDim)
        4. Build ConfiguredInterfaceModel instances
        5. Create ConfiguredKernelModel
        6. Validate variant constraints
        7. Return ConfiguredKernelModel

        Args:
            params: Parallelization parameters, e.g., {"SIMD": 64, "PE": 1}

        Returns:
            ConfiguredKernelModel with resolved stream shapes

        Raises:
            ValueError: If params invalid or missing required parameters
            ValidationError: If variant constraints fail

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
        configured_inputs = []
        interface_lookup = {}  # For DerivedDim resolution

        for inv_inp in self.inputs:
            if inv_inp.stream_tiling is None:
                # No stream tiling specified - use block shape
                stream_shape = inv_inp.block_shape
            else:
                stream_shape = resolve_template(
                    inv_inp.stream_tiling,
                    inv_inp.block_shape,  # reference shape for stream tiling
                    lambda name: params[name],
                    interface_lookup,  # Empty for inputs (no derivation yet)
                )

            cfg_inp = ConfiguredInterfaceModel(
                invariant=inv_inp,
                stream_shape=stream_shape,
            )
            configured_inputs.append(cfg_inp)
            interface_lookup[inv_inp.name] = cfg_inp

        # 3. Resolve stream shapes for outputs (can reference inputs via DerivedDim)
        configured_outputs = []

        for inv_out in self.outputs:
            if inv_out.stream_tiling is None:
                # No stream tiling specified - use block shape
                stream_shape = inv_out.block_shape
            else:
                stream_shape = resolve_template(
                    inv_out.stream_tiling,
                    inv_out.block_shape,  # reference shape for stream tiling
                    lambda name: params[name],
                    interface_lookup,  # Now includes inputs
                )

            cfg_out = ConfiguredInterfaceModel(
                invariant=inv_out,
                stream_shape=stream_shape,
            )
            configured_outputs.append(cfg_out)
            interface_lookup[inv_out.name] = cfg_out

        # 4. Create configured model
        configured_model = ConfiguredKernelModel(
            invariant=self,
            inputs=tuple(configured_inputs),
            outputs=tuple(configured_outputs),
            params=params,
        )

        # 5. Validate variant constraints
        # Import ConfiguredValidationContext here to avoid circular dependency
        from .validation import ConfiguredValidationContext

        ctx = ConfiguredValidationContext(configured_model, params)

        for constraint in self.variant_constraints:
            error = constraint.validate(ctx)
            if error:
                raise ValidationError(
                    f"Variant constraint failed for configuration {params}: {error}"
                )

        return configured_model

    def get_input(self, name: str) -> Optional[InvariantInterfaceModel]:
        """Get invariant input model by name."""
        for inp in self.inputs:
            if inp.name == name:
                return inp
        return None

    def get_output(self, name: str) -> Optional[InvariantInterfaceModel]:
        """Get invariant output model by name."""
        for out in self.outputs:
            if out.name == name:
                return out
        return None

    def get_interface(self, name: str) -> Optional[InvariantInterfaceModel]:
        """Get invariant interface model by name (searches inputs then outputs)."""
        interface = self.get_input(name)
        if interface is not None:
            return interface
        return self.get_output(name)


@dataclass(frozen=True)
class ConfiguredKernelModel:
    """Fully configured kernel model with resolved stream shapes.

    Holds reference to parent InvariantKernelModel to avoid duplicating
    invariant data. Provides same interface as KernelModel for compatibility
    with existing code.

    Created by InvariantKernelModel.configure() with specific parallelization
    parameters. Multiple ConfiguredKernelModel instances share the same
    InvariantKernelModel (flyweight pattern).

    Attributes:
        invariant: Reference to parent invariant model (shared)
        inputs: Configured input interfaces with resolved stream shapes
        outputs: Configured output interfaces with resolved stream shapes
        params: Current parallelization configuration
            Example: {"SIMD": 64, "PE": 1}
    """
    invariant: InvariantKernelModel
    inputs: Tuple[ConfiguredInterfaceModel, ...]
    outputs: Tuple[ConfiguredInterfaceModel, ...]
    params: Dict[str, int]

    # Convenience properties that delegate to invariant
    @property
    def name(self) -> str:
        """Kernel name from invariant model."""
        return self.invariant.name

    @property
    def internal_datatypes(self) -> Dict[str, BaseDataType]:
        """Internal datatypes from invariant model."""
        return self.invariant.internal_datatypes

    def get_input(self, name: str) -> Optional[ConfiguredInterfaceModel]:
        """Get configured input model by name."""
        for inp in self.inputs:
            if inp.name == name:
                return inp
        return None

    def get_output(self, name: str) -> Optional[ConfiguredInterfaceModel]:
        """Get configured output model by name."""
        for out in self.outputs:
            if out.name == name:
                return out
        return None

    def get_interface(self, name: str) -> Optional[ConfiguredInterfaceModel]:
        """Get configured interface by name (inputs or outputs).

        Args:
            name: Interface name to look up

        Returns:
            ConfiguredInterfaceModel if found, None otherwise
        """
        for interface in self.inputs:
            if interface.name == name:
                return interface
        for interface in self.outputs:
            if interface.name == name:
                return interface
        return None

    # Computed properties for compatibility with existing code
    @property
    def initiation_interval(self) -> int:
        """Kernel initiation interval in cycles."""
        if not self.inputs:
            return 1
        # ConfiguredInterfaceModel has initiation_interval property
        return max(inp.tensor_folding_factor * inp.block_folding_factor
                   for inp in self.inputs)

    @property
    def max_block_folding_factor(self) -> int:
        """Maximum block folding factor across all inputs."""
        if not self.inputs:
            return 1
        return max(inp.block_folding_factor for inp in self.inputs)

    @property
    def max_tensor_folding_factor(self) -> int:
        """Maximum tensor folding factor across all inputs."""
        if not self.inputs:
            return 1
        return max(inp.tensor_folding_factor for inp in self.inputs)

    @property
    def total_output_values(self) -> int:
        """Total output values across all outputs."""
        return sum(prod(out.tensor_shape) for out in self.outputs)

    def output_stream_width_bits(self, output_idx: int = 0) -> int:
        """Stream width in bits for output.

        Returns the actual stream width based on the output's stream_shape.
        """
        output = self.outputs[output_idx]
        return output.stream_width_bits

    def output_stream_shape(self, output_idx: int = 0) -> Shape:
        """Stream shape for output.

        Returns the output's stream_shape attribute (resolved during configure).
        """
        return self.outputs[output_idx].stream_shape
