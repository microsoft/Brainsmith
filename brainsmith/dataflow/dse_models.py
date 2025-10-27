############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""
Immutable kernel models for the dataflow system.

These models represent runtime instances built by DesignSpaceBuilder from:
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
    """Interface definition for kernel design space.

    Built once from ONNX context, reused across all parallelization configs during DSE.
    Stream tiling preserved as template for later resolution with specific parameters.
    """
    name: str
    tensor_shape: Shape
    block_shape: Shape
    stream_tiling: TilingSpec
    datatype: BaseDataType
    is_weight: bool = False
    tensor_name: Optional[str] = None  # ONNX tensor name for initializer lookups


@dataclass(frozen=True)
class InterfaceDesignPoint:
    """Interface with specific parallelization configuration.

    Flyweight pattern: references parent design space, stores only stream_shape.
    Ensures minimal memory overhead when exploring multiple configurations.
    """
    design_space: InterfaceDesignSpace
    stream_shape: Shape

    # Convenience properties (delegate to design space)
    @property
    def name(self) -> str:
        return self.design_space.name

    @property
    def tensor_shape(self) -> Shape:
        return self.design_space.tensor_shape

    @property
    def block_shape(self) -> Shape:
        return self.design_space.block_shape

    @property
    def datatype(self) -> BaseDataType:
        return self.design_space.datatype

    @property
    def is_weight(self) -> bool:
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
    def tensor_blocks_shape(self) -> Shape:
        """Per-dimension blocks needed to tile tensor.

        Returns shape where each dimension is ceil(tensor_dim / block_dim).
        Describes spatial decomposition: how we tile the problem.

        Example: tensor_shape=(100, 64), block_shape=(32, 16)
                 → tensor_blocks_shape=(4, 4)  # 4x4 grid of blocks
        """
        return tuple(
            math.ceil(tensor_dim / block_dim)
            for tensor_dim, block_dim in zip(self.tensor_shape, self.block_shape)
        )

    @property
    def stream_cycles_shape(self) -> Shape:
        """Per-dimension cycles needed to stream one block.

        Returns shape where each dimension is ceil(block_dim / stream_dim).
        Describes temporal execution: how we stream each tile.

        Example: block_shape=(32, 16), stream_shape=(8, 4)
                 → stream_cycles_shape=(4, 4)  # 4x4 cycles per block
        """
        return tuple(
            math.ceil(block_dim / stream_dim)
            for block_dim, stream_dim in zip(self.block_shape, self.stream_shape)
        )

    @property
    def tensor_folding_factor(self) -> int:
        """Number of blocks needed to cover full tensor.

        Product of tensor_blocks_shape.
        Uses ceiling division: a tensor of size 100 with block size 32
        requires ceil(100/32) = 4 blocks (3 full + 1 partial).
        """
        return math.prod(self.tensor_blocks_shape)

    @property
    def block_folding_factor(self) -> int:
        """Cycles to stream one block.

        Product of stream_cycles_shape.
        Uses ceiling division: a block of size 32 with stream width 10
        requires ceil(32/10) = 4 cycles (3 full + 1 partial).
        """
        return math.prod(self.stream_cycles_shape)

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

    Built once from ONNX context, acts as factory for KernelDesignPoint via configure().
    Contains properties constant during DSE plus valid ranges for all explorable dimensions.

    Construction: build() once (expensive), configure() many times (fast).
    """
    name: str
    inputs: Dict[str, InterfaceDesignSpace]
    outputs: Dict[str, InterfaceDesignSpace]
    internal_datatypes: Dict[str, BaseDataType]
    optimization_constraints: List['Constraint']
    dimensions: Dict[str, Set[Union[int, str]]]  # Unified: tiling + resource dimensions

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

    def configure(self, config: Dict[str, Union[int, str]]) -> 'KernelDesignPoint':
        """Instantiate kernel at specified point in design space.

        Creates a KernelDesignPoint with resolved stream shapes and validates
        all parametric constraints.

        Args:
            config: Dimension values (tiling + resource) specifying the instance point

        Returns:
            KernelDesignPoint with fully resolved configuration

        Raises:
            ValueError: If config invalid or missing dimensions
            ValidationError: If parametric constraints fail
        """
        self._validate_params(config)

        interface_lookup = {}
        inputs = self._instantiate_interfaces(self.inputs, config, interface_lookup)
        outputs = self._instantiate_interfaces(self.outputs, config, interface_lookup)

        instance = KernelDesignPoint(
            design_space=self,
            inputs=inputs,
            outputs=outputs,
            config=config,
        )

        self._validate_instance(instance, config)
        return instance

    def _validate_params(self, params: Dict[str, Union[int, str]]) -> None:
        """Validate parameters specify valid point in design space."""
        for param_name, value in params.items():
            if param_name not in self.dimensions:
                raise ValueError(
                    f"Unknown dimension: {param_name}. "
                    f"Known: {list(self.dimensions.keys())}"
                )
            if value not in self.dimensions[param_name]:
                valid = sorted(self.dimensions[param_name]) if all(isinstance(v, int) for v in self.dimensions[param_name]) else sorted(self.dimensions[param_name])
                raise ValueError(f"Invalid {param_name}={value}. Valid: {valid}")

        missing = set(self.dimensions.keys()) - set(params.keys())
        if missing:
            raise ValueError(f"Missing dimensions: {missing}")

    def _instantiate_interfaces(
        self,
        interfaces: Dict[str, Any],
        params: Dict[str, int],
        interface_lookup: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create instance interfaces with resolved stream shapes.

        Args:
            interfaces: Design space interfaces (inputs or outputs)
            params: Parallelization parameters for this instance
            interface_lookup: Lookup dict for DerivedDim resolution (mutated)

        Returns:
            Dict of InterfaceDesignPoint with resolved stream shapes
        """
        from .template_resolution import resolve_template

        configured = {}
        for interface in interfaces.values():
            stream_shape = (
                interface.block_shape if interface.stream_tiling is None
                else resolve_template(
                    interface.stream_tiling,
                    interface.block_shape,
                    lambda name: params[name],
                    interface_lookup,
                    model=None,  # Not available during configure (optional)
                    tensor_name=None,  # Not available during configure (optional)
                    hierarchy=ShapeHierarchy.STREAM  # Explicit: tuple shorthand uses STREAM hierarchy
                )
            )

            configured_interface = InterfaceDesignPoint(
                design_space=interface,
                stream_shape=stream_shape
            )
            configured[interface.name] = configured_interface
            interface_lookup[interface.name] = configured_interface

        return configured

    def _validate_instance(
        self,
        instance: 'KernelDesignPoint',
        config: Dict[str, Union[int, str]]
    ) -> None:
        """Validate instance satisfies parametric constraints.

        Args:
            instance: The instantiated kernel
            config: Dimension values used to create instance

        Raises:
            ValidationError: If any parametric constraint fails
        """
        from .validation import ConfigurationValidationContext, ValidationError

        ctx = ConfigurationValidationContext(configured_model=instance, params=config)
        for constraint in self.optimization_constraints:
            if error := constraint.check(ctx):
                raise ValidationError(
                    f"Constraint failed for {config}: {error}"
                )


@dataclass(frozen=True)
class KernelDesignPoint:
    """Specific kernel instance with resolved configuration.

    Flyweight pattern: references parent design space, stores only configuration-specific data.
    Created by KernelDesignSpace.configure() with specific dimension values (tiling + resource).
    """
    design_space: KernelDesignSpace
    inputs: Dict[str, InterfaceDesignPoint]
    outputs: Dict[str, InterfaceDesignPoint]
    config: Dict[str, Union[int, str]]  # Unified: tiling + resource dimensions

    @property
    def input_list(self) -> List[InterfaceDesignPoint]:
        """Inputs in declaration order (for ONNX positional mapping)."""
        return list(self.inputs.values())

    @property
    def output_list(self) -> List[InterfaceDesignPoint]:
        """Outputs in declaration order (for ONNX positional mapping)."""
        return list(self.outputs.values())

    # Convenience properties (delegate to design space)
    @property
    def name(self) -> str:
        return self.design_space.name

    @property
    def internal_datatypes(self) -> Dict[str, BaseDataType]:
        return self.design_space.internal_datatypes

    # Computed properties for compatibility with existing code
    @property
    def initiation_interval(self) -> int:
        """Kernel initiation interval in cycles."""
        if not self.inputs:
            return 1
        # InterfaceDesignPoint has tensor_folding_factor and block_folding_factor
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
