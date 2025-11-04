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
from typing import Iterator, List, Dict, Tuple, Optional, Union, Any, Set, Sequence, TYPE_CHECKING, FrozenSet, Literal
from abc import ABC
import math

from .types import Shape, ShapeHierarchy, prod
from .ordered_dimension import OrderedDimension
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
    """Interface design space built once, configured many times.

    Defines interface structure constant during DSE. Stream tiling preserved
    as template for resolution with specific parallelization parameters.

    Attributes:
        name: Interface name
        tensor_shape: Full tensor dimensions
        block_shape: Block dimensions (per-operation tile size)
        stream_tiling: Stream tiling template (e.g., ["SIMD"] or [1, 1, 1, "PE"])
        datatype: Interface datatype
        is_weight: Whether this is a weight tensor (constant)
        tensor_name: ONNX tensor name for initializer lookups
        parallelism_dimension: OrderedDimension for stream parameter (None if no parallelism)
        parallelism_param: Parameter name for stream dimension (e.g., "SIMD", "PE")
    """
    name: str
    tensor_shape: Shape
    block_shape: Shape
    stream_tiling: TilingSpec
    datatype: BaseDataType
    is_weight: bool = False
    tensor_name: Optional[str] = None  # ONNX tensor name for initializer lookups

    # Parallelism metadata (None if no stream parameters)
    parallelism_dimension: Optional[OrderedDimension] = None
    parallelism_param: Optional[str] = None


@dataclass(frozen=True)
class InterfaceDesignPoint:
    """Interface instance with resolved parallelization.

    Flyweight pattern: references parent design space, stores only configuration-
    specific stream_shape. Delegates tensor_shape, block_shape, and datatype
    to design space for minimal memory overhead.

    Attributes:
        design_space: Parent InterfaceDesignSpace
        stream_shape: Resolved stream dimensions for this configuration
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
    """Kernel design space built once, configured many times.

    Built by DesignSpaceBuilder from ONNX context, acts as factory for
    KernelDesignPoint via configure(). Contains structure constant during
    DSE plus valid ranges for all explorable dimensions.

    Attributes:
        name: Kernel name
        inputs: Input interface design spaces (by name)
        outputs: Output interface design spaces (by name)
        internal_datatypes: Internal datatypes (e.g., accumulator)
        optimization_constraints: Parametric constraints validated at configure()
        dimensions: Explorable dimensions - OrderedDimension (with navigation) or
                   frozenset (discrete categories like ram_style)
    """
    name: str
    inputs: Dict[str, InterfaceDesignSpace]
    outputs: Dict[str, InterfaceDesignSpace]
    internal_datatypes: Dict[str, BaseDataType]
    optimization_constraints: List['Constraint']
    dimensions: Dict[str, Union[OrderedDimension, FrozenSet]]  # OrderedDimension for ordered, frozenset for discrete

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

    # =========================================================================
    # Dimension Query Methods
    # =========================================================================

    def get_dimension(self, name: str) -> Union[OrderedDimension, FrozenSet]:
        """Get dimension by name.

        Args:
            name: Dimension name

        Returns:
            OrderedDimension for ordered dimensions, frozenset for discrete

        Raises:
            KeyError: If dimension not found
        """
        return self.dimensions[name]

    def get_ordered(self, name: str) -> OrderedDimension:
        """Get ordered dimension by name.

        Args:
            name: Dimension name

        Returns:
            OrderedDimension instance

        Raises:
            KeyError: If dimension not found
            TypeError: If dimension is discrete (not ordered)
        """
        dim = self.dimensions[name]
        if not isinstance(dim, OrderedDimension):
            raise TypeError(
                f"Dimension '{name}' is discrete (frozenset), not ordered. "
                f"Use get_dimension() for type-agnostic access."
            )
        return dim

    def is_ordered(self, name: str) -> bool:
        """Check if dimension is ordered.

        Args:
            name: Dimension name

        Returns:
            True if dimension is OrderedDimension, False if discrete (frozenset)

        Raises:
            KeyError: If dimension not found
        """
        return isinstance(self.dimensions[name], OrderedDimension)

    def is_discrete(self, name: str) -> bool:
        """Check if dimension is discrete.

        Args:
            name: Dimension name

        Returns:
            True if dimension is discrete (frozenset), False if ordered

        Raises:
            KeyError: If dimension not found
        """
        return isinstance(self.dimensions[name], frozenset)

    # =========================================================================
    # Delegation Methods (for OrderedDimension navigation)
    # =========================================================================

    def dim_min(self, name: str) -> int:
        """Get minimum value of ordered dimension.

        Args:
            name: Dimension name

        Returns:
            Minimum value

        Raises:
            KeyError: If dimension not found
            TypeError: If dimension is discrete
        """
        return self.get_ordered(name).min()

    def dim_max(self, name: str) -> int:
        """Get maximum value of ordered dimension.

        Args:
            name: Dimension name

        Returns:
            Maximum value

        Raises:
            KeyError: If dimension not found
            TypeError: If dimension is discrete
        """
        return self.get_ordered(name).max()

    def dim_at_percentage(
        self,
        name: str,
        percentage: float,
        rounding: str = 'natural'
    ) -> int:
        """Get value at percentage position in ordered dimension.

        Args:
            name: Dimension name
            percentage: Position in range [0.0, 1.0]
            rounding: 'natural', 'down', or 'up'

        Returns:
            Value at percentage position

        Raises:
            KeyError: If dimension not found
            TypeError: If dimension is discrete
            ValueError: If percentage out of range or invalid rounding
        """
        return self.get_ordered(name).at_percentage(percentage, rounding)

    def dim_at_index(self, name: str, idx: int) -> int:
        """Get value at index in ordered dimension.

        Args:
            name: Dimension name
            idx: Index position (supports negative indexing)

        Returns:
            Value at index

        Raises:
            KeyError: If dimension not found
            TypeError: If dimension is discrete
            IndexError: If index out of range
        """
        return self.get_ordered(name).at_index(idx)

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

            dim = self.dimensions[param_name]

            # Handle OrderedDimension
            if isinstance(dim, OrderedDimension):
                if value not in dim.values:
                    raise ValueError(
                        f"Invalid {param_name}={value}. "
                        f"Valid range: [{dim.min()}, {dim.max()}], "
                        f"values: {dim.values}"
                    )
            # Handle discrete (frozenset)
            elif isinstance(dim, frozenset):
                if value not in dim:
                    raise ValueError(
                        f"Invalid {param_name}={value}. "
                        f"Valid: {sorted(dim)}"
                    )
            else:
                # Fallback for backward compatibility (shouldn't happen)
                if value not in dim:
                    raise ValueError(f"Invalid {param_name}={value}. Valid: {dim}")

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
    """Immutable kernel instance at specific design point.

    Created by KernelDesignSpace.configure() with specific dimension values.
    Flyweight pattern minimizes memory - references parent design space,
    stores only configuration-specific data.

    Navigation methods return new instances - the design point itself is immutable.
    Use with_dimension(), with_step_up(), sweep_dimension() to explore the space.

    Attributes:
        design_space: Parent KernelDesignSpace
        inputs: Configured input interfaces (by name)
        outputs: Configured output interfaces (by name)
        config: Dimension values defining this point (e.g., {"SIMD": 16, "PE": 4})
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

    # =========================================================================
    # Navigation Methods (Immutable - Return New Instances)
    # =========================================================================

    def with_dimension(self, name: str, value: Union[int, str]) -> 'KernelDesignPoint':
        """Create new design point with specified dimension value.

        Works for both ordered and discrete dimensions.

        Args:
            name: Dimension name
            value: New value for dimension

        Returns:
            New KernelDesignPoint with updated dimension

        Raises:
            KeyError: If dimension not found
            ValueError: If value not valid for dimension

        Examples:
            >>> point = design_space.configure({"SIMD": 4, "PE": 1})
            >>> point2 = point.with_dimension("SIMD", 8)
            >>> point2.config["SIMD"]
            8
        """
        new_config = {**self.config, name: value}
        return self.design_space.configure(new_config)

    def with_min(self, name: str) -> 'KernelDesignPoint':
        """Create new design point with ordered dimension at minimum.

        Args:
            name: Dimension name (must be ordered)

        Returns:
            New KernelDesignPoint with dimension at minimum

        Raises:
            KeyError: If dimension not found
            TypeError: If dimension is discrete (not ordered)

        Examples:
            >>> point = design_space.configure({"SIMD": 8, "PE": 4})
            >>> point2 = point.with_min("SIMD")
            >>> point2.config["SIMD"]
            1
        """
        min_val = self.design_space.dim_min(name)
        return self.with_dimension(name, min_val)

    def with_max(self, name: str) -> 'KernelDesignPoint':
        """Create new design point with ordered dimension at maximum.

        Args:
            name: Dimension name (must be ordered)

        Returns:
            New KernelDesignPoint with dimension at maximum

        Raises:
            KeyError: If dimension not found
            TypeError: If dimension is discrete (not ordered)

        Examples:
            >>> point = design_space.configure({"SIMD": 8, "PE": 4})
            >>> point2 = point.with_max("SIMD")
            >>> point2.config["SIMD"]
            64
        """
        max_val = self.design_space.dim_max(name)
        return self.with_dimension(name, max_val)

    def with_percentage(
        self,
        name: str,
        percentage: float,
        rounding: str = 'natural'
    ) -> 'KernelDesignPoint':
        """Create new design point with ordered dimension at percentage.

        Args:
            name: Dimension name (must be ordered)
            percentage: Position in range [0.0, 1.0]
            rounding: 'natural', 'down', or 'up'

        Returns:
            New KernelDesignPoint with dimension at percentage

        Raises:
            KeyError: If dimension not found
            TypeError: If dimension is discrete (not ordered)
            ValueError: If percentage out of range

        Examples:
            >>> point = design_space.configure({"SIMD": 4, "PE": 1})
            >>> point2 = point.with_percentage("SIMD", 0.5)
            >>> point2.config["SIMD"]
            8
        """
        value = self.design_space.dim_at_percentage(name, percentage, rounding)
        return self.with_dimension(name, value)

    def with_step_up(self, name: str, n: int = 1) -> 'KernelDesignPoint':
        """Create new design point with ordered dimension stepped up.

        Clamps at maximum if n steps would exceed bounds.

        Args:
            name: Dimension name (must be ordered)
            n: Number of steps to move up (default 1)

        Returns:
            New KernelDesignPoint with dimension stepped up

        Raises:
            KeyError: If dimension not found
            TypeError: If dimension is discrete (not ordered)
            ValueError: If current value not in dimension or n < 0

        Examples:
            >>> point = design_space.configure({"SIMD": 4, "PE": 1})
            >>> point2 = point.with_step_up("SIMD", 2)
            >>> point2.config["SIMD"]
            16
        """
        dim = self.design_space.get_ordered(name)
        current = self.config[name]
        new_val = dim.step_up(current, n)
        return self.with_dimension(name, new_val)

    def with_step_down(self, name: str, n: int = 1) -> 'KernelDesignPoint':
        """Create new design point with ordered dimension stepped down.

        Clamps at minimum if n steps would go below bounds.

        Args:
            name: Dimension name (must be ordered)
            n: Number of steps to move down (default 1)

        Returns:
            New KernelDesignPoint with dimension stepped down

        Raises:
            KeyError: If dimension not found
            TypeError: If dimension is discrete (not ordered)
            ValueError: If current value not in dimension or n < 0

        Examples:
            >>> point = design_space.configure({"SIMD": 16, "PE": 4})
            >>> point2 = point.with_step_down("SIMD", 1)
            >>> point2.config["SIMD"]
            8
        """
        dim = self.design_space.get_ordered(name)
        current = self.config[name]
        new_val = dim.step_down(current, n)
        return self.with_dimension(name, new_val)

    # =========================================================================
    # Exploration Methods (Iterators)
    # =========================================================================

    def sweep_dimension(
        self,
        name: str,
        start: Optional[Union[int, str]] = None,
        stop: Optional[Union[int, str]] = None
    ) -> Iterator['KernelDesignPoint']:
        """Sweep through all valid values for a dimension.

        For ordered dimensions, iterates in order from start to stop.
        For discrete dimensions, iterates in sorted order (ignores start/stop).

        Args:
            name: Dimension to sweep
            start: Start value (None = use min/first), ordered dims only
            stop: Stop value (None = use max/last), ordered dims only

        Yields:
            KernelDesignPoint for each value in range

        Raises:
            KeyError: If dimension not found

        Examples:
            >>> # Full sweep (ordered)
            >>> for point in base.sweep_dimension("PE"):
            ...     evaluate(point)

            >>> # Partial sweep (ordered)
            >>> for point in base.sweep_dimension("SIMD", start=8, stop=64):
            ...     evaluate(point)

            >>> # Discrete sweep (ignores start/stop)
            >>> for point in base.sweep_dimension("ram_style"):
            ...     evaluate(point)
        """
        dim = self.design_space.get_dimension(name)

        if isinstance(dim, OrderedDimension):
            # Ordered: sweep in order from start to stop
            start_idx = 0 if start is None else dim.index_of(start)
            stop_idx = len(dim) - 1 if stop is None else dim.index_of(stop)

            for idx in range(start_idx, stop_idx + 1):
                value = dim.at_index(idx)
                yield self.with_dimension(name, value)
        else:  # frozenset
            # Discrete: iterate in sorted order
            for value in sorted(dim):
                yield self.with_dimension(name, value)

    def sweep_percentage(
        self,
        name: str,
        percentages: List[float],
        rounding: Literal['natural', 'down', 'up'] = 'natural'
    ) -> Iterator['KernelDesignPoint']:
        """Sweep through ordered dimension at specified percentage points.

        Only valid for ordered dimensions.

        Args:
            name: Ordered dimension to sweep
            percentages: List of percentage points (0.0-1.0)
            rounding: Rounding mode for fractional indices

        Yields:
            KernelDesignPoint for each percentage

        Raises:
            KeyError: If dimension not found
            TypeError: If dimension is discrete (not ordered)

        Examples:
            >>> # Quartile sweep
            >>> for point in base.sweep_percentage("PE", [0.0, 0.25, 0.5, 0.75, 1.0]):
            ...     evaluate(point)

            >>> # Decile sweep
            >>> deciles = [i/10 for i in range(11)]
            >>> for point in base.sweep_percentage("SIMD", deciles):
            ...     evaluate(point)
        """
        for pct in percentages:
            yield self.with_percentage(name, pct, rounding)

    # =========================================================================
    # Interface Stream Navigation (Index-Based)
    # =========================================================================

    def with_input_stream(self, index: int, value: int) -> 'KernelDesignPoint':
        """Set input interface stream parallelism by index.

        Convenience method for interface-agnostic parallelism navigation.
        Automatically resolves the parallelism parameter name from the interface.

        Args:
            index: Input interface index (0-based)
            value: Parallelism value

        Returns:
            New KernelDesignPoint with updated parallelism

        Raises:
            IndexError: If index out of range
            ValueError: If interface has no parallelism parameter or value invalid

        Example:
            >>> # Set first input to PE=16
            >>> point2 = point.with_input_stream(0, 16)
        """
        if index < 0 or index >= len(self.input_list):
            raise IndexError(
                f"Input index {index} out of range [0, {len(self.input_list)})"
            )

        param = self.input_list[index].design_space.parallelism_param
        if param is None:
            raise ValueError(
                f"Input interface {index} has no stream parallelism parameter"
            )

        return self.with_dimension(param, value)

    def with_output_stream(self, index: int, value: int) -> 'KernelDesignPoint':
        """Set output interface stream parallelism by index.

        Args:
            index: Output interface index (0-based)
            value: Parallelism value

        Returns:
            New KernelDesignPoint with updated parallelism

        Raises:
            IndexError: If index out of range
            ValueError: If interface has no parallelism parameter or value invalid
        """
        if index < 0 or index >= len(self.output_list):
            raise IndexError(
                f"Output index {index} out of range [0, {len(self.output_list)})"
            )

        param = self.output_list[index].design_space.parallelism_param
        if param is None:
            raise ValueError(
                f"Output interface {index} has no stream parallelism parameter"
            )

        return self.with_dimension(param, value)

    def increase_input_stream(self, index: int, n: int = 1) -> 'KernelDesignPoint':
        """Increase input stream parallelism by n steps.

        Args:
            index: Input interface index (0-based)
            n: Number of steps to increase (default 1)

        Returns:
            New KernelDesignPoint with increased parallelism

        Raises:
            IndexError: If index out of range
            ValueError: If interface has no parallelism parameter
        """
        if index < 0 or index >= len(self.input_list):
            raise IndexError(
                f"Input index {index} out of range [0, {len(self.input_list)})"
            )

        param = self.input_list[index].design_space.parallelism_param
        if param is None:
            raise ValueError(
                f"Input interface {index} has no stream parallelism parameter"
            )

        return self.increase(param, n)

    def decrease_input_stream(self, index: int, n: int = 1) -> 'KernelDesignPoint':
        """Decrease input stream parallelism by n steps.

        Args:
            index: Input interface index (0-based)
            n: Number of steps to decrease (default 1)

        Returns:
            New KernelDesignPoint with decreased parallelism

        Raises:
            IndexError: If index out of range
            ValueError: If interface has no parallelism parameter
        """
        if index < 0 or index >= len(self.input_list):
            raise IndexError(
                f"Input index {index} out of range [0, {len(self.input_list)})"
            )

        param = self.input_list[index].design_space.parallelism_param
        if param is None:
            raise ValueError(
                f"Input interface {index} has no stream parallelism parameter"
            )

        return self.decrease(param, n)

    def increase_output_stream(self, index: int, n: int = 1) -> 'KernelDesignPoint':
        """Increase output stream parallelism by n steps.

        Args:
            index: Output interface index (0-based)
            n: Number of steps to increase (default 1)

        Returns:
            New KernelDesignPoint with increased parallelism

        Raises:
            IndexError: If index out of range
            ValueError: If interface has no parallelism parameter
        """
        if index < 0 or index >= len(self.output_list):
            raise IndexError(
                f"Output index {index} out of range [0, {len(self.output_list)})"
            )

        param = self.output_list[index].design_space.parallelism_param
        if param is None:
            raise ValueError(
                f"Output interface {index} has no stream parallelism parameter"
            )

        return self.increase(param, n)

    def decrease_output_stream(self, index: int, n: int = 1) -> 'KernelDesignPoint':
        """Decrease output stream parallelism by n steps.

        Args:
            index: Output interface index (0-based)
            n: Number of steps to decrease (default 1)

        Returns:
            New KernelDesignPoint with decreased parallelism

        Raises:
            IndexError: If index out of range
            ValueError: If interface has no parallelism parameter
        """
        if index < 0 or index >= len(self.output_list):
            raise IndexError(
                f"Output index {index} out of range [0, {len(self.output_list)})"
            )

        param = self.output_list[index].design_space.parallelism_param
        if param is None:
            raise ValueError(
                f"Output interface {index} has no stream parallelism parameter"
            )

        return self.decrease(param, n)

    def with_input_stream_percentage(
        self,
        index: int,
        percentage: float,
        rounding: Literal['natural', 'down', 'up'] = 'natural'
    ) -> 'KernelDesignPoint':
        """Set input stream parallelism to percentage of range.

        Args:
            index: Input interface index (0-based)
            percentage: Value from 0.0 to 1.0 (0.0=min, 1.0=max)
            rounding: How to round fractional indices

        Returns:
            New KernelDesignPoint with parallelism at percentage

        Raises:
            IndexError: If index out of range
            ValueError: If interface has no parallelism parameter or percentage invalid
        """
        if index < 0 or index >= len(self.input_list):
            raise IndexError(
                f"Input index {index} out of range [0, {len(self.input_list)})"
            )

        param = self.input_list[index].design_space.parallelism_param
        if param is None:
            raise ValueError(
                f"Input interface {index} has no stream parallelism parameter"
            )

        return self.with_percentage(param, percentage, rounding)

    def with_output_stream_percentage(
        self,
        index: int,
        percentage: float,
        rounding: Literal['natural', 'down', 'up'] = 'natural'
    ) -> 'KernelDesignPoint':
        """Set output stream parallelism to percentage of range.

        Args:
            index: Output interface index (0-based)
            percentage: Value from 0.0 to 1.0 (0.0=min, 1.0=max)
            rounding: How to round fractional indices

        Returns:
            New KernelDesignPoint with parallelism at percentage

        Raises:
            IndexError: If index out of range
            ValueError: If interface has no parallelism parameter or percentage invalid
        """
        if index < 0 or index >= len(self.output_list):
            raise IndexError(
                f"Output index {index} out of range [0, {len(self.output_list)})"
            )

        param = self.output_list[index].design_space.parallelism_param
        if param is None:
            raise ValueError(
                f"Output interface {index} has no stream parallelism parameter"
            )

        return self.with_percentage(param, percentage, rounding)

    def sweep_input_stream(
        self,
        index: int,
        start: Optional[int] = None,
        stop: Optional[int] = None
    ) -> Iterator['KernelDesignPoint']:
        """Sweep input stream parallelism dimension.

        Args:
            index: Input interface index (0-based)
            start: Starting value (default: minimum)
            stop: Ending value inclusive (default: maximum)

        Yields:
            KernelDesignPoint for each parallelism value in range

        Raises:
            IndexError: If index out of range
            ValueError: If interface has no parallelism parameter
        """
        if index < 0 or index >= len(self.input_list):
            raise IndexError(
                f"Input index {index} out of range [0, {len(self.input_list)})"
            )

        param = self.input_list[index].design_space.parallelism_param
        if param is None:
            raise ValueError(
                f"Input interface {index} has no stream parallelism parameter"
            )

        yield from self.sweep_dimension(param, start, stop)

    def sweep_output_stream(
        self,
        index: int,
        start: Optional[int] = None,
        stop: Optional[int] = None
    ) -> Iterator['KernelDesignPoint']:
        """Sweep output stream parallelism dimension.

        Args:
            index: Output interface index (0-based)
            start: Starting value (default: minimum)
            stop: Ending value inclusive (default: maximum)

        Yields:
            KernelDesignPoint for each parallelism value in range

        Raises:
            IndexError: If index out of range
            ValueError: If interface has no parallelism parameter
        """
        if index < 0 or index >= len(self.output_list):
            raise IndexError(
                f"Output index {index} out of range [0, {len(self.output_list)})"
            )

        param = self.output_list[index].design_space.parallelism_param
        if param is None:
            raise ValueError(
                f"Output interface {index} has no stream parallelism parameter"
            )

        yield from self.sweep_dimension(param, start, stop)

    def sweep_input_stream_percentage(
        self,
        index: int,
        percentages: List[float],
        rounding: Literal['natural', 'down', 'up'] = 'natural'
    ) -> Iterator['KernelDesignPoint']:
        """Sweep input stream parallelism at percentage points.

        Args:
            index: Input interface index (0-based)
            percentages: List of percentage values (0.0 to 1.0)
            rounding: How to round fractional indices

        Yields:
            KernelDesignPoint for each percentage

        Raises:
            IndexError: If index out of range
            ValueError: If interface has no parallelism parameter
        """
        if index < 0 or index >= len(self.input_list):
            raise IndexError(
                f"Input index {index} out of range [0, {len(self.input_list)})"
            )

        param = self.input_list[index].design_space.parallelism_param
        if param is None:
            raise ValueError(
                f"Input interface {index} has no stream parallelism parameter"
            )

        yield from self.sweep_percentage(param, percentages, rounding)

    def sweep_output_stream_percentage(
        self,
        index: int,
        percentages: List[float],
        rounding: Literal['natural', 'down', 'up'] = 'natural'
    ) -> Iterator['KernelDesignPoint']:
        """Sweep output stream parallelism at percentage points.

        Args:
            index: Output interface index (0-based)
            percentages: List of percentage values (0.0 to 1.0)
            rounding: How to round fractional indices

        Yields:
            KernelDesignPoint for each percentage

        Raises:
            IndexError: If index out of range
            ValueError: If interface has no parallelism parameter
        """
        if index < 0 or index >= len(self.output_list):
            raise IndexError(
                f"Output index {index} out of range [0, {len(self.output_list)})"
            )

        param = self.output_list[index].design_space.parallelism_param
        if param is None:
            raise ValueError(
                f"Output interface {index} has no stream parallelism parameter"
            )

        yield from self.sweep_percentage(param, percentages, rounding)

    # =========================================================================
    # Interface Stream Query Helpers
    # =========================================================================

    def get_input_stream_param(self, index: int) -> Optional[str]:
        """Get parallelism parameter name for input interface.

        Args:
            index: Input interface index (0-based)

        Returns:
            Parameter name (e.g., "SIMD", "PE") or None if no parallelism

        Raises:
            IndexError: If index out of range
        """
        if index < 0 or index >= len(self.input_list):
            raise IndexError(
                f"Input index {index} out of range [0, {len(self.input_list)})"
            )

        return self.input_list[index].design_space.parallelism_param

    def get_output_stream_param(self, index: int) -> Optional[str]:
        """Get parallelism parameter name for output interface.

        Args:
            index: Output interface index (0-based)

        Returns:
            Parameter name or None if no parallelism

        Raises:
            IndexError: If index out of range
        """
        if index < 0 or index >= len(self.output_list):
            raise IndexError(
                f"Output index {index} out of range [0, {len(self.output_list)})"
            )

        return self.output_list[index].design_space.parallelism_param

    def get_input_stream_value(self, index: int) -> Optional[int]:
        """Get current parallelism value for input interface.

        Args:
            index: Input interface index (0-based)

        Returns:
            Current parallelism value or None if no parallelism

        Raises:
            IndexError: If index out of range
        """
        param = self.get_input_stream_param(index)
        return self.config.get(param) if param else None

    def get_output_stream_value(self, index: int) -> Optional[int]:
        """Get current parallelism value for output interface.

        Args:
            index: Output interface index (0-based)

        Returns:
            Current parallelism value or None if no parallelism

        Raises:
            IndexError: If index out of range
        """
        param = self.get_output_stream_param(index)
        return self.config.get(param) if param else None

    def get_input_stream_dimension(self, index: int) -> Optional['OrderedDimension']:
        """Get parallelism dimension for input interface.

        Args:
            index: Input interface index (0-based)

        Returns:
            OrderedDimension or None if no parallelism

        Raises:
            IndexError: If index out of range
        """
        if index < 0 or index >= len(self.input_list):
            raise IndexError(
                f"Input index {index} out of range [0, {len(self.input_list)})"
            )

        return self.input_list[index].design_space.parallelism_dimension

    def get_output_stream_dimension(self, index: int) -> Optional['OrderedDimension']:
        """Get parallelism dimension for output interface.

        Args:
            index: Output interface index (0-based)

        Returns:
            OrderedDimension or None if no parallelism

        Raises:
            IndexError: If index out of range
        """
        if index < 0 or index >= len(self.output_list):
            raise IndexError(
                f"Output index {index} out of range [0, {len(self.output_list)})"
            )

        return self.output_list[index].design_space.parallelism_dimension
