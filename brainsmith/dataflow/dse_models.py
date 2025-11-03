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
    """Interface definition for kernel design space.

    Built once from ONNX context, reused across all parallelization configs during DSE.
    Stream tiling preserved as template for later resolution with specific parameters.

    Parallelism metadata (linked during build):
    - parallelism_dimension: Shared reference to OrderedDimension from kernel dimensions dict
    - parallelism_param: Parameter name (e.g., "SIMD") for navigation

    Single-param only for now (multi-param will flatten to synthetic 1D in future).
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


class NavigableInterface:
    """Navigable interface view with parallelism control.

    Lightweight wrapper that combines InterfaceDesignPoint (structure)
    with KernelDesignPoint (configuration) to enable interface-agnostic
    parallelism navigation.

    Provides ergonomic API for controlling parallelism without knowing
    kernel-specific parameter names:
    - Query: `parallelism`, `parallelism_dimension`, `has_parallelism`
    - Navigate: `with_parallelism()`, `increase/decrease_parallelism()`, min/max
    - Explore: `sweep_parallelism()`, `sweep_parallelism_percentage()`

    All navigation methods return new KernelDesignPoint instances (immutable).

    Example:
        >>> # Interface-agnostic access (don't need to know param name)
        >>> point.input[0].parallelism  # Current value
        64
        >>> point.input[0].parallelism_dimension  # Dimension spec
        OrderedDimension(name='SIMD', values=(1, 2, 4, 8, 16, 32, 64, 128))

        >>> # Navigation (returns new kernel points)
        >>> point2 = point.input[0].with_parallelism(16)
        >>> point3 = point.input[0].increase_parallelism(2)

        >>> # Exploration
        >>> for p in point.input[0].sweep_parallelism():
        ...     print(p.input[0].parallelism)
        1
        2
        4
        ...
    """

    def __init__(self, interface_point: InterfaceDesignPoint, kernel_point: 'KernelDesignPoint'):
        """Create navigable interface wrapper.

        Args:
            interface_point: Interface structure (shape hierarchy, datatype)
            kernel_point: Kernel configuration (for navigation context)
        """
        self._point = interface_point
        self._kernel = kernel_point

    # =========================================================================
    # Structural Properties (delegated to InterfaceDesignPoint)
    # =========================================================================

    @property
    def name(self) -> str:
        """Interface name."""
        return self._point.name

    @property
    def tensor_shape(self) -> Shape:
        """Tensor shape (full unfolded shape)."""
        return self._point.tensor_shape

    @property
    def block_shape(self) -> Shape:
        """Block shape (spatial tile size)."""
        return self._point.block_shape

    @property
    def stream_shape(self) -> Shape:
        """Stream shape (parallelized bandwidth per cycle)."""
        return self._point.stream_shape

    @property
    def datatype(self) -> BaseDataType:
        """Interface datatype."""
        return self._point.datatype

    @property
    def is_weight(self) -> bool:
        """Whether interface is weight (constant tensor)."""
        return self._point.is_weight

    def get_shape(self, hierarchy: ShapeHierarchy) -> Shape:
        """Get shape at specified hierarchy level."""
        return self._point.get_shape(hierarchy)

    @property
    def tensor_blocks_shape(self) -> Shape:
        """Per-dimension blocks needed to tile tensor."""
        return self._point.tensor_blocks_shape

    @property
    def stream_cycles_shape(self) -> Shape:
        """Per-dimension cycles needed to stream one block."""
        return self._point.stream_cycles_shape

    @property
    def tensor_folding_factor(self) -> int:
        """Number of blocks needed to cover full tensor."""
        return self._point.tensor_folding_factor

    @property
    def block_folding_factor(self) -> int:
        """Cycles to stream one block."""
        return self._point.block_folding_factor

    @property
    def streaming_bandwidth(self) -> int:
        """Elements streamed per cycle."""
        return self._point.streaming_bandwidth

    @property
    def stream_width_bits(self) -> int:
        """Stream width in bits."""
        return self._point.stream_width_bits

    # =========================================================================
    # Parallelism Query Properties
    # =========================================================================

    @property
    def has_parallelism(self) -> bool:
        """Check if interface has parallelism dimension.

        Returns:
            True if interface has stream parallelism parameter, False otherwise
        """
        return self._point.design_space.parallelism_dimension is not None

    @property
    def parallelism(self) -> Optional[int]:
        """Current parallelism value.

        Returns:
            Current parallelism parameter value, or None if no parallelism
        """
        if self._point.design_space.parallelism_param is None:
            return None
        return self._kernel.config[self._point.design_space.parallelism_param]

    @property
    def parallelism_dimension(self) -> Optional[OrderedDimension]:
        """Parallelism dimension specification.

        Returns:
            OrderedDimension with valid values and navigation methods,
            or None if interface has no parallelism
        """
        return self._point.design_space.parallelism_dimension

    # =========================================================================
    # Navigation Methods (return new KernelDesignPoint)
    # =========================================================================

    def with_parallelism(self, value: int) -> 'KernelDesignPoint':
        """Set parallelism to specific value.

        Args:
            value: Target parallelism value (must be in dimension's valid values)

        Returns:
            New KernelDesignPoint with updated parallelism

        Raises:
            ValueError: If interface has no parallelism or value is invalid
        """
        self._require_parallelism()
        param = self._point.design_space.parallelism_param
        return self._kernel.with_dimension(param, value)

    def with_min_parallelism(self) -> 'KernelDesignPoint':
        """Set parallelism to minimum value.

        Returns:
            New KernelDesignPoint with minimum parallelism

        Raises:
            ValueError: If interface has no parallelism
        """
        self._require_parallelism()
        param = self._point.design_space.parallelism_param
        return self._kernel.with_min(param)

    def with_max_parallelism(self) -> 'KernelDesignPoint':
        """Set parallelism to maximum value.

        Returns:
            New KernelDesignPoint with maximum parallelism

        Raises:
            ValueError: If interface has no parallelism
        """
        self._require_parallelism()
        param = self._point.design_space.parallelism_param
        return self._kernel.with_max(param)

    def increase_parallelism(self, n: int = 1) -> 'KernelDesignPoint':
        """Increase parallelism by n steps.

        Args:
            n: Number of steps to increase (default 1)

        Returns:
            New KernelDesignPoint with increased parallelism
            (clamped to maximum if beyond range)

        Raises:
            ValueError: If interface has no parallelism
        """
        self._require_parallelism()
        dim = self.parallelism_dimension
        new_val = dim.step_up(self.parallelism, n)
        return self.with_parallelism(new_val)

    def decrease_parallelism(self, n: int = 1) -> 'KernelDesignPoint':
        """Decrease parallelism by n steps.

        Args:
            n: Number of steps to decrease (default 1)

        Returns:
            New KernelDesignPoint with decreased parallelism
            (clamped to minimum if beyond range)

        Raises:
            ValueError: If interface has no parallelism
        """
        self._require_parallelism()
        dim = self.parallelism_dimension
        new_val = dim.step_down(self.parallelism, n)
        return self.with_parallelism(new_val)

    def with_parallelism_percentage(
        self,
        percentage: float,
        rounding: Literal['natural', 'down', 'up'] = 'natural'
    ) -> 'KernelDesignPoint':
        """Set parallelism to percentage of range.

        Args:
            percentage: Value from 0.0 to 1.0 (0.0=min, 1.0=max)
            rounding: How to round fractional indices
                - 'natural': round() to nearest (default)
                - 'down': floor() to lower value
                - 'up': ceil() to higher value

        Returns:
            New KernelDesignPoint with parallelism at percentage

        Raises:
            ValueError: If interface has no parallelism or percentage invalid
        """
        self._require_parallelism()
        dim = self.parallelism_dimension
        value = dim.at_percentage(percentage, rounding)
        return self.with_parallelism(value)

    # =========================================================================
    # Exploration Methods (yield KernelDesignPoint)
    # =========================================================================

    def sweep_parallelism(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None
    ) -> Iterator['KernelDesignPoint']:
        """Sweep parallelism dimension.

        Args:
            start: Starting value (default: minimum)
            stop: Ending value inclusive (default: maximum)

        Yields:
            KernelDesignPoint for each parallelism value in range

        Raises:
            ValueError: If interface has no parallelism
        """
        self._require_parallelism()
        param = self._point.design_space.parallelism_param
        yield from self._kernel.sweep_dimension(param, start, stop)

    def sweep_parallelism_percentage(
        self,
        percentages: List[float],
        rounding: Literal['natural', 'down', 'up'] = 'natural'
    ) -> Iterator['KernelDesignPoint']:
        """Sweep parallelism at percentage points.

        Args:
            percentages: List of percentage values (0.0 to 1.0)
            rounding: How to round fractional indices

        Yields:
            KernelDesignPoint for each percentage

        Raises:
            ValueError: If interface has no parallelism
        """
        self._require_parallelism()
        param = self._point.design_space.parallelism_param
        yield from self._kernel.sweep_percentage(param, percentages, rounding)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _require_parallelism(self) -> None:
        """Validate that interface has parallelism dimension.

        Raises:
            ValueError: If interface has no parallelism
        """
        if not self.has_parallelism:
            raise ValueError(
                f"Interface '{self.name}' has no parallelism dimension. "
                f"Cannot navigate parallelism for interfaces without stream parameters."
            )


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

    @property
    def input(self) -> List[NavigableInterface]:
        """Navigable input interfaces with parallelism control.

        Returns list of NavigableInterface wrappers that combine structure
        (from InterfaceDesignPoint) with configuration (from KernelDesignPoint)
        to enable interface-agnostic parallelism navigation.

        Use this property when you want to control parallelism without knowing
        kernel-specific parameter names (SIMD, PE, MW, MH, etc.).

        Examples:
            >>> # Query parallelism (interface-agnostic)
            >>> point.input[0].parallelism  # Current value
            64
            >>> point.input[0].parallelism_dimension  # Dimension spec
            OrderedDimension(name='SIMD', values=(1, 2, 4, 8, 16, 32, 64, 128))

            >>> # Navigate parallelism (don't need to know param name!)
            >>> point2 = point.input[0].with_parallelism(16)  # Set to 16
            >>> point3 = point.input[0].increase_parallelism()  # Step up
            >>> point4 = point.input[0].with_min_parallelism()  # Minimum

            >>> # Explore parallelism
            >>> for p in point.input[0].sweep_parallelism():
            ...     print(p.input[0].parallelism)  # All valid values

        Note:
            Use `input_list` for backward compatibility or when you don't need
            parallelism navigation (e.g., just reading shapes/datatypes).
        """
        return [NavigableInterface(iface, self) for iface in self.input_list]

    @property
    def output(self) -> List[NavigableInterface]:
        """Navigable output interfaces with parallelism control.

        Returns list of NavigableInterface wrappers. See `input` property
        documentation for usage examples.

        Note:
            Output parallelism is less common than input parallelism, but
            available for kernels that do parallelize outputs (e.g., Split).
        """
        return [NavigableInterface(iface, self) for iface in self.output_list]

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
