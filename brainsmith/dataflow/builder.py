############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
from __future__ import annotations

"""Kernel design space builder - constructs immutable models from schemas and context (two-phase).

This module separates model construction logic from KernelOp (FINN integration).
The builder can be used independently for testing, tooling, or non-FINN contexts.

Key Components:
- BuildContext: Context data for building (schema, graph, accessors)
- DesignSpaceBuilder: Orchestrates two-phase model construction

Architecture:
    KernelOp (FINN adapter) → DesignSpaceBuilder → KernelDesignSpace → KernelDesignPoint

The builder follows a two-phase flow:
1. build(): Build KernelDesignSpace (tensor shapes, block shapes, datatypes, valid ranges)
2. design_space.configure(params): Build KernelDesignPoint (stream shapes for specific params)
"""

import logging
from dataclasses import dataclass
from functools import reduce
from math import gcd
from typing import TYPE_CHECKING, Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple, Union

from qonnx.core.datatype import BaseDataType
from qonnx.core.modelwrapper import ModelWrapper

from brainsmith._internal.math import divisors
from .ordered_dimension import OrderedDimension
from .schemas import KernelSchema
from .template_resolution import resolve_template, normalize_template
from .types import VALUE_OPTIMIZED, ShapeHierarchy
from .spec_helpers import derive_datatype, value_optimized_datatype

if TYPE_CHECKING:
    from .dse_models import InterfaceDesignSpace, KernelDesignSpace

logger = logging.getLogger(__name__)


@dataclass
class BuildContext:
    """Context for building a KernelDesignSpace.

    Encapsulates all data needed to build a KernelDesignSpace from a schema.
    Makes dependencies explicit and enables testing without FINN infrastructure.

    Attributes:
        schema: KernelSchema defining structure
        model_w: ModelWrapper for ONNX graph access
        node_inputs: ONNX node input tensor names
        node_outputs: ONNX node output tensor names
        param_getter: Function to retrieve nodeattr values (e.g., get_nodeattr)
        param_setter: Function to store nodeattr values (e.g., set_nodeattr)
        node_name: Node name for error messages (optional)
    """
    schema: KernelSchema
    model_w: ModelWrapper
    node_inputs: List[str]
    node_outputs: List[str]
    param_getter: Callable[[str], Any]
    param_setter: Callable[[str, Any], None]
    node_name: str = "<unknown>"


class DesignSpaceBuilder:
    """Builds kernel design space from schema + ONNX context.

    Separates model construction logic from KernelOp (FINN integration).
    Can be used independently for testing, tooling, or non-FINN contexts.

    Design space exploration (DSE) workflow:
    1. build() creates KernelDesignSpace once (properties that don't vary)
       - Tensor shapes from ONNX graph
       - Block shapes from block_tiling templates
       - Datatypes from ONNX graph or derivation
       - Valid parallelization parameter ranges
    2. design_space.configure() creates KernelDesignPoint many times (fast)
       - Stream shapes from stream_tiling templates + parallelization params
       - Parametric constraint validation

    Example:
        >>> builder = DesignSpaceBuilder()
        >>> context = BuildContext(
        ...     schema=kernel_schema,
        ...     model_w=model_wrapper,
        ...     node_inputs=list(node.input),
        ...     node_outputs=list(node.output),
        ...     param_getter=self.get_nodeattr,
        ...     param_setter=self.set_nodeattr,
        ...     node_name=node.name
        ... )
        >>> design_space = builder.build(context)
        >>> config = design_space.configure({"SIMD": 64, "PE": 1})
    """

    def _resolve_datatype_spec(
        self,
        spec: Any,
        tensor_name: Optional[str] = None,
        fallback_datatype: Optional[BaseDataType] = None
    ) -> BaseDataType:
        """Resolve DatatypeSpec union type to concrete datatype.

        Handles all DatatypeSpec union variants:
        - None: Use fallback_datatype (from graph)
        - BaseDataType: Use as-is
        - str: Shorthand for derive_datatype(interface_name)
        - VALUE_OPTIMIZED: Optimize from tensor values
        - Callable: Custom datatype function

        Args:
            spec: DatatypeSpec to resolve
            tensor_name: ONNX tensor name for VALUE_OPTIMIZED (optional)
            fallback_datatype: Datatype to use if spec is None

        Returns:
            Resolved BaseDataType

        Raises:
            ValueError: If resolution fails
        """
        # Strategy 1: None → use fallback
        if spec is None:
            if fallback_datatype is None:
                raise ValueError("DatatypeSpec is None but no fallback_datatype provided")
            return fallback_datatype

        # Strategy 2: Concrete type → use as-is
        if isinstance(spec, BaseDataType):
            return spec

        # Strategies 3-5: Resolver functions
        try:
            resolver = self._get_datatype_resolver(spec)
            return resolver(
                self._interfaces,
                self._ctx.param_getter,
                self._ctx.model_w,
                tensor_name
            )
        except Exception as e:
            spec_type = type(spec).__name__ if not isinstance(spec, str) else f"'{spec}'"
            raise ValueError(f"Datatype resolution failed for {spec_type}: {e}") from e

    def _get_datatype_resolver(self, spec: Any) -> Callable:
        """Get resolver function for datatype spec.

        Args:
            spec: DatatypeSpec to get resolver for (str, VALUE_OPTIMIZED, or callable)

        Returns:
            Resolver function with signature (interfaces, param_getter, model, tensor_name) -> BaseDataType

        Raises:
            ValueError: If spec is not a valid resolver type
        """
        if isinstance(spec, str):
            return derive_datatype(spec)

        if spec is VALUE_OPTIMIZED:
            return value_optimized_datatype()

        if callable(spec):
            return spec

        raise ValueError(
            f"Invalid DatatypeSpec: {spec} (type {type(spec).__name__}). "
            f"Must be None, BaseDataType, str, VALUE_OPTIMIZED, or callable."
        )

    def build(self, ctx: BuildContext) -> KernelDesignSpace:
        """Build kernel design space from ONNX context.

        Resolves all properties constant across parallelization configs:
        - Tensor shapes (from ONNX graph)
        - Block shapes (from block_tiling templates)
        - Datatypes (from ONNX graph + union type derivation)
        - Internal datatypes (from union type derivation)
        - Structural constraints (validated once)
        - Valid parallelization parameter ranges (divisor sets)

        Stream shapes are left as templates for later resolution via configure().

        Args:
            ctx: Build context with ONNX node and ModelWrapper

        Returns:
            KernelDesignSpace ready for configuration exploration

        Raises:
            ValueError: If structural constraints fail
        """
        from .dse_models import InterfaceDesignSpace, KernelDesignSpace
        from .validation import DesignSpaceValidationContext

        self._ctx = ctx
        self._interfaces: Dict[str, Any] = {}

        logger.debug(f"Building KernelDesignSpace for {ctx.node_name}")

        # Build input interfaces from ONNX graph
        inputs: Dict[str, InterfaceDesignSpace] = {}

        for i, inp_name in enumerate(ctx.node_inputs):
            if not inp_name:
                continue

            if i >= len(ctx.schema.inputs):
                logger.warning(
                    f"Node has input {i} but schema only defines {len(ctx.schema.inputs)} inputs"
                )
                continue

            schema = ctx.schema.inputs[i]

            try:
                interface = self._build_interface(
                    direction='input',
                    index=i,
                    tensor_name=inp_name,
                    schema=schema
                )
                inputs[schema.name] = interface
                self._interfaces[schema.name] = interface
            except ValueError as e:
                raise ValueError(f"Failed to build input '{schema.name}': {e}") from e

        # Derive internal datatypes from inputs and parameters
        internal_datatypes = {}

        if ctx.schema.internal_datatypes:
            for internal_name, datatype_spec in ctx.schema.internal_datatypes.items():
                try:
                    # Use unified DatatypeSpec resolver (supports union types)
                    datatype = self._resolve_datatype_spec(
                        spec=datatype_spec,
                        tensor_name=None,  # Internals have no ONNX tensor
                        fallback_datatype=None  # Internal datatypes must be explicit
                    )
                    ctx.param_setter(f"{internal_name}Datatype", datatype.name)

                    # Store datatype directly (no shapes for internal datatypes)
                    self._interfaces[internal_name] = datatype
                    internal_datatypes[internal_name] = datatype

                    logger.debug(f"  Internal '{internal_name}': dtype={datatype.name}")
                except ValueError as e:
                    raise ValueError(f"Failed to resolve internal datatype '{internal_name}': {e}") from e

        # Build output interfaces (may derive datatypes from inputs)
        outputs: Dict[str, InterfaceDesignSpace] = {}

        for i, out_name in enumerate(ctx.node_outputs):
            if i >= len(ctx.schema.outputs):
                logger.warning(
                    f"Node has output {i} but schema only defines {len(ctx.schema.outputs)} outputs"
                )
                continue

            schema = ctx.schema.outputs[i]

            try:
                interface = self._build_interface(
                    direction='output',
                    index=i,
                    tensor_name=out_name,
                    schema=schema
                )
                outputs[schema.name] = interface
                self._interfaces[schema.name] = interface
            except ValueError as e:
                raise ValueError(f"Failed to build output '{schema.name}': {e}") from e

        # Separate constraints by evaluation phase (structural vs optimization)
        structural_constraints = [
            c for c in ctx.schema.constraints
            if c.evaluation_phase == 'structural'
        ]
        optimization_constraints = [
            c for c in ctx.schema.constraints
            if c.evaluation_phase != 'structural'
        ]

        logger.debug(
            f"  Split {len(ctx.schema.constraints)} constraints: "
            f"{len(structural_constraints)} structural, {len(optimization_constraints)} optimization"
        )

        # Validate structural constraints against design space
        if structural_constraints:
            validation_ctx = DesignSpaceValidationContext(
                inputs=inputs,
                outputs=outputs,
                internal_datatypes=internal_datatypes,
                param_getter=ctx.param_getter
            )

            failed = [
                f"{c.describe()}: {e}"
                for c in structural_constraints
                if (e := c.check(validation_ctx))
            ]
            if failed:
                raise ValueError(
                    f"{ctx.node_name} validation failed:\n" + "\n".join(failed)
                )

            logger.debug(f"  All {len(structural_constraints)} structural constraints passed")

        # Compute valid dimension values (tiling from divisors + DSE from schema)
        all_dimensions = self._compute_dimension_ranges(inputs, outputs, ctx.schema)

        # Assemble immutable design space model
        # Note: structural_constraints validated above but not stored (never re-validated)
        design_space = KernelDesignSpace(
            name=ctx.schema.name,
            inputs=inputs,
            outputs=outputs,
            internal_datatypes=internal_datatypes,
            optimization_constraints=optimization_constraints,
            dimensions=all_dimensions,
        )

        logger.debug(f"KernelDesignSpace built successfully for {ctx.node_name}")
        return design_space

    def _resolve_output_datatype(self, schema: Any, out_name: str) -> BaseDataType:
        """Resolve output datatype from schema or graph.

        Supports DatatypeSpec union type:
        - None: Use ONNX graph datatype (pass-through)
        - BaseDataType: Fixed datatype
        - str: Shorthand for derive_datatype(interface)
        - VALUE_OPTIMIZED: Optimize from tensor values
        - Callable: Custom datatype function

        Args:
            schema: OutputSchema
            out_name: ONNX output tensor name

        Returns:
            Resolved DataType

        Raises:
            ValueError: If datatype resolution fails
        """
        graph_dt = self._ctx.model_w.get_tensor_datatype(out_name)

        # Use unified DatatypeSpec resolver
        derived_dt = self._resolve_datatype_spec(
            spec=schema.datatype,
            tensor_name=out_name,
            fallback_datatype=graph_dt  # Use graph datatype if spec is None
        )

        # Log if schema overrides graph
        if schema.datatype is not None and derived_dt != graph_dt:
            logger.info(
                f"Output '{schema.name}' datatype: schema derived {derived_dt.name}, "
                f"graph has {graph_dt.name} - using schema"
            )

        return derived_dt

    def _resolve_input_datatype(self, schema: Any, inp_name: str) -> BaseDataType:
        """Resolve input datatype from schema or graph.

        Supports DatatypeSpec union type:
        - None: Use ONNX graph datatype (default)
        - BaseDataType: Fixed datatype
        - str: Shorthand for derive_datatype(interface)
        - VALUE_OPTIMIZED: Optimize from tensor values
        - Callable: Custom datatype function

        Args:
            schema: InputSchema
            inp_name: ONNX input tensor name

        Returns:
            Resolved DataType

        Raises:
            ValueError: If datatype resolution fails
        """
        graph_dt = self._ctx.model_w.get_tensor_datatype(inp_name)

        # Use unified DatatypeSpec resolver
        derived_dt = self._resolve_datatype_spec(
            spec=schema.datatype,
            tensor_name=inp_name,
            fallback_datatype=graph_dt  # Use graph datatype if spec is None
        )

        # Log if schema optimizes graph datatype
        if schema.datatype is not None and derived_dt != graph_dt:
            logger.info(
                f"Input '{schema.name}': schema optimized {graph_dt.name} → {derived_dt.name}"
            )

        return derived_dt

    def _build_interface(
        self,
        direction: str,
        index: int,
        tensor_name: str,
        schema: Any,
    ) -> InterfaceDesignSpace:
        """Build single interface design space from ONNX tensor and schema.

        Args:
            direction: 'input' or 'output' (for error messages and nodeattr naming)
            index: Tensor index in node's input/output list
            tensor_name: ONNX tensor name
            schema: Input/OutputSchema defining structure

        Returns:
            InterfaceDesignSpace with resolved shapes and normalized stream_tiling

        Raises:
            ValueError: If building fails
        """
        from .dse_models import InterfaceDesignSpace

        ctx = self._ctx

        # Resolve datatype (both inputs and outputs can derive)
        if direction == 'input':
            datatype = self._resolve_input_datatype(schema, tensor_name)
        else:  # output
            datatype = self._resolve_output_datatype(schema, tensor_name)

        # Store datatype to nodeattrs for FINN
        ctx.param_setter(f"{direction}{index}Datatype", datatype.name)

        # Get tensor shape from graph
        tensor_shape = tuple(ctx.model_w.get_tensor_shape(tensor_name))

        # Resolve block shape from template
        try:
            block_shape = resolve_template(
                schema.block_tiling,
                tensor_shape,
                ctx.param_getter,
                self._interfaces,
                ctx.model_w,
                tensor_name,
                hierarchy=ShapeHierarchy.BLOCK  # Explicit: tuple shorthand uses BLOCK hierarchy
            )
        except ValueError as e:
            raise ValueError(
                f"Failed to resolve block_tiling for {direction} '{schema.name}': {e}"
            ) from e

        # Normalize stream_tiling to match block_shape rank
        normalized_stream_tiling = None
        if schema.stream_tiling is not None:
            try:
                normalized_stream_tiling = normalize_template(
                    schema.stream_tiling, block_shape
                )
            except ValueError as e:
                raise ValueError(
                    f"Failed to normalize stream_tiling for {direction} '{schema.name}': {e}"
                ) from e

        # Infer is_weight from ONNX initializer (only for inputs)
        is_weight = (
            direction == 'input' and
            ctx.model_w.get_initializer(tensor_name) is not None
        )

        # Build and return interface
        interface = InterfaceDesignSpace(
            name=schema.name,
            tensor_shape=tensor_shape,
            block_shape=block_shape,
            stream_tiling=normalized_stream_tiling,
            datatype=datatype,
            is_weight=is_weight,
            tensor_name=tensor_name
        )

        logger.debug(
            f"  {direction.capitalize()} '{schema.name}': tensor={tensor_shape}, "
            f"block={block_shape}, stream={schema.stream_tiling}, dtype={datatype.name}"
        )

        return interface

    # =========================================================================
    # Helper Methods for Two-Phase Construction
    # =========================================================================

    def _divisors(self, n: int) -> Set[int]:
        """Compute all divisors of a positive integer (wrapper for testing).

        Args:
            n: Positive integer

        Returns:
            Set of all divisors of n

        Raises:
            ValueError: If n is non-positive
        """
        return divisors(n)

    def _compute_dimension_ranges(
        self,
        inputs: Dict[str, InterfaceDesignSpace],
        outputs: Dict[str, InterfaceDesignSpace],
        schema: 'KernelSchema',
    ) -> Dict[str, Union[OrderedDimension, FrozenSet]]:
        """Compute valid values for all explorable dimensions (tiling + DSE).

        Combines:
        1. Tiling dimensions (PE, SIMD) - computed as divisors, wrapped in OrderedDimension
        2. DSE dimensions - from schema.dse_dimensions, auto-detected as ordered or discrete

        Tiling dimension logic:
        - A tiling parameter is any string appearing in stream_tiling
        - Valid values are divisors of the corresponding block dimension
        - For multi-dimensional cases, if a parameter appears in multiple
          dimensions or interfaces, valid values are divisors of GCD of all
          block dimensions where the parameter appears
        - Always wrapped in OrderedDimension (ordered sequences)

        DSE dimension auto-detection:
        - list/tuple → OrderedDimension (ordered sequences with navigation)
        - set/frozenset → frozenset (discrete categories)

        Example (tiling):
            stream_tiling=["SIMD"], block_shape=(768,)
            → SIMD must divide 768
            → OrderedDimension("SIMD", (1, 2, 3, 4, 6, 8, 12, 16, ..., 768))

        Example (ordered DSE):
            dse_dimensions={"depth": DSEDimension("depth", [128, 256, 512, 1024])}
            → OrderedDimension("depth", (128, 256, 512, 1024))

        Example (discrete DSE):
            dse_dimensions={"ram_style": DSEDimension("ram_style", {"distributed", "block"})}
            → frozenset({"distributed", "block"})

        Args:
            inputs: Input interfaces with resolved block shapes (dict or list)
            outputs: Output interfaces with resolved block shapes (dict or list)
            schema: KernelSchema containing dse_dimensions

        Returns:
            Dict mapping dimension name to OrderedDimension or frozenset
            Example: {
                "SIMD": OrderedDimension("SIMD", (1, 2, 3, 4, 6, 8)),
                "ram_style": frozenset({"distributed", "block"})
            }

        Raises:
            ValueError: If parameter appears in block_tiling (violates R1 from spec)
        """
        # Collect all block dimensions that each parameter must divide
        param_constraints = {}  # param_name -> list of block dimensions

        # Handle both dict and list inputs (for tests)
        inputs_iter = inputs.values() if isinstance(inputs, dict) else inputs
        outputs_iter = outputs.values() if isinstance(outputs, dict) else outputs
        all_interfaces = (*inputs_iter, *outputs_iter)

        for interface in all_interfaces:
            if interface.stream_tiling is None:
                continue

            for dim_idx, spec_elem in enumerate(interface.stream_tiling):
                # Check if this is a parallelization parameter (string)
                if isinstance(spec_elem, str):
                    param_name = spec_elem
                    block_dim = interface.block_shape[dim_idx]

                    if param_name not in param_constraints:
                        param_constraints[param_name] = []
                    param_constraints[param_name].append(block_dim)

                # DerivedDim doesn't introduce new parameters
                # Literals (1, FULL_DIM) don't create parameters

        # Each tiling param must divide GCD of all block dims where it appears
        # Wrap in OrderedDimension (always ordered sequences)
        tiling_dimensions = {}
        for param_name, block_dims in param_constraints.items():
            gcd_value = reduce(gcd, block_dims)
            divisor_list = sorted(divisors(gcd_value))
            tiling_dimensions[param_name] = OrderedDimension(
                name=param_name,
                values=tuple(divisor_list),
                default=None  # Will use minimum (first value)
            )

        logger.debug(
            f"Computed {len(tiling_dimensions)} tiling dimensions: "
            + ", ".join(f"{k}={len(v)} values" for k, v in tiling_dimensions.items())
        )

        # Add DSE dimensions from schema (auto-detect ordered vs discrete)
        dse_dimensions = {}
        for dim_name, dim_spec in schema.dse_dimensions.items():
            # Evaluate values (callable or direct)
            if callable(dim_spec.values):
                values = dim_spec.values(self._ctx)
            else:
                values = dim_spec.values

            # Auto-detect type based on container type
            if isinstance(values, (list, tuple)):
                # Ordered sequence → OrderedDimension
                dse_dimensions[dim_name] = OrderedDimension(
                    name=dim_name,
                    values=tuple(sorted(values)),  # Ensure sorted
                    default=dim_spec.default if hasattr(dim_spec, 'default') else None
                )
            elif isinstance(values, (set, frozenset)):
                # Discrete set → frozenset
                dse_dimensions[dim_name] = frozenset(values)
            else:
                # Fallback: treat as discrete
                logger.warning(
                    f"DSE dimension '{dim_name}' has unexpected type {type(values)}. "
                    f"Treating as discrete (frozenset)."
                )
                dse_dimensions[dim_name] = frozenset(values)

        if dse_dimensions:
            ordered_count = sum(1 for v in dse_dimensions.values() if isinstance(v, OrderedDimension))
            discrete_count = sum(1 for v in dse_dimensions.values() if isinstance(v, frozenset))
            logger.debug(
                f"Added {len(dse_dimensions)} DSE dimensions: "
                f"{ordered_count} ordered, {discrete_count} discrete"
            )

        # Combine tiling + DSE dimensions
        all_dimensions = {**tiling_dimensions, **dse_dimensions}

        return all_dimensions


__all__ = [
    'BuildContext',
    'DesignSpaceBuilder',
]
