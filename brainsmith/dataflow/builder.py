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
    KernelOp (FINN adapter) → DesignSpaceBuilder → KernelDesignSpace → KernelInstance

The builder follows a two-phase flow:
1. build(): Build KernelDesignSpace (tensor shapes, block shapes, datatypes, valid ranges)
2. design_space.configure(params): Build KernelInstance (stream shapes for specific params)
"""

import logging
from dataclasses import dataclass
from functools import reduce
from math import gcd
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

from qonnx.core.datatype import BaseDataType
from qonnx.core.modelwrapper import ModelWrapper

from brainsmith.utils import divisors
from .schemas import KernelSchema
from .template_resolution import resolve_template, normalize_template
from .datatype_sources import DatatypeSource

if TYPE_CHECKING:
    from .models import InterfaceDesignSpace, KernelDesignSpace

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
    2. design_space.configure() creates KernelInstance many times (fast)
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

    def build(self, ctx: BuildContext) -> KernelDesignSpace:
        """Build kernel design space from ONNX context.

        Resolves all properties constant across parallelization configs:
        - Tensor shapes (from ONNX graph)
        - Block shapes (from block_tiling templates)
        - Datatypes (from ONNX graph + DatatypeSource derivation)
        - Internal datatypes (from DatatypeSource)
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
        from .models import InterfaceDesignSpace, KernelDesignSpace
        from .validation import DesignSpaceValidationContext

        self._ctx = ctx
        self._interfaces: Dict[str, Any] = {}

        logger.debug(f"Building KernelDesignSpace for {ctx.node_name}")

        # Build input interfaces from ONNX graph
        interfaces_input: Dict[str, InterfaceDesignSpace] = {}

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
                interfaces_input[schema.name] = interface
                self._interfaces[schema.name] = interface
            except ValueError as e:
                raise ValueError(f"Failed to build input '{schema.name}': {e}") from e

        # Derive internal datatypes from inputs and parameters
        internal_datatypes = {}

        if ctx.schema.internal_datatypes:
            for internal_name, datatype_source in ctx.schema.internal_datatypes.items():
                try:
                    datatype = datatype_source.resolve(self._interfaces, ctx.param_getter)
                    ctx.param_setter(f"{internal_name}Datatype", datatype.name)

                    # Store datatype directly (no shapes for internal datatypes)
                    self._interfaces[internal_name] = datatype
                    internal_datatypes[internal_name] = datatype

                    logger.debug(f"  Internal '{internal_name}': dtype={datatype.name}")
                except ValueError as e:
                    raise ValueError(f"Failed to resolve internal datatype '{internal_name}': {e}") from e

        # Build output interfaces (may derive datatypes from inputs)
        interfaces_output: Dict[str, InterfaceDesignSpace] = {}

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
                interfaces_output[schema.name] = interface
                self._interfaces[schema.name] = interface
            except ValueError as e:
                raise ValueError(f"Failed to build output '{schema.name}': {e}") from e

        # Separate constraints by evaluation phase (structural vs parametric)
        structural_constraints = [
            c for c in ctx.schema.constraints
            if c.evaluation_phase == 'structural'
        ]
        parametric_constraints = [
            c for c in ctx.schema.constraints
            if c.evaluation_phase != 'structural'
        ]

        logger.debug(
            f"  Split {len(ctx.schema.constraints)} constraints: "
            f"{len(structural_constraints)} structural, {len(parametric_constraints)} parametric"
        )

        # Validate structural constraints against design space
        if structural_constraints:
            validation_ctx = DesignSpaceValidationContext(
                inputs=interfaces_input,
                outputs=interfaces_output,
                internal_datatypes=internal_datatypes,
                param_getter=ctx.param_getter
            )

            errors = []
            for constraint in structural_constraints:
                error = constraint.check(validation_ctx)
                if error is not None:
                    errors.append(f"  - {constraint.describe()}: {error}")

            if errors:
                error_msg = f"Design space validation failed for {ctx.node_name}:\n" + "\n".join(errors)
                raise ValueError(error_msg)

            logger.debug(f"  All {len(structural_constraints)} structural constraints passed")

        # Compute valid parameter values as divisors of block dimensions
        valid_ranges = self._compute_parameter_ranges(interfaces_input, interfaces_output)

        # Assemble immutable design space model
        # Note: structural_constraints validated above but not stored (never re-validated)
        design_space = KernelDesignSpace(
            name=ctx.schema.name,
            inputs=interfaces_input,
            outputs=interfaces_output,
            internal_datatypes=internal_datatypes,
            parametric_constraints=parametric_constraints,
            parallelization_params=valid_ranges,
        )

        logger.debug(f"KernelDesignSpace built successfully for {ctx.node_name}")
        return design_space

    def _resolve_output_datatype(self, schema: Any, out_name: str) -> BaseDataType:
        """Resolve output datatype from schema or graph.

        Three cases:
        1. schema.datatype is None: Use datatype from ONNX graph (pass-through)
        2. schema.datatype is DatatypeSource: Derive from inputs/internals
        3. schema.datatype is DataType: Use fixed datatype (rare)

        Args:
            schema: OutputSchema
            out_name: ONNX output tensor name

        Returns:
            Resolved DataType

        Raises:
            ValueError: If datatype resolution fails
        """
        if schema.datatype is None:
            # No schema derivation - use graph datatype (pass-through)
            return self._ctx.model_w.get_tensor_datatype(out_name)

        if isinstance(schema.datatype, DatatypeSource):
            # Derive from inputs or internal datatypes
            derived_dt = schema.datatype.resolve(self._interfaces, self._ctx.param_getter)
            graph_dt = self._ctx.model_w.get_tensor_datatype(out_name)

            if derived_dt != graph_dt:
                logger.info(
                    f"Output '{schema.name}' datatype: schema derived {derived_dt.name}, "
                    f"graph has {graph_dt.name} - using schema"
                )

            return derived_dt

        # Fixed datatype specified in schema (rare case)
        return schema.datatype

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
        from .models import InterfaceDesignSpace

        ctx = self._ctx

        # Resolve datatype (differs for inputs vs outputs)
        if direction == 'input':
            datatype = ctx.model_w.get_tensor_datatype(tensor_name)
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
                self._interfaces
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
            is_weight=is_weight
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

    def _compute_parameter_ranges(
        self,
        interfaces_input: Dict[str, InterfaceDesignSpace],
        interfaces_output: Dict[str, InterfaceDesignSpace],
    ) -> Dict[str, Set[int]]:
        """Compute valid divisor sets for each parallelization parameter.

        A parallelization parameter is any string appearing in stream_tiling.
        Valid values are divisors of the corresponding block dimension.

        For multi-dimensional cases, if a parameter appears in multiple
        dimensions or interfaces, valid values are divisors of GCD of all
        block dimensions where the parameter appears.

        Example (single appearance):
            stream_tiling=["SIMD"], block_shape=(768,)
            → SIMD must divide 768
            → valid SIMD = {1, 2, 3, 4, 6, 8, 12, 16, ..., 768}

        Example (multiple appearances):
            input: stream_tiling=["PE"], block_shape=(256,)
            output: stream_tiling=["PE"], block_shape=(512,)
            → PE must divide both 256 and 512
            → PE must divide gcd(256, 512) = 256
            → valid PE = {1, 2, 4, 8, 16, 32, 64, 128, 256}

        Args:
            interfaces_input: Input interfaces with resolved block shapes (dict or list)
            interfaces_output: Output interfaces with resolved block shapes (dict or list)

        Returns:
            Dict mapping parameter name to set of valid divisors
            Example: {"SIMD": {1, 2, 3, 4, 6, 8, ..., 768}, "PE": {1, 2, 4, 8}}

        Raises:
            ValueError: If parameter appears in block_tiling (violates R1 from spec)
        """
        # Collect all block dimensions that each parameter must divide
        param_constraints = {}  # param_name -> list of block dimensions

        # Handle both dict and list inputs (for tests)
        inputs_iter = interfaces_input.values() if isinstance(interfaces_input, dict) else interfaces_input
        outputs_iter = interfaces_output.values() if isinstance(interfaces_output, dict) else interfaces_output
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

        # Each param must divide GCD of all block dims where it appears
        valid_ranges = {}
        for param_name, block_dims in param_constraints.items():
            gcd_value = reduce(gcd, block_dims)
            valid_ranges[param_name] = divisors(gcd_value)

        logger.debug(
            f"Computed valid ranges for {len(valid_ranges)} parameters: "
            + ", ".join(f"{k}={len(v)} values" for k, v in valid_ranges.items())
        )

        return valid_ranges


# =============================================================================
# Module-Level Build Function (Recommended Entry Point)
# =============================================================================

def build_kernel_design_space(ctx: BuildContext) -> KernelDesignSpace:
    """Build kernel design space from ONNX context.

    This is the recommended way to build a design space. Creates a temporary
    DesignSpaceBuilder instance for this one build operation, then discards it.

    The builder is stateless - it only uses temporary state (_ctx, _interfaces)
    during the build process. There's no benefit to caching builder instances.

    Args:
        ctx: Build context with schema, ONNX graph data, and accessors

    Returns:
        Immutable KernelDesignSpace ready for configuration exploration

    Raises:
        ValueError: If structural constraints fail or build cannot complete

    Example:
        >>> ctx = BuildContext(
        ...     schema=kernel_schema,
        ...     model_w=model_wrapper,
        ...     node_inputs=list(node.input),
        ...     node_outputs=list(node.output),
        ...     param_getter=get_nodeattr,
        ...     param_setter=set_nodeattr,
        ...     node_name=node.name
        ... )
        >>> design_space = build_kernel_design_space(ctx)
        >>> config = design_space.configure({"SIMD": 64, "PE": 1})
    """
    from .models import KernelDesignSpace
    return DesignSpaceBuilder().build(ctx)


__all__ = [
    'BuildContext',
    'DesignSpaceBuilder',
    'build_kernel_design_space',
]
