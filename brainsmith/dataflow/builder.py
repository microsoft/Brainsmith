############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Kernel design space builder - constructs immutable models from schemas and context (two-phase).

This module separates model construction logic from KernelOp (FINN integration).
The builder can be used independently for testing, tooling, or non-FINN contexts.

Key Components:
- BuildContext: Context data for building (schema, graph, accessors)
- KernelModelBuilder: Orchestrates two-phase model construction

Architecture:
    KernelOp (FINN adapter) → KernelModelBuilder → KernelDesignSpace → KernelConfiguration

The builder follows a two-phase flow:
1. build(): Build KernelDesignSpace (tensor shapes, block shapes, datatypes, valid ranges)
2. design_space.configure(params): Build KernelConfiguration (stream shapes for specific params)
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from qonnx.core.datatype import BaseDataType
from qonnx.core.modelwrapper import ModelWrapper

from brainsmith.utils import divisors
from .schemas import KernelSchema
from .template_resolution import resolve_template, normalize_template
from .datatype_sources import DatatypeSource

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


class KernelModelBuilder:
    """Builds kernel design space from schema + ONNX context.

    Separates model construction logic from KernelOp (FINN integration).
    Can be used independently for testing, tooling, or non-FINN contexts.

    Design space exploration (DSE) workflow:
    1. build() creates KernelDesignSpace once (properties that don't vary)
       - Tensor shapes from ONNX graph
       - Block shapes from block_tiling templates
       - Datatypes from ONNX graph or derivation
       - Valid parallelization parameter ranges
    2. design_space.configure() creates KernelConfiguration many times (fast)
       - Stream shapes from stream_tiling templates + parallelization params
       - Parametric constraint validation

    Example:
        >>> builder = KernelModelBuilder()
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

    def build(self, ctx: BuildContext) -> 'KernelDesignSpace':
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

        # Phase 1: Build design space input models
        design_space_inputs = {}  # Dict[str, InterfaceDesignSpace]

        for i, inp_name in enumerate(ctx.node_inputs):
            if not inp_name:
                continue

            schema = ctx.schema.inputs[i]

            try:
                # Get datatype and tensor shape from graph
                datatype = ctx.model_w.get_tensor_datatype(inp_name)
                tensor_shape = tuple(ctx.model_w.get_tensor_shape(inp_name))

                # Store datatype to nodeattrs for FINN
                ctx.param_setter(f"input{i}Datatype", datatype.name)

                # Resolve block shape from template
                block_shape = resolve_template(
                    schema.block_tiling,
                    tensor_shape,
                    ctx.param_getter,
                    self._interfaces
                )

                # Infer is_weight from ONNX initializer presence
                is_weight = ctx.model_w.get_initializer(inp_name) is not None

                # Normalize stream_tiling to match block_shape rank
                normalized_stream_tiling = None
                if schema.stream_tiling is not None:
                    normalized_stream_tiling = normalize_template(
                        schema.stream_tiling, block_shape
                    )

                # Build InterfaceDesignSpace (stream_tiling normalized but not resolved)
                ds_input = InterfaceDesignSpace(
                    name=schema.name,
                    tensor_shape=tensor_shape,
                    block_shape=block_shape,
                    stream_tiling=normalized_stream_tiling,  # Normalized template, values not resolved
                    datatype=datatype,
                    is_weight=is_weight
                )

                design_space_inputs[schema.name] = ds_input  # Store in dict by name
                self._interfaces[schema.name] = ds_input  # Store for derivations

                logger.debug(
                    f"  Input '{schema.name}': tensor={tensor_shape}, "
                    f"block={block_shape}, stream_tiling={schema.stream_tiling}, dtype={datatype.name}"
                )
            except ValueError as e:
                raise ValueError(f"Input '{schema.name}': {e}") from e

        # Phase 2: Resolve internal datatypes
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
                    raise ValueError(f"Internal datatype '{internal_name}': {e}") from e

        # Phase 3: Build design space output models
        design_space_outputs = {}  # Dict[str, InterfaceDesignSpace]

        for i, out_name in enumerate(ctx.node_outputs):
            schema = ctx.schema.outputs[i]

            try:
                # Resolve or extract datatype
                datatype = self._resolve_output_datatype(schema, out_name)

                # Store datatype to nodeattrs for FINN
                ctx.param_setter(f"output{i}Datatype", datatype.name)

                # Get tensor shape from graph
                tensor_shape = tuple(ctx.model_w.get_tensor_shape(out_name))

                # Resolve block shape from template
                block_shape = resolve_template(
                    schema.block_tiling,
                    tensor_shape,
                    ctx.param_getter,
                    self._interfaces
                )

                # Normalize stream_tiling to match block_shape rank
                normalized_stream_tiling = None
                if schema.stream_tiling is not None:
                    normalized_stream_tiling = normalize_template(
                        schema.stream_tiling, block_shape
                    )

                # Build InterfaceDesignSpace (stream_tiling normalized but not resolved)
                ds_output = InterfaceDesignSpace(
                    name=schema.name,
                    tensor_shape=tensor_shape,
                    block_shape=block_shape,
                    stream_tiling=normalized_stream_tiling,  # Normalized template, values not resolved
                    datatype=datatype,
                    is_weight=False  # Outputs never have initializers
                )

                design_space_outputs[schema.name] = ds_output  # Store in dict by name
                self._interfaces[schema.name] = ds_output  # Store for derivations

                logger.debug(
                    f"  Output '{schema.name}': tensor={tensor_shape}, "
                    f"block={block_shape}, stream_tiling={schema.stream_tiling}, dtype={datatype.name}"
                )
            except ValueError as e:
                raise ValueError(f"Output '{schema.name}': {e}") from e

        # Phase 4: Split constraints into structural vs parametric
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

        # Phase 5: Validate structural constraints
        if structural_constraints:
            ds_ctx = DesignSpaceValidationContext(
                inputs=design_space_inputs,
                outputs=design_space_outputs,
                internal_datatypes=internal_datatypes,
                param_getter=ctx.param_getter
            )

            errors = []
            for constraint in structural_constraints:
                error = constraint.check(ds_ctx)
                if error is not None:
                    errors.append(f"  - {constraint.describe()}: {error}")

            if errors:
                error_msg = f"Design space validation failed for {ctx.node_name}:\n" + "\n".join(errors)
                raise ValueError(error_msg)

            logger.debug(f"  All {len(structural_constraints)} structural constraints passed")

        # Phase 6: Compute valid parallelization ranges
        valid_ranges = self._compute_parameter_ranges(design_space_inputs, design_space_outputs)

        # Phase 7: Create and return design space
        # Note: structural_constraints validated above but not stored (never re-validated)
        design_space = KernelDesignSpace(
            name=ctx.schema.name,
            inputs=design_space_inputs,
            outputs=design_space_outputs,
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

    # =========================================================================
    # Helper Methods for Two-Phase Construction
    # =========================================================================

    def _compute_parameter_ranges(
        self,
        design_space_inputs: Dict[str, 'InterfaceDesignSpace'],
        design_space_outputs: Dict[str, 'InterfaceDesignSpace'],
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
            design_space_inputs: Input interfaces with resolved block shapes (dict: name → interface)
            design_space_outputs: Output interfaces with resolved block shapes (dict: name → interface)

        Returns:
            Dict mapping parameter name to set of valid divisors
            Example: {"SIMD": {1, 2, 3, 4, 6, 8, ..., 768}, "PE": {1, 2, 4, 8}}

        Raises:
            ValueError: If parameter appears in block_tiling (violates R1 from spec)
        """
        from math import gcd
        from functools import reduce

        # Collect all block dimensions that each parameter must divide
        param_constraints = {}  # param_name -> list of block dimensions

        all_interfaces = (*design_space_inputs.values(), *design_space_outputs.values())

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

        # Compute valid ranges as divisors of GCD
        valid_ranges = {}
        for param_name, block_dims in param_constraints.items():
            # Must divide all block dimensions where param appears
            # Valid values = divisors(gcd(block_dims))
            combined = reduce(gcd, block_dims)
            valid_ranges[param_name] = divisors(combined)

        logger.debug(
            f"Computed valid ranges for {len(valid_ranges)} parameters: "
            + ", ".join(f"{k}={len(v)} values" for k, v in valid_ranges.items())
        )

        return valid_ranges


# =============================================================================
# Module-Level Build Function (Recommended Entry Point)
# =============================================================================

def build_kernel_design_space(ctx: BuildContext) -> 'KernelDesignSpace':
    """Build kernel design space from ONNX context.

    This is the recommended way to build a design space. Creates a temporary
    KernelModelBuilder instance for this one build operation, then discards it.

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
    return KernelModelBuilder().build(ctx)


__all__ = [
    'BuildContext',
    'KernelModelBuilder',
    'build_kernel_design_space',
]
