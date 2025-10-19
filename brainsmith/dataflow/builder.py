############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""KernelModel builder - constructs immutable models from schemas and context.

This module separates model construction logic from KernelOp (FINN integration).
The builder can be used independently for testing, tooling, or non-FINN contexts.

Key Components:
- BuildContext: Context data for building (schema, graph, accessors)
- KernelModelBuilder: Orchestrates multi-phase model construction

Architecture:
    KernelOp (FINN adapter) → KernelModelBuilder → KernelModel

The builder follows a single-pass flow:
1. Build InputModels (datatype + shapes from graph + templates)
2. Resolve internal datatypes (datatype-only stubs)
3. Build OutputModels (datatype + shapes, resolving any unset dimensions)
4. Create KernelModel (fully resolved, immutable)
5. Validate constraints and relationships
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from qonnx.core.datatype import BaseDataType
from qonnx.core.modelwrapper import ModelWrapper

from .schemas import KernelSchema
from .models import KernelModel, InputModel, OutputModel, InternalDatatypeModel
from .template_resolution import resolve_template, normalize_template
from .datatype_sources import DatatypeSource
from .validation import KernelValidationContext

logger = logging.getLogger(__name__)


@dataclass
class BuildContext:
    """Context for building a KernelModel.

    Encapsulates all data needed to build a KernelModel from a schema.
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
    """Builds immutable KernelModel from schema + context.

    Separates model construction logic from KernelOp (FINN integration).
    Can be used independently for testing, tooling, or non-FINN contexts.

    The builder orchestrates a multi-phase construction process:
    1. Build InputModels with shapes resolved from templates
    2. Resolve internal datatypes (accumulator, etc.)
    3. Build OutputModels with shapes, datatypes, and dimension resolution
    4. Create KernelModel (fully resolved, immutable)
    5. Validate all constraints and relationships

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
        >>> model = builder.build(context)
    """

    def build(self, ctx: BuildContext) -> KernelModel:
        """Build KernelModel from context (backward-compatible single-phase API).

        This method maintains backward compatibility during migration to
        two-phase construction. New code should use build_invariant() + configure().

        Implementation: Delegates to two-phase construction internally.

        Flow:
        1. Build invariant model via build_invariant()
        2. Extract current parallelization params
        3. Configure with current params via invariant.configure()
        4. Convert ConfiguredKernelModel to legacy KernelModel format
        5. Return KernelModel

        Args:
            ctx: BuildContext with schema, graph, and accessors

        Returns:
            Validated KernelModel instance (same as before)

        Raises:
            ValueError: If model cannot be built or validation fails
        """
        logger.debug(f"Building KernelModel for {ctx.node_name} (via two-phase construction)")

        # Phase 1: Build invariant model
        invariant = self.build_invariant(ctx)

        # Phase 2: Extract current parallelization params
        params = {}
        for param_name in invariant.parallelization_params.keys():
            params[param_name] = ctx.param_getter(param_name)

        # Phase 3: Configure with current params
        configured = invariant.configure(params)

        # Phase 4: Convert to legacy KernelModel format
        # Build traditional InputModel/OutputModel instances
        input_models = []
        for cfg_inp in configured.inputs:
            input_models.append(InputModel(
                name=cfg_inp.name,
                tensor_shape=cfg_inp.tensor_shape,
                block_shape=cfg_inp.block_shape,
                stream_shape=cfg_inp.stream_shape,
                datatype=cfg_inp.datatype,
                is_weight=cfg_inp.is_weight,
            ))

        output_models = []
        for cfg_out in configured.outputs:
            output_models.append(OutputModel(
                name=cfg_out.name,
                tensor_shape=cfg_out.tensor_shape,
                block_shape=cfg_out.block_shape,
                stream_shape=cfg_out.stream_shape,
                datatype=cfg_out.datatype,
            ))

        kernel_model = KernelModel(
            name=configured.name,
            inputs=tuple(input_models),
            outputs=tuple(output_models),
        )

        logger.debug(f"KernelModel built successfully for {ctx.node_name} (via two-phase)")
        return kernel_model

    def build_invariant(self, ctx: BuildContext) -> 'InvariantKernelModel':
        """Build invariant kernel model (Phase 1 of two-phase construction).

        Resolves all properties that don't vary during DSE:
        - Tensor shapes (from ONNX graph)
        - Block shapes (from block_tiling templates)
        - Datatypes (from ONNX graph + DatatypeSource derivation)
        - Internal datatypes (from DatatypeSource)
        - Invariant constraints (validated once)

        Defers resolution of stream shapes (variant properties).
        Stream shapes will be resolved later during configure().

        Args:
            ctx: Build context with ONNX node and ModelWrapper

        Returns:
            InvariantKernelModel ready for configuration

        Raises:
            ValueError: If invariant constraints fail
        """
        from .models import InvariantInterfaceModel, InvariantKernelModel, InternalDatatypeModel
        from .validation import InvariantValidationContext

        self._ctx = ctx
        self._interfaces: Dict[str, Any] = {}

        logger.debug(f"Building InvariantKernelModel for {ctx.node_name}")

        # Phase 1: Build invariant input models
        invariant_inputs = []

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

                # Build InvariantInterfaceModel (stream_tiling normalized but not resolved)
                inv_input = InvariantInterfaceModel(
                    name=schema.name,
                    tensor_shape=tensor_shape,
                    block_shape=block_shape,
                    stream_tiling=normalized_stream_tiling,  # Normalized template, values not resolved
                    datatype=datatype,
                    is_weight=is_weight
                )

                invariant_inputs.append(inv_input)
                self._interfaces[schema.name] = inv_input  # Store for derivations

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

                    # Store InternalDatatypeModel (no shapes for internal datatypes)
                    self._interfaces[internal_name] = InternalDatatypeModel(datatype=datatype)
                    internal_datatypes[internal_name] = datatype

                    logger.debug(f"  Internal '{internal_name}': dtype={datatype.name}")
                except ValueError as e:
                    raise ValueError(f"Internal datatype '{internal_name}': {e}") from e

        # Phase 3: Build invariant output models
        invariant_outputs = []

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

                # Build InvariantInterfaceModel (stream_tiling normalized but not resolved)
                inv_output = InvariantInterfaceModel(
                    name=schema.name,
                    tensor_shape=tensor_shape,
                    block_shape=block_shape,
                    stream_tiling=normalized_stream_tiling,  # Normalized template, values not resolved
                    datatype=datatype,
                    is_weight=False  # Outputs never have initializers
                )

                invariant_outputs.append(inv_output)
                self._interfaces[schema.name] = inv_output  # Store for derivations

                logger.debug(
                    f"  Output '{schema.name}': tensor={tensor_shape}, "
                    f"block={block_shape}, stream_tiling={schema.stream_tiling}, dtype={datatype.name}"
                )
            except ValueError as e:
                raise ValueError(f"Output '{schema.name}': {e}") from e

        # Phase 4: Split constraints into invariant vs variant
        invariant_constraints = []
        variant_constraints = []

        for constraint in ctx.schema.constraints:
            if self._is_invariant_constraint(constraint):
                invariant_constraints.append(constraint)
            else:
                variant_constraints.append(constraint)

        logger.debug(
            f"  Split {len(ctx.schema.constraints)} constraints: "
            f"{len(invariant_constraints)} invariant, {len(variant_constraints)} variant"
        )

        # Phase 5: Validate invariant constraints
        if invariant_constraints:
            inv_ctx = InvariantValidationContext(
                inputs=invariant_inputs,
                outputs=invariant_outputs,
                internal_datatypes=internal_datatypes,
                param_getter=ctx.param_getter
            )

            errors = []
            for constraint in invariant_constraints:
                error = constraint.check(inv_ctx)
                if error is not None:
                    errors.append(f"  - {constraint.describe()}: {error}")

            if errors:
                error_msg = f"Invariant validation failed for {ctx.node_name}:\n" + "\n".join(errors)
                raise ValueError(error_msg)

            logger.debug(f"  All {len(invariant_constraints)} invariant constraints passed")

        # Phase 6: Compute valid parallelization ranges
        valid_ranges = self._compute_valid_ranges(invariant_inputs, invariant_outputs)

        # Phase 7: Create and return invariant model
        invariant_model = InvariantKernelModel(
            name=ctx.schema.name,
            inputs=tuple(invariant_inputs),
            outputs=tuple(invariant_outputs),
            internal_datatypes=internal_datatypes,
            invariant_constraints=invariant_constraints,
            variant_constraints=variant_constraints,
            parallelization_params=valid_ranges,
        )

        logger.debug(f"InvariantKernelModel built successfully for {ctx.node_name}")
        return invariant_model

    def _build_inputs(self) -> List[InputModel]:
        """Build InputModels from schema and ONNX graph.

        For each input:
        1. Extract datatype and tensor shape from ONNX graph
        2. Store datatype to nodeattrs (for FINN)
        3. Resolve block and stream shapes from templates
        4. Create InputModel
        5. Store in interfaces dict (for cross-interface derivations)

        Returns:
            List of InputModel instances
        """
        input_models = []

        for i, inp_name in enumerate(self._ctx.node_inputs):
            if not inp_name:
                continue

            schema = self._ctx.schema.inputs[i]

            try:
                # Get datatype and tensor shape from graph
                datatype = self._ctx.model_w.get_tensor_datatype(inp_name)
                tensor_shape = tuple(self._ctx.model_w.get_tensor_shape(inp_name))

                # Store datatype to nodeattrs for FINN
                self._ctx.param_setter(f"input{i}Datatype", datatype.name)

                # Resolve block and stream shapes
                block_shape, stream_shape = self._resolve_interface_shapes(
                    schema, tensor_shape
                )

                # Infer is_weight from ONNX initializer presence
                is_weight = self._ctx.model_w.get_initializer(inp_name) is not None

                # Build full InputModel
                input_model = InputModel(
                    name=schema.name,
                    tensor_shape=tensor_shape,
                    block_shape=block_shape,
                    stream_shape=stream_shape,
                    datatype=datatype,
                    is_weight=is_weight  # Inferred from ONNX, not from schema
                )

                input_models.append(input_model)
                self._interfaces[schema.name] = input_model  # Store for derivations

                logger.debug(
                    f"  Input '{schema.name}': tensor={tensor_shape}, "
                    f"block={block_shape}, stream={stream_shape}, dtype={datatype.name}"
                )
            except ValueError as e:
                raise ValueError(f"Input '{schema.name}': {e}") from e

        return input_models

    def _resolve_internal_datatypes(self) -> None:
        """Resolve internal datatypes (datatype-only, no shapes).

        Internal datatypes represent intermediate computation values
        (e.g., accumulator precision) that don't correspond to ONNX tensors.
        They are derived from inputs/outputs using DatatypeSource patterns.
        """
        if not self._ctx.schema.internal_datatypes:
            return

        for internal_name, datatype_source in self._ctx.schema.internal_datatypes.items():
            try:
                datatype = datatype_source.resolve(self._interfaces, self._ctx.param_getter)
                self._ctx.param_setter(f"{internal_name}Datatype", datatype.name)

                # Store InternalDatatypeModel (no shapes for internal datatypes)
                self._interfaces[internal_name] = InternalDatatypeModel(datatype=datatype)

                logger.debug(f"  Internal '{internal_name}': dtype={datatype.name}")
            except ValueError as e:
                raise ValueError(f"Internal datatype '{internal_name}': {e}") from e

    def _build_outputs(self) -> List[OutputModel]:
        """Build OutputModels from schema and ONNX graph.

        For each output:
        1. Resolve datatype (derived from inputs/internals or from graph)
        2. Store datatype to nodeattrs (for FINN)
        3. Extract tensor shape from ONNX graph
        4. Resolve block and stream shapes from templates
        5. Create OutputModel
        6. Store in interfaces dict

        Returns:
            List of OutputModel instances
        """
        output_models = []

        for i, out_name in enumerate(self._ctx.node_outputs):
            schema = self._ctx.schema.outputs[i]

            try:
                # Resolve or extract datatype
                datatype = self._resolve_output_datatype(schema, out_name)

                # Store datatype to nodeattrs for FINN
                self._ctx.param_setter(f"output{i}Datatype", datatype.name)

                # Get tensor shape from graph
                tensor_shape = tuple(self._ctx.model_w.get_tensor_shape(out_name))

                # Resolve block and stream shapes
                block_shape, stream_shape = self._resolve_interface_shapes(
                    schema, tensor_shape
                )

                # Resolve any unset stream dimensions (None → 1 singleton default)
                stream_shape = self._resolve_unset_dimensions(
                    stream_shape, schema.name
                )

                # Build full OutputModel
                output_model = OutputModel(
                    name=schema.name,
                    tensor_shape=tensor_shape,
                    block_shape=block_shape,
                    stream_shape=stream_shape,
                    datatype=datatype
                )

                output_models.append(output_model)
                self._interfaces[schema.name] = output_model  # Store for derivations

                logger.debug(
                    f"  Output '{schema.name}': tensor={tensor_shape}, "
                    f"block={block_shape}, stream={stream_shape}, dtype={datatype.name}"
                )
            except ValueError as e:
                raise ValueError(f"Output '{schema.name}': {e}") from e

        return output_models

    def _resolve_interface_shapes(
        self,
        schema: Any,
        tensor_shape: Tuple[int, ...]
    ) -> Tuple[Tuple[int, ...], Tuple[Optional[int], ...]]:
        """Resolve both block and stream shapes for an interface.

        Extracts the common pattern of resolving block_tiling and stream_tiling.
        Handles the case where stream_tiling is None (returns unset dimensions).

        Args:
            schema: InputSchema or OutputSchema
            tensor_shape: Reference tensor shape

        Returns:
            Tuple of (block_shape, stream_shape)
        """
        # Resolve block shape
        block_shape = resolve_template(
            schema.block_tiling,
            tensor_shape,
            self._ctx.param_getter,
            self._interfaces
        )

        # Resolve stream shape
        if schema.stream_tiling is not None:
            stream_shape = resolve_template(
                schema.stream_tiling,
                block_shape,
                self._ctx.param_getter,
                self._interfaces
            )
        else:
            # No stream tiling specified - use unset dimensions
            stream_shape = tuple([None] * len(block_shape))

        return block_shape, stream_shape

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

    def _resolve_unset_dimensions(
        self,
        stream_shape: Tuple[Optional[int], ...],
        interface_name: str
    ) -> Tuple[int, ...]:
        """Resolve any unset dimensions (None) to concrete values.

        Default strategy: All unset dimensions → 1 (singleton)

        This provides a predictable fallback when stream_tiling is not
        explicitly specified in the schema. Kernels should specify
        stream_tiling explicitly for non-trivial streaming patterns.

        Args:
            stream_shape: Stream shape (may contain None values)
            interface_name: Interface name for logging

        Returns:
            Stream shape with all dimensions resolved
        """
        if not any(d is None for d in stream_shape):
            return stream_shape

        resolved = tuple(1 if d is None else d for d in stream_shape)
        logger.debug(
            f"  Resolved '{interface_name}' stream_shape: "
            f"{stream_shape} → {resolved} (singleton default)"
        )
        return resolved

    def _validate_model(self, kernel_model: KernelModel, ctx: BuildContext) -> None:
        """Validate kernel model using unified constraint system.

        Creates a KernelValidationContext and runs all schema constraints through it.

        Args:
            kernel_model: Built KernelModel to validate
            ctx: BuildContext with param_getter

        Raises:
            ValueError: If any constraint fails
        """
        if not ctx.schema.constraints:
            return  # No constraints to validate

        # Create kernel validation context
        validation_ctx = KernelValidationContext(
            kernel_model=kernel_model,
            param_getter=ctx.param_getter
        )

        # Run all constraints
        errors = []
        for constraint in ctx.schema.constraints:
            error = constraint.check(validation_ctx)
            if error is not None:
                errors.append(f"  - {constraint.describe()}: {error}")

        if errors:
            error_msg = f"Kernel validation failed for {ctx.node_name}:\n" + "\n".join(errors)
            raise ValueError(error_msg)

        logger.debug(f"  All {len(ctx.schema.constraints)} constraints passed")

    # =========================================================================
    # Helper Methods for Two-Phase Construction
    # =========================================================================

    def _divisors(self, n: int) -> set:
        """Compute all divisors of n efficiently.

        Uses sqrt optimization: only check up to sqrt(n), add both i and n/i.

        Args:
            n: Positive integer

        Returns:
            Set of all divisors of n
            Example: _divisors(12) -> {1, 2, 3, 4, 6, 12}

        Raises:
            ValueError: If n is not a positive integer

        Performance: O(√n) instead of O(n)
        """
        if n <= 0:
            raise ValueError(f"Cannot compute divisors of non-positive integer: {n}")

        divisors = set()
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divisors.add(i)
                divisors.add(n // i)  # Add the paired divisor
        return divisors

    def _compute_valid_ranges(
        self,
        invariant_inputs: List[Any],  # List[InvariantInterfaceModel]
        invariant_outputs: List[Any],  # List[InvariantInterfaceModel]
    ) -> Dict[str, set]:
        """Compute valid divisor sets for each parallelization parameter.

        A parallelization parameter is any string appearing in stream_tiling.
        Valid values are divisors of the corresponding block dimension.

        For multi-dimensional cases, if a parameter appears in multiple
        dimensions or interfaces, valid values are divisors of GCD of all
        block dimensions where the parameter appears.

        Args:
            invariant_inputs: Input interfaces with resolved block shapes
            invariant_outputs: Output interfaces with resolved block shapes

        Returns:
            Dict mapping parameter name to set of valid divisors
            Example: {"SIMD": {1, 2, 3, 4, 6, 8, ..., 768}, "PE": {1, 2, 4, 8}}

        Raises:
            ValueError: If parameter appears in block_tiling (violates R1 from spec)

        Performance: Target <10ms for typical kernel
        """
        from math import gcd
        from functools import reduce

        # Collect all block dimensions that each parameter must divide
        param_constraints = {}  # param_name -> list of block dimensions

        all_interfaces = list(invariant_inputs) + list(invariant_outputs)

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
            valid_ranges[param_name] = self._divisors(combined)

        logger.debug(
            f"Computed valid ranges for {len(valid_ranges)} parameters: "
            + ", ".join(f"{k}={len(v)} values" for k, v in valid_ranges.items())
        )

        return valid_ranges

    def _is_invariant_constraint(self, constraint: Any) -> bool:
        """Determine if constraint is invariant or variant.

        Invariant constraints are validated once during build_invariant().
        Variant constraints are validated per-configuration during configure().

        Uses the constraint's evaluation_phase property for classification.
        This property defaults to a heuristic (backward compatible), but can
        be explicitly overridden by constraint subclasses.

        Args:
            constraint: Constraint to classify

        Returns:
            True if invariant (validated once), False if variant (per-config)

        Examples:
            - DatatypeInteger: invariant (no hierarchy)
            - ShapesEqual(hierarchy=TENSOR): invariant
            - ShapesEqual(hierarchy=BLOCK): invariant
            - ShapesEqual(hierarchy=STREAM): variant
            - DimensionDivisible(hierarchy=STREAM): variant
        """
        # Use evaluation_phase property (Phase 4 enhancement)
        # Falls back to heuristic for backward compatibility
        return constraint.evaluation_phase == 'invariant'


__all__ = [
    'BuildContext',
    'KernelModelBuilder',
]
