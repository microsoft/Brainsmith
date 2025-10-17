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
from .template_resolution import resolve_template
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
        ctx: ModelWrapper for ONNX graph access
        node_inputs: ONNX node input tensor names
        node_outputs: ONNX node output tensor names
        param_getter: Function to retrieve nodeattr values (e.g., get_nodeattr)
        param_setter: Function to store nodeattr values (e.g., set_nodeattr)
        node_name: Node name for error messages (optional)
    """
    schema: KernelSchema
    ctx: ModelWrapper
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
        ...     ctx=model_wrapper,
        ...     node_inputs=list(node.input),
        ...     node_outputs=list(node.output),
        ...     param_getter=self.get_nodeattr,
        ...     param_setter=self.set_nodeattr,
        ...     node_name=node.name
        ... )
        >>> model = builder.build(context)
    """

    def build(self, ctx: BuildContext) -> KernelModel:
        """Build KernelModel from context.

        Single-pass flow:
        1. Build InputModels (datatype + shapes)
        2. Resolve internal datatypes (datatype-only stubs)
        3. Build OutputModels (datatype + shapes, resolve unset dims)
        4. Create KernelModel (fully resolved, immutable)
        5. Validate constraints and relationships

        Args:
            ctx: BuildContext with schema, graph, and accessors

        Returns:
            Validated KernelModel instance

        Raises:
            ValueError: If model cannot be built or validation fails
        """
        # Store context as instance state for duration of build
        self._ctx = ctx
        self._interfaces: Dict[str, Any] = {}

        logger.debug(f"Building KernelModel for {ctx.node_name}")

        # Phase 1: Build InputModels
        input_models = self._build_inputs()

        # Phase 2: Resolve internal datatypes
        self._resolve_internal_datatypes()

        # Phase 3: Build OutputModels
        output_models = self._build_outputs()

        # Phase 4: Create KernelModel (derives unset output dimensions)
        kernel_model = KernelModel(
            name=ctx.schema.name,
            inputs=tuple(input_models),
            outputs=tuple(output_models)
        )

        # Phase 5: Validate using unified constraint system
        self._validate_model(kernel_model, ctx)

        logger.debug(f"KernelModel built successfully for {ctx.node_name}")
        return kernel_model

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
                datatype = self._ctx.ctx.get_tensor_datatype(inp_name)
                tensor_shape = tuple(self._ctx.ctx.get_tensor_shape(inp_name))

                # Store datatype to nodeattrs for FINN
                self._ctx.param_setter(f"input{i}Datatype", datatype.name)

                # Resolve block and stream shapes
                block_shape, stream_shape = self._resolve_interface_shapes(
                    schema, tensor_shape
                )

                # Infer is_weight from ONNX initializer presence
                is_weight = self._ctx.ctx.get_initializer(inp_name) is not None

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
                tensor_shape = tuple(self._ctx.ctx.get_tensor_shape(out_name))

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
            return self._ctx.ctx.get_tensor_datatype(out_name)

        if isinstance(schema.datatype, DatatypeSource):
            # Derive from inputs or internal datatypes
            derived_dt = schema.datatype.resolve(self._interfaces, self._ctx.param_getter)
            graph_dt = self._ctx.ctx.get_tensor_datatype(out_name)

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


__all__ = [
    'BuildContext',
    'KernelModelBuilder',
]
