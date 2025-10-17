############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Kernel operator base class."""

import logging
import math
from abc import ABC
from typing import Any, List, Optional, Tuple

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from .schemas import KernelSchema
from .models import KernelModel, InputModel, OutputModel
from .datatype_sources import DatatypeSource
from .template_resolution import resolve_template

logger = logging.getLogger(__name__)


class KernelOpError(Exception):
    """Kernel operator exception with node context."""
    def __init__(self, node, message):
        self.node = node
        super().__init__(f"{node.name}: {message}")


class KernelOp(HWCustomOp, ABC):
    """Kernel operator base class (shapes extracted from ModelWrapper, never stored)."""

    kernel_schema: KernelSchema  # type: ignore[assignment]

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

        if self.kernel_schema is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define kernel_schema class attribute"
            )

        self._kernel_model: Optional[KernelModel] = None

    def _error(self, message: str) -> KernelOpError:
        """Create exception with node context."""
        return KernelOpError(self.onnx_node, message)

    # ====================================================================
    # Public API: FINN Integration
    # ====================================================================

    def get_nodeattr_types(self):
        """Return nodeattr registry (datatypes + user params)."""
        return {**super().get_nodeattr_types(), **self.kernel_schema.get_nodeattr_types()}

    def get_kernel_model(self, ctx: ModelWrapper) -> KernelModel:
        """Get KernelModel (cached or rebuilt from ctx + schema).

        Flow:
        1. Return cached model if available
        2. Extract shapes from context (source of truth)
        3. Resolve and store all datatypes
        4. Build immutable KernelModel from resolved data
        5. Validate constraints and relationships
        6. Cache and return
        """
        if self._kernel_model is not None:
            return self._kernel_model

        if ctx is None:
            raise self._error(
                "ModelWrapper (ctx) required to build KernelModel. "
                "KernelOp needs ModelWrapper to extract tensor shapes from the ONNX graph "
                "and validate hardware configurations."
            )

        logger.debug(f"Building KernelModel for {self.onnx_node.name}")

        # Phase 1: Extract shapes from context (source of truth)
        input_shapes = []
        for i, inp_name in enumerate(self.onnx_node.input):
            if inp_name:
                shape = tuple(ctx.get_tensor_shape(inp_name))
                input_shapes.append(shape)

        output_shapes = []
        for i, out_name in enumerate(self.onnx_node.output):
            shape = tuple(ctx.get_tensor_shape(out_name))
            output_shapes.append(shape)

        # Phase 2: Resolve and store all datatypes
        self._resolve_and_store_datatypes(ctx)

        # Phase 3: Build model from schema + resolved datatypes
        kernel_model = self._build_from_schema(input_shapes, output_shapes)

        # Phase 4: Validate
        self._validate_kernel_model(kernel_model)

        # Phase 5: Cache
        self._kernel_model = kernel_model
        logger.debug(f"KernelModel built successfully for {self.onnx_node.name}")

        return self._kernel_model

    def infer_node_datatype(self, ctx):
        """FINN compatibility wrapper."""
        self.get_kernel_model(ctx)

    @property
    def kernel_model(self) -> KernelModel:
        """Access cached KernelModel (requires prior get_kernel_model call)."""
        if self._kernel_model is None:
            raise RuntimeError(
                f"Cannot access kernel_model for {self.onnx_node.name}. "
                f"Call get_kernel_model(ctx) first to build the model."
            )
        return self._kernel_model

    def _resolve_and_store_datatypes(self, ctx: ModelWrapper) -> None:
        """Resolve and store all datatypes (inputs, outputs, internals).

        This is the authoritative datatype resolution phase:
        - Input datatypes: Always from graph (ONNX determines inputs)
        - Internal datatypes: Derived from schema (using input datatypes)
        - Output datatypes: From schema if specified, else from graph

        All resolved datatypes are stored in nodeattrs for FINN integration.
        """
        interfaces = {}

        # Phase 1: Input datatypes - extract from graph and store
        for i, inp_name in enumerate(self.onnx_node.input):
            if inp_name:
                dtype = ctx.get_tensor_datatype(inp_name)
                self.set_nodeattr(f"input{i}Datatype", dtype.name)

                # Add to interfaces for internal/output derivation
                class TempInterface:
                    def __init__(self, dt):
                        self.datatype = dt

                interfaces[self.kernel_schema.inputs[i].name] = TempInterface(dtype)

        # Phase 2: Internal datatypes - derive from schema and store
        if self.kernel_schema.internal_datatypes:
            for internal_name, datatype_source in self.kernel_schema.internal_datatypes.items():
                try:
                    resolved_dt = datatype_source.resolve(interfaces, self.get_nodeattr)
                    attr_name = f"{internal_name}Datatype"
                    self.set_nodeattr(attr_name, resolved_dt.name)

                    # Add to interfaces for output derivation
                    interfaces[internal_name] = TempInterface(resolved_dt)

                    logger.debug(f"  Internal '{internal_name}': {resolved_dt.name}")
                except ValueError as e:
                    raise self._error(
                        f"Internal datatype '{internal_name}' resolution failed: {e}"
                    )

        # Phase 3: Output datatypes - derive from schema or extract from graph
        for i, out_name in enumerate(self.onnx_node.output):
            schema = self.kernel_schema.outputs[i]

            if schema.datatype is not None:
                # Schema specifies derivation - use it
                if isinstance(schema.datatype, DatatypeSource):
                    try:
                        derived_dt = schema.datatype.resolve(interfaces, self.get_nodeattr)
                        graph_dt = ctx.get_tensor_datatype(out_name)

                        if derived_dt != graph_dt:
                            logger.info(
                                f"Output '{schema.name}' datatype: schema derived {derived_dt.name}, "
                                f"graph has {graph_dt.name} - using schema"
                            )

                        dtype = derived_dt
                    except ValueError as e:
                        raise self._error(
                            f"Output '{schema.name}' datatype resolution failed: {e}"
                        )
                else:
                    # Schema specifies fixed datatype (rare)
                    dtype = schema.datatype
            else:
                # No schema derivation - use graph datatype (pass-through)
                dtype = ctx.get_tensor_datatype(out_name)

            self.set_nodeattr(f"output{i}Datatype", dtype.name)
            logger.debug(f"  Output '{schema.name}': {dtype.name}")

    # ====================================================================
    # Public API: Shape/Datatype Queries
    # ====================================================================

    def get_input_datatype(self, ind=0) -> DataType:
        """Get input datatype."""
        return DataType[self.get_nodeattr(f"input{ind}Datatype")]

    def get_output_datatype(self, ind=0) -> DataType:
        """Get output datatype."""
        return DataType[self.get_nodeattr(f"output{ind}Datatype")]

    def get_normal_input_shape(self, ind=0, ctx: Optional[ModelWrapper] = None) -> List[int]:
        """Get input tensor shape."""
        return list(self.get_kernel_model(ctx).inputs[ind].tensor_shape)

    def get_normal_output_shape(self, ind=0, ctx: Optional[ModelWrapper] = None) -> List[int]:
        """Get output tensor shape."""
        return list(self.get_kernel_model(ctx).outputs[ind].tensor_shape)

    def get_folded_input_shape(self, ind=0, ctx: Optional[ModelWrapper] = None) -> Tuple[int, ...]:
        """Get FINN folded input shape (fold_factors + flattened_stream)."""
        km = self.get_kernel_model(ctx)
        tensor_shape = km.inputs[ind].tensor_shape
        stream_shape = km.inputs[ind].stream_shape
        fold_factors = [t // s for t, s in zip(tensor_shape, stream_shape)]
        flattened_stream = math.prod(stream_shape)
        return tuple(fold_factors + [flattened_stream])

    def get_folded_output_shape(self, ind=0, ctx: Optional[ModelWrapper] = None) -> Tuple[int, ...]:
        """Get FINN folded output shape (fold_factors + flattened_stream)."""
        km = self.get_kernel_model(ctx)
        tensor_shape = km.outputs[ind].tensor_shape
        stream_shape = km.outputs[ind].stream_shape
        fold_factors = [t // s for t, s in zip(tensor_shape, stream_shape)]
        flattened_stream = math.prod(stream_shape)
        return tuple(fold_factors + [flattened_stream])

    def get_instream_width(self, ind=0, ctx: Optional[ModelWrapper] = None) -> int:
        """Get input stream width in bits."""
        return self.get_kernel_model(ctx).inputs[ind].stream_width_bits

    def get_outstream_width(self, ind=0, ctx: Optional[ModelWrapper] = None) -> int:
        """Get output stream width in bits."""
        return self.get_kernel_model(ctx).outputs[ind].stream_width_bits

    def get_number_output_values(self, ctx: Optional[ModelWrapper] = None):
        """Get number of time-multiplexed output values."""
        folded_shape = self.get_folded_output_shape(ind=0, ctx=ctx)
        return math.prod(folded_shape[:-1])

    def get_exp_cycles(self, ctx: Optional[ModelWrapper] = None):
        """Get expected cycles (initiation interval)."""
        return self.get_kernel_model(ctx).initiation_interval

    def make_shape_compatible_op(self, ctx):
        """Create standard ONNX op for shape inference (auto-detects pattern)."""
        from onnx import helper

        num_out = len(self.onnx_node.output)
        num_in = len(self.onnx_node.input)

        if num_in == 1 and num_out > 1:
            return helper.make_node(
                "Split",
                inputs=[self.onnx_node.input[0]],
                outputs=list(self.onnx_node.output),
                axis=-1
            )

        if num_out == 1:
            input_shapes = [tuple(ctx.get_tensor_shape(inp))
                           for inp in self.onnx_node.input]

            if len(set(input_shapes)) == 1:
                return super().make_const_shape_op(input_shapes[0])
            else:
                raise NotImplementedError(
                    f"{self.__class__.__name__}: {num_in} inputs with different shapes "
                    f"{input_shapes}. Override make_shape_compatible_op()."
                )

        raise NotImplementedError(
            f"{self.__class__.__name__}: {num_in} inputs → {num_out} outputs. "
            f"Override make_shape_compatible_op()."
        )

    def set_nodeattr(self, name: str, value: Any) -> None:
        """Set nodeattr and invalidate cache."""

        old_value = None
        try:
            old_value = self.get_nodeattr(name)
        except (AttributeError, Exception):
            pass

        if old_value != value:
            super().set_nodeattr(name, value)
            self._invalidate_cache()

    # ====================================================================
    # Internal Implementation
    # ====================================================================

    def _invalidate_cache(self) -> None:
        """Invalidate cached KernelModel."""
        self._kernel_model = None

    def _build_from_schema(self, input_shapes, output_shapes) -> KernelModel:
        """Build KernelModel from schema templates and extracted shapes."""
        logger.debug(f"Building KernelModel from schema for {self.onnx_node.name}")

        interfaces = {}

        # Phase 1: Build InputModels
        input_models = []
        for i, schema in enumerate(self.kernel_schema.inputs):
            tensor_shape = input_shapes[i]
            datatype = DataType[self.get_nodeattr(f"input{i}Datatype")]

            block_shape = resolve_template(
                schema.block_tiling,
                tensor_shape,
                self.get_nodeattr,
                f"Input '{schema.name}' block",
                interfaces
            )

            stream_shape = resolve_template(
                schema.stream_tiling,
                block_shape,
                self.get_nodeattr,
                f"Input '{schema.name}' stream",
                interfaces
            )

            input_model = InputModel(
                name=schema.name,
                tensor_shape=tensor_shape,
                block_shape=block_shape,
                stream_shape=stream_shape,
                datatype=datatype,
                is_weight=schema.is_weight
            )

            input_models.append(input_model)
            interfaces[schema.name] = input_model

            logger.debug(
                f"  Input '{schema.name}': tensor={tensor_shape}, "
                f"block={block_shape}, stream={stream_shape}"
            )

        # Phase 2: Add resolved internal datatypes to interfaces
        # (Already resolved and stored in nodeattrs by _resolve_and_store_datatypes)
        if self.kernel_schema.internal_datatypes:
            for internal_name in self.kernel_schema.internal_datatypes.keys():
                try:
                    datatype = DataType[self.get_nodeattr(f"{internal_name}Datatype")]

                    class InternalDatatype:
                        def __init__(self, dt):
                            self.datatype = dt

                    interfaces[internal_name] = InternalDatatype(datatype)
                except (AttributeError, KeyError) as e:
                    raise self._error(
                        f"Internal datatype '{internal_name}' not found in nodeattrs. "
                        f"Should have been resolved in _resolve_and_store_datatypes(). Error: {e}"
                    )

        # Phase 3: Build OutputModels
        output_models = []
        for i, schema in enumerate(self.kernel_schema.outputs):
            tensor_shape = output_shapes[i]
            datatype = DataType[self.get_nodeattr(f"output{i}Datatype")]

            block_shape = resolve_template(
                schema.block_tiling,
                tensor_shape,
                self.get_nodeattr,
                f"Output '{schema.name}' block",
                interfaces
            )

            if schema.stream_tiling is not None:
                stream_shape = resolve_template(
                    schema.stream_tiling,
                    block_shape,
                    self.get_nodeattr,
                    f"Output '{schema.name}' stream",
                    interfaces
                )
            else:
                stream_shape = tuple([None] * len(block_shape))

            output_model = OutputModel(
                name=schema.name,
                tensor_shape=tensor_shape,
                block_shape=block_shape,
                stream_shape=stream_shape,
                datatype=datatype
            )

            output_models.append(output_model)
            interfaces[schema.name] = output_model

            logger.debug(
                f"  Output '{schema.name}': tensor={tensor_shape}, "
                f"block={block_shape}, stream={stream_shape}"
            )

        return KernelModel(
            name=self.kernel_schema.name,
            inputs=tuple(input_models),
            outputs=tuple(output_models)
        )

    def _validate_kernel_model(self, model: KernelModel) -> None:
        """Validate constraints and relationships."""
        logger.debug(f"Validating KernelModel for {self.onnx_node.name}")

        for i, input_model in enumerate(model.inputs):
            schema = self.kernel_schema.inputs[i]
            if schema.constraints:
                self._validate_interface_constraints(
                    schema.name, input_model, schema.constraints
                )

        for i, output_model in enumerate(model.outputs):
            schema = self.kernel_schema.outputs[i]
            if schema.constraints:
                self._validate_interface_constraints(
                    schema.name, output_model, schema.constraints
                )

        if self.kernel_schema.relationships:
            logger.debug(
                f"Validating {len(self.kernel_schema.relationships)} relationships"
            )
            for relationship in self.kernel_schema.relationships:
                error = relationship.check(model, self.get_nodeattr)
                if error:
                    raise self._error(str(error))

    def _validate_interface_constraints(
        self,
        interface_name: str,
        interface_model: Any,
        constraints: List
    ) -> None:
        """Validate single interface constraints."""
        for constraint in constraints:
            error = constraint.check(interface_model, self.get_nodeattr)
            if error:
                raise self._error(str(error))
