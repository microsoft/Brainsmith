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
from .models import KernelModel
from .builder import BuildContext, KernelModelBuilder

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
        self._builder = KernelModelBuilder()

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

        Delegates to KernelModelBuilder for construction. This method provides:
        - Caching of built models
        - Error context (node name)
        - FINN integration (nodeattr accessors)

        The builder handles:
        - Building InputModels with template resolution
        - Resolving internal datatypes
        - Building OutputModels with derived datatypes
        - Validation of constraints and relationships

        Args:
            ctx: ModelWrapper for ONNX graph access

        Returns:
            Cached or newly built KernelModel

        Raises:
            KernelOpError: If model cannot be built or validation fails
        """
        # Return cached model if available
        if self._kernel_model is not None:
            return self._kernel_model

        if ctx is None:
            raise self._error(
                "ModelWrapper (ctx) required to build KernelModel. "
                "KernelOp needs ModelWrapper to extract tensor shapes from the ONNX graph "
                "and validate hardware configurations."
            )

        # Build context for builder
        build_ctx = BuildContext(
            schema=self.kernel_schema,
            ctx=ctx,
            node_inputs=list(self.onnx_node.input),
            node_outputs=list(self.onnx_node.output),
            param_getter=self.get_nodeattr,
            param_setter=self.set_nodeattr,
            node_name=self.onnx_node.name
        )

        # Delegate to builder
        try:
            self._kernel_model = self._builder.build(build_ctx)
        except ValueError as e:
            raise self._error(str(e))

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
