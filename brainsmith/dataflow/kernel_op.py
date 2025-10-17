############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Kernel operator base class."""

import logging
import math
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from onnx import NodeProto

from .schemas import KernelSchema
from .models import KernelModel
from .builder import BuildContext, KernelModelBuilder

if TYPE_CHECKING:
    from .inference import InferenceResult, InferenceConfig

logger = logging.getLogger(__name__)


class KernelOpError(Exception):
    """Kernel operator exception with node context."""
    def __init__(self, node, message):
        self.node = node
        super().__init__(f"{node.name}: {message}")


class KernelOp(HWCustomOp, ABC):
    """Kernel operator base class (shapes extracted from ModelWrapper, never stored).

    Subclasses must implement kernel_schema as a property that returns a KernelSchema.

    Three common usage patterns:

    1. Static schema (most common):
        ```python
        SCHEMA = df.KernelSchema(name="LayerNorm", ...)

        @property
        def kernel_schema(self) -> KernelSchema:
            return SCHEMA
        ```

    2. Variable-input schema (e.g., Concat with N inputs):
        ```python
        @property
        def kernel_schema(self) -> KernelSchema:
            num_inputs = len(self.onnx_node.input)
            return make_concat_schema(num_inputs)
        ```

    3. Mode-dependent schema (e.g., MVAU with different memory modes):
        IMPORTANT: If schema construction needs to call get_nodeattr(), you MUST
        override get_nodeattr_types() explicitly to avoid circular dependency.

        ```python
        def get_nodeattr_types(self):
            # Define all nodeattrs BEFORE schema construction
            my_attrs = super().get_nodeattr_types()
            my_attrs.update({"mem_mode": ("s", True, "internal")})
            return my_attrs

        @property
        def kernel_schema(self) -> KernelSchema:
            mem_mode = self.get_nodeattr("mem_mode")
            return make_mvau_schema(mem_mode)
        ```
    """

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)
        self._kernel_model: Optional[KernelModel] = None
        self._builder = KernelModelBuilder()

    @property
    @abstractmethod
    def kernel_schema(self) -> KernelSchema:
        """Return kernel schema (enforced by abstract method).

        Must be implemented by all subclasses. See class docstring for usage patterns.
        """
        pass

    # ====================================================================
    # Inference Support (Optional - Kernels Can Opt-In)
    # ====================================================================

    @classmethod
    def get_class_schema(cls) -> Optional[KernelSchema]:
        """Get schema without instantiation (required for inference).

        Override to return the kernel's schema. If not overridden,
        kernel does not support automatic inference.

        Returns:
            KernelSchema if inference is supported, None otherwise

        Example:
            @classmethod
            def get_class_schema(cls):
                return ADDSTREAMS_SCHEMA
        """
        return None

    @classmethod
    def supports_inference(cls) -> bool:
        """Check if this kernel supports automatic inference.

        Returns True if the kernel has a class schema and either:
        - Has an inference_config in the schema, OR
        - Overrides can_infer_from() method

        Returns:
            True if kernel supports inference, False otherwise
        """
        schema = cls.get_class_schema()
        if schema is None:
            return False

        return (
            schema.inference_config is not None or
            hasattr(cls, 'can_infer_from') and
            cls.can_infer_from.__func__ is not KernelOp.can_infer_from.__func__
        )

    @classmethod
    def can_infer_from(cls, node: NodeProto, model: ModelWrapper) -> bool:
        """Check if this kernel can be inferred from the given ONNX node.

        Default implementation uses schema.inference_config for validation.
        Override for custom matching logic beyond declarative config.

        Args:
            node: ONNX node to check
            model: ModelWrapper for graph access

        Returns:
            True if this kernel can be inferred from the node

        Example:
            # Custom validation beyond InferenceConfig
            @classmethod
            def can_infer_from(cls, node, model):
                if node.op_type != "Add":
                    return False
                # Custom logic...
                return True
        """
        schema = cls.get_class_schema()
        if schema is None or schema.inference_config is None:
            return False

        return cls._check_inference_config(node, model, schema.inference_config)

    @classmethod
    def infer_from(
        cls,
        node: NodeProto,
        model: ModelWrapper,
        insert_index: int
    ) -> 'InferenceResult':
        """Create HW nodes from the given ONNX node.

        Must be implemented by subclasses that support inference.

        Args:
            node: ONNX node to convert
            model: ModelWrapper for graph access
            insert_index: Where to insert new nodes

        Returns:
            InferenceResult with nodes to insert/remove

        Raises:
            NotImplementedError: If kernel doesn't implement inference

        Example:
            @classmethod
            def infer_from(cls, node, model, insert_index):
                helper = InferenceHelper(model)
                hw_node = helper.make_node("AddStreams", ...)
                return InferenceResult(
                    nodes_to_insert=[hw_node],
                    nodes_to_remove=[node]
                )
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement infer_from(). "
            f"Override this method to support automatic inference."
        )

    @classmethod
    def _check_inference_config(
        cls,
        node: NodeProto,
        model: ModelWrapper,
        config: 'InferenceConfig'
    ) -> bool:
        """Default implementation checks inference config criteria.

        Validates ONNX node against InferenceConfig rules:
        - Op type matching
        - Static/dynamic input requirements
        - Datatype requirements
        - Shape requirements
        - Custom validator

        Args:
            node: ONNX node to validate
            model: ModelWrapper for graph access
            config: InferenceConfig with validation rules

        Returns:
            True if node matches all config criteria
        """
        # Check op type
        if node.op_type not in config.source_ops:
            return False

        # Check static input requirements
        if config.require_static_inputs:
            for idx in config.require_static_inputs:
                if idx >= len(node.input):
                    return False
                if model.get_initializer(node.input[idx]) is None:
                    return False

        # Check dynamic input requirements
        if config.require_dynamic_inputs:
            for idx in config.require_dynamic_inputs:
                if idx >= len(node.input):
                    return False
                if model.get_initializer(node.input[idx]) is not None:
                    return False

        # Check integer datatype requirement
        if config.require_integer_inputs:
            for inp in node.input:
                dt = model.get_tensor_datatype(inp)
                if dt is None or not dt.is_integer():
                    return False

        # Check same shape requirement
        if config.require_same_shapes:
            if len(node.input) < 2:
                return False
            shapes = [model.get_tensor_shape(inp) for inp in node.input]
            if not all(s == shapes[0] for s in shapes):
                return False

        # Custom validator
        if config.custom_validator is not None:
            if not config.custom_validator(node, model):
                return False

        return True

    def _error(self, message: str) -> KernelOpError:
        """Create exception with node context."""
        return KernelOpError(self.onnx_node, message)

    # ====================================================================
    # Public API: FINN Integration
    # ====================================================================

    def get_nodeattr_types(self):
        """Return nodeattr registry (datatypes + user params + kernel params).

        Auto-delegates to kernel_schema.get_nodeattr_types() which includes:
        - Interface datatypes (input0Datatype, output0Datatype, etc.)
        - Internal datatypes (accumulatorDatatype, etc.)
        - Template parameters (SIMD, PE, etc.)
        - Kernel-specific parameters (epsilon, algorithm, etc.)

        Only override if your schema property calls get_nodeattr() (circular dependency).
        See class docstring for the mode-dependent schema pattern.
        """
        my_attrs = super().get_nodeattr_types()

        try:
            my_attrs.update(self.kernel_schema.get_nodeattr_types())
        except RecursionError as e:
            raise RuntimeError(
                f"{self.__class__.__name__}.kernel_schema property calls get_nodeattr(), "
                f"creating circular dependency. You must override get_nodeattr_types() "
                f"explicitly to define nodeattrs before schema construction. "
                f"See KernelOp docstring for mode-dependent schema pattern."
            ) from e

        return my_attrs

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
