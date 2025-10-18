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
from .validation import OnnxValidationContext, KernelValidationContext

if TYPE_CHECKING:
    from .inference import InferenceResult, InferencePattern

logger = logging.getLogger(__name__)


class KernelOpError(Exception):
    """Kernel operator exception with node context."""
    def __init__(self, node, message):
        self.node = node
        super().__init__(f"{node.name}: {message}")


class KernelOp(HWCustomOp, ABC):
    """Kernel operator base class (shapes extracted from ModelWrapper, never stored).

    Subclasses must implement build_schema() to construct their KernelSchema.

    Two common usage patterns:

    1. Static schema (most common - LayerNorm, Softmax, AddStreams):
        ```python
        SCHEMA = df.KernelSchema(name="LayerNorm", ...)

        @classmethod
        def build_schema(cls, node, model):
            return SCHEMA  # Constant, ignores parameters
        ```

    2. Dynamic schema (variable I/O - Concat, MVAU):
        ```python
        @classmethod
        def build_schema(cls, node, model):
            # Inspect node structure to determine schema
            num_inputs = len(node.input)
            inputs = [InputSchema(name=f"input{i}", ...) for i in range(num_inputs)]
            return KernelSchema(name="Concat", inputs=inputs, outputs=[...])
        ```

    The schema is built once during __init__ and frozen as an attribute.
    During inference validation, build_schema() is called with full ModelWrapper context.
    """

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)
        # Build and freeze schema from node structure
        self.kernel_schema = self.build_schema(onnx_node, model=None)
        self._kernel_model: Optional[KernelModel] = None
        self._builder = KernelModelBuilder()

    @classmethod
    @abstractmethod
    def build_schema(cls, node: NodeProto, model: Optional[ModelWrapper]) -> KernelSchema:
        """Build kernel schema from ONNX node.

        Polymorphic method that handles both static and dynamic schemas:
        - Static schemas: return constant, ignore parameters
        - Dynamic schemas: inspect node structure to build schema

        Called in two contexts:
        1. During __init__: model=None (schema built for instance)
        2. During can_infer_from(): model provided (schema built for validation)

        Args:
            node: ONNX node (provides inputs, outputs, attributes)
            model: Optional ModelWrapper (provides shapes, datatypes for validation context)

        Returns:
            KernelSchema defining kernel structure

        Example (static schema):
            @classmethod
            def build_schema(cls, node, model):
                return LAYERNORM_SCHEMA

        Example (dynamic schema):
            @classmethod
            def build_schema(cls, node, model):
                num_inputs = len(node.input)
                inputs = [InputSchema(name=f"input{i}", ...) for i in range(num_inputs)]
                return KernelSchema(name="Concat", inputs=inputs, outputs=[...])
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement build_schema() as a classmethod "
            f"returning a KernelSchema. See KernelOp docstring for examples."
        )

    # ====================================================================
    # Inference Support (Required - All Kernels Must Define)
    # ====================================================================

    @classmethod
    @abstractmethod
    def get_inference_pattern(cls) -> 'InferencePattern':
        """Get ONNX inference pattern (required).

        This is separate from schema to maintain framework-agnostic structure.
        Schema defines WHAT the kernel IS (structure, constraints).
        InferencePattern defines WHERE to find it in ONNX (discovery).

        Returns:
            InferencePattern defining ONNX discovery

        Example:
            @classmethod
            def get_inference_pattern(cls):
                return LAYERNORM_INFERENCE
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement get_inference_pattern() as a classmethod "
            f"returning an InferencePattern. See KernelOp docstring for examples."
        )


    @classmethod
    def can_infer_from(cls, node: NodeProto, model: ModelWrapper) -> bool:
        """Check if this ONNX node can be converted to this hardware kernel.

        This is a validation gate called by InferKernels to determine which kernel
        (if any) should handle a given ONNX node. It performs three checks:

        1. Op type matching: node.op_type in InferencePattern.source_ops
        2. Schema constraints: All KernelSchema.constraints pass (using OnnxValidationContext)
        3. Custom matcher: Optional InferencePattern.matcher(node, model) returns True

        This method should NOT have side effects or raise exceptions - it's a pure
        boolean check. If this returns True, infer_from() will be called to perform
        the actual conversion.

        Override only for validation logic that cannot be expressed declaratively
        in schema constraints or the InferencePattern matcher.

        Args:
            node: ONNX node to validate
            model: ModelWrapper for graph context (shapes, datatypes, layouts)

        Returns:
            True if this kernel can convert the node, False otherwise

        Example:
            # Custom validation beyond declarative constraints
            @classmethod
            def can_infer_from(cls, node, model):
                # First run default validation
                if not super().can_infer_from(node, model):
                    return False

                # Additional imperative checks
                kernel_shape = get_attr(node, "kernel_shape")
                if kernel_shape[0] != kernel_shape[1]:
                    return False  # Only square kernels

                return True
        """
        schema = cls.build_schema(node, model)
        pattern = cls.get_inference_pattern()

        # Check op type matching (empty source_ops means no inference)
        if not pattern.source_ops or node.op_type not in pattern.source_ops:
            return False

        # Run unified constraints through ONNX validation context
        # Pass schema to enable mapping of interface names to ONNX tensor names
        ctx = OnnxValidationContext(node=node, model=model, schema=schema)
        for constraint in schema.constraints:
            error = constraint.check(ctx)
            if error is not None:
                # Constraint failed - this node doesn't match
                logger.debug(
                    f"{cls.__name__} cannot infer from {node.name}: {error}"
                )
                return False

        # Run custom matcher if provided
        if pattern.matcher is not None:
            if not pattern.matcher(node, model):
                logger.debug(
                    f"{cls.__name__} custom matcher rejected {node.name}"
                )
                return False

        return True

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

        Only override if build_schema() needs to read nodeattrs (circular dependency).
        In that case, define nodeattrs explicitly before calling build_schema().
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
            # Invalidate cached model
            self._kernel_model = None
