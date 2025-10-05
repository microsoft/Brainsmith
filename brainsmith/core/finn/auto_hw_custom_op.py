############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Base class for automatic hardware custom operators.

Two-level caching system:
1. _tensor_context: Cached from ModelWrapper (only changes with graph updates)
2. _kernel_model: Computed from schema + context + nodeattrs (invalidated on nodeattr changes)

Model Creation Flow:
    1. Extract TensorContext from ModelWrapper
    2. Validate context against schema constraints
    3. Resolve dimensions from templates + nodeattrs + context
    4. Build InputModel/OutputModel objects
    5. Create KernelModel with full validation

Cache invalidation:
  set_nodeattr() → invalidates _kernel_model only (tensor context unchanged)
  graph changes → refresh_tensor_context() updates both caches and syncs graph
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Set, Tuple, Union

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from brainsmith.core.dataflow import (
    KernelSchema,
    TensorContext,
    TensorInfo,
    InputSchema,
    OutputSchema,
    InputModel,
    OutputModel,
    KernelModel,
    create_kernel_model,
    validate_datatype_against_constraints,
    RelationType
)
from brainsmith.core.dataflow.types import prod


class HWCustomOpError(Exception):
    """Exception raised by hardware custom operators with automatic node context.

    Attributes:
        node: The ONNX node where the error occurred
        message: The error message
    """
    def __init__(self, node, message):
        self.node = node
        super().__init__(f"{node.name}: {message}")


class AutoHWCustomOp(HWCustomOp, ABC):
    """Base class for automatic hardware custom operators.

    Key features:
    - Direct model creation from schema + context + nodeattrs
    - Two-phase model creation separating nodeattr and ModelWrapper dependencies
    - Automatic cache invalidation on nodeattr changes
    - Efficient partial updates when only nodeattrs change

    Subclasses must:
    - Define kernel_schema class attribute with KernelSchema instance
    - Implement abstract methods from HWCustomOp base class
    """

    # =============================================================================
    # Class Attributes
    # =============================================================================

    # Subclasses must define this class attribute
    kernel_schema: KernelSchema = None

    # =============================================================================
    # Initialization
    # =============================================================================

    def __init__(self, onnx_node, **kwargs):
        """Initialize with two-level cache system."""
        super().__init__(onnx_node, **kwargs)

        if self.kernel_schema is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define kernel_schema class attribute"
            )

        # Two-level cache: tensor context (from graph) and kernel model (computed)
        self._tensor_context: Optional[TensorContext] = None
        self._kernel_model: Optional[KernelModel] = None

    def _error(self, message: str) -> HWCustomOpError:
        """Create exception with node context.

        Args:
            message: Error message (node name will be prepended automatically)

        Returns:
            HWCustomOpError with node context
        """
        return HWCustomOpError(self.onnx_node, message)

    @property
    def kernel_model(self) -> KernelModel:
        """Get the cached kernel model, building it lazily if needed.

        Returns cached model if available, otherwise builds it from current
        tensor context and nodeattrs.

        Raises:
            RuntimeError: If tensor context not initialized
        """
        if self._kernel_model is None:
            if self._tensor_context is None:
                raise RuntimeError(
                    "Tensor context not initialized. Call refresh_tensor_context() first."
                )
            self.build_model()
        return self._kernel_model

    def refresh_tensor_context(self, model: ModelWrapper) -> None:
        """Update tensor context from model and invalidate kernel model if changed.

        This updates the internal state to match the current graph state.
        The kernel model is only invalidated if the tensor context actually changed.

        Args:
            model: The ModelWrapper containing the current graph state
        """
        # Store old context for comparison
        old_context = self._tensor_context

        # Update tensor context from current graph
        new_context = TensorContext.from_model_wrapper(
            self.onnx_node, model
        )

        # Update tensor context
        self._tensor_context = new_context

        # Invalidate kernel model only if context changed
        if old_context != new_context:
            self._kernel_model = None

        # TAFK TODO: Consider propagating output datatypes back to graph

    # =============================================================================
    # Override - Nodeattr Management with Change Detection
    # =============================================================================

    def set_nodeattr(self, name: str, value: Any) -> None:
        """Set attribute and invalidate kernel model if needed."""
        protected_attrs = {'backend', 'op', 'version'}
        if name in self.kernel_schema.protected_attr_names:
            raise self._error(f"Cannot modify protected attribute '{name}'")
        
        # Get old value for comparison
        old_value = None
        try:
            old_value = self.get_nodeattr(name)
        except (AttributeError, Exception):
            # Attribute not set yet - this is expected for initial setup
            old_value = None

        # Set the new value
        super().set_nodeattr(name, value)

        # If value changed invalidate kernel model
        # (but NOT tensor context - that comes from the graph!)
        if old_value != value:
            # For now, we assume all node_attr updates invalidate kernel model
            # TODO: Consider if this is necessary in all cases 
            self._kernel_model = None

    # =============================================================================
    # Override - HWCustomOp Interface Methods (Delegating to Cached Model)
    # =============================================================================

    def get_folded_output_shape(self, ind=0):
        """Get folded output shape using cached model (returns stream_shape)."""
        return list(self.kernel_model.output_stream_shape(ind))

    def get_number_output_values(self):
        """Get total output values using cached model."""
        return self.kernel_model.total_output_values

    def get_exp_cycles(self):
        """Get expected cycles using cached model."""
        return self.kernel_model.initiation_interval

    def get_input_datatype(self, ind=0) -> DataType:
        """Get input datatype from cached model."""
        return self.kernel_model.inputs[ind].datatype

    def get_output_datatype(self, ind=0) -> DataType:
        """Get output datatype from cached model."""
        return self.kernel_model.outputs[ind].datatype

    def get_normal_input_shape(self, ind=0) -> List[int]:
        """Get normal input shape from cached model."""
        return list(self.kernel_model.inputs[ind].tensor_shape)

    def get_normal_output_shape(self, ind=0) -> List[int]:
        """Get normal output shape from cached model."""
        return list(self.kernel_model.outputs[ind].tensor_shape)

    def get_folded_input_shape(self, ind=0) -> List[int]:
        """Get folded input shape using cached model (returns stream_shape)."""
        return list(self.kernel_model.inputs[ind].stream_shape)

    def get_instream_width(self, ind=0) -> int:
        """Get input stream width in bits."""
        return self.kernel_model.inputs[ind].stream_width_bits

    def get_outstream_width(self, ind=0) -> int:
        """Get output stream width in bits."""
        return self.kernel_model.output_stream_width_bits(ind)
    
    def infer_node_datatype(self, model):
        """FINN compatibility: Infer and set node datatypes from graph context.

        This is a compatibility wrapper for legacy FINN code. Modern code should
        use refresh_tensor_context() directly.

        Args:
            model: The ModelWrapper containing the graph
        """
        self.refresh_tensor_context(model)

    # =============================================================================
    # Private - Model Creation Methods (formerly DirectKernelFactory)
    # =============================================================================

    def build_model(self) -> KernelModel:
        """
        Create a KernelModel directly from schema, context, and nodeattrs.

        This replaces DirectKernelFactory.build_model() but works directly
        with self.get_nodeattr() instead of a nodeattrs dict.

        Returns:
            KernelModel: Fully validated and ready-to-use model

        Raises:
            ValueError: If validation fails at any stage
        """
        if self._tensor_context is None:
            raise RuntimeError("Tensor context not initialized")

        # Build input models
        input_models = []
        for i in range(len(self.kernel_schema.inputs)):
            input_model = self._create_input_model(i)
            if input_model is not None:  # Skip optional inputs
                input_models.append(input_model)

        # Build output models
        output_models = []
        for i in range(len(self.kernel_schema.outputs)):
            output_models.append(self._create_output_model(i))

        # Build kernel model
        model = KernelModel(
            name=self.kernel_schema.name,
            inputs=tuple(input_models),
            outputs=tuple(output_models),
        )

        # Validate model against schema (includes relationships and streaming constraints)
        validation_result = self.kernel_schema.validate(model)
        if not validation_result.is_valid:
            error_msgs = [v.message for v in validation_result.violations if v.severity == "error"]
            raise self._error(f"Model validation failed: {'; '.join(error_msgs)}")

        self._kernel_model = model
        return model

    def _create_input_model(self, index: int) -> Optional[InputModel]:
        """Create input model from schema and tensor context."""
        # Skip optional inputs not present
        schema = self.kernel_schema.inputs[index]
        tensor = self._tensor_context.inputs[index]

        if schema.optional and tensor.name == "":
            return None

        block_shape = self._resolve_dimensions(
            schema.block_tiling,
            tensor.shape,
            f"Input '{schema.name}' block"
        )

        stream_shape = self._resolve_dimensions(
            schema.stream_tiling,
            block_shape,
            f"Input '{schema.name}' stream"
        )

        # Get datatype from nodeattr (already set and validated)
        datatype_attr = self.kernel_schema.get_datatype_attr(index)
        datatype = DataType[self.get_nodeattr(datatype_attr)]

        # Create model
        return InputModel(
            name=schema.name,
            tensor_shape=tensor.shape,
            block_shape=block_shape,
            stream_shape=stream_shape,
            datatype=datatype,
            is_weight=schema.is_weight
        )

    def _create_output_model(self, index: int) -> OutputModel:
        """Create output model from schema and tensor context."""
        # Resolve block dimensions
        schema = self.kernel_schema.outputs[index]
        tensor = self._tensor_context.outputs[index]

        block_shape = self._resolve_dimensions(
            schema.block_tiling,
            tensor.shape,
            f"Output '{schema.name}' block"
        )

        # Get datatype from nodeattr (already set and validated)
        datatype_attr = self.kernel_schema.get_datatype_attr(index, False)
        datatype = DataType[self.get_nodeattr(datatype_attr)]

        # Create model
        return OutputModel(
            name=schema.name,
            tensor_shape=tensor.shape,
            block_shape=block_shape,
            datatype=datatype
        )

    def _resolve_dimensions(
        self,
        template: List[Any],
        reference_shape: List[int],
        context_name: str
    ) -> List[int]:
        """Resolve template dimensions to concrete values."""
        # Pad with 1s at beginning if template is shorter than the reference
        if len(template) < len(reference_shape):
            padding = len(reference_shape) - len(template)
            template = [1] * padding + template
        elif len(template) > len(reference_shape):
            raise self._error(
                f"{context_name}: template length {len(template)} exceeds "
                f"tensor rank {len(reference_shape)}"
            )

        # Then process each dimension
        resolved = []
        for i, (dim, ref) in enumerate(zip(template, reference_shape)):
            if isinstance(dim, str):
                if dim == ":":  # Full dimension
                    value = ref
                else:
                    # Parameter name - resolve from nodeattrs
                    try:
                        value = self.get_nodeattr(dim)
                    except AttributeError:
                        raise self._error(
                            f"{context_name}[{i}]: parameter '{dim}' not found in node attributes"
                        )
                    # This is the only tiling check necessary, since slice and 1 are always valid
                    if ref % value != 0:
                        raise self._error(
                            f"{context_name}[{i}]: parameter '{dim}' value {value} "
                            f"does not divide parent dimension size {ref}"
                        )
            elif isinstance(dim, int):
                if dim == 1:
                    value = 1  # Singleton literal
                else:
                    raise self._error(
                        f"{context_name}[{i}]: only singleton (1) allowed for literals, "
                        f"got {dim}. Use parameters for other values."
                    )
                # TODO: Consider if any legitimate reason to loosen literal restrictions
            else:
                raise self._error(
                    f"{context_name}[{i}]: invalid template element '{dim}'"
                )

            resolved.append(value)

        return resolved
