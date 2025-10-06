############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Base class for hardware custom operators automated with Dataflow Modeling.

Two-level caching system:
1. _tensor_context: Cached from ModelWrapper (changes with graph updates)
2. _kernel_model: Computed from schema + context + nodeattrs (invalidated on nodeattr changes)

Cache invalidation:
  set_nodeattr() → invalidates _kernel_model only
  refresh_tensor_context() → updates _tensor_context, invalidates _kernel_model if context changed
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from brainsmith.core.dataflow import (
    KernelSchema,
    TensorContext,
    InputModel,
    OutputModel,
    KernelModel
)


class HWCustomOpError(Exception):
    """Exception raised by hardware custom operators with automatic node context."""
    def __init__(self, node, message):
        self.node = node
        super().__init__(f"{node.name}: {message}")


class AutoHWCustomOp(HWCustomOp, ABC):
    """Base class for automatic hardware custom operators.

    Subclasses must:
    - Define kernel_schema class attribute with KernelSchema instance
    - Implement abstract methods from HWCustomOp base class
    """

    kernel_schema: KernelSchema  # type: ignore[assignment]  # Subclasses must define

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

        if self.kernel_schema is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define kernel_schema class attribute"
            )

        self._tensor_context: Optional[TensorContext] = None
        self._kernel_model: Optional[KernelModel] = None

    def _error(self, message: str) -> HWCustomOpError:
        """Create exception with node context."""
        return HWCustomOpError(self.onnx_node, message)

    @property
    def tensor_context(self) -> TensorContext:
        """Get tensor context, raising if not initialized.

        Raises:
            RuntimeError: If tensor context not initialized
        """
        if self._tensor_context is None:
            raise RuntimeError(
                "Tensor context not initialized. Call refresh_tensor_context() first."
            )
        return self._tensor_context

    def refresh_tensor_context(self, model: ModelWrapper) -> None:
        """Update tensor context from model, invalidate kernel model if context changed."""
        new_context = TensorContext.from_model_wrapper(self.onnx_node, model)

        if self._tensor_context != new_context:
            self._tensor_context = new_context
            self._kernel_model = None

    @property
    def kernel_model(self) -> KernelModel:
        """Lazy-built kernel model from tensor context and nodeattrs.

        Raises:
            RuntimeError: If tensor context not initialized
        """
        if self._kernel_model is None:
            self.build_model()
            assert self._kernel_model is not None  # build_model() sets this
        return self._kernel_model

    def infer_node_datatype(self, model):
        """FINN compatibility wrapper. Modern code should use refresh_tensor_context()."""
        self.refresh_tensor_context(model)

    # Public API - FINN HWCustomOp Interface

    def get_input_datatype(self, ind=0) -> DataType:
        return self.kernel_model.input_datatype(ind)

    def get_output_datatype(self, ind=0) -> DataType:
        return self.kernel_model.output_datatype(ind)

    def get_normal_input_shape(self, ind=0) -> List[int]:
        return list(self.kernel_model.input_tensor_shape(ind))

    def get_normal_output_shape(self, ind=0) -> List[int]:
        return list(self.kernel_model.output_tensor_shape(ind))

    def get_folded_input_shape(self, ind=0) -> List[int]:
        return list(self.kernel_model.input_stream_shape(ind))

    def get_folded_output_shape(self, ind=0):
        return list(self.kernel_model.output_stream_shape(ind))

    def get_instream_width(self, ind=0) -> int:
        return self.kernel_model.input_stream_width_bits(ind)

    def get_outstream_width(self, ind=0) -> int:
        return self.kernel_model.output_stream_width_bits(ind)

    def get_number_output_values(self):
        return self.kernel_model.total_output_values

    def get_exp_cycles(self):
        return self.kernel_model.initiation_interval

    # Overrides

    def set_nodeattr(self, name: str, value: Any) -> None:
        """Set attribute and invalidate kernel model cache if value changed."""
        if name in self.kernel_schema.protected_attr_names:
            raise self._error(f"Cannot modify protected attribute '{name}'")

        old_value = None
        try:
            old_value = self.get_nodeattr(name)
        except (AttributeError, Exception):
            pass

        if old_value != value:
            super().set_nodeattr(name, value)
            self._kernel_model = None

    # Private Implementation

    def build_model(self) -> KernelModel:
        """Create KernelModel from schema, tensor context, and nodeattrs.

        Raises:
            RuntimeError: If tensor context not initialized
            HWCustomOpError: If validation fails
        """
        input_models = []
        for i in range(len(self.kernel_schema.inputs)):
            input_model = self._create_input_model(i)
            if input_model is not None:
                input_models.append(input_model)

        output_models = []
        for i in range(len(self.kernel_schema.outputs)):
            output_models.append(self._create_output_model(i))

        # KernelModel.__post_init__ handles dimension resolution and validation
        model = KernelModel(
            name=self.kernel_schema.name,
            inputs=tuple(input_models),
            outputs=tuple(output_models),
            relationships=self.kernel_schema.relationships,
        )

        self._kernel_model = model
        return model

    def _create_input_model(self, index: int) -> Optional[InputModel]:
        """Create input model from schema and tensor context."""
        schema = self.kernel_schema.inputs[index]
        tensor = self.tensor_context.inputs[index]

        if tensor is None:
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

        datatype_attr = self.kernel_schema.get_datatype_attr(index)
        datatype = DataType[self.get_nodeattr(datatype_attr)]

        input_model = InputModel(
            name=schema.name,
            tensor_shape=tensor.shape,
            block_shape=block_shape,
            stream_shape=stream_shape,
            datatype=datatype,
            is_weight=schema.is_weight
        )

        # Validate interface constraints AFTER creating model
        self._validate_interface_constraints(
            schema.name,
            input_model,
            schema.constraints
        )

        return input_model

    def _create_output_model(self, index: int) -> OutputModel:
        """Create output model from schema - stream_shape may be unset."""
        schema = self.kernel_schema.outputs[index]
        tensor = self.tensor_context.outputs[index]

        block_shape = self._resolve_dimensions(
            schema.block_tiling,
            tensor.shape,
            f"Output '{schema.name}' block"
        )

        # Resolve stream tiling if specified in schema
        if schema.stream_tiling is not None:
            stream_shape = self._resolve_dimensions(
                schema.stream_tiling,
                block_shape,
                f"Output '{schema.name}' stream"
            )
        else:
            # Leave unset - will be resolved by KernelModel.__post_init__
            stream_shape = tuple([None] * len(block_shape))

        datatype_attr = self.kernel_schema.get_datatype_attr(index, False)
        datatype = DataType[self.get_nodeattr(datatype_attr)]

        output_model = OutputModel(
            name=schema.name,
            tensor_shape=tensor.shape,
            block_shape=block_shape,
            stream_shape=stream_shape,
            datatype=datatype
        )

        # Validate interface constraints (only if stream_shape fully resolved)
        if not output_model.has_unset_dims():
            self._validate_interface_constraints(
                schema.name,
                output_model,
                schema.constraints
            )

        return output_model

    def _resolve_dimensions(
        self,
        template: Any,  # TilingSpec or similar sequence
        reference_shape: Tuple[int, ...],
        context_name: str
    ) -> List[int]:
        """Resolve template dimensions to concrete values."""
        if len(template) < len(reference_shape):
            padding = len(reference_shape) - len(template)
            template = [1] * padding + template
        elif len(template) > len(reference_shape):
            raise self._error(
                f"{context_name}: template length {len(template)} exceeds "
                f"tensor rank {len(reference_shape)}"
            )

        resolved = []
        for i, (dim, ref) in enumerate(zip(template, reference_shape)):
            if isinstance(dim, str):
                if dim == ":":
                    value = ref
                else:
                    try:
                        value = self.get_nodeattr(dim)
                    except AttributeError:
                        raise self._error(
                            f"{context_name}[{i}]: parameter '{dim}' not found in node attributes"
                        )
                    if ref % value != 0:
                        raise self._error(
                            f"{context_name}[{i}]: parameter '{dim}' value {value} "
                            f"does not divide parent dimension size {ref}"
                        )
            elif isinstance(dim, int):
                if dim == 1:
                    value = 1
                else:
                    raise self._error(
                        f"{context_name}[{i}]: only singleton (1) allowed for literals, "
                        f"got {dim}. Use parameters for other values."
                    )
            else:
                raise self._error(
                    f"{context_name}[{i}]: invalid template element '{dim}'"
                )

            resolved.append(value)

        return resolved

    def _validate_interface_constraints(
        self,
        interface_name: str,
        interface_model: Any,
        constraints: List['InterfaceConstraint']
    ) -> None:
        """Validate interface constraints for a single interface.

        Args:
            interface_name: Name of interface being validated
            interface_model: Interface model (InputModel or OutputModel) to validate
            constraints: List of InterfaceConstraints from schema

        Raises:
            HWCustomOpError: If any constraint is violated
        """
        for constraint in constraints:
            error_msg = constraint.check(interface_model, self.get_nodeattr)
            if error_msg:
                raise self._error(f"{interface_name}: {error_msg}")

