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

import logging
import math
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
    KernelModel,
    resolve_template
)

logger = logging.getLogger(__name__)


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
        """Get tensor context from cache or node metadata.

        Tries to read from:
        1. Instance cache (_tensor_context)
        2. Node metadata (persisted from previous initialization)

        Raises:
            RuntimeError: If tensor context not available from either source
        """
        if self._tensor_context is None:
            # Try to read from node metadata
            cached_tc = TensorContext.from_node_metadata(self.onnx_node)
            if cached_tc:
                self._tensor_context = cached_tc
            else:
                raise RuntimeError(
                    f"Tensor context not available for node '{self.onnx_node.name}'. "
                    f"Either attach it during node creation or call refresh_tensor_context(model)."
                )
        return self._tensor_context

    def refresh_tensor_context(self, model: ModelWrapper) -> None:
        """Update tensor context from model and persist to node metadata.

        This method:
        1. Extracts tensor context from ModelWrapper (shapes, dtypes)
        2. Caches it in the instance
        3. Attaches it to node.metadata_props for persistence
        4. Invalidates kernel_model if context changed

        Args:
            model: ModelWrapper to extract tensor information from
        """
        new_context = TensorContext.from_model_wrapper(self.onnx_node, model)

        if self._tensor_context != new_context:
            self._tensor_context = new_context
            self._kernel_model = None
            # Persist to node metadata for future use
            new_context.attach_to_node(self.onnx_node)

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
        """Get FINN-compatible folded input shape.

        FINN's dataflow infrastructure expects folded shapes to represent the full
        tensor with time-multiplexing dimensions. This method computes the folded
        shape by dividing each tensor dimension by its corresponding stream dimension,
        then appending the flattened stream parallelism as the final dimension.

        Algorithm:
        1. For each dimension i: fold_factor[i] = tensor_shape[i] / stream_shape[i]
        2. flattened_stream = product(stream_shape)
        3. folded_shape = fold_factors + [flattened_stream]

        Example for tensor=[1, 128, 768], stream=[1, 1, 8]:
            fold_factors = [1/1, 128/1, 768/8] = [1, 128, 96]
            flattened_stream = 1 * 1 * 8 = 8
            folded_shape = [1, 128, 96, 8]

        Returns:
            Folded shape with time-multiplexing and parallelism dimensions
        """
        tensor_shape = self.kernel_model.input_tensor_shape(ind)
        stream_shape = self.kernel_model.input_stream_shape(ind)

        # Compute fold factor for each dimension
        fold_factors = [t // s for t, s in zip(tensor_shape, stream_shape)]

        # Flatten stream parallelism into single dimension
        flattened_stream = math.prod(stream_shape)

        return tuple(fold_factors + [flattened_stream])

    def get_folded_output_shape(self, ind=0):
        """Get FINN-compatible folded output shape.

        See get_folded_input_shape() for explanation of folded shape computation.

        Returns:
            Folded shape with time-multiplexing and parallelism dimensions
        """
        tensor_shape = self.kernel_model.output_tensor_shape(ind)
        stream_shape = self.kernel_model.output_stream_shape(ind)

        # Compute fold factor for each dimension
        fold_factors = [t // s for t, s in zip(tensor_shape, stream_shape)]

        # Flatten stream parallelism into single dimension
        flattened_stream = math.prod(stream_shape)

        return tuple(fold_factors + [flattened_stream])

    def get_instream_width(self, ind=0) -> int:
        return self.kernel_model.input_stream_width_bits(ind)

    def get_outstream_width(self, ind=0) -> int:
        return self.kernel_model.output_stream_width_bits(ind)

    def get_number_output_values(self):
        """Get number of output values for FINN.

        FINN expects this to return the number of time-multiplexed output values,
        which is the product of the folded output shape excluding the parallelism
        dimension (i.e., folded_shape[:-1]).

        This represents the number of cycles/transfers needed to stream the output.

        Returns:
            Number of time-multiplexed output values
        """
        folded_shape = self.get_folded_output_shape()
        return math.prod(folded_shape[:-1])

    def get_exp_cycles(self):
        return self.kernel_model.initiation_interval

    def make_shape_compatible_op(self, model):
        """Create shape-compatible operation for InferShapes transformation.

        This is called by QONNX InferShapes before tensor context initialization,
        so it must access shapes directly from model rather than via kernel_model.

        Default implementation uses input[0] shape (assumes shape-preserving kernel).

        Subclasses should override for:
        - Multi-input kernels with shape dependencies
        - Shape-transforming kernels (Crop, Pool, Upsample, etc.)
        - Kernels without inputs (constant generators)

        Args:
            model: ModelWrapper instance

        Returns:
            ONNX node for shape inference (typically RandomNormal via make_const_shape_op)
        """
        input_shape = tuple(model.get_tensor_shape(self.onnx_node.input[0]))
        return super().make_const_shape_op(input_shape)

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
        logger.debug(f"Building KernelModel for {self.onnx_node.name} ({self.kernel_schema.name})")

        # Build interfaces dict incrementally for DerivedDim/ScaledDim resolution
        interfaces = {}

        input_models = []
        for i in range(len(self.kernel_schema.inputs)):
            input_model = self._create_input_model(i, interfaces)
            if input_model is not None:
                input_models.append(input_model)
                interfaces[input_model.name] = input_model
                logger.debug(
                    f"  Input '{input_model.name}': "
                    f"tensor={input_model.tensor_shape}, "
                    f"block={input_model.block_shape}, "
                    f"stream={input_model.stream_shape}"
                )

        output_models = []
        for i in range(len(self.kernel_schema.outputs)):
            output_model = self._create_output_model(i, interfaces)
            output_models.append(output_model)
            interfaces[output_model.name] = output_model
            logger.debug(
                f"  Output '{output_model.name}': "
                f"tensor={output_model.tensor_shape}, "
                f"block={output_model.block_shape}, "
                f"stream={output_model.stream_shape}"
            )

        # KernelModel.__post_init__ handles dimension resolution
        model = KernelModel(
            name=self.kernel_schema.name,
            inputs=tuple(input_models),
            outputs=tuple(output_models),
        )

        # Unified validation: validate all interfaces in one pass
        logger.debug(f"Validating all interfaces for {self.onnx_node.name}")

        # Validate all inputs
        for i, input_model in enumerate(model.inputs):
            schema = self.kernel_schema.inputs[i]
            if schema.constraints:
                self._validate_interface_constraints(
                    schema.name,
                    input_model,
                    schema.constraints
                )

        # Validate all outputs
        for i, output_model in enumerate(model.outputs):
            schema = self.kernel_schema.outputs[i]
            if schema.constraints:
                self._validate_interface_constraints(
                    schema.name,
                    output_model,
                    schema.constraints
                )

        logger.debug(f"KernelModel built successfully for {self.onnx_node.name}")
        self._kernel_model = model
        return model

    def _create_input_model(self, index: int, interfaces: dict) -> Optional[InputModel]:
        """Create input model from schema and tensor context.

        Args:
            index: Index of input in schema
            interfaces: Dict of already-built interface models for DerivedDim resolution
        """
        schema = self.kernel_schema.inputs[index]
        tensor = self.tensor_context.inputs[index]

        if tensor is None:
            return None

        block_shape = self._resolve_dimensions(
            schema.block_tiling,
            tensor.shape,
            f"Input '{schema.name}' block",
            interfaces
        )

        stream_shape = self._resolve_dimensions(
            schema.stream_tiling,
            block_shape,
            f"Input '{schema.name}' stream",
            interfaces
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

        return input_model

    def _create_output_model(self, index: int, interfaces: dict) -> OutputModel:
        """Create output model from schema - stream_shape may be unset.

        Args:
            index: Index of output in schema
            interfaces: Dict of already-built interface models for DerivedDim resolution
        """
        schema = self.kernel_schema.outputs[index]
        tensor = self.tensor_context.outputs[index]

        block_shape = self._resolve_dimensions(
            schema.block_tiling,
            tensor.shape,
            f"Output '{schema.name}' block",
            interfaces
        )

        # Resolve stream tiling if specified in schema
        if schema.stream_tiling is not None:
            stream_shape = self._resolve_dimensions(
                schema.stream_tiling,
                block_shape,
                f"Output '{schema.name}' stream",
                interfaces
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

        return output_model

    def _resolve_dimensions(
        self,
        template: Any,  # TilingSpec or similar sequence
        reference_shape: Tuple[int, ...],
        context_name: str,
        interfaces: dict
    ) -> Tuple[int, ...]:
        """Resolve template dimensions to concrete values.

        Args:
            template: Template specification (e.g., [":", "PE", DerivedDim("Q", 1)])
            reference_shape: Reference shape to resolve against
            context_name: Context string for error messages
            interfaces: Dict of already-built interface models for DerivedDim resolution
        """
        try:
            return resolve_template(
                template, reference_shape, self.get_nodeattr, context_name, interfaces
            )
        except ValueError as e:
            raise self._error(str(e))

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
            error = constraint.check(interface_model, self.get_nodeattr)
            if error:
                raise self._error(str(error))

