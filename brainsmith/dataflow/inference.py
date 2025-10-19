############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Kernel inference infrastructure for HW layer inference.

This module provides helper utilities for kernel inference:
- InferenceHelper: Utility methods for common inference operations

Discovery is handled by KernelOp.get_source_ops().
Validation is handled by unified Constraint system (see constraints.py).
Transformation is handled by TransformationSpec/TransformationResult (see transformation.py).

Example usage:
    # Using helper for graph manipulation
    helper = InferenceHelper(model, domain="brainsmith.kernels")
    in0 = helper.ensure_layout(node.input[0], "NHWC", insert_index)
    new_node = helper.make_node("MyKernel", [in0], [out], {"PE": 1})
"""

import logging
import warnings
from typing import Optional, Dict, Any, List

from onnx import NodeProto, TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
import qonnx.core.data_layout as DataLayout
from qonnx.util.onnx import nchw_to_nhwc

logger = logging.getLogger(__name__)


class InferenceHelper:
    """Helper utilities for kernel inference operations.

    Encapsulates common operations needed during kernel inference:
    - Layout conversion with automatic Transpose insertion
    - Intermediate tensor creation
    - HW node creation with standard attributes
    - Shape/datatype extraction

    Example:
        helper = InferenceHelper(model)

        # Ensure input is NHWC (inserts Transpose if needed)
        nhwc_input = helper.ensure_layout(node.input[0], "NHWC", insert_index)

        # Create intermediate tensor
        intermediate = helper.make_intermediate_tensor(
            shape=[1, 224, 224, 64],
            dtype=DataType["INT8"],
            layout="NHWC"
        )

        # Create HW node
        hw_node = helper.make_node(
            "AddStreams",
            inputs=[nhwc_input, node.input[1]],
            outputs=[node.output[0]],
            attributes={"NumChannels": 64, "PE": 1}
        )
    """

    def __init__(self, model: ModelWrapper, domain: str = "brainsmith.kernels"):
        """Initialize helper with model context.

        Args:
            model: QONNX ModelWrapper for graph access
            domain: ONNX domain for created nodes (default: "brainsmith.kernels")
                   Use "finn.custom_op.fpgadataflow" for FINN kernels
        """
        self.model = model
        self.domain = domain
        self._insert_count = 0

    def ensure_layout(
        self,
        tensor_name: str,
        target_layout: str,
        insert_index: int
    ) -> str:
        """Ensure tensor is in target layout, inserting Transpose if needed.

        If the tensor is already in the target layout, returns the tensor name
        unchanged. Otherwise, inserts a Transpose node and returns the new
        tensor name.

        Args:
            tensor_name: Name of tensor to check/convert
            target_layout: Desired data layout ("NCHW" or "NHWC")
            insert_index: Where to insert Transpose node if needed

        Returns:
            Tensor name (original if no conversion needed, new if converted)

        Example:
            # Convert NCHW input to NHWC (may insert Transpose)
            nhwc_input = helper.ensure_layout(node.input[0], "NHWC", insert_index)
        """
        current_layout = self.model.get_tensor_layout(tensor_name)

        # Convert target_layout string to DataLayout enum value
        target_layout_enum = getattr(DataLayout, target_layout, None)
        if target_layout_enum is None:
            warnings.warn(f"Unknown target layout: {target_layout}")
            return tensor_name

        if current_layout == target_layout_enum:
            return tensor_name

        # Convert using qonnx utility (handles node insertion)
        if current_layout == DataLayout.NCHW and target_layout == "NHWC":
            new_tensor = nchw_to_nhwc(tensor_name, self.model, insert_index)
        elif current_layout == DataLayout.NHWC and target_layout == "NCHW":
            new_tensor = nchw_to_nhwc(tensor_name, self.model, insert_index, reverse=True)
        else:
            warnings.warn(
                f"Unsupported layout conversion: {current_layout} -> {target_layout}. "
                f"Returning tensor unchanged."
            )
            return tensor_name

        self._insert_count += 1
        logger.debug(f"Converted {tensor_name} from {current_layout} to {target_layout}")
        return new_tensor

    def make_intermediate_tensor(
        self,
        shape: List[int],
        dtype: DataType,
        layout: Optional[str] = None
    ) -> str:
        """Create a new intermediate tensor with given shape and datatype.

        Args:
            shape: Tensor shape (e.g., [1, 224, 224, 64])
            dtype: QONNX DataType
            layout: Optional data layout annotation

        Returns:
            Name of newly created tensor

        Example:
            intermediate = helper.make_intermediate_tensor(
                shape=[1, 224, 224, 64],
                dtype=DataType["INT8"],
                layout="NHWC"
            )
        """
        tensor = helper.make_tensor_value_info(
            self.model.make_new_valueinfo_name(),
            TensorProto.FLOAT,
            shape
        )
        self.model.graph.value_info.append(tensor)
        self.model.set_tensor_datatype(tensor.name, dtype)

        if layout is not None:
            self.model.set_tensor_layout(tensor.name, DataLayout[layout])

        logger.debug(f"Created intermediate tensor {tensor.name} with shape {shape}")
        return tensor.name

    def make_node(
        self,
        op_type: str,
        inputs: List[str],
        outputs: List[str],
        attributes: Dict[str, Any],
        name_prefix: Optional[str] = None
    ) -> NodeProto:
        """Create HW node with standard domain/backend attributes.

        Args:
            op_type: HW operator type (e.g., "AddStreams", "MVAU")
            inputs: List of input tensor names
            outputs: List of output tensor names
            attributes: Node attributes (PE, NumChannels, etc.)
            name_prefix: Optional prefix for node name (defaults to op_type)

        Returns:
            Created ONNX NodeProto

        Example:
            node = helper.make_node(
                "AddStreams",
                inputs=[in0, in1],
                outputs=[out],
                attributes={"NumChannels": 64, "PE": 1},
                name_prefix="AddStreams_layer1"
            )
        """
        node_name = name_prefix or f"{op_type}_{hash(tuple(inputs)) & 0xFFFFFFFF}"

        return helper.make_node(
            op_type,
            inputs,
            outputs,
            domain=self.domain,
            backend="fpgadataflow",
            name=node_name,
            **attributes
        )

    def get_num_channels(self, tensor_name: str) -> int:
        """Get number of channels from tensor (assumes NHWC layout).

        Args:
            tensor_name: Name of tensor to query

        Returns:
            Number of channels (last dimension)

        Example:
            num_ch = helper.get_num_channels(node.input[0])  # Returns 64 for (1,224,224,64)
        """
        shape = self.model.get_tensor_shape(tensor_name)
        return int(shape[-1])

    def get_num_input_vectors(self, tensor_name: str) -> List[int]:
        """Get input vector shape (all dimensions except last).

        Args:
            tensor_name: Name of tensor to query

        Returns:
            Shape excluding last dimension (e.g., [1, 224, 224] for (1,224,224,64))

        Example:
            num_vecs = helper.get_num_input_vectors(node.input[0])
        """
        shape = self.model.get_tensor_shape(tensor_name)
        return list(shape[:-1])

    # ===================================================================
    # Type Checking Utilities (multi-tensor operations)
    # ===================================================================

    def is_integer_tensor(self, tensor_name: str) -> bool:
        """Check if tensor has integer datatype.

        Args:
            tensor_name: Name of tensor to check

        Returns:
            True if tensor has integer datatype

        Example:
            if not helper.is_integer_tensor(node.input[0]):
                return None  # Skip inference
        """
        dt = self.model.get_tensor_datatype(tensor_name)
        return dt is not None and dt.is_integer()

    def all_integer_tensors(self, tensor_names: List[str]) -> bool:
        """Check if all tensors have integer datatypes.

        Multi-tensor convenience method.
        Reduces: 4 lines → 1 line

        Args:
            tensor_names: List of tensor names to check

        Returns:
            True if all tensors are integers, False otherwise

        Example:
            # Before
            dt0 = model.get_tensor_datatype(in0)
            dt1 = model.get_tensor_datatype(in1)
            if not dt0.is_integer() or not dt1.is_integer():
                return None

            # After
            if not helper.all_integer_tensors([in0, in1]):
                return None
        """
        return all(
            self.model.get_tensor_datatype(t) is not None and
            self.model.get_tensor_datatype(t).is_integer()
            for t in tensor_names
        )

    def datatypes_match(self, *tensor_names: str) -> bool:
        """Check if all tensors have identical datatypes.

        Multi-tensor comparison utility.
        Reduces: 3 lines → 1 line

        Args:
            *tensor_names: Variable number of tensor names

        Returns:
            True if all tensors have same datatype, False otherwise

        Example:
            # Before
            dt0 = model.get_tensor_datatype(in0)
            dt1 = model.get_tensor_datatype(in1)
            if dt0 != dt1:
                return None

            # After
            if not helper.datatypes_match(in0, in1):
                return None
        """
        if not tensor_names:
            return True
        datatypes = [self.model.get_tensor_datatype(t) for t in tensor_names]
        if any(dt is None for dt in datatypes):
            return False
        return len(set(datatypes)) == 1

    # ===================================================================
    # Static/Dynamic Detection (readability predicates)
    # ===================================================================

    def is_static(self, tensor_name: str) -> bool:
        """Check if tensor has initializer (is static/constant).

        Eliminates double-negative pattern.
        Reduces: 1 line → 1 line (but more readable)

        Args:
            tensor_name: Name of tensor to check

        Returns:
            True if tensor has initializer

        Example:
            # Before
            in0_static = not (model.get_initializer(in0) is None)

            # After
            in0_static = helper.is_static(in0)
        """
        return self.model.get_initializer(tensor_name) is not None

    def is_dynamic(self, tensor_name: str) -> bool:
        """Check if tensor is dynamic (no initializer).

        Args:
            tensor_name: Name of tensor to check

        Returns:
            True if tensor is dynamic (no initializer)
        """
        return self.model.get_initializer(tensor_name) is None

    def any_static(self, tensor_names: List[str]) -> bool:
        """Check if any tensors are static.

        Multi-tensor convenience method.
        Reduces: 3-4 lines → 1 line

        Args:
            tensor_names: List of tensor names to check

        Returns:
            True if at least one tensor is static

        Example:
            # Before
            in0_static = not (model.get_initializer(in0) is None)
            in1_static = not (model.get_initializer(in1) is None)
            if in0_static or in1_static:
                return None  # Skip - need all dynamic

            # After
            if helper.any_static([in0, in1]):
                return None  # Skip - need all dynamic
        """
        return any(self.model.get_initializer(t) is not None for t in tensor_names)

    def all_dynamic(self, tensor_names: List[str]) -> bool:
        """Check if all tensors are dynamic.

        Multi-tensor convenience method.

        Args:
            tensor_names: List of tensor names

        Returns:
            True if all tensors are dynamic (no initializers)
        """
        return all(self.model.get_initializer(t) is None for t in tensor_names)

    # ===================================================================
    # Shape Validation (comparison & computed properties)
    # ===================================================================

    def shapes_match(self, *tensor_names: str) -> bool:
        """Check if all tensors have identical shapes.

        Multi-tensor comparison utility.
        Reduces: 3 lines → 1 line

        Args:
            *tensor_names: Variable number of tensor names

        Returns:
            True if all tensors have same shape, False otherwise

        Example:
            # Before
            shape0 = model.get_tensor_shape(in0)
            shape1 = model.get_tensor_shape(in1)
            if shape0 != shape1:
                return None

            # After
            if not helper.shapes_match(in0, in1):
                return None
        """
        if not tensor_names:
            return True
        shapes = [self.model.get_tensor_shape(t) for t in tensor_names]
        if any(s is None for s in shapes):
            return False
        # Convert to tuples for set comparison
        return len(set(tuple(s) for s in shapes)) == 1

    def is_4d_tensor(self, tensor_name: str) -> bool:
        """Check if tensor is 4D (common for conv/pool operations).

        Computed predicate (more readable than len(shape) == 4).

        Args:
            tensor_name: Name of tensor to check

        Returns:
            True if tensor has 4 dimensions
        """
        shape = self.model.get_tensor_shape(tensor_name)
        return shape is not None and len(shape) == 4

    def get_spatial_dims(self, tensor_name: str):
        """Get spatial dimensions (H, W) for 4D NHWC tensor.

        Computed property - extracts H, W from 4D shape.

        Args:
            tensor_name: Name of tensor

        Returns:
            Tuple of (height, width)

        Raises:
            ValueError: If tensor is not 4D

        Example:
            # Before
            shape = model.get_tensor_shape(tensor_name)
            h, w = shape[1], shape[2]

            # After
            h, w = helper.get_spatial_dims(tensor_name)
        """
        shape = self.model.get_tensor_shape(tensor_name)
        if shape is None or len(shape) != 4:
            raise ValueError(f"{tensor_name} is not 4D (shape: {shape})")
        return (shape[1], shape[2])

    # ===================================================================
    # Graph Topology (pattern matching helpers)
    # ===================================================================

    def has_consumer_of_type(self, tensor_name: str, op_type: str) -> bool:
        """Check if tensor has consumer with given op_type.

        Pattern matching convenience (not a direct wrapper).
        Useful for fusion detection.

        Args:
            tensor_name: Name of tensor
            op_type: ONNX op_type to match

        Returns:
            True if tensor has single consumer with matching op_type

        Example:
            # Before
            consumer = model.find_consumer(mm_output)
            if consumer is not None and consumer.op_type == "MultiThreshold":
                # Merge nodes into single MVAU
                ...

            # After
            if helper.has_consumer_of_type(mm_output, "MultiThreshold"):
                consumer = model.find_consumer(mm_output)
                # Merge nodes into single MVAU
                ...
        """
        consumer = self.model.find_consumer(tensor_name)
        return consumer is not None and consumer.op_type == op_type

    def has_multiple_consumers(self, tensor_name: str) -> bool:
        """Check if tensor has multiple consumers (fanout > 1).

        Computed predicate (more readable than get_tensor_fanout() > 1).

        Args:
            tensor_name: Name of tensor

        Returns:
            True if fanout > 1
        """
        return self.model.get_tensor_fanout(tensor_name) > 1

    # ===================================================================
    # Transformation Support (for TransformationSpec system)
    # ===================================================================

    def get_layout(self, tensor_name: str) -> Optional[str]:
        """Get current layout of tensor as string.

        Args:
            tensor_name: Name of tensor to query

        Returns:
            Layout as string ("NCHW", "NHWC", etc.) or None if not set

        Example:
            layout = helper.get_layout(node.input[0])  # Returns "NHWC"
        """
        layout_enum = self.model.get_tensor_layout(tensor_name)
        if layout_enum is None:
            return None
        # Convert DataLayout enum to string
        return layout_enum.name

    def get_node_attr_value(self, node: NodeProto, attr_name: str) -> Any:
        """Get ONNX node attribute value.

        Convenience wrapper around onnx.helper.get_attribute_value().

        Args:
            node: ONNX node to query
            attr_name: Attribute name to retrieve

        Returns:
            Attribute value

        Raises:
            AttributeError: If attribute not found

        Example:
            epsilon = helper.get_node_attr_value(node, "epsilon")
            axis = helper.get_node_attr_value(node, "axis")
        """
        from onnx import helper as onnx_helper
        return onnx_helper.get_attribute_value(
            next((attr for attr in node.attribute if attr.name == attr_name), None)
        )

