############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Kernel inference infrastructure for HW layer inference.

This module provides the core infrastructure for kernel inference:
- InferencePattern: Defines ONNX → HW kernel pattern matching
- InferenceResult: Structured result from inference operations
- InferenceHelper: Utility methods for common inference operations

Validation is handled by unified Constraint system (see constraints.py).

Example usage:
    # Define inference pattern
    pattern = InferencePattern(
        source_ops=["Add"],
        layout_conversions={"input0": "NHWC", "input1": "NHWC"},
    )

    # Using helper for graph manipulation
    helper = InferenceHelper(model)
    in0 = helper.ensure_layout(node.input[0], "NHWC", insert_index)
    new_node = helper.make_node("MyKernel", [in0], [out], {"PE": 1})
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any

from onnx import NodeProto, TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
import qonnx.core.data_layout as DataLayout
from qonnx.util.onnx import nchw_to_nhwc

logger = logging.getLogger(__name__)


@dataclass
class InferencePattern:
    """Defines ONNX → HW kernel pattern matching.

    Specifies which ONNX operations can be converted to this kernel and
    provides hints for layout conversion. Validation is handled by the
    unified Constraint system (see KernelSchema.constraints).

    This is NOT about validation - that's in schema.constraints.
    This is about:
    - Which ONNX ops to match
    - Layout conversion hints
    - Custom matching logic beyond constraints

    Attributes:
        source_ops: ONNX op types that can be converted to this kernel
        layout_conversions: Optional dict mapping tensor names to target layouts
            Example: {"input0": "NHWC", "input1": "NHWC", "output": "NHWC"}
        matcher: Optional custom matching function beyond constraints
            Signature: (node: NodeProto, model: ModelWrapper) -> bool

    Example:
        # AddStreams: Convert ONNX Add to AddStreams HW kernel
        InferencePattern(
            source_ops=["Add"],
            layout_conversions={
                "input0": "NHWC",
                "input1": "NHWC",
                "output": "NHWC",
            }
        )

        # MVAU: MatMul with custom matching
        InferencePattern(
            source_ops=["MatMul"],
            matcher=lambda node, model: check_weight_is_static(node, model)
        )
    """

    source_ops: List[str]
    layout_conversions: Optional[Dict[str, str]] = None
    matcher: Optional[Callable[[NodeProto, ModelWrapper], bool]] = None


@dataclass
class InferenceResult:
    """Result of kernel inference operation.

    Contains the graph modifications needed to replace an ONNX node
    with hardware-specific custom ops.

    Attributes:
        nodes_to_insert: New nodes to insert into the graph
        nodes_to_remove: Existing nodes to remove from the graph
        metadata: Optional metadata about the inference (for debugging/logging)

    Example:
        InferenceResult(
            nodes_to_insert=[addstreams_node],
            nodes_to_remove=[original_add_node],
            metadata={"num_channels": 128, "layout_converted": True}
        )
    """

    nodes_to_insert: List[NodeProto]
    nodes_to_remove: List[NodeProto]
    metadata: Dict[str, Any] = field(default_factory=dict)


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

    def __init__(self, model: ModelWrapper):
        """Initialize helper with model context.

        Args:
            model: QONNX ModelWrapper for graph access
        """
        self.model = model
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
            domain="finn.custom_op.fpgadataflow",
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
