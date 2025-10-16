############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
InferSoftmax - Transform ONNX Softmax nodes to hardware Softmax operations.

Transforms standard ONNX Softmax nodes into hardware Softmax operations
using the modern KernelOp infrastructure with Dataflow Modeling.

Key features:
- Creates "Softmax" nodes using KernelOp base class
- Does NOT set ifm_dim or NumChannels (inferred from tensor context via kernel_model)
- Relies on declarative KernelSchema for shape information
- Automatic shape inference and validation

The transformation process:
1. Find Softmax nodes in the graph
2. Validate axis attribute (must be -1 or None)
3. Create Softmax node with minimal attributes (SIMD=1)
4. Remove old Softmax node
5. Rerun shape and datatype inference
6. Initialize tensor context for all Softmax nodes (enables kernel_model)

Example:
    from brainsmith.kernels.softmax.infer_softmax import InferSoftmax

    model = ModelWrapper(...)
    model = model.transform(InferSoftmax())

    # Result: ONNX Softmax nodes replaced with hardware Softmax nodes
"""

from onnx import helper
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.custom_op.registry import getCustomOp
from brainsmith.core.plugins import transform


@transform(
    kernel="Softmax",
    description="Convert ONNX Softmax nodes to hardware Softmax operations",
    author="Thomas Keller"
)
class InferSoftmax(Transformation):
    """Convert standard ONNX Softmax into hardware Softmax operations.

    This transform uses the modern KernelOp infrastructure with
    Dataflow Modeling system for automatic shape inference.

    The key benefit is that shape information (ifm_dim, NumChannels) is inferred
    automatically from the tensor context via kernel_model, eliminating redundancy
    and potential inconsistency.

    Only processes Softmax nodes with axis=-1 or axis=None (last dimension).

    Attributes explicitly set on Softmax node:
    - SIMD: Parallelization factor (default 1)

    Attributes set automatically by refresh_df_model():
    - _input0Datatype: Populated from model graph
    - _output0Datatype: Populated from model graph (always FLOAT32 for Softmax)
    - _input0TensorShape, _input0BlockShape, _input0StreamShape
    - _output0TensorShape, _output0BlockShape, _output0StreamShape
    """

    def apply(self, model):
        """Apply InferSoftmax transformation to model.

        Args:
            model: ModelWrapper containing ONNX graph

        Returns:
            Tuple of (transformed_model, graph_modified_flag)
        """
        graph = model.graph
        node_ind = 0
        graph_modified = False

        for node in graph.node:
            node_ind += 1
            # Only process ONNX Softmax nodes (not our hardware Softmax nodes)
            if node.op_type == "Softmax" and node.domain != "brainsmith.kernels":
                # Get input/output tensor names
                input_tensor = node.input[0]
                output_tensor = node.output[0]

                # Skip if shape not available (shouldn't happen with proper inference)
                input_shape = model.get_tensor_shape(input_tensor)
                if input_shape is None or len(input_shape) == 0:
                    continue

                # Only support normalization over last dimension (axis=-1)
                axis = helper.get_node_attr_value(node, "axis")
                if axis is not None and axis != -1:
                    continue

                # Get channel count
                channels = input_shape[-1]

                # Create node with no parallelization first (SIMD=1)
                simd = 1
                assert channels % simd == 0, f"Requirement channel ({channels}) divisible by SIMD ({simd}) is violated."

                # Create and insert Softmax node
                new_node = helper.make_node(
                    "Softmax",
                    [input_tensor],
                    [output_tensor],
                    domain="brainsmith.kernels",
                    backend="fpgadataflow",
                    SIMD=simd,
                    name="Softmax_" + node.name,
                )
                graph.node.insert(node_ind, new_node)

                # Remove old node
                graph.node.remove(node)
                graph_modified = True

        if graph_modified:
            # Re-infer shapes and datatypes after transformation
            # This ensures intermediate tensors have consistent shapes, which is critical
            # when multiple Softmax nodes are connected (e.g., node1 -> mid -> node2)
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())

            # Initialize tensor context for newly created Softmax nodes
            # This must happen AFTER InferShapes to ensure intermediate tensors have valid shapes
            for node in model.graph.node:
                if node.op_type == "Softmax" and node.domain == "brainsmith.kernels":
                    op_inst = getCustomOp(node)
                    op_inst.refresh_df_model(model)

        return (model, graph_modified)
