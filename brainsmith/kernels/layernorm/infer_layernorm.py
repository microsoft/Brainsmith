############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
InferLayerNorm - Transform FuncLayerNorm to LayerNorm hardware operations.

Transforms FuncLayerNorm functional nodes into hardware LayerNorm operations
using the modern AutoHWCustomOp infrastructure with Dataflow Modeling.

Key features:
- Creates "LayerNorm" nodes using AutoHWCustomOp base class
- Does NOT set ifm_dim or NumChannels (inferred from tensor context via kernel_model)
- Relies on declarative KernelSchema for shape information
- Automatic shape inference and validation

The transformation process:
1. Find FuncLayerNorm nodes in the graph
2. Extract epsilon and axis attributes
3. Handle NCHW layout conversion if needed
4. Create LayerNorm node with minimal attributes
5. Remove old FuncLayerNorm node
6. Rerun shape and datatype inference
7. Initialize tensor context for all LayerNorm nodes (enables kernel_model)

Example:
    from brainsmith.kernels.layernorm.infer_layernorm import InferLayerNorm

    model = ModelWrapper(...)
    model = model.transform(InferLayerNorm())

    # Result: FuncLayerNorm nodes replaced with LayerNorm nodes
"""

import qonnx.core.data_layout as DataLayout
from onnx import helper
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.onnx import nchw_to_nhwc
from qonnx.custom_op.registry import getCustomOp
from brainsmith.core.plugins import transform


@transform(
    kernel="LayerNorm",
    description="Convert FuncLayerNorm to LayerNorm hardware operations",
    author="Thomas Keller"
)
class InferLayerNorm(Transformation):
    """Convert FuncLayerNorm into LayerNorm HW operations.

    This transform uses the modern AutoHWCustomOp infrastructure with
    Dataflow Modeling system for automatic shape inference.

    The key benefit is that shape information (ifm_dim, NumChannels) is inferred
    automatically from the tensor context via kernel_model, eliminating redundancy
    and potential inconsistency.

    Only normalizes over the channel dimension (last axis).

    Attributes explicitly set on LayerNorm node:
    - SIMD: Parallelization factor (default 1)
    - epsilon: Small value to prevent division by zero

    Attributes set automatically by refresh_tensor_context():
    - inputDataType: Populated from tensor context
    - outputDataType: Populated from tensor context

    Attributes inferred dynamically from tensor context (not stored):
    - ifm_dim: Input feature map dimensions (from tensor context)
    - NumChannels: Number of channels (from tensor context)
    """

    def apply(self, model):
        """Apply InferLayerNorm transformation to model.

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
            if node.op_type == "FuncLayerNorm":
                act_in = node.input[0]
                act_out = node.output[0]

                # Skip if shape not available (shouldn't happen with proper inference)
                shape_in = model.get_tensor_shape(act_in)
                if shape_in is None or len(shape_in) == 0:
                    continue

                # Handle NCHW layout conversion
                norm_axis = helper.get_node_attr_value(node, "axis")
                if model.get_tensor_layout(act_in) == DataLayout.NCHW:
                    act_in = nchw_to_nhwc(act_in, model, node_ind)
                    node_ind += 1
                    shape_in = model.get_tensor_shape(act_in)
                    # Shift axis for norm appropriately
                    norm_axis = (norm_axis + 2) % 4

                ch = shape_in[-1]

                # Keep track of where we need to insert the HLS Op
                # It has to be ahead of the output transform
                insert_point = node_ind
                if model.get_tensor_layout(act_out) == DataLayout.NCHW:
                    act_out = nchw_to_nhwc(act_out, model, node_ind, reverse=True)
                    node_ind += 1

                # Check if 1D, norming on channel axis
                if not (norm_axis == -1 or norm_axis == len(shape_in) - 1):
                    continue

                # Create node with no parallelization first
                simd = 1
                assert ch % simd == 0, "Requirement channel divisible by SIMD is violated."

                # Create and insert LayerNorm node
                new_node = helper.make_node(
                    "LayerNorm",
                    [act_in],
                    [act_out],
                    domain="brainsmith.kernels",
                    backend="fpgadataflow",
                    SIMD=simd,
                    epsilon=helper.get_node_attr_value(node, "epsilon"),
                    name="LayerNorm_" + node.name,
                )
                graph.node.insert(insert_point, new_node)

                # Remove old node
                graph.node.remove(node)
                graph_modified = True

        if graph_modified:
            # Re-infer shapes and datatypes after transformation
            # This ensures intermediate tensors have consistent shapes, which is critical
            # when multiple LayerNorm nodes are connected (e.g., node1 -> mid -> node2)
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())

            # Initialize tensor context for newly created LayerNorm nodes
            # This must happen AFTER InferShapes to ensure intermediate tensors have valid shapes
            for node in model.graph.node:
                if node.op_type == "LayerNorm" and node.domain == "brainsmith.kernels":
                    op_inst = getCustomOp(node)
                    op_inst.refresh_tensor_context(model)

        return (model, graph_modified)
