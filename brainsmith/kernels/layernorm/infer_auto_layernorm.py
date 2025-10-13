############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
InferAutoLayerNorm - Transform FuncLayerNorm to AutoLayerNorm hardware operations.

This is the modern version of InferLayerNorm that targets AutoLayerNorm instead of
the legacy LayerNorm implementation.

Key differences from InferLayerNorm:
- Creates "AutoLayerNorm" nodes (not "LayerNorm")
- Does NOT set ifm_dim or NumChannels (inferred from tensor context via kernel_model)
- Relies on declarative KernelSchema for shape information
- Uses AutoHWCustomOp base class with automatic shape inference

The transformation process:
1. Find FuncLayerNorm nodes in the graph
2. Extract epsilon and axis attributes
3. Handle NCHW layout conversion if needed
4. Create AutoLayerNorm node with minimal attributes
5. Remove old FuncLayerNorm node
6. Rerun shape and datatype inference

Example:
    from brainsmith.kernels.layernorm.infer_auto_layernorm import InferAutoLayerNorm

    model = ModelWrapper(...)
    model = model.transform(InferAutoLayerNorm())

    # Result: FuncLayerNorm nodes replaced with AutoLayerNorm nodes
"""

import qonnx.core.data_layout as DataLayout
from onnx import helper
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.onnx import nchw_to_nhwc
from brainsmith.core.plugins import transform


@transform(
    kernel="LayerNorm",
    description="Convert FuncLayerNorm to AutoLayerNorm hardware operations",
    author="Thomas Keller"
)
class InferAutoLayerNorm(Transformation):
    """Convert FuncLayerNorm into AutoLayerNorm HW operations.

    This transform targets the modern AutoLayerNorm implementation which uses
    AutoHWCustomOp and the Dataflow Modeling system for automatic shape inference.

    The key benefit is that shape information (ifm_dim, NumChannels) is inferred
    automatically from the tensor context via kernel_model, eliminating redundancy
    and potential inconsistency.

    Only normalizes over the channel dimension (last axis).

    Attributes created on AutoLayerNorm node:
    - SIMD: Parallelization factor (default 1)
    - epsilon: Small value to prevent division by zero
    - inputDataType: FINN DataType name (e.g., "INT8")
    - outputDataType: FINN DataType name
    - exec_mode: Execution mode (default "python")

    Attributes NOT created (inferred automatically):
    - ifm_dim: Input feature map dimensions (from tensor context)
    - NumChannels: Number of channels (from tensor context)
    """

    def apply(self, model):
        """Apply InferAutoLayerNorm transformation to model.

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

                # Get shape and datatypes from tensor context
                shape_in = model.get_tensor_shape(act_in)
                idt = model.get_tensor_datatype(act_in)
                odt = model.get_tensor_datatype(act_out)

                # Skip if shape not available (shouldn't happen with proper inference)
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

                # Create and insert AutoLayerNorm node
                # IMPORTANT: No ifm_dim or NumChannels - these are inferred automatically!
                new_node = helper.make_node(
                    "AutoLayerNorm",  # Modern implementation
                    [act_in],
                    [act_out],
                    domain="brainsmith.kernels",
                    backend="fpgadataflow",
                    SIMD=simd,
                    epsilon=helper.get_node_attr_value(node, "epsilon"),
                    inputDataType=idt.name,
                    outputDataType=odt.name,
                    name="AutoLayerNorm_" + node.name,
                )
                graph.node.insert(insert_point, new_node)

                # Remove old node
                graph.node.remove(node)
                graph_modified = True

        if graph_modified:
            # Re-infer shapes and datatypes after transformation
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())

        return (model, graph_modified)
