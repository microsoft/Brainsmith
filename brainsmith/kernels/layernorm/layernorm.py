# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
import torch.nn.functional as F
from onnx import NodeProto
from onnx.helper import make_node
from typing import Optional

from qonnx.core.modelwrapper import ModelWrapper
from brainsmith.dataflow import KernelOp
import brainsmith.dataflow as df
from brainsmith.dataflow import DerivedDatatype, DerivedDim, FULL_DIM
from brainsmith.core.plugins import kernel


# Helper function for axis validation in inference
def _check_channel_norm_axis(node, model):
    """Validate that FuncLayerNorm normalizes over channel dimension (last axis).

    Args:
        node: ONNX NodeProto for FuncLayerNorm
        model: ModelWrapper

    Returns:
        True if axis == -1 or axis == len(shape) - 1, False otherwise
    """
    from onnx import helper

    shape_in = model.get_tensor_shape(node.input[0])
    if shape_in is None or len(shape_in) == 0:
        return False

    norm_axis = helper.get_node_attr_value(node, "axis")
    # Accept -1 or last dimension index
    return norm_axis == -1 or norm_axis == len(shape_in) - 1


# Module-level KernelSchema definition (structure only)
LAYERNORM_SCHEMA = df.KernelSchema(
    name="LayerNorm",
    inputs=[
        df.InputSchema(
            name="input",
            block_tiling=[FULL_DIM],         # (1, 1, channels)
            stream_tiling=["SIMD"],          # Stream channels with SIMD parallelism
            # Datatype always from ONNX: input0Datatype
        )
    ],
    outputs=[
        df.OutputSchema(
            name="output",
            block_tiling=[FULL_DIM],                  # (1, 1, channels)
            stream_tiling=[DerivedDim("input", -1)],  # Output streams at same rate as input
            datatype=DerivedDatatype("input")         # Output datatype same as input
        )
    ],
    kernel_params={
        "epsilon": ("f", True, 1e-5),  # Normalization epsilon for numerical stability
    },
    constraints=[
        # Input must be dynamic (no initializers)
        df.IsDynamic("input"),
        # Channel dimension must be divisible by SIMD
        df.DimensionDivisible("input", -1, "SIMD", hierarchy=df.ShapeHierarchy.TENSOR),
    ]
)

# Module-level InferencePattern (ONNX discovery)
LAYERNORM_INFERENCE = df.InferencePattern(
    source_ops=["FuncLayerNorm"],
    layout_conversions={"input": "NHWC"},  # Convert NCHW → NHWC if needed
    matcher=_check_channel_norm_axis  # Validate axis == -1 (channel norm)
)


@kernel(
    description="Hardware LayerNorm using Dataflow Modeling",
    author="Shane Fleming"
)
class LayerNorm(KernelOp):
    """Abstraction layer for HW implementation of the LayerNorm layer.

    Schema auto-generates:
    - "SIMD" from stream_tiling=["SIMD"]
    - "input0Datatype" from input interface
    - "output0Datatype" from output interface
    - "epsilon" from kernel_params
    """

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    @classmethod
    def build_schema(cls, node: NodeProto, model: Optional[ModelWrapper]) -> df.KernelSchema:
        """Build LayerNorm schema (constant for all instances)."""
        return LAYERNORM_SCHEMA

    @classmethod
    def get_inference_pattern(cls) -> df.InferencePattern:
        """Return LayerNorm inference pattern (ONNX discovery)."""
        return LAYERNORM_INFERENCE

    @classmethod
    def infer_from(cls, node, model, insert_index):
        """Infer LayerNorm kernel from FuncLayerNorm ONNX node.

        Handles layout conversion (NCHW → NHWC) and initializes kernel with:
        - SIMD=1 (default parallelization)
        - epsilon from FuncLayerNorm node

        Args:
            node: ONNX NodeProto for FuncLayerNorm
            model: ModelWrapper
            insert_index: Index where to insert new nodes

        Returns:
            InferenceResult with nodes to insert/remove
        """
        from onnx import helper
        import qonnx.core.data_layout as DataLayout
        from qonnx.util.onnx import nchw_to_nhwc
        from brainsmith.dataflow.inference import InferenceResult

        act_in = node.input[0]
        act_out = node.output[0]

        # Get input shape
        shape_in = model.get_tensor_shape(act_in)
        if shape_in is None or len(shape_in) == 0:
            raise ValueError(f"Cannot infer LayerNorm from {node.name}: input shape not available")

        # Handle NCHW layout conversion on input (modifies graph in-place)
        norm_axis = helper.get_node_attr_value(node, "axis")
        if model.get_tensor_layout(act_in) == DataLayout.NCHW:
            act_in = nchw_to_nhwc(act_in, model, insert_index)
            shape_in = model.get_tensor_shape(act_in)

        # Handle NCHW layout conversion on output (modifies graph in-place)
        if model.get_tensor_layout(act_out) == DataLayout.NCHW:
            act_out = nchw_to_nhwc(act_out, model, insert_index, reverse=True)

        # Create LayerNorm node with default SIMD=1
        simd = 1
        epsilon = helper.get_node_attr_value(node, "epsilon")

        new_node = helper.make_node(
            "LayerNorm",
            [act_in],
            [act_out],
            domain="brainsmith.kernels",
            backend="fpgadataflow",
            SIMD=simd,
            epsilon=epsilon,
            name="LayerNorm_" + node.name,
        )

        return InferenceResult(
            nodes_to_insert=[new_node],
            nodes_to_remove=[node],
            metadata={"layout_conversion": model.get_tensor_layout(node.input[0]) == DataLayout.NCHW}
        )

    def execute_node(self, context, graph):
        node = self.onnx_node
        in_values = context[node.input[0]]

        # Get epsilon from nodeattr
        epsilon = self.get_nodeattr("epsilon")

        # PyTorch LayerNorm over last dimension
        # normalized_shape must be the dimensions to normalize over
        in_tensor = torch.from_numpy(in_values)
        out_tensor = F.layer_norm(
            in_tensor,
            normalized_shape=[in_values.shape[-1]],  # Normalize over channels
            eps=epsilon
        )

        # Store result
        context[node.output[0]] = out_tensor.numpy().astype(np.float32)
