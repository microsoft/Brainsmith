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
        # Must normalize over last axis (channel dimension)
        df.NodeAttributeEquals("axis", -1),
    ]
)

# Module-level InferencePattern (ONNX discovery)
LAYERNORM_INFERENCE = df.InferencePattern(
    source_ops=["FuncLayerNorm"],
    layout_conversions={"input": "NHWC"},  # Convert NCHW → NHWC if needed
    # Axis validation handled by NodeAttributeEquals constraint in schema
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
        from brainsmith.dataflow.inference import InferenceHelper, InferenceResult

        # Use InferenceHelper for cleaner layout conversion
        # Use Brainsmith domain since LayerNorm is a Brainsmith kernel
        inference_helper = InferenceHelper(model, domain="brainsmith.kernels")

        # Ensure NHWC layout (handles conversion automatically)
        act_in = inference_helper.ensure_layout(node.input[0], "NHWC", insert_index)
        act_out = inference_helper.ensure_layout(node.output[0], "NHWC", insert_index)

        # Extract epsilon from source node
        epsilon = helper.get_node_attr_value(node, "epsilon")

        # Create LayerNorm node using InferenceHelper
        # (domain is already set to "brainsmith.kernels" in helper initialization)
        new_node = inference_helper.make_node(
            "LayerNorm",
            [act_in],
            [act_out],
            {
                "SIMD": 1,
                "epsilon": epsilon,
            },
            name_prefix=f"LayerNorm_{node.name}"
        )

        return inference_helper.make_inference_result(
            new_node,
            node,
            layout_conversion=act_in != node.input[0] or act_out != node.output[0]
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
