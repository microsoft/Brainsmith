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


# Module-level unified KernelSchema (structure + transformation)
LAYERNORM_SCHEMA = df.KernelSchema(
    name="LayerNorm",
    domain="brainsmith.kernels",
    inputs=[
        df.InputSchema(
            name="input",
            block_tiling=[FULL_DIM],         # (1, 1, channels)
            stream_tiling=["SIMD"],          # Stream channels with SIMD parallelism
            required_layout="NHWC",          # Required layout (embedded)
        )
    ],
    outputs=[
        df.OutputSchema(
            name="output",
            block_tiling=[FULL_DIM],                  # (1, 1, channels)
            stream_tiling=[DerivedDim("input", -1)],  # Output streams at same rate as input
            datatype=DerivedDatatype("input"),        # Output datatype same as input
            required_layout="NHWC",                   # Required layout (embedded)
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
    ],
    # Transformation specification (unified)
    source_ops=["FuncLayerNorm"],
    attribute_mapping={"epsilon": "epsilon"},
    initial_parallelization={"SIMD": 1},
)


@kernel(
    description="Hardware LayerNorm w/out Bias/Scale",
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
