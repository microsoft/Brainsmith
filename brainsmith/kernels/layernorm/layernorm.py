# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
import torch.nn.functional as F
from onnx.helper import make_node

from brainsmith.core.finn import AutoHWCustomOp
import brainsmith.core.dataflow as df
from brainsmith.core.dataflow import DerivedDatatype, DerivedDim
from brainsmith.core.plugins import kernel


# Module-level KernelSchema definition
LAYERNORM_SCHEMA = df.KernelSchema(
    name="LayerNorm",
    inputs=[
        df.InputSchema(
            name="input",
            block_tiling=[":"],              # (1, 1, channels)
            stream_tiling=["SIMD"],          # Stream channels with SIMD parallelism
            datatype="inputDataType",        # Custom name (FINN compatibility)
        )
    ],
    outputs=[
        df.OutputSchema(
            name="output",
            block_tiling=[":"],                        # (1, 1, channels)
            stream_tiling=[DerivedDim("input", -1)],   # Output streams at same rate as input
            datatype="outputDataType",                 # Custom name for FINN compatibility
        )
    ]
)


@kernel(
    description="Hardware LayerNorm using AutoHWCustomOp and Dataflow Modeling",
    author="Shane Fleming"
)
class LayerNorm(AutoHWCustomOp):
    """Abstraction layer for HW implementation of the LayerNorm layer."""

    kernel_schema = LAYERNORM_SCHEMA

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = super().get_nodeattr_types()
        my_attrs.update({
            "SIMD": ("i", True, 1),
            "epsilon": ("f", True, 1e-5),
            "inputDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
        })
        return my_attrs

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
