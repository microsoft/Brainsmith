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
from brainsmith.core.plugins import kernel

# Import clean schema (no ONNX knowledge)
from .layernorm_schema import LAYERNORM_SCHEMA


@kernel(
    description="Hardware LayerNorm w/out Bias/Scale",
    author="Shane Fleming"
)
class LayerNorm(KernelOp):
    """Abstraction layer for HW implementation of the LayerNorm layer."""

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
