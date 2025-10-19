# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from scipy.special import softmax
from onnx import NodeProto
from typing import Optional

from qonnx.core.modelwrapper import ModelWrapper
from brainsmith.dataflow import KernelOp
import brainsmith.dataflow as df
from brainsmith.dataflow import (
    DerivedDatatype,
    DerivedDim,
    FULL_DIM
)
from brainsmith.core.plugins import kernel


# Module-level unified KernelSchema (structure + transformation)
SOFTMAX_SCHEMA = df.KernelSchema(
    name="Softmax",
    domain="brainsmith.kernels",
    inputs=[
        df.InputSchema(
            name="input",
            block_tiling=[FULL_DIM],       # One Softmax op: (1, 1, channels)
            stream_tiling=["SIMD"],        # Stream channels with SIMD parallelism
            # No required_layout - Softmax works with any layout
        )
    ],
    outputs=[
        df.OutputSchema(
            name="output",
            block_tiling=[FULL_DIM],                   # Same as input: (1, 1, channels)
            stream_tiling=[DerivedDim("input", -1)],   # Output streams at same rate as input
            datatype=DerivedDatatype("input"),         # Derive FLOAT32 from input
            # No required_layout - preserves input layout
        )
    ],
    constraints=[
        # Input must be floating-point datatype
        df.DatatypeFloat(("input",)),
        # Input must be dynamic (no initializers)
        df.IsDynamic("input"),
        # Must operate on last axis (channel dimension), or None (defaults to -1)
        df.NodeAttributeEquals("axis", [None, -1]),
        # Don't re-convert already-converted hardware nodes
        # (ONNX Softmax has domain="", hardware Softmax has domain="brainsmith.kernels")
        df.Custom(
            lambda ctx: None if getattr(ctx.node, 'domain', '') != 'brainsmith.kernels'
                       else "Already a hardware Softmax node",
            "Node must be ONNX Softmax, not hardware Softmax"
        ),
    ],
    # Transformation specification (unified)
    source_ops=["Softmax"],
    initial_parallelization={"SIMD": 1},
)


@kernel(
    description="Float32 Softmax using Dataflow Modeling",
    author="Shane Fleming"
)
class Softmax(KernelOp):
    """Abstraction layer for HW implementation of Softmax layers.

    Schema auto-generates:
    - "SIMD" from stream_tiling=["SIMD"]
    - "input0Datatype" from input interface
    - "output0Datatype" from output interface
    """

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    @classmethod
    def build_schema(cls, node: NodeProto, model: Optional[ModelWrapper]) -> df.KernelSchema:
        """Build Softmax schema (constant for all instances)."""
        return SOFTMAX_SCHEMA

    # No infer_from() override - default handles it!
    # Default implementation:
    # - Discovers Softmax nodes (from spec.source_ops)
    # - Creates node with SIMD=1 (from spec.initial_parallelization)
    # - No layout conversions needed (no layout requirements in spec)
    # - Verifies automatically

    def execute_node(self, context, graph):
        node = self.onnx_node
        input_data = context[node.input[0]]

        # scipy.special.softmax with axis=-1 (last dimension)
        # Handles numerical stability automatically
        output_data = softmax(input_data, axis=-1)

        # Store result as float32
        context[node.output[0]] = output_data.astype(np.float32)
