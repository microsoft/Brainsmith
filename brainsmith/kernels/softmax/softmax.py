# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from scipy.special import softmax

from brainsmith.core.finn import AutoHWCustomOp
import brainsmith.core.dataflow as df 
from brainsmith.core.dataflow import (
    DatatypeConstraint,
    DerivedDatatype,
    DerivedDim
)
from brainsmith.core.plugins import kernel


# Module-level KernelSchema definition
SOFTMAX_SCHEMA = df.KernelSchema(
    name="Softmax",
    inputs=[
        df.InputSchema(
            name="input",
            block_tiling=[":"],            # One Softmax op: (1, 1, channels)
            stream_tiling=["SIMD"],        # Stream channels with SIMD parallelism
            # Datatype always from ONNX: input0Datatype
            constraints=[
                DatatypeConstraint("input", "FLOAT", 32, 32),
            ]
        )
    ],
    outputs=[
        df.OutputSchema(
            name="output",
            block_tiling=[":"],                        # Same as input: (1, 1, channels)
            stream_tiling=[DerivedDim("input", -1)],   # Output streams at same rate as input
            datatype=DerivedDatatype("input"),         # Derive FLOAT32 from input
        )
    ]
)


@kernel(
    description="Float32 Softmax",
    author="Shane Fleming"
)
class Softmax(AutoHWCustomOp):
    """Abstraction layer for HW implementation of Softmax layers."""

    kernel_schema = SOFTMAX_SCHEMA

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    # All nodeattrs auto-generated from schema:
    # - "SIMD" from stream_tiling=["SIMD"]
    # - "input0Datatype" from input interface (default naming)
    # - "output0Datatype" from output interface (default naming, derived from input)

    def execute_node(self, context, graph):
        node = self.onnx_node
        input_data = context[node.input[0]]

        # scipy.special.softmax with axis=-1 (last dimension)
        # Handles numerical stability automatically
        output_data = softmax(input_data, axis=-1)

        # Store result as float32
        context[node.output[0]] = output_data.astype(np.float32)
