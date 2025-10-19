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


# Module-level KernelSchema definition (structure only)
SOFTMAX_SCHEMA = df.KernelSchema(
    name="Softmax",
    inputs=[
        df.InputSchema(
            name="input",
            block_tiling=[FULL_DIM],       # One Softmax op: (1, 1, channels)
            stream_tiling=["SIMD"],        # Stream channels with SIMD parallelism
        )
    ],
    outputs=[
        df.OutputSchema(
            name="output",
            block_tiling=[FULL_DIM],                   # Same as input: (1, 1, channels)
            stream_tiling=[DerivedDim("input", -1)],   # Output streams at same rate as input
            datatype=DerivedDatatype("input"),         # Derive FLOAT32 from input
        )
    ],
    constraints=[
        # Input must be floating-point datatype
        df.DatatypeFloat(("input",)),
        # Input must be dynamic (no initializers)
        df.IsDynamic("input"),
        # Must operate on last axis (channel dimension), or None (defaults to -1)
        df.NodeAttributeEquals("axis", [None, -1]),
    ]
)

# Module-level InferencePattern (ONNX discovery)
SOFTMAX_INFERENCE = df.InferencePattern(
    source_ops=["Softmax"],  # ONNX Softmax nodes
    matcher=lambda node, model: node.domain != "brainsmith.kernels"  # Skip already-converted hardware nodes
    # Axis validation handled by NodeAttributeEquals constraint in schema
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

    @classmethod
    def get_inference_pattern(cls) -> df.InferencePattern:
        """Return Softmax inference pattern (ONNX discovery)."""
        return SOFTMAX_INFERENCE

    @classmethod
    def infer_from(cls, node, model, insert_index):
        """Infer Softmax kernel from ONNX Softmax node.

        Axis validation is handled by the matcher (_check_softmax_axis).
        Initializes kernel with SIMD=1 (default parallelization).

        Args:
            node: ONNX NodeProto for Softmax
            model: ModelWrapper
            insert_index: Index where to insert new nodes

        Returns:
            InferenceResult with nodes to insert/remove
        """
        from brainsmith.dataflow.inference import InferenceHelper

        # Use InferenceHelper for cleaner node creation
        # Use Brainsmith domain since Softmax is a Brainsmith kernel
        helper = InferenceHelper(model, domain="brainsmith.kernels")

        # Get input/output tensor names (no layout conversion needed for Softmax)
        input_tensor = node.input[0]
        output_tensor = node.output[0]

        # Get channel count from input shape
        channels = helper.get_num_channels(input_tensor)

        # Create Softmax node with default SIMD=1
        # (Axis validation already done by matcher)
        new_node = helper.make_node(
            "Softmax",
            [input_tensor],
            [output_tensor],
            {
                "SIMD": 1,  # Default parallelization
            },
            name_prefix=f"Softmax_{node.name}"
        )

        return helper.make_inference_result(
            new_node,
            node,
            channels=channels
        )

    def execute_node(self, context, graph):
        node = self.onnx_node
        input_data = context[node.input[0]]

        # scipy.special.softmax with axis=-1 (last dimension)
        # Handles numerical stability automatically
        output_data = softmax(input_data, axis=-1)

        # Store result as float32
        context[node.output[0]] = output_data.astype(np.float32)
