# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""AddStreams hardware kernel for element-wise addition of two streams.

This kernel implements element-wise addition of two integer streams with
identical shapes. It demonstrates the unified constraint system with
declarative validation.

Example ONNX pattern:
    Add(input0: INT8[1,224,224,64], input1: INT8[1,224,224,64])
    -> output: INT8[1,224,224,64]

Hardware mapping:
    AddStreams with PE parallelism for channel-wise processing
"""

import numpy as np
from onnx import NodeProto

from brainsmith.dataflow import KernelOp, InferenceHelper, InferenceResult
import brainsmith.dataflow as df
from brainsmith.core.plugins import kernel
from qonnx.core.modelwrapper import ModelWrapper


# Module-level KernelSchema definition with unified constraints
ADDSTREAMS_SCHEMA = df.KernelSchema(
    name="AddStreams",
    inputs=[
        df.InputSchema(
            name="input0",
            block_tiling=[df.FULL_DIM, df.FULL_DIM, df.FULL_DIM, df.FULL_DIM],
            stream_tiling=[1, 1, 1, "PE"],
        ),
        df.InputSchema(
            name="input1",
            block_tiling=[df.FULL_DIM, df.FULL_DIM, df.FULL_DIM, df.FULL_DIM],
            stream_tiling=[1, 1, 1, "PE"],
        ),
    ],
    outputs=[
        df.OutputSchema(
            name="output",
            block_tiling=[df.FULL_DIM, df.FULL_DIM, df.FULL_DIM, df.FULL_DIM],
            stream_tiling=[1, 1, 1, df.DerivedDim("input0", -1)],
            datatype=df.DerivedDatatype("input0"),  # Output datatype same as input0
        )
    ],
    constraints=[
        # Both inputs must be dynamic (not initializers/weights)
        df.IsDynamic("input0"),
        df.IsDynamic("input1"),
        # Both inputs must be integers
        df.DatatypeInteger("input0"),
        df.DatatypeInteger("input1"),
        # Inputs must have same shape
        df.ShapesEqual("input0", "input1"),
    ],
    kernel_params={
        "PE": ("i", True, 1),
        "NumChannels": ("i", True, 1),
    },
    inference=df.InferencePattern(
        source_ops=["Add"],
        layout_conversions={
            "input0": "NHWC",
            "input1": "NHWC",
            "output": "NHWC",
        }
    )
)


@kernel(
    description="Element-wise addition of two integer streams",
    author="Thomas Keller"
)
class AddStreams(KernelOp):
    """Hardware kernel for element-wise addition of two streams.

    Adds two integer streams element-wise with configurable parallelism.

    Schema auto-generates:
    - "PE" from stream_tiling=[1, 1, 1, "PE"]
    - "input0Datatype" from input0 interface
    - "input1Datatype" from input1 interface
    - "output0Datatype" from output interface (derived from input0)
    - "NumChannels" from kernel_params (set during inference)

    Validation (unified constraints):
    - IsDynamic("input0"), IsDynamic("input1"): Both inputs must be dynamic tensors
    - DatatypeInteger("input0"), DatatypeInteger("input1"): Integer datatypes required
    - ShapesEqual("input0", "input1"): Inputs must have identical shapes

    Inference pattern:
    - Matches ONNX Add nodes
    - Automatically converts to NHWC layout if needed
    """

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    # ====================================================================
    # Schema (Required by KernelOp)
    # ====================================================================

    @classmethod
    def get_class_schema(cls) -> df.KernelSchema:
        """Return schema for inference (class method, no instantiation needed)."""
        return ADDSTREAMS_SCHEMA

    @property
    def kernel_schema(self) -> df.KernelSchema:
        """Return AddStreams schema (instance property)."""
        return ADDSTREAMS_SCHEMA

    # ====================================================================
    # Inference (Opt-in)
    # ====================================================================

    @classmethod
    def infer_from(
        cls,
        node: NodeProto,
        model: ModelWrapper,
        insert_index: int
    ) -> InferenceResult:
        """Create AddStreams HW node from ONNX Add node.

        Args:
            node: ONNX Add node to convert
            model: ModelWrapper for graph access
            insert_index: Where to insert new nodes

        Returns:
            InferenceResult with AddStreams node and removed Add node
        """
        helper = InferenceHelper(model)

        # Handle layout conversion (guided by InferencePattern)
        in0 = helper.ensure_layout(node.input[0], "NHWC", insert_index)
        in1 = helper.ensure_layout(node.input[1], "NHWC", insert_index)
        result = helper.ensure_layout(node.output[0], "NHWC", insert_index)

        # Extract parameters from ONNX graph
        in0_shape = model.get_tensor_shape(in0)
        num_channels = helper.get_num_channels(in0)
        num_input_vectors = helper.get_num_input_vectors(in0)

        # Get datatypes
        idt0 = model.get_tensor_datatype(in0)
        idt1 = model.get_tensor_datatype(in1)

        # Create AddStreams HW node
        hw_node = helper.make_node(
            "AddStreams",
            inputs=[in0, in1],
            outputs=[result],
            attributes={
                "NumChannels": num_channels,
                "PE": 1,  # Default PE=1 (no parallelization initially)
                "inputDataTypes": [idt0.name, idt1.name],
                "numInputVectors": num_input_vectors,
            },
            name_prefix=f"AddStreams_{node.name}"
        )

        return InferenceResult(
            nodes_to_insert=[hw_node],
            nodes_to_remove=[node],
            metadata={
                "num_channels": num_channels,
                "input_vectors": num_input_vectors,
                "layout_converted": in0 != node.input[0]
            }
        )

    # ====================================================================
    # Execution (CPU implementation for testing/validation)
    # ====================================================================

    def execute_node(self, context, graph):
        """Execute AddStreams on CPU for testing/validation.

        Performs element-wise addition: output = input0 + input1

        Args:
            context: Execution context with tensor values
            graph: ONNX graph
        """
        node = self.onnx_node

        # Get input data
        input0_data = context[node.input[0]]
        input1_data = context[node.input[1]]

        # Element-wise addition
        output_data = input0_data + input1_data

        # Get output datatype and cast
        output_dt = self.get_output_datatype(ind=0)
        output_data = output_data.astype(output_dt.to_numpy_dt())

        # Store result
        context[node.output[0]] = output_data
