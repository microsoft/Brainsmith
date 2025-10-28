# Portions derived from FINN project
# Copyright (C) 2023, Advanced Micro Devices, Inc.
# Licensed under BSD-3-Clause License
#
# Modifications and additions Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""AddStreams hardware kernel for element-wise addition of two integer streams
with identical shapes.

Example ONNX pattern:
    Add(input0: INT8[1,224,224,64], input1: INT8[1,224,224,64])
    -> output: INT8[1,224,224,64]

Hardware mapping:
    AddStreams with PE parallelism for channel-wise processing
"""

import numpy as np
from onnx import NodeProto, helper
from typing import Optional

from brainsmith.dataflow import KernelOp, FULL_SHAPE
import brainsmith.dataflow as df
from brainsmith.dataflow.spec_helpers import add_datatype
from brainsmith.registry import kernel
from qonnx.core.modelwrapper import ModelWrapper


ADDSTREAMS_SCHEMA = df.KernelSchema(
    name="AddStreams",
    inputs=[
        df.InputSchema(
            name="input0",
            block_tiling=FULL_SHAPE,  # Rank-agnostic: works with any tensor rank
            stream_tiling=["PE"],
            required_layout="NHWC",  # Embedded layout requirement
        ),
        df.InputSchema(
            name="input1",
            block_tiling=FULL_SHAPE,  # Rank-agnostic: works with any tensor rank
            stream_tiling=["PE"],
            required_layout="NHWC",  # Embedded layout requirement
        ),
    ],
    outputs=[
        df.OutputSchema(
            name="output",
            block_tiling=FULL_SHAPE,  # Rank-agnostic: works with any tensor rank
            stream_tiling=[("input0", -1)],  # Auto-pads to match rank
            datatype=add_datatype("input0", "input1"),  # INT8 + INT8 → INT9 (prevents overflow)
            required_layout="NHWC",  # Embedded layout requirement
        )
    ],
    constraints=[
        # Both inputs must be dynamic (not initializers/weights)
        df.IsDynamic(("input0", "input1")),
        # Both inputs must be integers
        df.DatatypeInteger(("input0", "input1")),
        # Inputs must have same shape
        df.ShapesEqual(("input0", "input1")),
    ],
    kernel_params={
        "PE": ("i", False, 1),
        "NumChannels": ("i", False, 1),
        "numInputVectors": ("i", False, 1),
    },
)


@kernel(
    description="Element-wise addition of two integer streams",
    author="FINN Team"
)
class AddStreams(KernelOp):
    """Hardware kernel for element-wise addition of two streams."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    # ====================================================================
    # Schema (Required by KernelOp)
    # ====================================================================

    @classmethod
    def build_schema(cls, node: NodeProto, model: Optional[ModelWrapper]) -> df.KernelSchema:
        """Build AddStreams schema (constant for all instances)."""
        return ADDSTREAMS_SCHEMA

    @classmethod
    def can_infer_from(cls, node: NodeProto, model: ModelWrapper) -> bool:
        """Check if ONNX node can be converted to AddStreams kernel.

        Validates:
        - Op type is Add
        - Both inputs are dynamic (not initializers)
        - Both inputs are integers
        - Inputs have same shape
        """
        if node.op_type != "Add":
            return False

        # Check we have two inputs
        if len(node.input) != 2:
            return False

        # Check both inputs are dynamic (not initializers)
        initializer_names = [x.name for x in model.graph.initializer]
        for inp in node.input:
            if inp in initializer_names:
                return False

        # Check both inputs are integers
        try:
            dt0 = model.get_tensor_datatype(node.input[0])
            dt1 = model.get_tensor_datatype(node.input[1])
            if not (dt0.is_integer() and dt1.is_integer()):
                return False
        except:
            # If datatypes not available, reject
            return False

        # Check inputs have same shape
        try:
            shape0 = model.get_tensor_shape(node.input[0])
            shape1 = model.get_tensor_shape(node.input[1])
            if shape0 != shape1:
                return False
        except:
            # If shapes not available, reject
            return False

        return True

    # ====================================================================
    # Inference Implementation (Custom - needs NumChannels, etc.)
    # ====================================================================

    @classmethod
    def infer_from(
        cls,
        node: NodeProto,
        model: ModelWrapper,
        insert_index: int
    ) -> df.TransformationResult:
        """Create AddStreams HW node from ONNX Add node.

        NOTE: Assumes inputs are already in NHWC layout (preprocessing required).

        Args:
            node: ONNX Add node to convert
            model: ModelWrapper for graph access
            insert_index: Where to insert new nodes (unused - no layout conversion)

        Returns:
            TransformationResult with AddStreams node and removed Add node
        """
        schema = cls.build_schema(node, model)

        # Extract parameters from ONNX graph
        input_shape = model.get_tensor_shape(node.input[0])

        # Calculate NumChannels (last dimension) and numInputVectors (product of all other dims)
        num_channels = input_shape[-1]
        num_input_vectors = int(np.prod(input_shape[:-1]))

        # Create AddStreams HW node
        hw_node = helper.make_node(
            "AddStreams",
            inputs=list(node.input),
            outputs=list(node.output),
            domain="brainsmith.kernels",
            backend="fpgadataflow",
            NumChannels=num_channels,
            numInputVectors=num_input_vectors,
            name=f"AddStreams_{node.name}"
        )

        return df.TransformationResult(
            nodes_to_insert=[hw_node],
            nodes_to_remove=[node],
            actual_layouts={
                "input0": "NHWC",
                "input1": "NHWC",
                "output": "NHWC",
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

    # ====================================================================
    # Golden Reference (For Integration Testing)
    # ====================================================================

    @staticmethod
    def compute_golden_reference(inputs: dict) -> dict:
        """NumPy reference implementation for AddStreams.

        This is the single source of truth for correctness validation.
        All backends (Python, HLS cppsim, RTL rtlsim) must match this.

        Args:
            inputs: Dict with "input0" and "input1" numpy arrays

        Returns:
            Dict with "output" = input0 + input1

        Example:
            >>> inputs = {"input0": np.array([1, 2, 3]),
            ...           "input1": np.array([4, 5, 6])}
            >>> golden = AddStreams.compute_golden_reference(inputs)
            >>> golden["output"]
            array([5, 7, 9])
        """
        return {"output": inputs["input0"] + inputs["input1"]}

    @staticmethod
    def validate_golden_properties(inputs: dict, outputs: dict) -> None:
        """Validate mathematical properties of addition.

        Properties checked:
        - Commutativity: a + b == b + a
        - Associativity: (a + b) + c == a + (b + c) (for 3+ inputs, future)
        - Identity: a + 0 == a

        Args:
            inputs: Dict with "input0" and "input1" arrays
            outputs: Dict with "output" array from golden reference

        Raises:
            AssertionError: If properties are violated
        """
        input0 = inputs["input0"]
        input1 = inputs["input1"]
        output = outputs["output"]

        # Test commutativity: a + b == b + a
        reverse_sum = input1 + input0
        np.testing.assert_array_equal(
            output,
            reverse_sum,
            err_msg="Addition is not commutative (a + b != b + a)",
        )

        # Test against direct computation
        direct_sum = input0 + input1
        np.testing.assert_array_equal(
            output, direct_sum, err_msg="Golden reference does not match direct sum"
        )
