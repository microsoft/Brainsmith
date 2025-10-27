"""Integration tests for Thresholding kernel.

This module validates the modern KernelOp-based Thresholding implementation,
including:
1. Inference from MultiThreshold nodes
2. Shape method correctness
3. Datatype handling
4. HLS and RTL backend instantiation
5. Basic execution validation

Test Pattern:
    Input: [1, 28, 28, 128] (4D tensor, INT8)
    Thresholds: 7 thresholds per channel (for 3-bit quantization)
    Output: [1, 28, 28, 128] (UINT4)
    PE: 16 (128 channels / 16 = 8 folding)
"""

import pytest
import numpy as np

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes

from tests.fixtures.model_utils import create_multithreshold_model
from brainsmith.kernels.thresholding import Thresholding, Thresholding_hls, Thresholding_rtl


class TestThresholdingInference:
    """Test Thresholding inference from MultiThreshold nodes."""

    def test_can_infer_from_multithreshold(self):
        """Test that Thresholding.can_infer_from() correctly identifies MultiThreshold nodes."""
        # Create MultiThreshold model
        model_proto = create_multithreshold_model(
            input_shape=[1, 28, 28, 128],
            num_thresholds=7,
            input_dtype="INT8",
            threshold_dtype="INT8",
            output_dtype="UINT4",
            out_scale=1.0
        )

        model = ModelWrapper(model_proto)
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        # Set tensor datatypes
        model.set_tensor_datatype("input", DataType["INT8"])
        model.set_tensor_datatype("thresholds", DataType["INT8"])
        model.set_tensor_datatype("output", DataType["UINT4"])

        # Find MultiThreshold node
        mt_node = None
        for node in model.graph.node:
            if node.op_type == "MultiThreshold":
                mt_node = node
                break

        assert mt_node is not None, "MultiThreshold node not found in test model"

        # Test can_infer_from
        can_infer = Thresholding.can_infer_from(mt_node, model)
        assert can_infer, "Thresholding.can_infer_from() should return True for INT8→UINT4"

    def test_can_infer_rejects_float_output(self):
        """Test that transformation is rejected for float output datatypes.

        Float outputs are rejected during validation (build_design_space()),
        not during can_infer_from(). This test verifies the full transformation
        fails gracefully via InferKernel's try-validate pattern.
        """
        from brainsmith.transforms.infer_kernel import InferKernel

        # Create MultiThreshold model with float output
        model_proto = create_multithreshold_model(
            input_shape=[1, 28, 28, 128],
            num_thresholds=7,
            input_dtype="INT8",
            output_dtype="FLOAT32",  # Float output should be rejected
            out_scale=1.0
        )

        model = ModelWrapper(model_proto)
        model = model.transform(InferShapes())

        # Set tensor datatypes
        model.set_tensor_datatype("input", DataType["INT8"])
        model.set_tensor_datatype("thresholds", DataType["INT8"])
        model.set_tensor_datatype("output", DataType["FLOAT32"])

        # Apply InferKernel transform
        transform = InferKernel(Thresholding)
        model, modified = transform.apply(model)

        # Transformation should be skipped (validation fails due to float output)
        assert not modified, "Graph should not be modified (validation should reject float output)"

        # Original MultiThreshold node should still be present
        mt_node = None
        for node in model.graph.node:
            if node.op_type == "MultiThreshold":
                mt_node = node
                break
        assert mt_node is not None, "MultiThreshold node should still exist (transformation rejected)"

    def test_infer_from_creates_thresholding_node(self):
        """Test that infer_from() creates a Thresholding node with correct attributes."""
        # Create MultiThreshold model
        model_proto = create_multithreshold_model(
            input_shape=[1, 28, 28, 128],
            num_thresholds=7,
            input_dtype="INT8",
            threshold_dtype="INT8",
            output_dtype="UINT4",
            out_scale=1.0,
            out_bias=0
        )

        model = ModelWrapper(model_proto)
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        # Set tensor datatypes
        model.set_tensor_datatype("input", DataType["INT8"])
        model.set_tensor_datatype("thresholds", DataType["INT8"])
        model.set_tensor_datatype("output", DataType["UINT4"])

        # Find MultiThreshold node
        mt_node = None
        for node in model.graph.node:
            if node.op_type == "MultiThreshold":
                mt_node = node
                break

        # Apply inference
        result = Thresholding.infer_from(mt_node, model, insert_index=0)

        # Validate result
        assert len(result.nodes_to_insert) > 0, "infer_from() should create new nodes"
        assert len(result.nodes_to_remove) == 1, "infer_from() should remove MultiThreshold node"
        assert result.nodes_to_remove[0] == mt_node, "Should remove the MultiThreshold node"

        # Find the Thresholding node in nodes_to_insert
        thresholding_node = None
        for node in result.nodes_to_insert:
            if node.op_type == "Thresholding":
                thresholding_node = node
                break

        assert thresholding_node is not None, "Should create a Thresholding node"
        assert thresholding_node.domain == "brainsmith.kernels", "Should use brainsmith.kernels domain"

        # Create Thresholding instance to check attributes
        op = Thresholding(thresholding_node)
        assert op.get_nodeattr("PE") == 1, "Default PE should be 1"
        assert op.get_nodeattr("num_steps") == 7, "Should have 7 threshold steps"
        assert op.get_nodeattr("act_val") == 0, "ActVal should be 0"


class TestThresholdingShapes:
    """Test Thresholding shape methods."""

    @pytest.fixture
    def thresholding_model(self):
        """Create a Thresholding model for testing."""
        # Create MultiThreshold model
        model_proto = create_multithreshold_model(
            input_shape=[1, 28, 28, 128],
            num_thresholds=7,
            input_dtype="INT8",
            threshold_dtype="INT8",
            output_dtype="UINT4"
        )

        model = ModelWrapper(model_proto)
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        # Set tensor datatypes
        model.set_tensor_datatype("input", DataType["INT8"])
        model.set_tensor_datatype("thresholds", DataType["INT8"])
        model.set_tensor_datatype("output", DataType["UINT4"])

        # Find and transform MultiThreshold node
        mt_node = None
        insert_index = 0
        for i, node in enumerate(model.graph.node):
            if node.op_type == "MultiThreshold":
                mt_node = node
                insert_index = i
                break

        result = Thresholding.infer_from(mt_node, model, insert_index)

        # Apply transformation
        model.graph.node.remove(mt_node)
        for i, node in enumerate(result.nodes_to_insert):
            model.graph.node.insert(insert_index + i, node)

        # Find Thresholding node
        thresholding_node = None
        for node in model.graph.node:
            if node.op_type == "Thresholding":
                thresholding_node = node
                break

        op = Thresholding(thresholding_node)

        # Set PE for testing
        op.set_nodeattr("PE", 16)

        # Build design space
        op.build_design_space(model)

        return op, model

    def test_normal_input_shape(self, thresholding_model):
        """Test that get_normal_input_shape() returns correct shape."""
        op, model = thresholding_model

        # Get shape via design_point
        ki = op.design_point
        expected_shape = list(ki.inputs["input"].tensor_shape)

        # Thresholding should extract shape from design_point
        actual_shape = op.get_normal_input_shape()

        assert list(actual_shape) == expected_shape, (
            f"Normal input shape mismatch: expected {expected_shape}, got {actual_shape}"
        )
        assert list(actual_shape) == [1, 28, 28, 128], "Should match input shape [1, 28, 28, 128]"

    def test_folded_input_shape(self, thresholding_model):
        """Test that get_folded_input_shape() returns correct folded shape."""
        op, model = thresholding_model

        # Expected: [batch, height, width, fold, PE]
        # fold = num_channels // PE = 128 // 16 = 8
        expected_shape = [1, 28, 28, 8, 16]
        actual_shape = op.get_folded_input_shape()

        assert list(actual_shape) == expected_shape, (
            f"Folded input shape mismatch: expected {expected_shape}, got {actual_shape}"
        )

    def test_normal_output_shape(self, thresholding_model):
        """Test that get_normal_output_shape() returns correct shape (same as input)."""
        op, model = thresholding_model

        # Output shape should match input shape
        input_shape = op.get_normal_input_shape()
        output_shape = op.get_normal_output_shape()

        assert output_shape == input_shape, "Output shape should match input shape"

    def test_folded_output_shape(self, thresholding_model):
        """Test that get_folded_output_shape() returns correct folded shape."""
        op, model = thresholding_model

        # Output folded shape should match input folded shape
        input_folded = op.get_folded_input_shape()
        output_folded = op.get_folded_output_shape()

        assert output_folded == input_folded, "Folded output shape should match folded input shape"


class TestThresholdingStreamWidths:
    """Test Thresholding stream width methods."""

    @pytest.fixture
    def thresholding_model(self):
        """Create a Thresholding model for testing."""
        model_proto = create_multithreshold_model(
            input_shape=[1, 28, 28, 128],
            num_thresholds=7,
            input_dtype="INT8",
            threshold_dtype="INT8",
            output_dtype="UINT4"
        )

        model = ModelWrapper(model_proto)
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        model.set_tensor_datatype("input", DataType["INT8"])
        model.set_tensor_datatype("thresholds", DataType["INT8"])
        model.set_tensor_datatype("output", DataType["UINT4"])

        # Transform to Thresholding
        mt_node = model.graph.node[0]
        result = Thresholding.infer_from(mt_node, model, 0)
        model.graph.node.remove(mt_node)
        model.graph.node.insert(0, result.nodes_to_insert[0])

        op = Thresholding(model.graph.node[0])
        op.set_nodeattr("PE", 16)
        op.build_design_space(model)

        return op, model

    def test_instream_width(self, thresholding_model):
        """Test that get_instream_width() returns correct width."""
        op, model = thresholding_model

        # Input stream width = PE * input_bitwidth
        # PE = 16, INT8 = 8 bits
        expected_width = 16 * 8
        actual_width = op.get_instream_width()

        assert actual_width == expected_width, (
            f"Input stream width mismatch: expected {expected_width}, got {actual_width}"
        )

    def test_outstream_width(self, thresholding_model):
        """Test that get_outstream_width() returns correct width."""
        op, model = thresholding_model

        # Output stream width = PE * output_bitwidth
        # PE = 16, UINT4 = 4 bits
        expected_width = 16 * 4
        actual_width = op.get_outstream_width()

        assert actual_width == expected_width, (
            f"Output stream width mismatch: expected {expected_width}, got {actual_width}"
        )


class TestThresholdingBackends:
    """Test Thresholding backend instantiation."""

    def test_hls_backend_instantiation(self):
        """Test that Thresholding_hls can be instantiated."""
        from onnx import helper

        # Create minimal Thresholding node
        node = helper.make_node(
            "Thresholding",
            inputs=["input", "thresholds"],
            outputs=["output"],
            domain="brainsmith.kernels",
            PE=16,
            num_steps=7,
            act_val=0
        )

        # Instantiate HLS backend
        op = Thresholding_hls(node)

        assert op is not None, "Should create Thresholding_hls instance"
        assert op.get_nodeattr("PE") == 16, "Should preserve PE attribute"
        assert hasattr(op, "get_nodeattr_types"), "Should have HLS-specific methods"

        # Check that HLS-specific attributes exist
        nodeattr_types = op.get_nodeattr_types()
        assert "mem_mode" in nodeattr_types, "Should have mem_mode attribute"
        assert "ram_style" in nodeattr_types, "Should have ram_style attribute"

    def test_rtl_backend_instantiation(self):
        """Test that Thresholding_rtl can be instantiated."""
        from onnx import helper

        # Create minimal Thresholding node
        node = helper.make_node(
            "Thresholding",
            inputs=["input", "thresholds"],
            outputs=["output"],
            domain="brainsmith.kernels",
            PE=16,
            num_steps=7,
            act_val=0
        )

        # Instantiate RTL backend
        op = Thresholding_rtl(node)

        assert op is not None, "Should create Thresholding_rtl instance"
        assert op.get_nodeattr("PE") == 16, "Should preserve PE attribute"
        assert hasattr(op, "get_nodeattr_types"), "Should have RTL-specific methods"

        # Check that RTL-specific attributes exist
        nodeattr_types = op.get_nodeattr_types()
        assert "depth_trigger_uram" in nodeattr_types, "Should have depth_trigger_uram attribute"
        assert "depth_trigger_bram" in nodeattr_types, "Should have depth_trigger_bram attribute"


class TestThresholdingExecution:
    """Test Thresholding execution."""

    def test_execute_node_basic(self):
        """Test that execute_node() produces expected output shape and datatype."""
        # Create and transform model
        model_proto = create_multithreshold_model(
            input_shape=[1, 4, 4, 8],  # Small size for fast test
            num_thresholds=3,  # 2-bit output (4 levels)
            input_dtype="INT8",
            threshold_dtype="INT8",
            output_dtype="UINT2",
            out_bias=0
        )

        model = ModelWrapper(model_proto)
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        model.set_tensor_datatype("input", DataType["INT8"])
        model.set_tensor_datatype("thresholds", DataType["INT8"])
        model.set_tensor_datatype("output", DataType["UINT2"])

        # Transform to Thresholding
        mt_node = model.graph.node[0]
        result = Thresholding.infer_from(mt_node, model, 0)
        model.graph.node.remove(mt_node)
        for node in result.nodes_to_insert:
            model.graph.node.insert(0, node)

        # Find Thresholding node
        thresholding_node = None
        for node in model.graph.node:
            if node.op_type == "Thresholding":
                thresholding_node = node
                break

        op = Thresholding(thresholding_node)
        op.build_design_space(model)  # Initialize datatypes and design space

        # Create input data
        input_shape = (1, 4, 4, 8)
        input_data = np.random.randint(-128, 127, input_shape, dtype=np.int8).astype(np.float32)

        # Get thresholds from model initializer
        thresholds = model.get_initializer("thresholds")

        # Execute node
        context = {"input": input_data, "thresholds": thresholds}
        op.execute_node(context, model.graph)

        # Check output exists and has correct shape
        assert "output" in context, "Output should be in context after execution"
        output_data = context["output"]
        assert output_data.shape == input_shape, "Output shape should match input shape"

        # Output values should be in range [0, 3] for UINT2 with 3 thresholds
        assert output_data.min() >= 0, "Output should be non-negative"
        assert output_data.max() <= 3, "Output should be <= 3 for UINT2"
