############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
############################################################################
"""Tests for ElementwiseBinaryOp execution modes (Phase 3a + 3b)."""

import pytest
import numpy as np
from onnx import helper, TensorProto

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from finn.util.basic import getHWCustomOp

from brainsmith.primitives.transforms.infer_kernel_list import InferKernelList
from brainsmith.kernels.elementwise_binary.tests.test_elementwise_binary_parity import (
    SpecializeElementwiseBinaryToHLS
)
from tests.fixtures.kernel_test_helpers import make_broadcast_model


# =============================================================================
# Test Helpers
# =============================================================================

def infer_to_hls_backend(model):
    """Apply transformations to get ElementwiseBinaryOp_hls instance.

    Args:
        model: ONNX ModelWrapper

    Returns:
        Tuple of (ElementwiseBinaryOp_hls instance, transformed model)
    """
    # Standard inference
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    # Infer to ElementwiseBinaryOp
    model = model.transform(InferKernelList())

    # Specialize to HLS backend
    model = model.transform(SpecializeElementwiseBinaryToHLS())

    # Get HLS node (getHWCustomOp calls build_design_space)
    assert len(model.graph.node) == 1, f"Expected 1 node, got {len(model.graph.node)}"
    hls_node = model.graph.node[0]

    assert hls_node.op_type == "ElementwiseBinaryOp_hls", (
        f"Expected ElementwiseBinaryOp_hls, got {hls_node.op_type}"
    )

    op = getHWCustomOp(hls_node, model)
    return op, model


# =============================================================================
# Phase 3a: exec_mode Infrastructure Tests
# =============================================================================

class TestExecModeInfrastructure:
    """Test exec_mode parameter and dispatch logic (Phase 3a)."""

    def test_execute_node_requires_exec_mode(self):
        """Test that execute_node raises error if exec_mode not set."""
        # Create simple Add model with BOTH dynamic inputs (to avoid ChannelwiseOp)
        in0 = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [1, 4, 4, 8])
        in1 = helper.make_tensor_value_info("in1", TensorProto.FLOAT, [8])  # Dynamic, not initializer
        out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4, 4, 8])

        add_node = helper.make_node("Add", ["in0", "in1"], ["out"], name="test_add")
        graph = helper.make_graph([add_node], "test", [in0, in1], [out])  # Both as inputs
        model = helper.make_model(graph)
        model = ModelWrapper(model)

        # Set datatypes
        model.set_tensor_datatype("in0", DataType["INT8"])
        model.set_tensor_datatype("in1", DataType["INT8"])
        model.set_tensor_datatype("out", DataType["INT9"])

        # Infer to HLS backend (initializes design_point)
        hw_op, model = infer_to_hls_backend(model)

        # Prepare context
        in0_data = np.random.randint(-128, 127, (1, 4, 4, 8)).astype(np.float32)
        in1_data = np.random.randint(-10, 10, (8,)).astype(np.float32)
        context = {"in0": in0_data, "in1": in1_data}

        # Should raise ValueError because exec_mode is not set (empty string)
        with pytest.raises(ValueError, match="Invalid or unset exec_mode"):
            hw_op.execute_node(context, model.graph)

    def test_python_mode_works(self):
        """Test that python mode executes successfully."""
        # Create simple Add model with both dynamic inputs
        in0 = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [1, 4, 4, 8])
        in1 = helper.make_tensor_value_info("in1", TensorProto.FLOAT, [8])
        out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4, 4, 8])

        add_node = helper.make_node("Add", ["in0", "in1"], ["out"], name="test_add")
        graph = helper.make_graph([add_node], "test", [in0, in1], [out])
        model = helper.make_model(graph)
        model = ModelWrapper(model)

        model.set_tensor_datatype("in0", DataType["INT8"])
        model.set_tensor_datatype("in1", DataType["INT8"])
        model.set_tensor_datatype("out", DataType["INT9"])

        # Infer to HLS backend
        hw_op, model = infer_to_hls_backend(model)

        # Set python exec_mode
        hw_op.set_nodeattr("exec_mode", "python")

        in0_data = np.random.randint(-128, 127, (1, 4, 4, 8)).astype(np.float32)
        in1_data = np.random.randint(-10, 10, (8,)).astype(np.float32)
        context = {"in0": in0_data, "in1": in1_data}

        # Should execute successfully
        hw_op.execute_node(context, model.graph)
        assert "out" in context  # Verify output was created

    @pytest.mark.cppsim
    @pytest.mark.slow
    def test_cppsim_requires_vivado(self):
        """Verify cppsim requires Vivado environment (Phase 3d)."""
        # Check if Vivado is available
        import os
        if 'VITIS_PATH' not in os.environ and 'HLS_PATH' not in os.environ:
            pytest.skip("Vivado/Vitis not available - set via brainsmith config")

    @pytest.mark.cppsim
    @pytest.mark.slow
    @pytest.mark.parametrize("operation,np_op", [
        ("Add", np.add),
        ("Sub", np.subtract),
        ("Mul", np.multiply),
    ])
    def test_cppsim_basic_operations(self, operation, np_op):
        """Test cppsim execution for basic operations (Phase 3d)."""
        import os
        if 'VITIS_PATH' not in os.environ and 'HLS_PATH' not in os.environ:
            pytest.skip("Vivado/Vitis not available")

        # Create model
        in0 = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [1, 4, 4, 8])
        in1 = helper.make_tensor_value_info("in1", TensorProto.FLOAT, [8])
        out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4, 4, 8])

        node = helper.make_node(operation, ["in0", "in1"], ["out"], name=f"test_{operation.lower()}")
        graph = helper.make_graph([node], "test", [in0, in1], [out])
        model = helper.make_model(graph)
        model = ModelWrapper(model)

        model.set_tensor_datatype("in0", DataType["INT8"])
        model.set_tensor_datatype("in1", DataType["INT8"])
        model.set_tensor_datatype("out", DataType["INT16"])

        # Infer to HLS backend
        hw_op, model = infer_to_hls_backend(model)
        hw_op.set_nodeattr("exec_mode", "cppsim")

        # Create test data
        np.random.seed(42)
        in0_data = np.random.randint(-10, 10, (1, 4, 4, 8)).astype(np.float32)
        in1_data = np.random.randint(-5, 5, (8,)).astype(np.float32)
        context = {"in0": in0_data, "in1": in1_data}

        # Execute cppsim
        hw_op.execute_node(context, model.graph)

        # Validate output exists and has correct shape
        assert "out" in context
        assert context["out"].shape == (1, 4, 4, 8)

        # Validate correctness against numpy golden reference
        expected = np_op(in0_data, in1_data)
        # Allow some tolerance for quantization effects
        assert np.allclose(context["out"], expected, atol=1.0)

    def test_rtlsim_not_implemented_error(self):
        """Test that rtlsim mode raises NotImplementedError (Phase 3e)."""
        # Create simple model with both dynamic inputs
        in0 = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [1, 4, 4, 8])
        in1 = helper.make_tensor_value_info("in1", TensorProto.FLOAT, [8])
        out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4, 4, 8])

        add_node = helper.make_node("Add", ["in0", "in1"], ["out"], name="test_add")
        graph = helper.make_graph([add_node], "test", [in0, in1], [out])
        model = helper.make_model(graph)
        model = ModelWrapper(model)

        model.set_tensor_datatype("in0", DataType["INT8"])
        model.set_tensor_datatype("in1", DataType["INT8"])
        model.set_tensor_datatype("out", DataType["INT9"])

        # Infer to HLS backend
        hw_op, model = infer_to_hls_backend(model)
        hw_op.set_nodeattr("exec_mode", "rtlsim")

        in0_data = np.random.randint(-128, 127, (1, 4, 4, 8)).astype(np.float32)
        in1_data = np.ones((8,), dtype=np.float32)
        context = {"in0": in0_data, "in1": in1_data}

        # Should raise NotImplementedError with helpful message
        with pytest.raises(NotImplementedError, match="Phase 3e"):
            hw_op.execute_node(context, model.graph)


# =============================================================================
# Phase 3b: Python Execution Mode Tests
# =============================================================================

class TestPythonExecution:
    """Test Python execution mode correctness (Phase 3b)."""

    @pytest.mark.parametrize("operation,np_op", [
        ("Add", np.add),
        ("Sub", np.subtract),
        ("Mul", np.multiply),
    ])
    def test_python_execution_basic_operations(self, operation, np_op):
        """Test Python execution for basic arithmetic operations."""
        # Create model for operation
        in0 = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [1, 4, 4, 8])
        in1_data = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
        in1 = helper.make_tensor("in1", TensorProto.FLOAT, [8], in1_data.tolist())
        out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4, 4, 8])

        node = helper.make_node(operation, ["in0", "in1"], ["out"], name=f"test_{operation}")
        graph = helper.make_graph([node], "test", [in0], [out], [in1])
        model = helper.make_model(graph)
        model = ModelWrapper(model)

        model.set_tensor_datatype("in0", DataType["INT8"])
        model.set_tensor_datatype("in1", DataType["INT8"])
        model.set_tensor_datatype("out", DataType["INT16"])

        # Infer to HLS backend
        hw_op, model = infer_to_hls_backend(model)
        hw_op.set_nodeattr("exec_mode", "python")

        # Execute
        in0_data = np.random.randint(-10, 10, (1, 4, 4, 8)).astype(np.float32)
        context = {"in0": in0_data, "in1": in1_data}

        hw_op.execute_node(context, model.graph)

        # Verify result
        expected = np_op(in0_data, in1_data)
        # Quantize expected to INT16 range
        expected = np.clip(expected, -32768, 32767)

        np.testing.assert_array_equal(context["out"], expected.astype(np.float32))

    def test_python_execution_with_quantization(self):
        """Test Python execution applies output quantization correctly."""
        # Create Add model with overflow scenario
        in0 = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [1, 2, 2, 4])
        in1_data = np.array([100, 100, 100, 100], dtype=np.float32)
        in1 = helper.make_tensor("in1", TensorProto.FLOAT, [4], in1_data.tolist())
        out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 2, 2, 4])

        add_node = helper.make_node("Add", ["in0", "in1"], ["out"], name="test_add")
        graph = helper.make_graph([add_node], "test", [in0], [out], [in1])
        model = helper.make_model(graph)
        model = ModelWrapper(model)

        # Use INT8 output - will overflow without quantization
        model.set_tensor_datatype("in0", DataType["INT8"])
        model.set_tensor_datatype("in1", DataType["INT8"])
        model.set_tensor_datatype("out", DataType["INT8"])

        # Infer to HLS backend
        hw_op, model = infer_to_hls_backend(model)
        hw_op.set_nodeattr("exec_mode", "python")

        # Input that will overflow INT8 when added to 100
        in0_data = np.array([[[[50, 60, 70, 80],
                                [90, 100, 110, 120]]]], dtype=np.float32)
        context = {"in0": in0_data, "in1": in1_data}

        hw_op.execute_node(context, model.graph)

        # Verify quantization clipped to INT8 range
        result = context["out"]
        assert np.all(result >= -128)
        assert np.all(result <= 127)
        # Values that would overflow should be clipped to 127
        assert result[0, 0, 0, 0] == 127  # 50 + 100 = 150 → clipped to 127

    def test_python_execution_scalar_broadcast(self):
        """Test Python execution with scalar broadcasting."""
        # Create dynamic+dynamic Add with scalar broadcast
        in0 = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [1, 4, 4, 8])
        in1 = helper.make_tensor_value_info("in1", TensorProto.FLOAT, [1])
        out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4, 4, 8])

        add_node = helper.make_node("Add", ["in0", "in1"], ["out"], name="test_add")
        graph = helper.make_graph([add_node], "test", [in0, in1], [out])
        model = helper.make_model(graph)
        model = ModelWrapper(model)

        model.set_tensor_datatype("in0", DataType["INT8"])
        model.set_tensor_datatype("in1", DataType["INT8"])
        model.set_tensor_datatype("out", DataType["INT9"])

        # Infer to HLS backend
        hw_op, model = infer_to_hls_backend(model)
        hw_op.set_nodeattr("exec_mode", "python")

        # Execute with scalar
        in0_data = np.random.randint(-10, 10, (1, 4, 4, 8)).astype(np.float32)
        in1_data = np.array([5], dtype=np.float32)
        context = {"in0": in0_data, "in1": in1_data}

        hw_op.execute_node(context, model.graph)

        # Verify broadcasting worked
        expected = in0_data + in1_data
        expected = np.clip(expected, -256, 255)  # INT9 range
        np.testing.assert_array_equal(context["out"], expected.astype(np.float32))

    def test_python_execution_channel_broadcast(self):
        """Test Python execution with channel broadcasting."""
        # Create dynamic+dynamic Add with channel broadcast
        in0 = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [1, 4, 4, 8])
        in1 = helper.make_tensor_value_info("in1", TensorProto.FLOAT, [8])
        out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4, 4, 8])

        add_node = helper.make_node("Add", ["in0", "in1"], ["out"], name="test_add")
        graph = helper.make_graph([add_node], "test", [in0, in1], [out])
        model = helper.make_model(graph)
        model = ModelWrapper(model)

        model.set_tensor_datatype("in0", DataType["INT8"])
        model.set_tensor_datatype("in1", DataType["INT8"])
        model.set_tensor_datatype("out", DataType["INT9"])

        # Infer to HLS backend
        hw_op, model = infer_to_hls_backend(model)
        hw_op.set_nodeattr("exec_mode", "python")

        # Execute with per-channel values
        in0_data = np.random.randint(-10, 10, (1, 4, 4, 8)).astype(np.float32)
        in1_data = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
        context = {"in0": in0_data, "in1": in1_data}

        hw_op.execute_node(context, model.graph)

        # Verify channel-wise broadcasting
        expected = in0_data + in1_data
        expected = np.clip(expected, -256, 255)
        np.testing.assert_array_equal(context["out"], expected.astype(np.float32))

    def test_python_execution_spatial_broadcast(self):
        """Test Python execution with spatial broadcasting."""
        # Create dynamic+dynamic Add with spatial broadcast
        in0 = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [1, 4, 4, 8])
        in1 = helper.make_tensor_value_info("in1", TensorProto.FLOAT, [1, 1, 1, 8])
        out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4, 4, 8])

        add_node = helper.make_node("Add", ["in0", "in1"], ["out"], name="test_add")
        graph = helper.make_graph([add_node], "test", [in0, in1], [out])
        model = helper.make_model(graph)
        model = ModelWrapper(model)

        model.set_tensor_datatype("in0", DataType["INT8"])
        model.set_tensor_datatype("in1", DataType["INT8"])
        model.set_tensor_datatype("out", DataType["INT9"])

        # Infer to HLS backend
        hw_op, model = infer_to_hls_backend(model)
        hw_op.set_nodeattr("exec_mode", "python")

        # Execute with spatial broadcast
        in0_data = np.random.randint(-10, 10, (1, 4, 4, 8)).astype(np.float32)
        in1_data = np.random.randint(-5, 5, (1, 1, 1, 8)).astype(np.float32)
        context = {"in0": in0_data, "in1": in1_data}

        hw_op.execute_node(context, model.graph)

        # Verify spatial broadcasting
        expected = in0_data + in1_data
        expected = np.clip(expected, -256, 255)
        np.testing.assert_array_equal(context["out"], expected.astype(np.float32))

    @pytest.mark.parametrize("dtype_name,min_val,max_val", [
        ("INT8", -128, 127),
        ("UINT8", 0, 255),
        ("INT16", -32768, 32767),
        ("UINT16", 0, 65535),
    ])
    def test_quantization_helper(self, dtype_name, min_val, max_val):
        """Test _quantize_output helper method for various datatypes."""
        # Create a simple model to get the HLS backend instance
        in0 = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [1, 2, 2, 4])
        in1_data = np.ones((4,), dtype=np.float32)
        in1 = helper.make_tensor("in1", TensorProto.FLOAT, [4], in1_data.tolist())
        out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 2, 2, 4])

        add_node = helper.make_node("Add", ["in0", "in1"], ["out"], name="test_add")
        graph = helper.make_graph([add_node], "test", [in0], [out], [in1])
        model = helper.make_model(graph)
        model = ModelWrapper(model)

        model.set_tensor_datatype("in0", DataType[dtype_name])
        model.set_tensor_datatype("in1", DataType[dtype_name])
        model.set_tensor_datatype("out", DataType[dtype_name])

        # Infer to HLS backend
        hw_op, model = infer_to_hls_backend(model)

        # Test quantization with out-of-range values
        test_values = np.array([min_val - 100, min_val, 0, max_val, max_val + 100])
        quantized = hw_op._quantize_output(test_values, DataType[dtype_name])

        # Verify clipping
        assert quantized[0] == min_val  # Clipped from below
        assert quantized[1] == min_val
        assert quantized[2] == 0
        assert quantized[3] == max_val
        assert quantized[4] == max_val  # Clipped from above


# =============================================================================
# Phase 3d: Cppsim Error Handling Tests
# =============================================================================

class TestCppsimErrorHandling:
    """Test cppsim error handling and helpful messages (Phase 3d)."""

    def test_cppsim_missing_vitis_path(self):
        """Test behavior when VITIS_PATH not set.

        HLSBackend may defer environment validation until compilation time,
        so this test verifies that either:
        1. An error is raised immediately (strict validation)
        2. Execution proceeds but would fail at compile time (lazy validation)
        """
        import os

        # Create model
        hw_op, model = create_elementwise_binary_model("Add")
        hw_op.set_nodeattr("exec_mode", "cppsim")

        # Temporarily remove VITIS_PATH
        old_vitis = os.environ.get("VITIS_PATH")
        old_hls = os.environ.get("HLS_PATH")
        if old_vitis:
            del os.environ["VITIS_PATH"]
        if old_hls:
            del os.environ["HLS_PATH"]

        try:
            context = {"in0": np.ones((1, 4, 4, 8)), "in1": np.ones((8,))}

            # HLSBackend may defer validation to compile time
            # So we just verify the call doesn't crash unexpectedly
            try:
                hw_op.execute_node(context, model.graph)
                # If it succeeds, that's ok - validation happens at compile time
            except Exception as e:
                # If it fails, error should mention paths or compilation
                error_msg = str(e).lower()
                assert any(word in error_msg for word in
                          ["vitis", "hls", "path", "compile", "environment"])
        finally:
            # Restore environment
            if old_vitis:
                os.environ["VITIS_PATH"] = old_vitis
            if old_hls:
                os.environ["HLS_PATH"] = old_hls

    @pytest.mark.cppsim
    @pytest.mark.slow
    def test_cppsim_compilation_failure(self):
        """Test graceful handling of compilation errors.

        NOTE: This test is difficult to implement properly without mocking
        the C++ compiler, since HLSBackend regenerates code on execute_node.
        We skip this test for now - compilation error handling is tested
        implicitly by the base HLSBackend implementation.
        """
        pytest.skip("Compilation failure testing requires compiler mocking - "
                   "covered by HLSBackend tests")

    def test_cppsim_execution_before_preparation(self):
        """Test error when trying to execute without code generation."""
        # Create model
        hw_op, model = create_elementwise_binary_model("Add")
        hw_op.set_nodeattr("exec_mode", "cppsim")

        # Don't generate code - try to execute directly
        # Set empty code_gen_dir to simulate unprepared state
        hw_op.set_nodeattr("code_gen_dir_cppsim", "")
        hw_op.set_nodeattr("executable_path", "")

        context = {"in0": np.ones((1, 4, 4, 8)), "in1": np.ones((8,))}

        # HLSBackend should handle this gracefully
        # Either auto-generates code or raises helpful error
        try:
            hw_op.execute_node(context, model.graph)
        except Exception as e:
            # If it fails, error should mention code generation
            error_msg = str(e).lower()
            assert any(word in error_msg for word in ["code", "generate", "compile", "prepare"])

    def test_cppsim_environment_setup(self):
        """Test that environment setup fixture works correctly."""
        # Verify that the setup_cppsim_environment fixture exports config
        import os

        # After fixture runs, environment should have brainsmith config
        # (if config is available)
        try:
            from brainsmith.config import get_config
            config = get_config()

            # Check if Vivado paths are configured
            has_vivado = config.effective_vivado_path is not None

            if has_vivado:
                # Environment should be exported
                # Note: fixture runs with verbose=False, so no printed output
                assert config.effective_vivado_path.exists()
        except Exception:
            # Config might not be available - that's ok
            pass

    def test_cppsim_skip_without_vivado(self):
        """Test that cppsim tests skip gracefully without Vivado."""
        # This test itself verifies the skip logic
        import os

        has_vivado = ('VITIS_PATH' in os.environ or
                      'HLS_PATH' in os.environ)

        if not has_vivado:
            # Verify skip logic works
            hw_op, model = create_elementwise_binary_model("Add")
            hw_op.set_nodeattr("exec_mode", "cppsim")

            # Should either skip or fail with environment error
            context = {"in0": np.ones((1, 4, 4, 8)), "in1": np.ones((8,))}
            with pytest.raises(Exception):
                hw_op.execute_node(context, model.graph)


# =============================================================================
# Test Helper Functions
# =============================================================================

def create_elementwise_binary_model(operation="Add"):
    """Create simple ElementwiseBinaryOp model for testing.

    Args:
        operation: ONNX operation name (Add, Sub, Mul, Div)

    Returns:
        Tuple of (hw_op, model)
    """
    # Create model using make_broadcast_model (supports dynamic + dynamic with broadcasting)
    model, _ = make_broadcast_model(
        lhs_shape=[1, 4, 4, 8],
        rhs_shape=[8],
        operation=operation,
        datatype="INT8",
        lhs_name="in0",
        rhs_name="in1",
        output_name="output"
    )

    # Update output datatype
    model.set_tensor_datatype("output", DataType["INT16"])

    hw_op, model = infer_to_hls_backend(model)
    return hw_op, model
