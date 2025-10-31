"""Integration tests for AddStreams kernel pipeline.

Tests complete ONNX → Hardware → Execution flow with golden reference validation.

Test Coverage:
- Pipeline validation (ONNX Add → AddStreams HW node)
- Shape/datatype preservation through transforms
- Python execution vs NumPy golden reference
- HLS C++ simulation vs NumPy golden reference (slow)

Example Usage:
    # Run all fast tests
    pytest tests/pipeline/test_addstreams_integration.py -v

    # Run with HLS simulation (slow, requires VITIS_PATH)
    pytest tests/pipeline/test_addstreams_integration.py -v --run-slow

    # Run only golden reference tests
    pytest tests/pipeline/test_addstreams_integration.py -v -m golden
"""

import numpy as np
import pytest
from onnx import helper, TensorProto
from typing import Dict, Tuple, Type

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.transformation.base import Transformation

from brainsmith.kernels.addstreams import AddStreams
from brainsmith.primitives.transforms.infer_kernel_list import InferKernelList
from brainsmith.dataflow.kernel_op import KernelOp
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp

from tests.frameworks.single_kernel_test import SingleKernelTest
from tests.support.context import make_execution_context


class TestAddStreamsIntegration(SingleKernelTest):
    """Complete pipeline integration test for AddStreams kernel.

    Validates:
    1. ONNX Add node → AddStreams HW node transformation
    2. Shape/datatype inference correctness
    3. Python execution matches NumPy golden reference
    4. HLS C++ simulation matches NumPy golden reference
    5. Mathematical properties (commutativity)

    Test Shapes:
    - Default: [1, 64] (batch=1, channels=64)
    - Tests NHWC layout requirement
    - Tests PE parallelism (configurable)
    """

    # ================================================================
    # Test Configuration
    # ================================================================

    def get_test_shape(self) -> Tuple[int, ...]:
        """Shape for test inputs (NHWC format).

        Returns:
            Tuple of (batch, channels) or (batch, height, width, channels)

        Override for different test shapes.
        """
        return (1, 64)  # Batch=1, Channels=64

    def get_test_datatype(self) -> DataType:
        """Datatype for test inputs.

        Returns:
            QONNX DataType (e.g., DataType.INT8)

        Override for different datatypes.
        """
        return DataType["INT8"]

    def get_num_inputs(self) -> int:
        """AddStreams has 2 inputs."""
        return 2

    def get_num_outputs(self) -> int:
        """AddStreams has 1 output."""
        return 1

    # ================================================================
    # Required Abstract Methods
    # ================================================================

    def make_test_model(self) -> Tuple[ModelWrapper, str]:
        """Create ONNX model with Add node (standard ONNX, not hardware).

        Creates:
        - Two inputs: input0, input1 (same shape, same datatype)
        - One Add node
        - One output: output

        Returns:
            (model, node_name): ModelWrapper and name of Add node
        """
        shape = self.get_test_shape()
        dtype = self.get_test_datatype()

        # Convert DataType to ONNX TensorProto type
        # For integer types, use FLOAT as container (FINN convention)
        onnx_dtype = TensorProto.FLOAT

        # Create inputs
        input0 = helper.make_tensor_value_info("input0", onnx_dtype, shape)
        input1 = helper.make_tensor_value_info("input1", onnx_dtype, shape)

        # Create output
        output = helper.make_tensor_value_info("output", onnx_dtype, shape)

        # Create Add node
        add_node = helper.make_node(
            "Add", ["input0", "input1"], ["output"], name="Add_test"
        )

        # Create graph
        graph = helper.make_graph(
            [add_node], "test_addstreams", [input0, input1], [output]
        )

        # Create model
        model = helper.make_model(graph)
        model_w = ModelWrapper(model)

        # Set datatypes (FINN convention: store in model annotations)
        model_w.set_tensor_datatype("input0", dtype)
        model_w.set_tensor_datatype("input1", dtype)
        # Output datatype will be inferred (INT8 + INT8 → INT9)

        return model_w, "Add_test"

    def get_kernel_inference_transform(self) -> Type[Transformation]:
        """Return InferKernelList transform.

        InferKernelList will detect the Add node and convert it to AddStreams
        if it meets the constraints (integer inputs, same shape, both dynamic).
        """
        return InferKernelList

    def compute_golden_reference(
        self, inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Test-owned golden reference for AddStreams.

        Implements element-wise addition: output = input0 + input1

        This is TEST LOGIC that defines what "correct" means for AddStreams.
        The test owns this definition, not the production kernel code.

        Args:
            inputs: {"input0": array, "input1": array}

        Returns:
            {"output": sum_array}
        """
        input0 = inputs["input0"]
        input1 = inputs["input1"]
        output = input0 + input1
        return {"output": output}

    # ================================================================
    # Optional Configuration
    # ================================================================

    def configure_kernel_node(self, op: HWCustomOp, model: ModelWrapper) -> None:
        """Configure AddStreams node after inference.

        Sets PE (parallelism) to test parallel execution.
        Default PE=1 (fully serial), override to test parallelism.

        Args:
            op: AddStreams HW node
            model: Model containing the node
        """
        # Set PE for parallelism testing
        # PE=4 means process 4 channels in parallel
        op.set_nodeattr("PE", 4)

    # ================================================================
    # Additional Tests (Beyond Base Class)
    # ================================================================

    @pytest.mark.pipeline
    @pytest.mark.golden
    @pytest.mark.phase1
    def test_golden_reference_properties(self):
        """Test that golden reference satisfies mathematical properties.

        Validates the golden reference implementation itself:
        - Commutativity: a + b == b + a
        - Correctness: matches direct NumPy addition
        """
        # Create test inputs
        shape = self.get_test_shape()
        np.random.seed(42)
        input0 = np.random.randint(-128, 128, shape).astype(np.float32)
        input1 = np.random.randint(-128, 128, shape).astype(np.float32)

        # Test commutativity: a + b == b + a
        forward_inputs = {"input0": input0, "input1": input1}
        reverse_inputs = {"input0": input1, "input1": input0}

        forward_output = self.compute_golden_reference(forward_inputs)["output"]
        reverse_output = self.compute_golden_reference(reverse_inputs)["output"]

        np.testing.assert_array_equal(
            forward_output,
            reverse_output,
            err_msg="Golden reference should be commutative (a+b == b+a)",
        )

        # Test correctness: matches direct NumPy addition
        expected_output = input0 + input1
        np.testing.assert_array_equal(
            forward_output,
            expected_output,
            err_msg="Golden reference should match direct NumPy addition",
        )

    @pytest.mark.pipeline
    @pytest.mark.phase1
    def test_pe_parallelism_configured(self):
        """Test that PE parallelism is correctly configured.

        Validates that configure_kernel_node() sets PE attribute.
        """
        op, model = self.run_inference_pipeline()

        # Check PE was set by configure_kernel_node()
        pe = op.get_nodeattr("PE")
        assert pe == 4, f"PE should be 4, got {pe}"

    @pytest.mark.pipeline
    @pytest.mark.phase1
    def test_num_channels_inferred(self):
        """Test that NumChannels is correctly inferred from input shape.

        NumChannels should equal the last dimension of input shape (NHWC).
        """
        op, model = self.run_inference_pipeline()

        # Get NumChannels
        num_channels = op.get_nodeattr("NumChannels")

        # Should equal last dimension of input shape
        shape = self.get_test_shape()
        expected_channels = shape[-1]

        assert num_channels == expected_channels, (
            f"NumChannels mismatch: expected {expected_channels}, got {num_channels}"
        )

    @pytest.mark.pipeline
    @pytest.mark.phase1
    def test_output_datatype_widened(self):
        """Test that output datatype is widened to prevent overflow.

        INT8 + INT8 should produce INT9 output (extra bit for overflow).
        """
        op, model = self.run_inference_pipeline()

        # Get input and output datatypes
        input0_dt = op.get_input_datatype(0)
        input1_dt = op.get_input_datatype(1)
        output_dt = op.get_output_datatype(0)

        # Check inputs are INT8
        assert input0_dt == DataType["INT8"], f"Input0 should be INT8, got {input0_dt}"
        assert input1_dt == DataType["INT8"], f"Input1 should be INT8, got {input1_dt}"

        # Check output is INT9 (widened)
        assert output_dt == DataType["INT9"], f"Output should be INT9, got {output_dt}"

    @pytest.mark.pipeline
    @pytest.mark.phase1
    def test_folded_shapes_account_for_pe(self):
        """Test that folded shapes account for PE parallelism.

        With PE=4, the stream should process 4 channels at once,
        affecting the folded shape calculation.
        """
        op, model = self.run_inference_pipeline()

        # Get folded input shape
        folded_input_shape = op.get_folded_input_shape(0)
        normal_input_shape = op.get_normal_input_shape(0)

        # With PE=4, the stream should process 4 channels at once
        # Example: [1, 64] normal → [1, 16, 4] folded
        #   - Batch dimension (1) preserved
        #   - 64 channels / 4 PE = 16 iterations
        #   - Last dimension = PE (4 channels per iteration)
        pe = op.get_nodeattr("PE")
        num_channels = normal_input_shape[-1]

        # Folded shape should preserve batch dimensions
        # [1, 64] → [1, 16, 4] (3D: batch, iterations, PE)
        assert len(folded_input_shape) >= 2, (
            f"Folded shape should be at least 2D, got {folded_input_shape}"
        )

        # Check that PE parallelism affects the shape
        # The folded shape should include PE in one of its dimensions
        assert pe in folded_input_shape, (
            f"PE value {pe} should appear in folded shape {folded_input_shape}"
        )

        # Check that total elements are preserved
        import numpy as np
        normal_elements = np.prod(normal_input_shape)
        folded_elements = np.prod(folded_input_shape)
        assert normal_elements == folded_elements, (
            f"Element count mismatch: normal={normal_elements}, folded={folded_elements}"
        )

    @pytest.mark.pipeline
    @pytest.mark.phase1
    def test_stream_widths_calculated_correctly(self):
        """Test stream width calculations with PE parallelism.

        Stream width should equal PE × bitwidth for efficient hardware packing.
        This validates get_instream_width() and get_outstream_width().
        """
        op, model = self.run_inference_pipeline()

        pe = op.get_nodeattr("PE")
        input0_bitwidth = op.get_input_datatype(0).bitwidth()
        input1_bitwidth = op.get_input_datatype(1).bitwidth()
        output_bitwidth = op.get_output_datatype().bitwidth()

        # Stream widths should be PE × bitwidth
        assert op.get_instream_width(0) == pe * input0_bitwidth, (
            f"Input0 stream width mismatch: expected {pe * input0_bitwidth}, "
            f"got {op.get_instream_width(0)}"
        )
        assert op.get_instream_width(1) == pe * input1_bitwidth, (
            f"Input1 stream width mismatch: expected {pe * input1_bitwidth}, "
            f"got {op.get_instream_width(1)}"
        )
        assert op.get_outstream_width() == pe * output_bitwidth, (
            f"Output stream width mismatch: expected {pe * output_bitwidth}, "
            f"got {op.get_outstream_width()}"
        )

    @pytest.mark.pipeline
    @pytest.mark.phase1
    def test_expected_cycles_calculation(self):
        """Test get_exp_cycles() returns correct cycle count.

        Expected cycles should equal the number of iterations through
        the folded tensor (batch × spatial × channel_iterations).
        """
        op, model = self.run_inference_pipeline()

        cycles = op.get_exp_cycles()
        folded_shape = op.get_folded_output_shape()

        # Cycles = product of all dimensions except last (PE dimension)
        expected_cycles = int(np.prod(folded_shape[:-1]))

        assert cycles == expected_cycles, (
            f"Expected cycles mismatch: expected {expected_cycles}, got {cycles}"
        )

    @pytest.mark.pipeline
    @pytest.mark.phase1
    def test_infer_node_datatype_syncs_datatypes(self):
        """Test infer_node_datatype() correctly syncs and calculates datatypes.

        Validates:
        1. Input datatypes are synced from model to node attributes
        2. Output datatype is calculated with overflow prevention (INT8 + INT8 → INT9)
        3. Output datatype is propagated back to model graph
        4. Method can be called explicitly (not just via InferDataTypes transform)
        """
        # Run pipeline to get AddStreams node
        op, model = self.run_inference_pipeline()

        # Get initial datatypes
        input0_name = op.onnx_node.input[0]
        input1_name = op.onnx_node.input[1]
        output_name = op.onnx_node.output[0]

        # Explicitly call infer_node_datatype (normally called by InferDataTypes transform)
        # This should be idempotent - calling again shouldn't break anything
        op.infer_node_datatype(model)

        # Verify input datatypes synced to nodeattrs
        assert op.get_input_datatype(0) == DataType["INT8"], (
            f"Input0 datatype should be INT8, got {op.get_input_datatype(0)}"
        )
        assert op.get_input_datatype(1) == DataType["INT8"], (
            f"Input1 datatype should be INT8, got {op.get_input_datatype(1)}"
        )

        # Verify output datatype calculated with overflow prevention
        # INT8 range: [-128, 127]
        # INT8 + INT8 worst case: -128 + (-128) = -256, 127 + 127 = 254
        # Requires INT9 range: [-256, 255]
        assert op.get_output_datatype(0) == DataType["INT9"], (
            f"Output datatype should be INT9 (INT8 + INT8 requires widening), "
            f"got {op.get_output_datatype(0)}"
        )

        # Verify output datatype propagated to model
        model_output_dt = model.get_tensor_datatype(output_name)
        assert model_output_dt == DataType["INT9"], (
            f"Model output datatype should be INT9, got {model_output_dt}"
        )


# ================================================================
# Edge Case Tests (Phase 2)
# ================================================================


class TestAddStreamsEdgeCases(TestAddStreamsIntegration):
    """Edge case tests for AddStreams kernel.

    Tests error handling and boundary conditions.
    """

    @pytest.mark.pipeline
    @pytest.mark.phase2
    def test_pe_must_divide_channels(self):
        """Test that PE must evenly divide NumChannels.

        KernelOp design space validation enforces this constraint.
        Valid PE values are divisors of NumChannels.
        """
        # Override to create shape where PE=4 is invalid
        self.get_test_shape = lambda: (1, 65)  # 65 divisors: [1, 5, 13, 65]

        # Should fail during design space configuration (PE=4 not valid)
        with pytest.raises(
            Exception,  # KernelOpError or ValueError
            match="Invalid PE=4"
        ):
            op, model = self.run_inference_pipeline()

    @pytest.mark.pipeline
    @pytest.mark.phase2
    def test_complex_num_input_vectors(self):
        """Test with multi-dimensional numInputVectors (conv-like shapes).

        Tests 4D tensors [batch, height, width, channels] which map to
        numInputVectors=[batch, height, width] in AddStreams.
        """
        # Override to create 4D tensor shape
        original_shape = self.get_test_shape
        self.get_test_shape = lambda: (1, 4, 4, 64)  # [B, H, W, C]

        original_model = self.make_test_model

        def make_4d_model():
            """Create ONNX model with 4D tensors."""
            shape = self.get_test_shape()
            dtype = self.get_test_datatype()

            # Convert DataType to ONNX TensorProto type
            onnx_dtype = TensorProto.FLOAT

            # Create inputs
            input0 = helper.make_tensor_value_info("input0", onnx_dtype, shape)
            input1 = helper.make_tensor_value_info("input1", onnx_dtype, shape)
            output = helper.make_tensor_value_info("output", onnx_dtype, shape)

            # Create Add node
            add_node = helper.make_node(
                "Add", ["input0", "input1"], ["output"], name="Add_4d_test"
            )

            # Create graph and model
            graph = helper.make_graph(
                [add_node], "test_addstreams_4d", [input0, input1], [output]
            )
            model = helper.make_model(graph)
            model_w = ModelWrapper(model)

            # Set datatypes
            model_w.set_tensor_datatype("input0", dtype)
            model_w.set_tensor_datatype("input1", dtype)

            return model_w, "Add_4d_test"

        self.make_test_model = make_4d_model

        try:
            # Should handle 4D tensors correctly
            # numInputVectors should be [1, 4, 4] (product = 16)
            op, model = self.run_inference_pipeline()

            # Verify numInputVectors (product of all dims except last)
            num_input_vectors = op.get_nodeattr("numInputVectors")
            # For 4D [1, 4, 4, 64], numInputVectors = 1*4*4 = 16
            expected_vectors = 1 * 4 * 4
            assert num_input_vectors == expected_vectors, (
                f"numInputVectors should be {expected_vectors}, got {num_input_vectors}"
            )

            # Verify shapes
            normal_shape = op.get_normal_input_shape()
            assert normal_shape == (1, 4, 4, 64), (
                f"Normal shape should be (1, 4, 4, 64), got {normal_shape}"
            )

            # Test execution with 4D tensors
            self.test_python_execution_vs_golden()

        finally:
            # Restore originals
            self.get_test_shape = original_shape
            self.make_test_model = original_model


# ================================================================
# Parametric Tests (Phase 2 Preview)
# ================================================================


class TestAddStreamsIntegrationParametric(TestAddStreamsIntegration):
    """Parametric tests for AddStreams across different configurations.

    Phase 2 feature preview: Tests multiple shapes and datatypes.

    Usage:
        pytest tests/pipeline/test_addstreams_integration.py::TestAddStreamsIntegrationParametric -v
    """

    @pytest.mark.pipeline
    @pytest.mark.phase2
    @pytest.mark.parametrize(
        "shape",
        [
            (1, 32),  # Small
            (1, 64),  # Medium
            (1, 128),  # Large
            (8, 64),  # Multi-batch
        ],
    )
    def test_golden_reference_various_shapes(self, shape):
        """Test golden reference with various input shapes."""
        # Override get_test_shape temporarily
        original_shape = self.get_test_shape
        self.get_test_shape = lambda: shape

        try:
            # Run standard golden reference test
            self.test_python_execution_vs_golden()
        finally:
            # Restore original
            self.get_test_shape = original_shape

    @pytest.mark.pipeline
    @pytest.mark.phase2
    @pytest.mark.parametrize(
        "datatype",
        [
            DataType["INT4"],
            DataType["INT8"],
            DataType["INT16"],
        ],
    )
    def test_golden_reference_various_datatypes(self, datatype):
        """Test golden reference with various integer datatypes."""
        # Override get_test_datatype temporarily
        original_dtype = self.get_test_datatype
        self.get_test_datatype = lambda: datatype

        try:
            # Run standard golden reference test
            self.test_python_execution_vs_golden()
        finally:
            # Restore original
            self.get_test_datatype = original_dtype


# ================================================================
# HLS RTL Simulation Tests (Phase 2)
# ================================================================


class TestAddStreamsHLSRTLSim(TestAddStreamsIntegration):
    """Test HLS backend with RTL simulation (not just cppsim).

    IMPORTANT: HLSBackend can produce both:
    1. C++ simulation (cppsim) - behavioral simulation of HLS code
    2. RTL simulation (rtlsim) - simulation of synthesized Verilog from HLS

    This test class validates the full HLS → RTL synthesis flow.
    RTLBackend would only support rtlsim (hand-written Verilog).
    """

    @pytest.mark.pipeline
    @pytest.mark.golden
    @pytest.mark.rtlsim
    @pytest.mark.slow
    @pytest.mark.phase2
    def test_hls_rtlsim_execution_vs_golden(self):
        """Test HLS RTL simulation matches golden reference.

        This validates the complete HLS → Verilog synthesis flow:
        1. Kernel inference creates HW node
        2. HLS specialization creates HLS node
        3. HLS synthesis generates Verilog (not just C++)
        4. RTL simulation of synthesized Verilog
        5. Results match golden reference

        This is more comprehensive than cppsim because it validates
        the actual RTL that would go on the FPGA.

        Requires:
            - Vivado installation with XSim
            - HLS synthesis (slow, ~minutes per test)
        """
        # Run pipeline to get base kernel
        base_op, base_model = self.run_inference_pipeline()

        # Specialize to HLS backend
        hls_op, hls_model = self.run_hls_specialization(base_op, base_model)

        # Generate test inputs (deterministic)
        np.random.seed(42)
        inputs = make_execution_context(hls_model, hls_op)

        # Compute golden reference
        golden_outputs = self.compute_golden_reference(inputs)

        # Execute via HLS rtlsim (synthesized Verilog, not C++)
        # This requires prepare_rtlsim() to synthesize HLS to Verilog first
        actual_outputs = self.execute_rtlsim(hls_op, hls_model, inputs)

        # Validate against golden
        tolerance = self.get_golden_tolerance_cppsim()  # Same tolerance as cppsim
        self.validate_against_golden(
            actual_outputs, golden_outputs, "HLS RTL simulation (rtlsim)", tolerance
        )

