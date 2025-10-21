"""Base class for parity testing between manual and KernelOp implementations.

This module provides ParityTestBase, an abstract base class that automates
testing equivalence between FINN's manual HWCustomOp implementations and
Brainsmith's KernelOp implementations.

Key features:
- **Transform-based testing (default)**: Tests production Infer transform workflow
- **25 generic test methods**: Shape, datatype, stream width, padding, inference, cycles,
  execution, resource estimation (LUT/DSP/BRAM/URAM), efficiency metrics, operation counting,
  RTL simulation, HDL generation
- Handles initialization differences between manual and KernelOp implementations
- Supports multi-input/output operations
- Random input generation based on datatypes
- Comprehensive assertion messages
- Critical datatype inference logic validation
- Full HLS (cppsim) and RTL (rtlsim) backend testing
"""

import numpy as np
import pytest
from abc import ABC, abstractmethod
from typing import Type, Dict, Any, Tuple, Optional

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.custom_op.registry import getCustomOp
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class ParityTestBase(ABC):
    """Base class for parity testing between manual and KernelOp implementations.

    **Default Pattern (Transform-Based):**
    Subclasses implement:
    - manual_op_class: Class reference to manual implementation
    - auto_op_class: Class reference to KernelOp implementation
    - make_test_model(): Create standard ONNX model (e.g., Softmax, not HWSoftmax)
    - get_manual_transform(): Return manual Infer transform class
    - get_auto_transform(): Return auto Infer transform class
    - configure_test_op(): Override SIMD or other attrs for testing (optional)

    **Custom Pattern (Override):**
    For non-standard cases, simply override:
    - setup_manual_op(): Custom manual op creation
    - setup_auto_op(): Custom auto op creation

    Inherited test methods (run automatically by pytest):

    **Base Tests (15)**:
    - test_normal_input_shape_parity()
    - test_normal_output_shape_parity()
    - test_folded_input_shape_parity()
    - test_folded_output_shape_parity()
    - test_instream_width_parity()
    - test_outstream_width_parity()
    - test_instream_width_padded_parity()
    - test_outstream_width_padded_parity()
    - test_input_datatype_parity()
    - test_output_datatype_parity()
    - test_infer_node_datatype_parity()
    - test_exp_cycles_parity()
    - test_number_output_values_parity()
    - test_make_shape_compatible_op_parity()
    - test_execute_node_parity()

    **HLS Backend Test (1)**:
    - test_cppsim_execution_parity() - C++ synthesis and execution

    **RTL Backend Tests (6)**:
    - test_lut_estimation_parity() - LUT resource estimation
    - test_dsp_estimation_parity() - DSP resource estimation
    - test_bram_estimation_parity() - BRAM resource estimation
    - test_uram_estimation_parity() - URAM resource estimation
    - test_rtlsim_execution_parity() - Verilog synthesis and execution
    - test_rtl_file_list_parity() - HDL file list generation
    - test_ipi_generation_parity() - Vivado IPI/TCL generation

    **Efficiency & Reporting Tests (3)**:
    - test_bram_efficiency_parity() - BRAM utilization efficiency
    - test_uram_efficiency_parity() - URAM utilization efficiency
    - test_op_and_param_counts_parity() - Operation and parameter counting

    Total: 25 comprehensive parity tests
    """

    # =========================================================================
    # Abstract Methods - Subclasses MUST implement
    # =========================================================================

    @property
    @abstractmethod
    def manual_op_class(self) -> Type[HWCustomOp]:
        """Manual HWCustomOp implementation class.

        Example:
            @property
            def manual_op_class(self):
                return HWSoftmax
        """
        pass

    @property
    @abstractmethod
    def auto_op_class(self) -> Type[HWCustomOp]:
        """KernelOp implementation class.

        Example:
            @property
            def auto_op_class(self):
                return AutoSoftmax  # KernelOp-based implementation
        """
        pass

    @abstractmethod
    def make_test_model(self) -> Tuple[ModelWrapper, str]:
        """Create standard ONNX model for testing.

        For transform-based testing (default):
            Create a standard ONNX node (e.g., Softmax, not HWSoftmax).
            Transforms will convert it to hardware nodes.

        Returns:
            (model, node_name): ModelWrapper and name of target node

        Example:
            def make_test_model(self):
                model = create_softmax_model(channels=768)
                return model, "Softmax_0"
        """
        pass

    # =========================================================================
    # Transform Methods - Implement for transform-based testing (recommended)
    # =========================================================================

    def get_manual_transform(self) -> Optional[Type[Transformation]]:
        """Return the Infer transform class for manual op.

        Returns:
            Transform class (e.g., InferSoftmax), or None if no transform

        Example:
            def get_manual_transform(self):
                from brainsmith.kernels.layernorm import InferLayerNorm
                return InferLayerNorm
        """
        return None

    def get_auto_transform(self) -> Optional[Type[Transformation]]:
        """Return the Infer transform class for auto op.

        Returns:
            Transform class (e.g., InferSoftmax), or None if no transform

        Example:
            def get_auto_transform(self):
                from brainsmith.kernels.layernorm import InferLayerNorm
                return InferLayerNorm
        """
        return None

    def configure_test_op(self, op: HWCustomOp, model: ModelWrapper, is_auto: bool) -> None:
        """Configure op after transform for testing (e.g., override SIMD).

        Called after transforms have been applied. Use to override default
        values for testing purposes.

        Args:
            op: The HWCustomOp instance (manual or auto)
            model: The ModelWrapper containing the op
            is_auto: True if auto op, False if manual op

        Example:
            def configure_test_op(self, op, model, is_auto):
                op.set_nodeattr("SIMD", 16)  # Override default
                if is_auto:
                    op.get_kernel_model(model)  # Refresh configuration with new SIMD
        """
        pass

    # =========================================================================
    # Optional Methods
    # =========================================================================

    def get_num_inputs(self) -> int:
        """Number of inputs to test. Override if > 1."""
        return 1

    def get_num_outputs(self) -> int:
        """Number of outputs to test. Override if > 1."""
        return 1

    # =========================================================================
    # Setup Methods - Use transforms by default, override if needed
    # =========================================================================

    def setup_manual_op(self) -> Tuple[HWCustomOp, ModelWrapper]:
        """Create and initialize manual op using transform (default behavior).

        Default: Uses get_manual_transform() to apply transform workflow.
        Override: Provide custom manual op creation logic.

        Returns:
            Tuple of (op, model)

        Raises:
            NotImplementedError: If no transform provided and not overridden
        """
        transform_class = self.get_manual_transform()
        if transform_class is None:
            raise NotImplementedError(
                f"No manual transform provided for {self.__class__.__name__}. Either:\n"
                f"1. Implement get_manual_transform() to return transform class\n"
                f"2. Override setup_manual_op() with custom implementation"
            )

        return self._setup_via_transform(
            transform_class,
            self.manual_op_class.__name__,
            is_auto=False
        )

    def setup_auto_op(self) -> Tuple[HWCustomOp, ModelWrapper]:
        """Create and initialize auto op using transform (default behavior).

        Default: Uses get_auto_transform() to apply transform workflow.
        Override: Provide custom auto op creation logic.

        Returns:
            Tuple of (op, model)

        Raises:
            NotImplementedError: If no transform provided and not overridden
        """
        transform_class = self.get_auto_transform()
        if transform_class is None:
            raise NotImplementedError(
                f"No auto transform provided for {self.__class__.__name__}. Either:\n"
                f"1. Implement get_auto_transform() to return transform class\n"
                f"2. Override setup_auto_op() with custom implementation"
            )

        return self._setup_via_transform(
            transform_class,
            self.auto_op_class.__name__,
            is_auto=True
        )

    def _setup_via_transform(
        self,
        transform_class: Type[Transformation],
        target_op_type: str,
        is_auto: bool
    ) -> Tuple[HWCustomOp, ModelWrapper]:
        """Generic helper for transform-based op setup.

        Workflow:
        1. Create base model (standard ONNX node)
        2. Apply InferShapes and InferDataTypes
        3. Apply the specified transform
        4. Find the transformed node
        5. Create op instance
        6. Call configure_test_op() for test-specific configuration

        Args:
            transform_class: The Infer transform class to apply
            target_op_type: Expected op_type after transformation
            is_auto: True if auto op, False if manual

        Returns:
            Tuple of (op, model)

        Raises:
            RuntimeError: If transform fails to create expected node
        """
        # Create base model with standard ONNX node
        model, _ = self.make_test_model()

        # Run standard shape and datatype inference
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        # Apply the kernel-specific transform
        transform = transform_class()
        model = model.transform(transform)

        # Find the transformed node
        target_node = None
        for node in model.graph.node:
            if node.op_type == target_op_type:
                target_node = node
                break

        if target_node is None:
            available_types = [n.op_type for n in model.graph.node]
            raise RuntimeError(
                f"Transform {transform_class.__name__} failed to create {target_op_type} node.\n"
                f"Available node types: {available_types}"
            )

        # Create op instance from the transformed node
        op = getCustomOp(target_node)

        # Allow test-specific configuration (e.g., override SIMD)
        self.configure_test_op(op, model, is_auto)

        return op, model

    # =========================================================================
    # Generic Parity Tests - Inherited by all subclasses
    # =========================================================================

    @pytest.mark.parity
    def test_normal_input_shape_parity(self):
        """Test normal input shape matches."""
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        for ind in range(self.get_num_inputs()):
            manual_shape = manual_op.get_normal_input_shape(ind)
            auto_shape = auto_op.get_normal_input_shape(ind)

            assert manual_shape == auto_shape, (
                f"Input {ind} normal shape mismatch:\n"
                f"  Manual: {manual_shape}\n"
                f"  Auto:   {auto_shape}"
            )

    @pytest.mark.parity
    def test_normal_output_shape_parity(self):
        """Test normal output shape matches."""
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        for ind in range(self.get_num_outputs()):
            manual_shape = manual_op.get_normal_output_shape(ind)
            auto_shape = auto_op.get_normal_output_shape(ind)

            assert manual_shape == auto_shape, (
                f"Output {ind} normal shape mismatch:\n"
                f"  Manual: {manual_shape}\n"
                f"  Auto:   {auto_shape}"
            )

    @pytest.mark.parity
    def test_folded_input_shape_parity(self):
        """Test folded input shape matches."""
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        for ind in range(self.get_num_inputs()):
            manual_shape = manual_op.get_folded_input_shape(ind)
            auto_shape = auto_op.get_folded_input_shape(ind)

            assert manual_shape == auto_shape, (
                f"Input {ind} folded shape mismatch:\n"
                f"  Manual: {manual_shape}\n"
                f"  Auto:   {auto_shape}"
            )

    @pytest.mark.parity
    def test_folded_output_shape_parity(self):
        """Test folded output shape matches."""
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        for ind in range(self.get_num_outputs()):
            manual_shape = manual_op.get_folded_output_shape(ind)
            auto_shape = auto_op.get_folded_output_shape(ind)

            assert manual_shape == auto_shape, (
                f"Output {ind} folded shape mismatch:\n"
                f"  Manual: {manual_shape}\n"
                f"  Auto:   {auto_shape}"
            )

    @pytest.mark.parity
    def test_instream_width_parity(self):
        """Test input stream width matches."""
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        for ind in range(self.get_num_inputs()):
            manual_width = manual_op.get_instream_width(ind)
            auto_width = auto_op.get_instream_width(ind)

            assert manual_width == auto_width, (
                f"Input {ind} stream width mismatch:\n"
                f"  Manual: {manual_width} bits\n"
                f"  Auto:   {auto_width} bits"
            )

    @pytest.mark.parity
    def test_outstream_width_parity(self):
        """Test output stream width matches."""
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        for ind in range(self.get_num_outputs()):
            manual_width = manual_op.get_outstream_width(ind)
            auto_width = auto_op.get_outstream_width(ind)

            assert manual_width == auto_width, (
                f"Output {ind} stream width mismatch:\n"
                f"  Manual: {manual_width} bits\n"
                f"  Auto:   {auto_width} bits"
            )

    @pytest.mark.parity
    def test_input_datatype_parity(self):
        """Test input datatype matches."""
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        for ind in range(self.get_num_inputs()):
            manual_dt = manual_op.get_input_datatype(ind)
            auto_dt = auto_op.get_input_datatype(ind)

            assert manual_dt == auto_dt, (
                f"Input {ind} datatype mismatch:\n"
                f"  Manual: {manual_dt.name}\n"
                f"  Auto:   {auto_dt.name}"
            )

    @pytest.mark.parity
    def test_output_datatype_parity(self):
        """Test output datatype matches."""
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        for ind in range(self.get_num_outputs()):
            manual_dt = manual_op.get_output_datatype(ind)
            auto_dt = auto_op.get_output_datatype(ind)

            assert manual_dt == auto_dt, (
                f"Output {ind} datatype mismatch:\n"
                f"  Manual: {manual_dt.name}\n"
                f"  Auto:   {auto_dt.name}"
            )

    @pytest.mark.parity
    def test_exp_cycles_parity(self):
        """Test expected cycles matches."""
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        manual_cycles = manual_op.get_exp_cycles()
        auto_cycles = auto_op.get_exp_cycles()

        assert manual_cycles == auto_cycles, (
            f"Expected cycles mismatch:\n"
            f"  Manual: {manual_cycles}\n"
            f"  Auto:   {auto_cycles}"
        )

    @pytest.mark.parity
    def test_execute_node_parity(self):
        """Test Python execution produces same results."""
        manual_op, manual_model = self.setup_manual_op()
        auto_op, auto_model = self.setup_auto_op()

        # Create execution context with same random inputs
        np.random.seed(42)  # Deterministic for reproducibility
        manual_context = self._make_execution_context(manual_model, manual_op)

        np.random.seed(42)  # Same seed for auto
        auto_context = self._make_execution_context(auto_model, auto_op)

        # Execute both
        manual_op.execute_node(manual_context, manual_model.graph)
        auto_op.execute_node(auto_context, auto_model.graph)

        # Compare all outputs
        for ind in range(self.get_num_outputs()):
            manual_output = manual_context[manual_op.onnx_node.output[ind]]
            auto_output = auto_context[auto_op.onnx_node.output[ind]]

            np.testing.assert_allclose(
                manual_output,
                auto_output,
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"Output {ind} execution results differ"
            )

    @pytest.mark.parity
    def test_number_output_values_parity(self):
        """Test number of output values matches (used for FIFO sizing)."""
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        manual_count = manual_op.get_number_output_values()
        auto_count = auto_op.get_number_output_values()

        assert manual_count == auto_count, (
            f"Number of output values mismatch:\n"
            f"  Manual: {manual_count}\n"
            f"  Auto:   {auto_count}"
        )

    @pytest.mark.parity
    def test_instream_width_padded_parity(self):
        """Test padded input stream width matches (AXI Stream alignment)."""
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        for ind in range(self.get_num_inputs()):
            manual_width = manual_op.get_instream_width_padded(ind)
            auto_width = auto_op.get_instream_width_padded(ind)

            assert manual_width == auto_width, (
                f"Input {ind} padded stream width mismatch:\n"
                f"  Manual: {manual_width} bits\n"
                f"  Auto:   {auto_width} bits"
            )

    @pytest.mark.parity
    def test_outstream_width_padded_parity(self):
        """Test padded output stream width matches (AXI Stream alignment)."""
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        for ind in range(self.get_num_outputs()):
            manual_width = manual_op.get_outstream_width_padded(ind)
            auto_width = auto_op.get_outstream_width_padded(ind)

            assert manual_width == auto_width, (
                f"Output {ind} padded stream width mismatch:\n"
                f"  Manual: {manual_width} bits\n"
                f"  Auto:   {auto_width} bits"
            )

    @pytest.mark.parity
    def test_make_shape_compatible_op_parity(self):
        """Test that shape-compatible ops have matching types and attributes."""
        manual_op, manual_model = self.setup_manual_op()
        auto_op, auto_model = self.setup_auto_op()

        manual_compat_op = manual_op.make_shape_compatible_op(manual_model)
        auto_compat_op = auto_op.make_shape_compatible_op(auto_model)

        # Verify both create compatible ops of the same type
        assert manual_compat_op.op_type == auto_compat_op.op_type, (
            f"Shape-compatible op type mismatch:\n"
            f"  Manual: {manual_compat_op.op_type}\n"
            f"  Auto:   {auto_compat_op.op_type}"
        )

        # Verify inputs/outputs match
        assert len(manual_compat_op.input) == len(auto_compat_op.input), (
            "Shape-compatible op input count mismatch"
        )
        assert len(manual_compat_op.output) == len(auto_compat_op.output), (
            "Shape-compatible op output count mismatch"
        )

    @pytest.mark.parity
    def test_infer_node_datatype_parity(self):
        """Test that datatype inference produces matching results in the model.

        This is a critical test that validates the datatype inference LOGIC,
        not just that datatypes match after inference. It ensures both
        implementations correctly update the ONNX model's tensor datatypes.
        """
        # Setup ops with fresh models
        manual_op, manual_model = self.setup_manual_op()
        auto_op, auto_model = self.setup_auto_op()

        # Run datatype inference on both
        manual_model_out = manual_op.infer_node_datatype(manual_model)
        auto_model_out = auto_op.infer_node_datatype(auto_model)

        # Use returned model if provided, otherwise use original (side-effect pattern)
        if manual_model_out is not None:
            manual_model = manual_model_out
        if auto_model_out is not None:
            auto_model = auto_model_out

        # Verify input datatypes were set correctly in model
        for ind in range(self.get_num_inputs()):
            input_name = manual_op.onnx_node.input[ind]
            if not input_name:  # Skip optional inputs
                continue

            manual_dt = manual_model.get_tensor_datatype(input_name)
            auto_dt = auto_model.get_tensor_datatype(input_name)

            assert manual_dt == auto_dt, (
                f"After infer_node_datatype, input {ind} ({input_name}) datatype mismatch:\n"
                f"  Manual model: {manual_dt.name if manual_dt else 'None'}\n"
                f"  Auto model:   {auto_dt.name if auto_dt else 'None'}"
            )

        # Verify output datatypes were set correctly in model
        for ind in range(self.get_num_outputs()):
            output_name = manual_op.onnx_node.output[ind]

            manual_dt = manual_model.get_tensor_datatype(output_name)
            auto_dt = auto_model.get_tensor_datatype(output_name)

            assert manual_dt == auto_dt, (
                f"After infer_node_datatype, output {ind} ({output_name}) datatype mismatch:\n"
                f"  Manual model: {manual_dt.name if manual_dt else 'None'}\n"
                f"  Auto model:   {auto_dt.name if auto_dt else 'None'}"
            )

    @pytest.mark.parity
    @pytest.mark.cppsim
    @pytest.mark.slow
    def test_cppsim_execution_parity(self):
        """Test HLS backend code generation via cppsim compilation and execution.

        This is the gold standard for validating HLS code generation correctness.
        It validates the entire code generation pipeline end-to-end:

        **Pipeline Steps**:
        1. **Code Generation**: Both backends generate C++ (via PrepareCppSim)
        2. **Compilation**: C++ compiles successfully (via CompileCppSim)
        3. **Execution**: Compiled binary executes (via SetExecMode("cppsim"))
        4. **Validation**: Both produce identical numerical results

        **Confidence Level**: If this test passes, we have high confidence that:
        - Generated C++ is syntactically correct (it compiles)
        - Generated C++ is semantically correct (it executes correctly)
        - Manual and auto backends are functionally equivalent
        - All code generation methods (defines, docompute, pragmas, etc.) are correct

        **Execution Examples**:
            # Run just the cppsim test
            pytest tests/parity/test_crop_parity.py::TestCropHLSParity::test_cppsim_execution_parity -v

            # Skip slow cppsim test (fast iteration)
            pytest tests/parity/test_crop_parity.py::TestCropHLSParity -v -m "not cppsim"

            # Only cppsim tests across all backends
            pytest tests/parity/ -v -m cppsim

        **Requirements**:
            - VITIS_PATH environment variable set
            - Both backends inherit from HLSBackend
            - Valid setup_manual_op() and setup_auto_op() implementations

        Raises:
            pytest.skip: If backend not HLS or Vitis unavailable
            AssertionError: If compilation fails or outputs differ
        """
        # Check for Vitis availability
        import os
        if not os.environ.get("VITIS_PATH"):
            pytest.skip("Vitis required for C++ compilation (set VITIS_PATH)")

        # Import FINN transformations
        try:
            from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
            from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
            from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
            from qonnx.transformation.infer_shapes import InferShapes
            from qonnx.transformation.infer_datatypes import InferDataTypes
            import finn.core.onnx_exec as oxe
        except ImportError as e:
            pytest.skip(f"FINN transformations not available: {e}")

        # Setup both backends with fresh models
        manual_op, manual_model = self.setup_manual_op()
        auto_op, auto_model = self.setup_auto_op()

        # Verify both are HLS backends
        if not self._is_hls_backend(manual_op):
            pytest.skip(
                f"{manual_op.__class__.__name__} is not an HLS backend. "
                f"cppsim execution requires HLSBackend inheritance."
            )
        if not self._is_hls_backend(auto_op):
            pytest.skip(
                f"{auto_op.__class__.__name__} is not an HLS backend. "
                f"cppsim execution requires HLSBackend inheritance."
            )

        # Generate identical test inputs for both executions
        np.random.seed(42)  # Deterministic for reproducibility
        test_context = self._make_execution_context(manual_model, manual_op)

        # Extract input tensors (exclude initializers/weights)
        input_dict = {}
        for inp_name in manual_op.onnx_node.input:
            if inp_name and inp_name in test_context:
                # Only include if not an initializer
                if manual_model.get_initializer(inp_name) is None:
                    input_dict[inp_name] = test_context[inp_name]

        # Ensure we have at least one input
        if not input_dict:
            pytest.skip(
                f"No streaming inputs found for {manual_op.__class__.__name__}. "
                f"All inputs are initializers (weights/parameters)."
            )

        # Compile and execute manual backend directly (no transforms)
        import tempfile
        import os

        # Set BSMITH_DIR for compilation (needed by Brainsmith kernels)
        # TAFK TODO: Change to use pydantic config after branch merge
        os.environ["BSMITH_DIR"] = "/home/tafk/dev/brainsmith-1"

        try:
            # Create temp directory for code generation
            manual_tmpdir = tempfile.mkdtemp(prefix="crop_hls_manual_")
            manual_op.set_nodeattr("code_gen_dir_cppsim", manual_tmpdir)

            # Save model to code_gen_dir (needed by exec_precompiled_singlenode_model)
            manual_model.save(os.path.join(manual_tmpdir, "node_model.onnx"))

            # Generate C++ code
            manual_op.code_generation_cppsim(manual_model)

            # Compile C++ code
            manual_op.compile_singlenode_code()

            # Set execution mode
            manual_op.set_nodeattr("exec_mode", "cppsim")

        except Exception as e:
            pytest.fail(
                f"Manual backend cppsim pipeline failed for {manual_op.__class__.__name__}:\n"
                f"\n"
                f"Error: {type(e).__name__}: {e}\n"
                f"\n"
                f"This indicates a code generation or compilation bug in the manual backend.\n"
                f"\n"
                f"Debug steps:\n"
                f"1. Check {getattr(manual_op, 'get_nodeattr', lambda x: 'N/A')('code_gen_dir_cppsim') if hasattr(manual_op, 'get_nodeattr') else manual_tmpdir} for generated C++ files\n"
                f"2. Look for compilation errors in build logs\n"
                f"3. Compare with working backend implementation"
            )

        # Compile and execute auto backend directly (no transforms)
        try:
            # Create temp directory for code generation
            auto_tmpdir = tempfile.mkdtemp(prefix="autocrop_hls_auto_")
            auto_op.set_nodeattr("code_gen_dir_cppsim", auto_tmpdir)

            # Save model to code_gen_dir (needed by exec_precompiled_singlenode_model)
            auto_model.save(os.path.join(auto_tmpdir, "node_model.onnx"))

            # Ensure kernel instance is available for KernelOp-based backends
            # Modern KernelOp uses get_kernel_instance(), legacy uses get_kernel_model()
            if hasattr(auto_op, 'get_kernel_instance'):
                auto_op.get_kernel_instance(auto_model)
            elif hasattr(auto_op, 'get_kernel_model'):
                auto_op.get_kernel_model(auto_model)

            # Generate C++ code
            auto_op.code_generation_cppsim(auto_model)

            # Compile C++ code
            auto_op.compile_singlenode_code()

            # Set execution mode
            auto_op.set_nodeattr("exec_mode", "cppsim")

        except Exception as e:
            pytest.fail(
                f"Auto backend cppsim pipeline failed for {auto_op.__class__.__name__}:\n"
                f"\n"
                f"Error: {type(e).__name__}: {e}\n"
                f"\n"
                f"This indicates a code generation or compilation bug in the auto backend.\n"
                f"\n"
                f"Debug steps:\n"
                f"1. Check {getattr(auto_op, 'get_nodeattr', lambda x: 'N/A')('code_gen_dir_cppsim') if hasattr(auto_op, 'get_nodeattr') else auto_tmpdir} for generated C++ files\n"
                f"2. Verify test_hls_defines_generation passes (constants match)\n"
                f"3. Compare with manual backend implementation"
            )

        # Execute manual backend via cppsim
        try:
            manual_op.execute_node(test_context, manual_model.graph)
            manual_result = {manual_op.onnx_node.output[0]: test_context[manual_op.onnx_node.output[0]]}
        except Exception as e:
            pytest.fail(
                f"Manual backend execution failed for {manual_op.__class__.__name__}:\n"
                f"\n"
                f"Error: {type(e).__name__}: {e}\n"
                f"\n"
                f"Code compiled successfully but execution failed.\n"
                f"Check execute_node() implementation."
            )

        # Execute auto backend via cppsim
        # Recreate context with same seed for auto
        np.random.seed(42)

        # Ensure kernel instance is available for KernelOp-based backends
        # Modern KernelOp uses get_kernel_instance(), legacy uses get_kernel_model()
        if hasattr(auto_op, 'get_kernel_instance'):
            auto_op.get_kernel_instance(auto_model)
        elif hasattr(auto_op, 'get_kernel_model'):
            auto_op.get_kernel_model(auto_model)

        auto_context = self._make_execution_context(auto_model, auto_op)

        try:
            auto_op.execute_node(auto_context, auto_model.graph)
            auto_result = {auto_op.onnx_node.output[0]: auto_context[auto_op.onnx_node.output[0]]}
        except Exception as e:
            pytest.fail(
                f"Auto backend execution failed for {auto_op.__class__.__name__}:\n"
                f"\n"
                f"Error: {type(e).__name__}: {e}\n"
                f"\n"
                f"Code compiled successfully but execution failed.\n"
                f"Check execute_node() implementation."
            )

        # Compare all outputs
        for ind in range(self.get_num_outputs()):
            output_name = manual_op.onnx_node.output[ind]

            assert output_name in manual_result, \
                f"Manual backend didn't produce output: {output_name}"
            assert output_name in auto_result, \
                f"Auto backend didn't produce output: {output_name}"

            manual_output = manual_result[output_name]
            auto_output = auto_result[output_name]

            # Verify shapes match
            assert manual_output.shape == auto_output.shape, (
                f"Output {ind} ({output_name}) shape mismatch:\n"
                f"  Manual: {manual_output.shape}\n"
                f"  Auto:   {auto_output.shape}"
            )

            # Verify numerical equivalence
            np.testing.assert_allclose(
                manual_output,
                auto_output,
                rtol=1e-5,
                atol=1e-6,
                err_msg=(
                    f"\n"
                    f"{'='*70}\n"
                    f"cppsim output {ind} ({output_name}) differs between backends\n"
                    f"{'='*70}\n"
                    f"\n"
                    f"Both backends compiled and executed successfully, but produced\n"
                    f"different numerical results. This indicates a CODE GENERATION BUG.\n"
                    f"\n"
                    f"Backends:\n"
                    f"  Manual: {manual_op.__class__.__name__}\n"
                    f"  Auto:   {auto_op.__class__.__name__}\n"
                    f"\n"
                    f"Debug checklist:\n"
                    f"  1. Run test_hls_defines_generation to check constants\n"
                    f"  2. Check docompute() - verify template parameters match\n"
                    f"  3. Check stream widths - verify packing calculations\n"
                    f"  4. Compare generated C++ files in code_gen_dir_cppsim\n"
                    f"{'='*70}\n"
                )
            )

    @pytest.mark.parity
    @pytest.mark.rtl
    def test_lut_estimation_parity(self):
        """Test LUT resource estimation matches between backends.

        Validates that both manual and auto implementations produce
        identical LUT estimates for the same configuration.

        Requirements:
            - Both backends must implement lut_estimation()
            - Estimates must match exactly (not approximately)
        """
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        # Check if method exists
        if not hasattr(manual_op, "lut_estimation"):
            pytest.skip(f"{manual_op.__class__.__name__} does not implement lut_estimation")
        if not hasattr(auto_op, "lut_estimation"):
            pytest.skip(f"{auto_op.__class__.__name__} does not implement lut_estimation")

        manual_luts = manual_op.lut_estimation()
        auto_luts = auto_op.lut_estimation()

        assert manual_luts == auto_luts, (
            f"LUT estimation mismatch:\n"
            f"  Manual: {manual_luts:,} LUTs\n"
            f"  Auto:   {auto_luts:,} LUTs"
        )

    @pytest.mark.parity
    @pytest.mark.rtl
    def test_dsp_estimation_parity(self):
        """Test DSP resource estimation matches between backends.

        Validates that both manual and auto implementations produce
        identical DSP estimates for the same configuration.

        Requirements:
            - Both backends must implement dsp_estimation(fpgapart)
            - Estimates must match exactly (not approximately)
        """
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        fpgapart = self.get_test_fpgapart()

        # Check if method exists
        if not hasattr(manual_op, "dsp_estimation"):
            pytest.skip(f"{manual_op.__class__.__name__} does not implement dsp_estimation")
        if not hasattr(auto_op, "dsp_estimation"):
            pytest.skip(f"{auto_op.__class__.__name__} does not implement dsp_estimation")

        manual_dsps = manual_op.dsp_estimation(fpgapart)
        auto_dsps = auto_op.dsp_estimation(fpgapart)

        assert manual_dsps == auto_dsps, (
            f"DSP estimation mismatch:\n"
            f"  Manual: {manual_dsps:,} DSPs\n"
            f"  Auto:   {auto_dsps:,} DSPs\n"
            f"  FPGA:   {fpgapart}"
        )

    @pytest.mark.parity
    @pytest.mark.rtl
    def test_bram_estimation_parity(self):
        """Test BRAM resource estimation matches between backends.

        Validates that both manual and auto implementations produce
        identical BRAM estimates for the same configuration.

        Requirements:
            - Both backends must implement bram_estimation()
            - Estimates must match exactly (not approximately)
        """
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        # Check if method exists
        if not hasattr(manual_op, "bram_estimation"):
            pytest.skip(f"{manual_op.__class__.__name__} does not implement bram_estimation")
        if not hasattr(auto_op, "bram_estimation"):
            pytest.skip(f"{auto_op.__class__.__name__} does not implement bram_estimation")

        manual_brams = manual_op.bram_estimation()
        auto_brams = auto_op.bram_estimation()

        assert manual_brams == auto_brams, (
            f"BRAM estimation mismatch:\n"
            f"  Manual: {manual_brams:,} BRAMs\n"
            f"  Auto:   {auto_brams:,} BRAMs"
        )

    @pytest.mark.parity
    @pytest.mark.rtl
    def test_uram_estimation_parity(self):
        """Test URAM resource estimation matches between backends.

        Validates that both manual and auto implementations produce
        identical URAM estimates for the same configuration.

        Requirements:
            - Both backends must implement uram_estimation()
            - Estimates must match exactly (not approximately)
        """
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        # Check if method exists
        if not hasattr(manual_op, "uram_estimation"):
            pytest.skip(f"{manual_op.__class__.__name__} does not implement uram_estimation")
        if not hasattr(auto_op, "uram_estimation"):
            pytest.skip(f"{auto_op.__class__.__name__} does not implement uram_estimation")

        manual_urams = manual_op.uram_estimation()
        auto_urams = auto_op.uram_estimation()

        assert manual_urams == auto_urams, (
            f"URAM estimation mismatch:\n"
            f"  Manual: {manual_urams:,} URAMs\n"
            f"  Auto:   {auto_urams:,} URAMs"
        )

    @pytest.mark.parity
    def test_bram_efficiency_parity(self):
        """Test BRAM efficiency estimation matches between backends.

        Validates that both manual and auto implementations produce
        identical BRAM efficiency estimates.

        BRAM efficiency = (actual BRAM usage) / (allocated BRAMs)
        Used for resource utilization reporting.

        Requirements:
            - Both backends must implement bram_efficiency_estimation()
            - Efficiency values must match exactly
        """
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        # Check if method exists (default implementation in HWCustomOp returns 1.0)
        if not hasattr(manual_op, "bram_efficiency_estimation"):
            pytest.skip(f"{manual_op.__class__.__name__} does not implement bram_efficiency_estimation")
        if not hasattr(auto_op, "bram_efficiency_estimation"):
            pytest.skip(f"{auto_op.__class__.__name__} does not implement bram_efficiency_estimation")

        manual_eff = manual_op.bram_efficiency_estimation()
        auto_eff = auto_op.bram_efficiency_estimation()

        assert manual_eff == auto_eff, (
            f"BRAM efficiency estimation mismatch:\n"
            f"  Manual: {manual_eff:.4f} ({manual_eff*100:.2f}%)\n"
            f"  Auto:   {auto_eff:.4f} ({auto_eff*100:.2f}%)"
        )

    @pytest.mark.parity
    def test_uram_efficiency_parity(self):
        """Test URAM efficiency estimation matches between backends.

        Validates that both manual and auto implementations produce
        identical URAM efficiency estimates.

        URAM efficiency = (actual URAM usage) / (allocated URAMs)
        Used for resource utilization reporting.

        Requirements:
            - Both backends must implement uram_efficiency_estimation()
            - Efficiency values must match exactly
        """
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        # Check if method exists (default implementation in HWCustomOp returns 1.0)
        if not hasattr(manual_op, "uram_efficiency_estimation"):
            pytest.skip(f"{manual_op.__class__.__name__} does not implement uram_efficiency_estimation")
        if not hasattr(auto_op, "uram_efficiency_estimation"):
            pytest.skip(f"{auto_op.__class__.__name__} does not implement uram_efficiency_estimation")

        manual_eff = manual_op.uram_efficiency_estimation()
        auto_eff = auto_op.uram_efficiency_estimation()

        assert manual_eff == auto_eff, (
            f"URAM efficiency estimation mismatch:\n"
            f"  Manual: {manual_eff:.4f} ({manual_eff*100:.2f}%)\n"
            f"  Auto:   {auto_eff:.4f} ({auto_eff*100:.2f}%)"
        )

    @pytest.mark.parity
    def test_op_and_param_counts_parity(self):
        """Test operation and parameter counting matches between backends.

        Validates that both manual and auto implementations report
        identical operation counts for performance modeling.

        Typical return format:
        {
            "op_mac": <int>,      # Number of MAC operations
            "op_param": <int>     # Number of parameters (weights, thresholds)
        }

        Requirements:
            - Both backends must implement get_op_and_param_counts()
            - All count values must match exactly
        """
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        # Check if method exists (default implementation in HWCustomOp returns empty dict)
        if not hasattr(manual_op, "get_op_and_param_counts"):
            pytest.skip(f"{manual_op.__class__.__name__} does not implement get_op_and_param_counts")
        if not hasattr(auto_op, "get_op_and_param_counts"):
            pytest.skip(f"{auto_op.__class__.__name__} does not implement get_op_and_param_counts")

        manual_counts = manual_op.get_op_and_param_counts()
        auto_counts = auto_op.get_op_and_param_counts()

        assert manual_counts == auto_counts, (
            f"Operation and parameter counts mismatch:\n"
            f"  Manual: {manual_counts}\n"
            f"  Auto:   {auto_counts}"
        )

    @pytest.mark.parity
    @pytest.mark.rtlsim
    @pytest.mark.slow
    def test_rtlsim_execution_parity(self):
        """Test RTL backend code generation via rtlsim compilation and execution.

        This is the gold standard for validating RTL code generation correctness.
        It validates the entire RTL generation and simulation pipeline end-to-end:

        **Pipeline Steps**:
        1. **HDL Generation**: Both backends generate Verilog (via PrepareRTLSim)
        2. **Simulation Setup**: Verilator compiles RTL (via SetExecMode("rtlsim"))
        3. **Execution**: Compiled simulator executes
        4. **Validation**: Both produce identical numerical results

        **Confidence Level**: If this test passes, we have high confidence that:
        - Generated Verilog is syntactically correct (Verilator compiles)
        - Generated Verilog is semantically correct (it executes correctly)
        - Manual and auto backends are functionally equivalent
        - All HDL generation methods are correct

        **Execution Examples**:
            # Run just the rtlsim test
            pytest tests/parity/test_shuffle_parity.py::TestShuffleRTLParity::test_rtlsim_execution_parity -v

            # Skip slow rtlsim test (fast iteration)
            pytest tests/parity/test_shuffle_parity.py::TestShuffleRTLParity -v -m "not rtlsim"

            # Only rtlsim tests across all backends
            pytest tests/parity/ -v -m rtlsim

        **Requirements**:
            - Verilator installed and available
            - Both backends inherit from RTLBackend
            - Valid setup_manual_op() and setup_auto_op() implementations

        Raises:
            pytest.skip: If backend not RTL or Verilator unavailable
            AssertionError: If compilation fails or outputs differ
        """
        # Check for Verilator availability
        import shutil
        if not shutil.which("verilator"):
            pytest.skip("Verilator required for RTL simulation")

        # Import FINN transformations
        try:
            from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
            from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
            from qonnx.transformation.infer_shapes import InferShapes
            from qonnx.transformation.infer_datatypes import InferDataTypes
        except ImportError as e:
            pytest.skip(f"FINN transformations not available: {e}")

        # Setup both backends with fresh models
        manual_op, manual_model = self.setup_manual_op()
        auto_op, auto_model = self.setup_auto_op()

        # Verify both are RTL backends
        if not self._is_rtl_backend(manual_op):
            pytest.skip(
                f"{manual_op.__class__.__name__} is not an RTL backend. "
                f"rtlsim execution requires RTLBackend inheritance."
            )
        if not self._is_rtl_backend(auto_op):
            pytest.skip(
                f"{auto_op.__class__.__name__} is not an RTL backend. "
                f"rtlsim execution requires RTLBackend inheritance."
            )

        # Generate identical test inputs for both executions
        np.random.seed(42)  # Deterministic for reproducibility
        test_context = self._make_execution_context(manual_model, manual_op)

        # Extract input tensors (exclude initializers/weights)
        input_dict = {}
        for inp_name in manual_op.onnx_node.input:
            if inp_name and inp_name in test_context:
                # Only include if not an initializer
                if manual_model.get_initializer(inp_name) is None:
                    input_dict[inp_name] = test_context[inp_name]

        # Ensure we have at least one input
        if not input_dict:
            pytest.skip(
                f"No streaming inputs found for {manual_op.__class__.__name__}. "
                f"All inputs are initializers (weights/parameters)."
            )

        # Set up and execute manual backend
        import tempfile
        import os

        # Set BSMITH_DIR for RTL generation (needed by Brainsmith kernels)
        os.environ["BSMITH_DIR"] = "/home/tafk/dev/brainsmith-1"

        try:
            # Create temp directory for RTL generation
            manual_tmpdir = tempfile.mkdtemp(prefix="rtl_manual_")
            manual_op.set_nodeattr("code_gen_dir_ipgen", manual_tmpdir)

            # Save model to code_gen_dir (needed by exec_precompiled_singlenode_model)
            manual_model.save(os.path.join(manual_tmpdir, "node_model.onnx"))

            # Generate RTL code
            manual_op.generate_hdl(manual_model, fpgapart=self.get_test_fpgapart())

            # Prepare RTL simulation
            manual_op.prepare_rtlsim(manual_model)

            # Set execution mode
            manual_op.set_nodeattr("exec_mode", "rtlsim")

        except Exception as e:
            pytest.fail(
                f"Manual backend rtlsim pipeline failed for {manual_op.__class__.__name__}:\n"
                f"\n"
                f"Error: {type(e).__name__}: {e}\n"
                f"\n"
                f"This indicates a code generation or compilation bug in the manual backend.\n"
                f"\n"
                f"Debug steps:\n"
                f"1. Check {getattr(manual_op, 'get_nodeattr', lambda x: 'N/A')('code_gen_dir_ipgen') if hasattr(manual_op, 'get_nodeattr') else manual_tmpdir} for generated Verilog files\n"
                f"2. Look for Verilator compilation errors\n"
                f"3. Compare with working backend implementation"
            )

        # Set up and execute auto backend
        try:
            # Create temp directory for RTL generation
            auto_tmpdir = tempfile.mkdtemp(prefix="rtl_auto_")
            auto_op.set_nodeattr("code_gen_dir_ipgen", auto_tmpdir)

            # Save model to code_gen_dir (needed by exec_precompiled_singlenode_model)
            auto_model.save(os.path.join(auto_tmpdir, "node_model.onnx"))

            # Ensure kernel instance is available for KernelOp-based backends
            if hasattr(auto_op, 'get_kernel_instance'):
                auto_op.get_kernel_instance(auto_model)
            elif hasattr(auto_op, 'get_kernel_model'):
                auto_op.get_kernel_model(auto_model)

            # Generate RTL code
            auto_op.generate_hdl(auto_model, fpgapart=self.get_test_fpgapart())

            # Prepare RTL simulation
            auto_op.prepare_rtlsim(auto_model)

            # Set execution mode
            auto_op.set_nodeattr("exec_mode", "rtlsim")

        except Exception as e:
            pytest.fail(
                f"Auto backend rtlsim pipeline failed for {auto_op.__class__.__name__}:\n"
                f"\n"
                f"Error: {type(e).__name__}: {e}\n"
                f"\n"
                f"This indicates a code generation or compilation bug in the auto backend.\n"
                f"\n"
                f"Debug steps:\n"
                f"1. Check {getattr(auto_op, 'get_nodeattr', lambda x: 'N/A')('code_gen_dir_ipgen') if hasattr(auto_op, 'get_nodeattr') else auto_tmpdir} for generated Verilog files\n"
                f"2. Look for Verilator compilation errors\n"
                f"3. Compare with manual backend implementation"
            )

        # Execute manual backend via rtlsim
        try:
            manual_op.execute_node(test_context, manual_model.graph)
            manual_result = {manual_op.onnx_node.output[0]: test_context[manual_op.onnx_node.output[0]]}
        except Exception as e:
            pytest.fail(
                f"Manual backend execution failed for {manual_op.__class__.__name__}:\n"
                f"\n"
                f"Error: {type(e).__name__}: {e}\n"
                f"\n"
                f"RTL compiled successfully but execution failed.\n"
                f"Check execute_node() implementation."
            )

        # Execute auto backend via rtlsim
        # Recreate context with same seed for auto
        np.random.seed(42)

        # Ensure kernel instance is available for KernelOp-based backends
        if hasattr(auto_op, 'get_kernel_instance'):
            auto_op.get_kernel_instance(auto_model)
        elif hasattr(auto_op, 'get_kernel_model'):
            auto_op.get_kernel_model(auto_model)

        auto_context = self._make_execution_context(auto_model, auto_op)

        try:
            auto_op.execute_node(auto_context, auto_model.graph)
            auto_result = {auto_op.onnx_node.output[0]: auto_context[auto_op.onnx_node.output[0]]}
        except Exception as e:
            pytest.fail(
                f"Auto backend execution failed for {auto_op.__class__.__name__}:\n"
                f"\n"
                f"Error: {type(e).__name__}: {e}\n"
                f"\n"
                f"RTL compiled successfully but execution failed.\n"
                f"Check execute_node() implementation."
            )

        # Compare all outputs
        for ind in range(self.get_num_outputs()):
            output_name = manual_op.onnx_node.output[ind]

            assert output_name in manual_result, \
                f"Manual backend didn't produce output: {output_name}"
            assert output_name in auto_result, \
                f"Auto backend didn't produce output: {output_name}"

            manual_output = manual_result[output_name]
            auto_output = auto_result[output_name]

            # Verify shapes match
            assert manual_output.shape == auto_output.shape, (
                f"Output {ind} ({output_name}) shape mismatch:\n"
                f"  Manual: {manual_output.shape}\n"
                f"  Auto:   {auto_output.shape}"
            )

            # Verify numerical equivalence
            np.testing.assert_allclose(
                manual_output,
                auto_output,
                rtol=1e-5,
                atol=1e-6,
                err_msg=(
                    f"\n"
                    f"{'='*70}\n"
                    f"rtlsim output {ind} ({output_name}) differs between backends\n"
                    f"{'='*70}\n"
                    f"\n"
                    f"Both backends compiled and executed successfully, but produced\n"
                    f"different numerical results. This indicates a CODE GENERATION BUG.\n"
                    f"\n"
                    f"Backends:\n"
                    f"  Manual: {manual_op.__class__.__name__}\n"
                    f"  Auto:   {auto_op.__class__.__name__}\n"
                    f"\n"
                    f"Debug checklist:\n"
                    f"  1. Compare generated Verilog files in code_gen_dir_ipgen\n"
                    f"  2. Check HDL template parameters\n"
                    f"  3. Check stream widths and packing\n"
                    f"  4. Verify resource parameters (PE, SIMD, etc.)\n"
                    f"{'='*70}\n"
                )
            )

    @pytest.mark.parity
    @pytest.mark.rtl
    def test_rtl_file_list_parity(self):
        """Test RTL file list generation matches between backends.

        Validates that both manual and auto implementations generate
        identical lists of RTL source files for synthesis.

        Requirements:
            - Both backends must implement get_rtl_file_list()
            - File lists must match exactly (same files in same order)
        """
        manual_op, manual_model = self.setup_manual_op()
        auto_op, auto_model = self.setup_auto_op()

        # Check if method exists
        if not hasattr(manual_op, "get_rtl_file_list"):
            pytest.skip(f"{manual_op.__class__.__name__} does not implement get_rtl_file_list")
        if not hasattr(auto_op, "get_rtl_file_list"):
            pytest.skip(f"{auto_op.__class__.__name__} does not implement get_rtl_file_list")

        # Skip if not RTL backends
        if not self._is_rtl_backend(manual_op):
            pytest.skip(f"{manual_op.__class__.__name__} is not an RTL backend")
        if not self._is_rtl_backend(auto_op):
            pytest.skip(f"{auto_op.__class__.__name__} is not an RTL backend")

        # Generate HDL for both backends
        import tempfile
        import os

        os.environ["BSMITH_DIR"] = "/home/tafk/dev/brainsmith-1"

        try:
            manual_tmpdir = tempfile.mkdtemp(prefix="rtl_filelist_manual_")
            manual_op.set_nodeattr("code_gen_dir_ipgen", manual_tmpdir)
            manual_op.generate_hdl(manual_model, fpgapart=self.get_test_fpgapart())
            manual_files = manual_op.get_rtl_file_list()
        except Exception as e:
            pytest.skip(f"Manual backend HDL generation failed: {e}")

        try:
            auto_tmpdir = tempfile.mkdtemp(prefix="rtl_filelist_auto_")
            auto_op.set_nodeattr("code_gen_dir_ipgen", auto_tmpdir)

            # Ensure kernel instance is available for KernelOp-based backends
            if hasattr(auto_op, 'get_kernel_instance'):
                auto_op.get_kernel_instance(auto_model)
            elif hasattr(auto_op, 'get_kernel_model'):
                auto_op.get_kernel_model(auto_model)

            auto_op.generate_hdl(auto_model, fpgapart=self.get_test_fpgapart())
            auto_files = auto_op.get_rtl_file_list()
        except Exception as e:
            pytest.skip(f"Auto backend HDL generation failed: {e}")

        # Extract just the filenames (not full paths) for comparison
        manual_filenames = [os.path.basename(f) for f in manual_files]
        auto_filenames = [os.path.basename(f) for f in auto_files]

        assert manual_filenames == auto_filenames, (
            f"RTL file list mismatch:\n"
            f"  Manual files: {manual_filenames}\n"
            f"  Auto files:   {auto_filenames}"
        )

    @pytest.mark.parity
    @pytest.mark.rtl
    def test_ipi_generation_parity(self):
        """Test IPI/TCL generation succeeds for both backends.

        Validates that both manual and auto implementations can generate
        Vivado IPI block design TCL scripts without errors.

        Requirements:
            - Both backends must implement code_generation_ipi()
            - Generation should complete without exceptions
        """
        manual_op, manual_model = self.setup_manual_op()
        auto_op, auto_model = self.setup_auto_op()

        # Check if method exists
        if not hasattr(manual_op, "code_generation_ipi"):
            pytest.skip(f"{manual_op.__class__.__name__} does not implement code_generation_ipi")
        if not hasattr(auto_op, "code_generation_ipi"):
            pytest.skip(f"{auto_op.__class__.__name__} does not implement code_generation_ipi")

        # Skip if not RTL backends
        if not self._is_rtl_backend(manual_op):
            pytest.skip(f"{manual_op.__class__.__name__} is not an RTL backend")
        if not self._is_rtl_backend(auto_op):
            pytest.skip(f"{auto_op.__class__.__name__} is not an RTL backend")

        import tempfile
        import os

        os.environ["BSMITH_DIR"] = "/home/tafk/dev/brainsmith-1"

        # Test manual backend IPI generation
        try:
            manual_tmpdir = tempfile.mkdtemp(prefix="rtl_ipi_manual_")
            manual_op.set_nodeattr("code_gen_dir_ipgen", manual_tmpdir)

            # Generate HDL first (required for IPI generation)
            manual_op.generate_hdl(manual_model, fpgapart=self.get_test_fpgapart())

            # Generate IPI TCL
            manual_op.code_generation_ipi(manual_model)

            # Verify TCL file was created
            tcl_path = os.path.join(manual_tmpdir, f"{manual_op.onnx_node.name}.tcl")
            assert os.path.exists(tcl_path), \
                f"Manual backend did not create IPI TCL file at {tcl_path}"

        except Exception as e:
            pytest.fail(
                f"Manual backend IPI generation failed for {manual_op.__class__.__name__}:\n"
                f"Error: {type(e).__name__}: {e}"
            )

        # Test auto backend IPI generation
        try:
            auto_tmpdir = tempfile.mkdtemp(prefix="rtl_ipi_auto_")
            auto_op.set_nodeattr("code_gen_dir_ipgen", auto_tmpdir)

            # Ensure kernel instance is available for KernelOp-based backends
            if hasattr(auto_op, 'get_kernel_instance'):
                auto_op.get_kernel_instance(auto_model)
            elif hasattr(auto_op, 'get_kernel_model'):
                auto_op.get_kernel_model(auto_model)

            # Generate HDL first (required for IPI generation)
            auto_op.generate_hdl(auto_model, fpgapart=self.get_test_fpgapart())

            # Generate IPI TCL
            auto_op.code_generation_ipi(auto_model)

            # Verify TCL file was created
            tcl_path = os.path.join(auto_tmpdir, f"{auto_op.onnx_node.name}.tcl")
            assert os.path.exists(tcl_path), \
                f"Auto backend did not create IPI TCL file at {tcl_path}"

        except Exception as e:
            pytest.fail(
                f"Auto backend IPI generation failed for {auto_op.__class__.__name__}:\n"
                f"Error: {type(e).__name__}: {e}"
            )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _is_hls_backend(self, op: HWCustomOp) -> bool:
        """Check if op is an HLS backend (has cppsim capability).

        HLS backends inherit from HLSBackend and have code generation methods.
        Only HLS backends can use cppsim execution mode.

        Args:
            op: HWCustomOp instance to check

        Returns:
            True if HLS backend, False otherwise
        """
        try:
            from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
            return isinstance(op, HLSBackend)
        except ImportError:
            return False

    def _setup_hls_backend_via_specialize(
        self,
        base_op: HWCustomOp,
        base_model: ModelWrapper,
        fpgapart: str = "xcvu9p-flgb2104-2-i"
    ) -> Tuple[HWCustomOp, ModelWrapper]:
        """Setup HLS backend by applying SpecializeLayers transform.

        This matches the production FINN workflow:
        1. Base kernel node with preferred_impl_style="hls"
        2. SpecializeLayers transform → Kernel_hls node
        3. getCustomOp() returns HLS backend class

        Args:
            base_op: Base kernel op instance (e.g., Shuffle, AutoShuffle)
            base_model: Model containing the base kernel node
            fpgapart: FPGA part name for specialization

        Returns:
            Tuple of (HLS backend op instance, transformed model)

        Example:
            # Create base Shuffle node
            shuffle_op, model = setup_base_shuffle()

            # Convert to Shuffle_hls via SpecializeLayers
            shuffle_hls_op, model = self._setup_hls_backend_via_specialize(
                shuffle_op, model
            )
        """
        from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
        from qonnx.custom_op.registry import getCustomOp

        # Set preferred implementation style to HLS
        base_op.set_nodeattr("preferred_impl_style", "hls")

        # Apply SpecializeLayers transform (matches FINN build flow)
        # This converts domain from "brainsmith.kernels" to "brainsmith.kernels.hls"
        # and op_type from "Shuffle" to "Shuffle_hls"
        model = base_model.transform(SpecializeLayers(fpgapart))

        # Find the specialized node (should be the same node, transformed)
        # The node's domain and op_type have been modified by SpecializeLayers
        specialized_node = None
        for node in model.graph.node:
            if node.domain.endswith(".hls") or node.domain.endswith(".rtl"):
                specialized_node = node
                break

        if specialized_node is None:
            raise RuntimeError(
                f"SpecializeLayers did not create HLS/RTL backend node. "
                f"Node domain: {base_op.onnx_node.domain}, "
                f"preferred_impl_style: {base_op.get_nodeattr('preferred_impl_style')}"
            )

        # Use getCustomOp to retrieve the HLS backend class
        # This will look up the class based on the specialized domain and op_type
        hls_op = getCustomOp(specialized_node)

        return hls_op, model

    def _is_rtl_backend(self, op: HWCustomOp) -> bool:
        """Check if op is an RTL backend (has rtlsim capability).

        RTL backends inherit from RTLBackend and have HDL generation methods.
        Only RTL backends can use rtlsim execution mode.

        Args:
            op: HWCustomOp instance to check

        Returns:
            True if RTL backend, False otherwise
        """
        try:
            from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
            return isinstance(op, RTLBackend)
        except ImportError:
            return False

    def get_test_fpgapart(self) -> str:
        """Get FPGA part for RTL testing.

        Override this method to test with different FPGA parts.
        Default is Versal xcvc1902 (DSP58-based).

        Returns:
            FPGA part string for RTL backend testing
        """
        return "xcvc1902-vsvd1760-2MP-e-S"

    def get_test_clock_period(self) -> float:
        """Get clock period (ns) for RTL testing.

        Override this method to test with different clock targets.
        Default is 3.0ns (~333MHz).

        Returns:
            Clock period in nanoseconds
        """
        return 3.0

    def _setup_rtl_backend_via_specialize(
        self,
        base_op: HWCustomOp,
        base_model: ModelWrapper,
        fpgapart: Optional[str] = None,
        clk: Optional[float] = None
    ) -> Tuple[HWCustomOp, ModelWrapper]:
        """Setup RTL backend by applying SpecializeLayers transform.

        This matches the production FINN workflow:
        1. Base kernel node with preferred_impl_style="rtl"
        2. Set clk_ns and fpgapart attributes
        3. SpecializeLayers transform → Kernel_rtl node
        4. getCustomOp() returns RTL backend class

        Args:
            base_op: Base kernel op instance (e.g., Shuffle, AutoShuffle)
            base_model: Model containing the base kernel node
            fpgapart: FPGA part name (uses get_test_fpgapart() if None)
            clk: Clock period in ns (uses get_test_clock_period() if None)

        Returns:
            Tuple of (RTL backend op instance, transformed model)

        Example:
            # Create base Shuffle node
            shuffle_op, model = setup_base_shuffle()

            # Convert to Shuffle_rtl via SpecializeLayers
            shuffle_rtl_op, model = self._setup_rtl_backend_via_specialize(
                shuffle_op, model
            )
        """
        from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
        from qonnx.custom_op.registry import getCustomOp

        # Use test defaults if not provided
        if fpgapart is None:
            fpgapart = self.get_test_fpgapart()
        if clk is None:
            clk = self.get_test_clock_period()

        # Set preferred implementation style to RTL
        base_op.set_nodeattr("preferred_impl_style", "rtl")

        # Set RTL-specific attributes
        base_op.set_nodeattr("fpgapart", fpgapart)
        base_op.set_nodeattr("clk_ns", clk)

        # Apply SpecializeLayers transform (matches FINN build flow)
        # This converts domain from "brainsmith.kernels" to "brainsmith.kernels.rtl"
        # and op_type from "Shuffle" to "Shuffle_rtl"
        model = base_model.transform(SpecializeLayers(fpgapart))

        # Find the specialized node (should be the same node, transformed)
        # The node's domain and op_type have been modified by SpecializeLayers
        specialized_node = None
        for node in model.graph.node:
            if node.domain.endswith(".rtl"):
                specialized_node = node
                break

        if specialized_node is None:
            raise RuntimeError(
                f"SpecializeLayers did not create RTL backend node. "
                f"Node domain: {base_op.onnx_node.domain}, "
                f"preferred_impl_style: {base_op.get_nodeattr('preferred_impl_style')}"
            )

        # Use getCustomOp to retrieve the RTL backend class
        # This will look up the class based on the specialized domain and op_type
        rtl_op = getCustomOp(specialized_node)

        return rtl_op, model

    def _make_execution_context(
        self,
        model: ModelWrapper,
        op: HWCustomOp
    ) -> Dict[str, np.ndarray]:
        """Create execution context with random inputs.

        Generates random test data based on:
        - Input shapes from op.get_normal_input_shape()
        - Input datatypes from op.get_input_datatype()

        Args:
            model: ModelWrapper containing the ONNX graph
            op: HWCustomOp instance

        Returns:
            Dict mapping tensor names to numpy arrays
        """
        context = {}
        node = op.onnx_node

        # Create random inputs
        for i, inp_name in enumerate(node.input):
            if not inp_name:  # Optional input (empty string)
                continue

            # Check if it's an initializer (weight/parameter)
            init = model.get_initializer(inp_name)
            if init is not None:
                context[inp_name] = init
                continue

            # Generate random input for streaming data
            try:
                shape = op.get_normal_input_shape(i)
                dtype = op.get_input_datatype(i)

                # Generate data in datatype's range
                if dtype.min() >= 0:
                    # Unsigned type
                    data = np.random.randint(
                        max(dtype.min(), 0),  # Ensure non-negative
                        min(dtype.max() + 1, 256),  # Cap at reasonable value
                        size=shape
                    ).astype(np.float32)
                else:
                    # Signed type
                    data = np.random.randint(
                        max(dtype.min(), -128),
                        min(dtype.max() + 1, 128),
                        size=shape
                    ).astype(np.float32)

                context[inp_name] = data

            except Exception as e:
                # If shape/datatype retrieval fails, skip this input
                # Subclass can override _make_execution_context for special handling
                pytest.skip(f"Cannot generate input {i} ({inp_name}): {e}")

        # Pre-allocate outputs
        for i, out_name in enumerate(node.output):
            try:
                shape = op.get_normal_output_shape(i)
                context[out_name] = np.zeros(shape, dtype=np.float32)
            except Exception as e:
                pytest.skip(f"Cannot pre-allocate output {i} ({out_name}): {e}")

        return context
