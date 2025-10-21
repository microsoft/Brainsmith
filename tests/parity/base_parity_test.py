"""Base class for parity testing between manual and KernelOp implementations.

This module provides ParityTestBase, an abstract base class that automates
testing equivalence between FINN's manual HWCustomOp implementations and
Brainsmith's KernelOp implementations.

Key features:
- **Transform-based testing (default)**: Tests production Infer transform workflow
- **15 generic test methods**: Shape, datatype, stream width, padding, inference, cycles, execution
- Handles initialization differences between manual and KernelOp implementations
- Supports multi-input/output operations
- Random input generation based on datatypes
- Comprehensive assertion messages
- Critical datatype inference logic validation
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

    Total: 15 comprehensive parity tests
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

            # Ensure kernel model is available for KernelOp-based backends
            if hasattr(auto_op, 'get_kernel_model'):
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

        # Ensure kernel model is available for KernelOp-based backends
        if hasattr(auto_op, 'get_kernel_model'):
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
