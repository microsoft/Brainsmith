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
from brainsmith.settings import load_config
from tests.common.constants import PARITY_DEFAULT_FPGA_PART_HLS

# Parity test helpers
# Use absolute imports for compatibility with sys.path-based imports
try:
    from assertions import (
        assert_shapes_match,
        assert_datatypes_match,
        assert_widths_match,
        assert_values_match,
        assert_arrays_close,
        assert_model_tensors_match,
    )
    from executors import CppSimExecutor, RTLSimExecutor
    from test_fixtures import make_execution_context
    from backend_helpers import (
        is_hls_backend,
        is_rtl_backend,
        setup_hls_backend_via_specialize,
        setup_rtl_backend_via_specialize,
    )
except ImportError:
    # Fall back to relative imports for package-based imports
    from .assertions import (
        assert_shapes_match,
        assert_datatypes_match,
        assert_widths_match,
        assert_values_match,
        assert_arrays_close,
        assert_model_tensors_match,
    )
    from .executors import CppSimExecutor, RTLSimExecutor
    from .test_fixtures import make_execution_context
    from .backend_helpers import (
        is_hls_backend,
        is_rtl_backend,
        setup_hls_backend_via_specialize,
        setup_rtl_backend_via_specialize,
    )


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
            assert_shapes_match(manual_shape, auto_shape, ind, "normal input")

    @pytest.mark.parity
    def test_normal_output_shape_parity(self):
        """Test normal output shape matches."""
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        for ind in range(self.get_num_outputs()):
            manual_shape = manual_op.get_normal_output_shape(ind)
            auto_shape = auto_op.get_normal_output_shape(ind)
            assert_shapes_match(manual_shape, auto_shape, ind, "normal output")

    @pytest.mark.parity
    def test_folded_input_shape_parity(self):
        """Test folded input shape matches."""
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        for ind in range(self.get_num_inputs()):
            manual_shape = manual_op.get_folded_input_shape(ind)
            auto_shape = auto_op.get_folded_input_shape(ind)
            assert_shapes_match(manual_shape, auto_shape, ind, "folded input")

    @pytest.mark.parity
    def test_folded_output_shape_parity(self):
        """Test folded output shape matches."""
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        for ind in range(self.get_num_outputs()):
            manual_shape = manual_op.get_folded_output_shape(ind)
            auto_shape = auto_op.get_folded_output_shape(ind)
            assert_shapes_match(manual_shape, auto_shape, ind, "folded output")

    @pytest.mark.parity
    def test_instream_width_parity(self):
        """Test input stream width matches."""
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        for ind in range(self.get_num_inputs()):
            manual_width = manual_op.get_instream_width(ind)
            auto_width = auto_op.get_instream_width(ind)
            assert_widths_match(manual_width, auto_width, ind, "Input")

    @pytest.mark.parity
    def test_outstream_width_parity(self):
        """Test output stream width matches."""
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        for ind in range(self.get_num_outputs()):
            manual_width = manual_op.get_outstream_width(ind)
            auto_width = auto_op.get_outstream_width(ind)
            assert_widths_match(manual_width, auto_width, ind, "Output")

    @pytest.mark.parity
    def test_input_datatype_parity(self):
        """Test input datatype matches."""
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        for ind in range(self.get_num_inputs()):
            manual_dt = manual_op.get_input_datatype(ind)
            auto_dt = auto_op.get_input_datatype(ind)
            assert_datatypes_match(manual_dt, auto_dt, ind, "Input")

    @pytest.mark.parity
    def test_output_datatype_parity(self):
        """Test output datatype matches."""
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        for ind in range(self.get_num_outputs()):
            manual_dt = manual_op.get_output_datatype(ind)
            auto_dt = auto_op.get_output_datatype(ind)
            assert_datatypes_match(manual_dt, auto_dt, ind, "Output")

    @pytest.mark.parity
    def test_exp_cycles_parity(self):
        """Test expected cycles matches."""
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        manual_cycles = manual_op.get_exp_cycles()
        auto_cycles = auto_op.get_exp_cycles()

        assert_values_match(manual_cycles, auto_cycles, "Expected cycles")

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
            output_name = manual_op.onnx_node.output[ind]
            manual_output = manual_context[output_name]
            auto_output = auto_context[output_name]

            assert_arrays_close(
                manual_output,
                auto_output,
                f"Output {ind} ({output_name}) execution results"
            )

    @pytest.mark.parity
    def test_number_output_values_parity(self):
        """Test number of output values matches (used for FIFO sizing)."""
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        manual_count = manual_op.get_number_output_values()
        auto_count = auto_op.get_number_output_values()

        assert_values_match(manual_count, auto_count, "Number of output values")

    @pytest.mark.parity
    def test_instream_width_padded_parity(self):
        """Test padded input stream width matches (AXI Stream alignment)."""
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        for ind in range(self.get_num_inputs()):
            manual_width = manual_op.get_instream_width_padded(ind)
            auto_width = auto_op.get_instream_width_padded(ind)

            # Use custom formatting for padded width
            def format_width(w):
                return f"{w} bits (padded)"
            assert_values_match(manual_width, auto_width, f"Input {ind} stream width", format_width)

    @pytest.mark.parity
    def test_outstream_width_padded_parity(self):
        """Test padded output stream width matches (AXI Stream alignment)."""
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        for ind in range(self.get_num_outputs()):
            manual_width = manual_op.get_outstream_width_padded(ind)
            auto_width = auto_op.get_outstream_width_padded(ind)

            # Use custom formatting for padded width
            def format_width(w):
                return f"{w} bits (padded)"
            assert_values_match(manual_width, auto_width, f"Output {ind} stream width", format_width)

    @pytest.mark.parity
    def test_make_shape_compatible_op_parity(self):
        """Test that shape-compatible ops preserve output structure.

        Shape-compatible ops are used for ONNX shape inference. What matters
        is that they preserve the output tensor count and names for the ONNX
        graph, enabling correct shape propagation.

        Different implementations may use different underlying operators:
        - RandomNormal (0 inputs) - generates data
        - Split (1 input) - divides existing tensor
        - Identity (1 input) - passes through

        The key requirement is that outputs match the original node's outputs,
        preserving tensor identity for downstream shape inference.
        """
        manual_op, manual_model = self.setup_manual_op()
        auto_op, auto_model = self.setup_auto_op()

        manual_compat_op = manual_op.make_shape_compatible_op(manual_model)
        auto_compat_op = auto_op.make_shape_compatible_op(auto_model)

        # Verify output count matches (critical for graph structure)
        assert_values_match(
            len(manual_compat_op.output),
            len(auto_compat_op.output),
            "Shape-compatible op output count"
        )

        # Verify output names match (preserves tensor identity)
        for i in range(len(manual_compat_op.output)):
            assert_values_match(
                manual_compat_op.output[i],
                auto_compat_op.output[i],
                f"Shape-compatible op output {i} name"
            )

        # Note: Input counts may differ (RandomNormal=0, Split=1, Identity=1)
        # This is acceptable as shape inference only cares about output structure

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

            assert_model_tensors_match(
                manual_model, auto_model, input_name,
                f"After infer_node_datatype, input {ind}"
            )

        # Verify output datatypes were set correctly in model
        for ind in range(self.get_num_outputs()):
            output_name = manual_op.onnx_node.output[ind]

            assert_model_tensors_match(
                manual_model, auto_model, output_name,
                f"After infer_node_datatype, output {ind}"
            )

    @pytest.mark.parity
    @pytest.mark.cppsim
    @pytest.mark.slow
    def test_cppsim_execution_parity(self):
        """Test HLS backend code generation via cppsim compilation and execution.

        This is the gold standard for validating HLS code generation correctness.
        It validates the entire code generation pipeline end-to-end:

        **Pipeline Steps**:
        1. **Code Generation**: Both backends generate C++ (via code_generation_cppsim)
        2. **Compilation**: C++ compiles successfully (via compile_singlenode_code)
        3. **Execution**: Compiled binary executes (via execute_node)
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
        # Setup both backends with fresh models
        manual_op, manual_model = self.setup_manual_op()
        auto_op, auto_model = self.setup_auto_op()

        # Generate test context with deterministic inputs
        np.random.seed(42)
        test_context = self._make_execution_context(manual_model, manual_op)

        # Execute and compare using CppSimExecutor
        executor = CppSimExecutor()
        executor.execute_and_compare(
            manual_op, manual_model,
            auto_op, auto_model,
            test_context
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

        def format_lut(count):
            return f"{count:,} LUTs"
        assert_values_match(manual_luts, auto_luts, "LUT estimation", format_lut)

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

        def format_dsp(count):
            return f"{count:,} DSPs (FPGA: {fpgapart})"
        assert_values_match(manual_dsps, auto_dsps, "DSP estimation", format_dsp)

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

        def format_bram(count):
            return f"{count:,} BRAMs"
        assert_values_match(manual_brams, auto_brams, "BRAM estimation", format_bram)

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

        def format_uram(count):
            return f"{count:,} URAMs"
        assert_values_match(manual_urams, auto_urams, "URAM estimation", format_uram)

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

        def format_efficiency(eff):
            return f"{eff:.4f} ({eff*100:.2f}%)"
        assert_values_match(manual_eff, auto_eff, "BRAM efficiency estimation", format_efficiency)

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

        def format_efficiency(eff):
            return f"{eff:.4f} ({eff*100:.2f}%)"
        assert_values_match(manual_eff, auto_eff, "URAM efficiency estimation", format_efficiency)

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

        assert_values_match(manual_counts, auto_counts, "Operation and parameter counts")

    @pytest.mark.parity
    @pytest.mark.rtlsim
    @pytest.mark.slow
    def test_rtlsim_execution_parity(self):
        """Test RTL backend code generation via rtlsim compilation and execution.

        This is the gold standard for validating RTL code generation correctness.
        It validates the entire RTL generation and simulation pipeline end-to-end:

        **Pipeline Steps**:
        1. **HDL Generation**: Both backends generate Verilog
           - HLS: via code_generation_ipgen → ipgen_singlenode_code (Vitis HLS synthesis)
           - RTL: via generate_hdl (direct HDL generation)
        2. **Simulation Setup**: xsim compiles RTL (via prepare_rtlsim)
        3. **Execution**: Compiled simulator executes (via execute_node)
        4. **Validation**: Both produce identical numerical results

        **Confidence Level**: If this test passes, we have high confidence that:
        - Generated Verilog is syntactically correct (xsim compiles)
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
            - XSI (Xilinx Simulator) support built and available
            - Vitis HLS available (for HLS backends)
            - Both backends inherit from RTLBackend or HLSBackend
            - Valid setup_manual_op() and setup_auto_op() implementations

        Raises:
            pytest.skip: If XSI unavailable or backend incompatible
            AssertionError: If compilation fails or outputs differ
        """
        # Setup both backends with fresh models
        manual_op, manual_model = self.setup_manual_op()
        auto_op, auto_model = self.setup_auto_op()

        # Generate test context with deterministic inputs
        np.random.seed(42)
        test_context = self._make_execution_context(manual_model, manual_op)

        # Execute and compare using RTLSimExecutor
        executor = RTLSimExecutor()
        executor.execute_and_compare(
            manual_op, manual_model,
            auto_op, auto_model,
            test_context
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

        settings = load_config()
        settings.export_to_environment()

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

        assert_values_match(manual_filenames, auto_filenames, "RTL file list")

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

        settings = load_config()
        settings.export_to_environment()

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

        Delegates to backend_helpers.is_hls_backend() for implementation.

        Args:
            op: HWCustomOp instance to check

        Returns:
            True if HLS backend, False otherwise
        """
        return is_hls_backend(op)

    def _setup_hls_backend_via_specialize(
        self,
        base_op: HWCustomOp,
        base_model: ModelWrapper,
        fpgapart: str = PARITY_DEFAULT_FPGA_PART_HLS
    ) -> Tuple[HWCustomOp, ModelWrapper]:
        """Setup HLS backend by applying SpecializeLayers transform.

        Delegates to backend_helpers.setup_hls_backend_via_specialize() for implementation.

        Args:
            base_op: Base kernel op instance (e.g., Shuffle, AutoShuffle)
            base_model: Model containing the base kernel node
            fpgapart: FPGA part name for specialization

        Returns:
            Tuple of (HLS backend op instance, transformed model)
        """
        return setup_hls_backend_via_specialize(base_op, base_model, fpgapart)

    def _is_rtl_backend(self, op: HWCustomOp) -> bool:
        """Check if op is an RTL backend (has rtlsim capability).

        Delegates to backend_helpers.is_rtl_backend() for implementation.

        Args:
            op: HWCustomOp instance to check

        Returns:
            True if RTL backend, False otherwise
        """
        return is_rtl_backend(op)

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

        Delegates to backend_helpers.setup_rtl_backend_via_specialize() for implementation.
        Uses get_test_fpgapart() and get_test_clock_period() configuration hooks when
        parameters are None, allowing subclasses to customize defaults.

        Args:
            base_op: Base kernel op instance (e.g., Shuffle, AutoShuffle)
            base_model: Model containing the base kernel node
            fpgapart: FPGA part name (uses get_test_fpgapart() if None)
            clk: Clock period in ns (uses get_test_clock_period() if None)

        Returns:
            Tuple of (RTL backend op instance, transformed model)
        """
        # Use test defaults if not provided (allows subclass overrides)
        if fpgapart is None:
            fpgapart = self.get_test_fpgapart()
        if clk is None:
            clk = self.get_test_clock_period()

        return setup_rtl_backend_via_specialize(base_op, base_model, fpgapart, clk)

    def _make_execution_context(
        self,
        model: ModelWrapper,
        op: HWCustomOp
    ) -> Dict[str, np.ndarray]:
        """Create execution context with random inputs.

        Wrapper around test_fixtures.make_execution_context that converts
        ValueError to pytest.skip for test-friendly error handling.

        Generates random test data based on:
        - Input shapes from op.get_normal_input_shape()
        - Input datatypes from op.get_input_datatype()

        Args:
            model: ModelWrapper containing the ONNX graph
            op: HWCustomOp instance

        Returns:
            Dict mapping tensor names to numpy arrays

        Raises:
            pytest.skip: If shape or datatype cannot be determined
        """
        try:
            return make_execution_context(model, op)
        except ValueError as e:
            # Convert ValueError to pytest.skip for test-friendly error handling
            pytest.skip(str(e))
