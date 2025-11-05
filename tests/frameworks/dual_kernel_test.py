"""DualKernelTest framework for manual vs auto parity + both vs golden.

This module provides DualKernelTest, a composition-based test framework that
replaces CoreParityTest, HWEstimationParityTest, and DualPipelineParityTest
(1065 lines → ~400 lines, 62% reduction).

Design Philosophy:
- Composition over inheritance: uses Phase 1 utilities (PipelineRunner, GoldenValidator, Executors)
- No diamond inheritance: single clean inheritance chain
- Dual testing: manual (FINN) vs auto (Brainsmith) parity + both vs golden
- Single configure_kernel_node: unified signature, no is_manual parameter confusion
- Backend specialization: supports 3-stage pipeline (ONNX → Base Kernel → Backend)

Replaces:
- CoreParityTest (411 lines) - structural parity
- HWEstimationParityTest (333 lines) - HW estimation parity
- DualPipelineParityTest (321 lines) - execution parity
- Total: 1065 lines → ~400 lines

Pipeline Architecture (3 stages):
- Stage 1: ONNX Node (e.g., Add, Mul)
- Stage 2: Base Kernel (e.g., AddStreams) - tested with Python execution
- Stage 3: Backend (e.g., AddStreams_hls) - tested with cppsim/rtlsim execution

Usage:
    from tests.frameworks.dual_kernel_test import DualKernelTest
    from finn.transformation.fpgadataflow.convert_to_hw_layers import InferAddStreamsLayer
    from brainsmith.primitives.transforms.infer_kernels import InferKernels

    class TestAddStreamsParity(DualKernelTest):
        def make_test_model(self):
            return model, "Add_0"

        def get_manual_transform(self):
            return InferAddStreamsLayer  # FINN

        def get_auto_transform(self):
            return InferKernels  # Brainsmith

        def compute_golden_reference(self, inputs):
            return {"output": inputs["input0"] + inputs["input1"]}

        def get_num_inputs(self):
            return 2

        def get_num_outputs(self):
            return 1

        # Optional: Enable backend testing (Stage 3)
        def get_backend_fpgapart(self):
            return "xc7z020clg400-1"

Inherited Tests (18):
- 7 core parity tests (shapes, widths, datatypes at Stage 2)
- 5 HW estimation tests (cycles, resources at Stage 2)
- 6 golden execution tests:
  * 2 Python tests (Stage 2: manual/auto vs golden)
  * 2 cppsim tests (Stage 3: manual/auto vs golden)
  * 2 rtlsim tests (Stage 3: manual/auto vs golden)
"""

import pytest
import numpy as np
from abc import abstractmethod
from typing import Dict, Type, Tuple

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp

# Import Phase 1 utilities
from tests.support.pipeline import PipelineRunner
from tests.support.validator import GoldenValidator, TolerancePresets
from tests.support.executors import PythonExecutor, CppSimExecutor, RTLSimExecutor

# Import backend specialization utilities
from tests.support.backend_utils import specialize_to_backend

# Import test fixtures and assertions
from tests.support.context import make_execution_context_qonnx
from tests.support.assertions import (
    assert_shapes_match,
    assert_widths_match,
    assert_values_match,
    assert_datatypes_match,
    assert_model_tensors_match,
)

# Import base config
from tests.frameworks.kernel_test_base import KernelTestConfig


class DualKernelTest(KernelTestConfig):
    """Test manual vs auto parity + both against golden reference.

    Subclasses implement:
    - make_test_model(): Create ONNX model (from KernelTestConfig)
    - get_num_inputs/outputs(): I/O counts (from KernelTestConfig)
    - get_manual_transform(): Returns FINN transform class
    - get_auto_transform(): Returns Brainsmith transform class
    - compute_golden_reference(): Test-owned golden reference

    Optional backend configuration:
    - get_backend_fpgapart(): Enable backend testing (Stage 3)
    - get_backend_variants(): Backend classes to try in priority order

    Pipeline Architecture (3 stages):
    - Stage 1: ONNX Node (e.g., Add, Mul)
    - Stage 2: Base Kernel (e.g., AddStreams) - Python execution
    - Stage 3: Backend (e.g., AddStreams_hls) - cppsim/rtlsim execution

    Provides 18 inherited tests:
    - 7 core parity tests (structural comparison at Stage 2)
    - 5 HW estimation tests (cycles, resources at Stage 2)
    - 6 golden execution tests:
      * 2 Python tests (Stage 2: base kernel vs golden)
      * 2 cppsim tests (Stage 3: backend vs golden)
      * 2 rtlsim tests (Stage 3: backend vs golden)
    """

    # ========================================================================
    # Abstract Methods - Subclasses MUST implement (3 additional)
    # ========================================================================

    @abstractmethod
    def get_manual_transform(self) -> Type[Transformation]:
        """Return FINN's manual transform for this kernel.

        Returns:
            Transform class (e.g., InferAddStreamsLayer)

        Example:
            def get_manual_transform(self):
                from finn.transformation.fpgadataflow.convert_to_hw_layers import InferAddStreamsLayer
                return InferAddStreamsLayer
        """
        pass

    @abstractmethod
    def get_auto_transform(self) -> Type[Transformation]:
        """Return Brainsmith's unified transform.

        Returns:
            Transform class (typically InferKernels)

        Example:
            def get_auto_transform(self):
                from brainsmith.primitives.transforms.infer_kernels import InferKernels
                return InferKernels
        """
        pass

    @abstractmethod
    def get_manual_backend_variants(self):
        """Backend variants for manual (FINN) pipeline.

        REQUIRED: Must override for all DualKernelTest subclasses.

        Manual pipeline uses FINN transforms which create nodes with FINN
        attributes. These MUST use FINN backends, not Brainsmith backends.

        Auto-detection is NOT possible because op.onnx_node.op_type lacks
        source context, causing registry to return Brainsmith backends
        (due to source priority: [project, brainsmith, finn]).

        IMPORTANT: FINN and Brainsmith pipelines must remain independent.
        - Manual pipeline: FINN transform → FINN backend (uses "Func" attribute)
        - Auto pipeline: Brainsmith transform → Brainsmith backend (uses "func" attribute)

        Returns:
            List[Type]: Backend classes in priority order

        Example (FINN ChannelwiseOp):
            def get_manual_backend_variants(self):
                from finn.custom_op.fpgadataflow.hls.channelwise_op_hls import ChannelwiseOp_hls
                return [ChannelwiseOp_hls]

        Raises:
            NotImplementedError: If not overridden (abstract method)
        """
        pass

    def get_auto_backend_variants(self):
        """Backend variants for auto (Brainsmith) pipeline.

        Override to specify Brainsmith backends explicitly, or leave as None
        to auto-detect from the registry (recommended for Brainsmith).

        Returns:
            List of backend classes to try in priority order, or None.
            None = auto-detect from Brainsmith registry (default, works well)

        Default:
            None (auto-detect from registry, recommended for Brainsmith)

        Example (explicit Brainsmith backend):
            def get_auto_backend_variants(self):
                from brainsmith.kernels.channelwise import ChannelwiseOp_hls
                return [ChannelwiseOp_hls]
        """
        return None

    @abstractmethod
    def compute_golden_reference(
        self, inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Compute test-owned golden reference.

        Each test defines what "correct" means for its specific test case.
        This is TEST LOGIC, not production code!

        Args:
            inputs: Dict mapping input names → numpy arrays

        Returns:
            Dict mapping output names → expected numpy arrays

        Example:
            def compute_golden_reference(self, inputs):
                return {"output": inputs["input0"] + inputs["input1"]}
        """
        pass

    # ========================================================================
    # Pipeline Execution (uses Phase 1 PipelineRunner)
    # ========================================================================

    def run_manual_pipeline(self, to_backend: bool = False) -> Tuple[HWCustomOp, ModelWrapper]:
        """Run manual (FINN) pipeline: ONNX → Base Kernel → Backend (optional).

        Uses PipelineRunner (Phase 1) instead of duplicating pipeline logic.

        Args:
            to_backend: If True, specialize to backend (Stage 3).
                       If False, return base kernel (Stage 2).
                       Default: False

        Returns:
            (op, model): Manual operator and model
                        - Stage 2: Base kernel (e.g., AddStreams)
                        - Stage 3: Backend (e.g., AddStreams_hls)

        Raises:
            RuntimeError: If pipeline fails to create HW node
            pytest.skip: If to_backend=True but backend not configured

        Example:
            # Stage 2: Base kernel (Python execution)
            op, model = self.run_manual_pipeline(to_backend=False)
            executor = PythonExecutor()
            outputs = executor.execute(op, model, inputs)

            # Stage 3: Backend (cppsim execution)
            op, model = self.run_manual_pipeline(to_backend=True)
            executor = CppSimExecutor()
            outputs = executor.execute(op, model, inputs)
        """
        # Stage 1 → Stage 3: ONNX → QONNX → FINN Kernel
        runner = PipelineRunner()
        op, model = runner.run(
            model_factory=self.make_onnx_model,
            transform=self.get_manual_transform(),
            configure_fn=lambda op, model: self.configure_kernel_node(op, model),
            qonnx_annotations=self.get_qonnx_annotations(),
            qonnx_layouts=self.get_qonnx_layouts()
        )

        # Stage 2 → Stage 3: Base Kernel → Backend (optional)
        if to_backend:
            fpgapart = self.get_backend_fpgapart()
            if fpgapart is None:
                pytest.skip(
                    "Backend specialization not configured. "
                    "Override get_backend_fpgapart() to enable backend testing."
                )

            backend_variants = self.get_manual_backend_variants()
            if backend_variants is None:
                # Auto-detect from registry (works for Brainsmith, may fail for FINN)
                backend_variants = self._auto_detect_backends(op)

            op, model = specialize_to_backend(op, model, fpgapart, backend_variants)

        return op, model

    def run_auto_pipeline(self, to_backend: bool = False) -> Tuple[HWCustomOp, ModelWrapper]:
        """Run auto (Brainsmith) pipeline: ONNX → Base Kernel → Backend (optional).

        Uses PipelineRunner (Phase 1) instead of duplicating pipeline logic.

        Args:
            to_backend: If True, specialize to backend (Stage 3).
                       If False, return base kernel (Stage 2).
                       Default: False

        Returns:
            (op, model): Auto operator and model
                        - Stage 2: Base kernel (e.g., AddStreams)
                        - Stage 3: Backend (e.g., AddStreams_hls)

        Raises:
            RuntimeError: If pipeline fails to create HW node
            pytest.skip: If to_backend=True but backend not configured

        Example:
            # Stage 2: Base kernel (Python execution)
            op, model = self.run_auto_pipeline(to_backend=False)
            executor = PythonExecutor()
            outputs = executor.execute(op, model, inputs)

            # Stage 3: Backend (cppsim execution)
            op, model = self.run_auto_pipeline(to_backend=True)
            executor = CppSimExecutor()
            outputs = executor.execute(op, model, inputs)
        """
        # Stage 1 → Stage 3: ONNX → QONNX → Brainsmith Kernel
        runner = PipelineRunner()
        op, model = runner.run(
            model_factory=self.make_onnx_model,
            transform=self.get_auto_transform(),
            configure_fn=lambda op, model: self.configure_kernel_node(op, model),
            qonnx_annotations=self.get_qonnx_annotations(),
            qonnx_layouts=self.get_qonnx_layouts()
        )

        # Stage 2 → Stage 3: Base Kernel → Backend (optional)
        if to_backend:
            fpgapart = self.get_backend_fpgapart()
            if fpgapart is None:
                pytest.skip(
                    "Backend specialization not configured. "
                    "Override get_backend_fpgapart() to enable backend testing."
                )

            backend_variants = self.get_auto_backend_variants()
            if backend_variants is None:
                # Auto-detect from registry (works well for Brainsmith)
                backend_variants = self._auto_detect_backends(op)

            op, model = specialize_to_backend(op, model, fpgapart, backend_variants)

        return op, model

    def _auto_detect_backends(self, op):
        """Auto-detect backend variants from Brainsmith registry.

        Args:
            op: HWCustomOp instance to find backends for

        Returns:
            List of backend classes

        Raises:
            pytest.skip: If no backends found

        Note:
            This uses Brainsmith's registry. FINN backends are not registered,
            so this will only find Brainsmith backends.
        """
        from brainsmith.registry import list_backends_for_kernel, get_backend
        backend_names = list_backends_for_kernel(op.onnx_node.op_type, language='hls')
        if not backend_names:
            pytest.skip(f"No HLS backend found for {op.onnx_node.op_type}")
        return [get_backend(name) for name in backend_names]

    # ========================================================================
    # Validation (uses Phase 1 GoldenValidator)
    # ========================================================================

    def validate_against_golden(
        self,
        actual_outputs: Dict[str, np.ndarray],
        golden_outputs: Dict[str, np.ndarray],
        backend_name: str,
        tolerance: Dict[str, float],
    ) -> None:
        """Validate actual outputs match golden reference.

        Uses GoldenValidator (Phase 1) instead of duplicating validation logic.

        Args:
            actual_outputs: Outputs from backend execution
            golden_outputs: Expected outputs from golden reference
            backend_name: Name of backend for error messages
            tolerance: Dict with 'rtol' and 'atol' keys

        Raises:
            AssertionError: If outputs don't match within tolerance
        """
        validator = GoldenValidator()
        validator.validate(
            actual_outputs=actual_outputs,
            golden_outputs=golden_outputs,
            backend_name=backend_name,
            rtol=tolerance["rtol"],
            atol=tolerance["atol"],
        )

    # ========================================================================
    # Core Parity Tests (7 tests)
    # ========================================================================

    @pytest.mark.parity
    @pytest.mark.core
    @pytest.mark.dual_kernel
    def test_normal_shapes_parity(self):
        """Test normal input/output shapes match between implementations."""
        manual_op, _ = self.run_manual_pipeline()
        auto_op, _ = self.run_auto_pipeline()

        # Input shapes
        for i in range(self.get_num_inputs()):
            manual_shape = manual_op.get_normal_input_shape(i)
            auto_shape = auto_op.get_normal_input_shape(i)
            assert_shapes_match(manual_shape, auto_shape, i, "normal input")

        # Output shapes
        for i in range(self.get_num_outputs()):
            manual_shape = manual_op.get_normal_output_shape(i)
            auto_shape = auto_op.get_normal_output_shape(i)
            assert_shapes_match(manual_shape, auto_shape, i, "normal output")

    @pytest.mark.parity
    @pytest.mark.core
    @pytest.mark.dual_kernel
    def test_folded_shapes_parity(self):
        """Test folded input/output shapes match between implementations."""
        manual_op, _ = self.run_manual_pipeline()
        auto_op, _ = self.run_auto_pipeline()

        # Input shapes
        for i in range(self.get_num_inputs()):
            manual_shape = manual_op.get_folded_input_shape(i)
            auto_shape = auto_op.get_folded_input_shape(i)
            assert_shapes_match(manual_shape, auto_shape, i, "folded input")

        # Output shapes
        for i in range(self.get_num_outputs()):
            manual_shape = manual_op.get_folded_output_shape(i)
            auto_shape = auto_op.get_folded_output_shape(i)
            assert_shapes_match(manual_shape, auto_shape, i, "folded output")

    @pytest.mark.parity
    @pytest.mark.core
    @pytest.mark.dual_kernel
    def test_stream_widths_parity(self):
        """Test input/output stream widths match between implementations."""
        manual_op, _ = self.run_manual_pipeline()
        auto_op, _ = self.run_auto_pipeline()

        # Input stream widths
        for i in range(self.get_num_inputs()):
            manual_width = manual_op.get_instream_width(i)
            auto_width = auto_op.get_instream_width(i)
            assert_widths_match(manual_width, auto_width, i, "Input")

        # Output stream widths
        for i in range(self.get_num_outputs()):
            manual_width = manual_op.get_outstream_width(i)
            auto_width = auto_op.get_outstream_width(i)
            assert_widths_match(manual_width, auto_width, i, "Output")

    @pytest.mark.parity
    @pytest.mark.core
    @pytest.mark.dual_kernel
    def test_stream_widths_padded_parity(self):
        """Test padded stream widths match (AXI alignment)."""
        manual_op, _ = self.run_manual_pipeline()
        auto_op, _ = self.run_auto_pipeline()

        # Input stream widths padded
        for i in range(self.get_num_inputs()):
            manual_width = manual_op.get_instream_width_padded(i)
            auto_width = auto_op.get_instream_width_padded(i)

            def format_width(w):
                return f"{w} bits (padded)"
            assert_values_match(
                manual_width, auto_width, f"Input {i} stream width", format_width
            )

        # Output stream widths padded
        for i in range(self.get_num_outputs()):
            manual_width = manual_op.get_outstream_width_padded(i)
            auto_width = auto_op.get_outstream_width_padded(i)

            def format_width(w):
                return f"{w} bits (padded)"
            assert_values_match(
                manual_width, auto_width, f"Output {i} stream width", format_width
            )

    @pytest.mark.parity
    @pytest.mark.core
    @pytest.mark.dual_kernel
    def test_datatypes_parity(self):
        """Test input/output datatypes match between implementations."""
        manual_op, _ = self.run_manual_pipeline()
        auto_op, _ = self.run_auto_pipeline()

        # Input datatypes
        for i in range(self.get_num_inputs()):
            manual_dt = manual_op.get_input_datatype(i)
            auto_dt = auto_op.get_input_datatype(i)
            assert_datatypes_match(manual_dt, auto_dt, i, "Input")

        # Output datatypes
        for i in range(self.get_num_outputs()):
            manual_dt = manual_op.get_output_datatype(i)
            auto_dt = auto_op.get_output_datatype(i)
            assert_datatypes_match(manual_dt, auto_dt, i, "Output")

    @pytest.mark.parity
    @pytest.mark.core
    @pytest.mark.dual_kernel
    def test_datatype_inference_parity(self):
        """Test datatype inference produces matching results.

        Note: This test compares datatypes only, not tensor names.
        Some kernels (e.g., DuplicateStreams) may rename tensors during
        infer_node_datatype(), so we compare datatypes by position, not name.
        """
        # Setup ops with fresh models
        manual_op, manual_model = self.run_manual_pipeline()
        auto_op, auto_model = self.run_auto_pipeline()

        # Run datatype inference
        manual_model_out = manual_op.infer_node_datatype(manual_model)
        auto_model_out = auto_op.infer_node_datatype(auto_model)

        # Use returned model if provided
        if manual_model_out is not None:
            manual_model = manual_model_out
        if auto_model_out is not None:
            auto_model = auto_model_out

        # Verify input datatypes (compare by position, not name)
        for i in range(self.get_num_inputs()):
            manual_input_name = manual_op.onnx_node.input[i]
            auto_input_name = auto_op.onnx_node.input[i]

            if not manual_input_name or not auto_input_name:
                continue

            # Get datatypes from each model using its own tensor names
            manual_dt = manual_model.get_tensor_datatype(manual_input_name)
            auto_dt = auto_model.get_tensor_datatype(auto_input_name)

            # Compare datatypes (names may differ)
            assert_datatypes_match(
                manual_dt, auto_dt, i,
                f"After infer_node_datatype, input"
            )

        # Verify output datatypes (compare by position, not name)
        for i in range(self.get_num_outputs()):
            manual_output_name = manual_op.onnx_node.output[i]
            auto_output_name = auto_op.onnx_node.output[i]

            # Get datatypes from each model using its own tensor names
            manual_dt = manual_model.get_tensor_datatype(manual_output_name)
            auto_dt = auto_model.get_tensor_datatype(auto_output_name)

            # Compare datatypes (names may differ)
            assert_datatypes_match(
                manual_dt, auto_dt, i,
                f"After infer_node_datatype, output"
            )

    @pytest.mark.parity
    @pytest.mark.core
    @pytest.mark.dual_kernel
    def test_make_shape_compatible_op_parity(self):
        """Test shape-compatible ops preserve output structure.

        Note: make_shape_compatible_op() returns an ONNX NodeProto (per FINN API),
        not a wrapped HWCustomOp. This is used for shape inference.
        """
        manual_op, manual_model = self.run_manual_pipeline()
        auto_op, auto_model = self.run_auto_pipeline()

        # Returns ONNX NodeProto for shape inference
        manual_compat_node = manual_op.make_shape_compatible_op(manual_model)
        auto_compat_node = auto_op.make_shape_compatible_op(auto_model)

        # Verify output count matches (NodeProto.output is a list of output names)
        assert len(manual_compat_node.output) == len(auto_compat_node.output), (
            f"Shape-compatible op output count mismatch: "
            f"manual={len(manual_compat_node.output)}, "
            f"auto={len(auto_compat_node.output)}"
        )

        # Verify output names match (both should use same output names as original op)
        for i in range(self.get_num_outputs()):
            manual_output_name = manual_compat_node.output[i] if i < len(manual_compat_node.output) else None
            auto_output_name = auto_compat_node.output[i] if i < len(auto_compat_node.output) else None

            assert manual_output_name == manual_op.onnx_node.output[i], (
                f"Manual shape-compatible op output {i} name mismatch"
            )
            assert auto_output_name == auto_op.onnx_node.output[i], (
                f"Auto shape-compatible op output {i} name mismatch"
            )

    # ========================================================================
    # Hardware Estimation Parity Tests (5 tests)
    # ========================================================================

    @pytest.mark.parity
    @pytest.mark.hw_estimation
    @pytest.mark.dual_kernel
    def test_expected_cycles_parity(self):
        """Test expected cycle counts match between implementations."""
        manual_op, _ = self.run_manual_pipeline()
        auto_op, _ = self.run_auto_pipeline()

        manual_cycles = manual_op.get_exp_cycles()
        auto_cycles = auto_op.get_exp_cycles()

        assert_values_match(manual_cycles, auto_cycles, "Expected cycles")

    @pytest.mark.parity
    @pytest.mark.hw_estimation
    @pytest.mark.dual_kernel
    def test_number_output_values_parity(self):
        """Test number of output values match (for FIFO sizing)."""
        manual_op, _ = self.run_manual_pipeline()
        auto_op, _ = self.run_auto_pipeline()

        manual_count = manual_op.get_number_output_values()
        auto_count = auto_op.get_number_output_values()

        assert_values_match(manual_count, auto_count, "Number of output values")

    @pytest.mark.parity
    @pytest.mark.hw_estimation
    @pytest.mark.dual_kernel
    def test_resource_estimates_parity(self):
        """Test resource estimates match between implementations."""
        manual_op, _ = self.run_manual_pipeline()
        auto_op, _ = self.run_auto_pipeline()

        # LUT estimation
        if hasattr(manual_op, "lut_estimation") and hasattr(auto_op, "lut_estimation"):
            manual_luts = manual_op.lut_estimation()
            auto_luts = auto_op.lut_estimation()

            def format_lut(count):
                return f"{count:,} LUTs"
            assert_values_match(manual_luts, auto_luts, "LUT estimation", format_lut)

        # DSP estimation (requires fpgapart parameter per FINN API)
        if hasattr(manual_op, "dsp_estimation") and hasattr(auto_op, "dsp_estimation"):
            # Use default fpgapart for estimation comparison
            from tests.support.constants import PARITY_DEFAULT_FPGA_PART_HLS
            fpgapart = PARITY_DEFAULT_FPGA_PART_HLS
            manual_dsps = manual_op.dsp_estimation(fpgapart)
            auto_dsps = auto_op.dsp_estimation(fpgapart)

            def format_dsp(count):
                return f"{count:,} DSPs"
            assert_values_match(manual_dsps, auto_dsps, "DSP estimation", format_dsp)

        # BRAM estimation
        if hasattr(manual_op, "bram_estimation") and hasattr(auto_op, "bram_estimation"):
            manual_brams = manual_op.bram_estimation()
            auto_brams = auto_op.bram_estimation()

            def format_bram(count):
                return f"{count:,} BRAMs"
            assert_values_match(manual_brams, auto_brams, "BRAM estimation", format_bram)

        # URAM estimation
        if hasattr(manual_op, "uram_estimation") and hasattr(auto_op, "uram_estimation"):
            manual_urams = manual_op.uram_estimation()
            auto_urams = auto_op.uram_estimation()

            def format_uram(count):
                return f"{count:,} URAMs"
            assert_values_match(manual_urams, auto_urams, "URAM estimation", format_uram)

    @pytest.mark.parity
    @pytest.mark.hw_estimation
    @pytest.mark.dual_kernel
    def test_efficiency_metrics_parity(self):
        """Test BRAM/URAM efficiency estimates match."""
        manual_op, _ = self.run_manual_pipeline()
        auto_op, _ = self.run_auto_pipeline()

        # BRAM efficiency
        if hasattr(manual_op, "bram_efficiency_estimation") and hasattr(auto_op, "bram_efficiency_estimation"):
            manual_eff = manual_op.bram_efficiency_estimation()
            auto_eff = auto_op.bram_efficiency_estimation()

            def format_efficiency(eff):
                return f"{eff:.4f} ({eff*100:.2f}%)"
            assert_values_match(manual_eff, auto_eff, "BRAM efficiency", format_efficiency)

        # URAM efficiency
        if hasattr(manual_op, "uram_efficiency_estimation") and hasattr(auto_op, "uram_efficiency_estimation"):
            manual_eff = manual_op.uram_efficiency_estimation()
            auto_eff = auto_op.uram_efficiency_estimation()

            def format_efficiency(eff):
                return f"{eff:.4f} ({eff*100:.2f}%)"
            assert_values_match(manual_eff, auto_eff, "URAM efficiency", format_efficiency)

    @pytest.mark.parity
    @pytest.mark.hw_estimation
    @pytest.mark.dual_kernel
    def test_operation_counts_parity(self):
        """Test operation and parameter counts match."""
        manual_op, _ = self.run_manual_pipeline()
        auto_op, _ = self.run_auto_pipeline()

        if hasattr(manual_op, "get_op_and_param_counts") and hasattr(auto_op, "get_op_and_param_counts"):
            manual_counts = manual_op.get_op_and_param_counts()
            auto_counts = auto_op.get_op_and_param_counts()

            assert_values_match(manual_counts, auto_counts, "Operation and parameter counts")

    # ========================================================================
    # Golden Execution Tests (6 tests)
    # ========================================================================

    @pytest.mark.golden
    @pytest.mark.dual_kernel
    def test_manual_python_vs_golden(self):
        """Test manual (FINN) Python execution matches golden reference."""
        manual_op, manual_model = self.run_manual_pipeline()

        # Generate test inputs
        np.random.seed(42)
        inputs = make_execution_context_qonnx(manual_model, manual_op)

        # Compute golden reference
        golden_outputs = self.compute_golden_reference(inputs)

        # Execute manual via Python
        executor = PythonExecutor()
        actual_outputs = executor.execute(manual_op, manual_model, inputs)

        # Validate
        tolerance = self.get_tolerance_python()
        self.validate_against_golden(
            actual_outputs, golden_outputs, "Manual Python execution", tolerance
        )

    @pytest.mark.golden
    @pytest.mark.dual_kernel
    def test_auto_python_vs_golden(self):
        """Test auto (Brainsmith) Python execution matches golden reference."""
        auto_op, auto_model = self.run_auto_pipeline()

        # Generate test inputs
        np.random.seed(42)
        inputs = make_execution_context_qonnx(auto_model, auto_op)

        # Compute golden reference
        golden_outputs = self.compute_golden_reference(inputs)

        # Execute auto via Python
        executor = PythonExecutor()
        actual_outputs = executor.execute(auto_op, auto_model, inputs)

        # Validate
        tolerance = self.get_tolerance_python()
        self.validate_against_golden(
            actual_outputs, golden_outputs, "Auto Python execution", tolerance
        )

    @pytest.mark.golden
    @pytest.mark.cppsim
    @pytest.mark.slow
    @pytest.mark.dual_kernel
    def test_manual_cppsim_vs_golden(self):
        """Test manual (FINN) cppsim execution matches golden reference."""
        manual_op, manual_model = self.run_manual_pipeline(to_backend=True)

        # Generate test inputs
        np.random.seed(42)
        inputs = make_execution_context_qonnx(manual_model, manual_op)

        # Compute golden reference
        golden_outputs = self.compute_golden_reference(inputs)

        # Execute manual via cppsim
        executor = CppSimExecutor()
        actual_outputs = executor.execute(manual_op, manual_model, inputs)

        # Validate
        tolerance = self.get_tolerance_cppsim()
        self.validate_against_golden(
            actual_outputs, golden_outputs, "Manual HLS cppsim", tolerance
        )

    @pytest.mark.golden
    @pytest.mark.cppsim
    @pytest.mark.slow
    @pytest.mark.dual_kernel
    def test_auto_cppsim_vs_golden(self):
        """Test auto (Brainsmith) cppsim execution matches golden reference."""
        auto_op, auto_model = self.run_auto_pipeline(to_backend=True)

        # Generate test inputs
        np.random.seed(42)
        inputs = make_execution_context_qonnx(auto_model, auto_op)

        # Compute golden reference
        golden_outputs = self.compute_golden_reference(inputs)

        # Execute auto via cppsim
        executor = CppSimExecutor()
        actual_outputs = executor.execute(auto_op, auto_model, inputs)

        # Validate
        tolerance = self.get_tolerance_cppsim()
        self.validate_against_golden(
            actual_outputs, golden_outputs, "Auto HLS cppsim", tolerance
        )

    @pytest.mark.golden
    @pytest.mark.rtlsim
    @pytest.mark.slow
    @pytest.mark.dual_kernel
    def test_manual_rtlsim_vs_golden(self):
        """Test manual (FINN) rtlsim execution matches golden reference."""
        manual_op, manual_model = self.run_manual_pipeline(to_backend=True)

        # Generate test inputs
        np.random.seed(42)
        inputs = make_execution_context_qonnx(manual_model, manual_op)

        # Compute golden reference
        golden_outputs = self.compute_golden_reference(inputs)

        # Execute manual via rtlsim
        executor = RTLSimExecutor()
        actual_outputs = executor.execute(manual_op, manual_model, inputs)

        # Validate
        tolerance = self.get_tolerance_rtlsim()
        self.validate_against_golden(
            actual_outputs, golden_outputs, "Manual RTL rtlsim", tolerance
        )

    @pytest.mark.golden
    @pytest.mark.rtlsim
    @pytest.mark.slow
    @pytest.mark.dual_kernel
    def test_auto_rtlsim_vs_golden(self):
        """Test auto (Brainsmith) rtlsim execution matches golden reference."""
        auto_op, auto_model = self.run_auto_pipeline(to_backend=True)

        # Generate test inputs
        np.random.seed(42)
        inputs = make_execution_context_qonnx(auto_model, auto_op)

        # Compute golden reference
        golden_outputs = self.compute_golden_reference(inputs)

        # Execute auto via rtlsim
        executor = RTLSimExecutor()
        actual_outputs = executor.execute(auto_op, auto_model, inputs)

        # Validate
        tolerance = self.get_tolerance_rtlsim()
        self.validate_against_golden(
            actual_outputs, golden_outputs, "Auto RTL rtlsim", tolerance
        )

    # Removed redundant parity tests that violate the golden reference principle:
    # - test_manual_auto_parity_python
    # - test_manual_auto_parity_cppsim
    #
    # Rationale: These tests are mathematically redundant via transitive property:
    #   If: manual == golden (validated by test_manual_python_vs_golden, test_manual_cppsim_vs_golden)
    #   And: auto == golden (validated by test_auto_python_vs_golden, test_auto_cppsim_vs_golden)
    #   Then: manual == auto (by transitivity)
    #
    # The principle: Use independent golden reference tests to validate correctness.
    # Only use direct manual vs auto comparison for HW concerns with no ONNX/NumPy
    # equivalent (e.g., folding shapes, stream widths, resource estimates).
