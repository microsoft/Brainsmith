"""DualKernelTest_v2 framework for manual vs auto parity + both vs golden (v2.3).

This module provides DualKernelTest_v2, a fixture-based test framework that
tests parity between manual (FINN) and auto (Brainsmith) implementations,
plus validates both against golden reference.

Design Philosophy (v2.3):
- Inherits from KernelTestBase_v2: Zero duplication of shared utilities
- Direct DataType annotations: NO Quant nodes (v2.3 standard)
- Attribute-based parameterization: NOT fixture-based (unlike SingleKernelTest)
- Composition over inheritance: Uses Phase 1 utilities (PipelineRunner, Executors)
- Dual testing: Manual (FINN) vs auto (Brainsmith) parity + both vs golden

Key Improvements from v2.2:
- ✅ Inherits validate_against_golden() from KernelTestBase_v2 (zero duplication)
- ✅ Inherits _auto_detect_backends() from KernelTestBase_v2 (zero duplication)
- ✅ Inherits _specialize_to_backend_stage() from KernelTestBase_v2 (unified specialization)
- ✅ Direct DataType annotations (no Quant nodes)
- ✅ Automatic golden reference from QONNX execution
- Result: ~100 lines shorter than v2.2 DualKernelTest

Architecture (v2.3):
    KernelTestConfig (abstract interface)
        ↓
    KernelTestBase_v2 (shared utilities) ← Phase 2
        ↓
    DualKernelTest_v2 (dual pipeline testing) ← THIS CLASS
        - Inherits validate_against_golden()
        - Inherits _auto_detect_backends()
        - Inherits _specialize_to_backend_stage()

Pipeline Architecture (3 stages):
- Stage 1: ONNX Node (e.g., Add, Mul)
- Stage 2: Base Kernel (e.g., AddStreams) - Python execution
- Stage 3: Backend (e.g., AddStreams_hls) - cppsim/rtlsim execution

Usage:
    from tests.frameworks.dual_kernel_test_v2 import DualKernelTest_v2
    from qonnx.core.datatype import DataType

    # Define test with attribute-based parameters
    class TestAddStreamsParity(DualKernelTest_v2):
        batch = 1
        channels = 64
        input_dtype = DataType["INT8"]

        def make_test_model(self, input_shapes):
            # Use shapes from attributes
            shape = [self.batch, self.channels]
            inp0 = helper.make_tensor_value_info("input0", TensorProto.FLOAT, shape)
            # ... build graph
            return model, ["input0", "input1"]

        def get_manual_transform(self):
            return InferAddStreamsLayer  # FINN

        def get_auto_transform(self):
            return InferKernels  # Brainsmith

        def get_manual_backend_variants(self):
            from finn.custom_op.fpgadataflow.hls.addstreams_hls import AddStreams_hls
            return [AddStreams_hls]

    # Result: 18 tests automatically!
    # - 7 core parity tests (shapes, widths, datatypes)
    # - 5 HW estimation tests (cycles, resources)
    # - 6 golden execution tests (manual/auto × Python/cppsim/rtlsim)

Inherited Tests (18):
1. Core Parity (7 tests):
   - test_normal_shapes_parity - Input/output normal shapes match
   - test_folded_shapes_parity - Input/output folded shapes match
   - test_stream_widths_parity - Input/output stream widths match
   - test_stream_widths_padded_parity - AXI-aligned widths match
   - test_datatypes_parity - Input/output datatypes match
   - test_datatype_inference_parity - infer_node_datatype() results match
   - test_make_shape_compatible_op_parity - Shape-compatible ops match

2. HW Estimation (5 tests):
   - test_expected_cycles_parity - Cycle counts match
   - test_number_output_values_parity - Output value counts match
   - test_resource_estimates_parity - LUT/DSP/BRAM/URAM estimates match
   - test_efficiency_metrics_parity - BRAM/URAM efficiency match
   - test_operation_counts_parity - Op/param counts match

3. Golden Execution (6 tests):
   - test_manual_python_vs_golden - FINN Python execution matches golden
   - test_auto_python_vs_golden - Brainsmith Python execution matches golden
   - test_manual_cppsim_vs_golden - FINN cppsim matches golden (slow)
   - test_auto_cppsim_vs_golden - Brainsmith cppsim matches golden (slow)
   - test_manual_rtlsim_vs_golden - FINN rtlsim matches golden (slow)
   - test_auto_rtlsim_vs_golden - Brainsmith rtlsim matches golden (slow)
"""

import pytest
import numpy as np
from abc import abstractmethod
from typing import Dict, Type, Tuple, Optional, List

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp

# Import base utilities
from tests.frameworks.kernel_test_base_v2 import KernelTestBase_v2

# Import Phase 1 utilities
from tests.support.pipeline import PipelineRunner
from tests.support.executors import PythonExecutor, CppSimExecutor, RTLSimExecutor
from tests.support.context import make_execution_context_qonnx
from tests.support.assertions import (
    assert_shapes_match,
    assert_widths_match,
    assert_values_match,
    assert_datatypes_match,
    assert_model_tensors_match,
)


class DualKernelTest_v2(KernelTestBase_v2):
    """Test manual vs auto parity + both against golden reference (v2.3).

    Subclasses implement:
    - make_test_model(input_shapes): Create model with concrete shapes (from KernelTestBase_v2)
    - get_kernel_inference_transform(): Return transform class (from KernelTestBase_v2)
    - get_manual_transform(): Returns FINN transform class (NEW)
    - get_auto_transform(): Returns Brainsmith transform class (NEW)
    - get_manual_backend_variants(): Backend classes for FINN pipeline (NEW)

    Optional overrides:
    - configure_kernel_node(): Configure PE, SIMD, etc. (from KernelTestBase_v2)
    - configure_backend_node(): Configure backend params (from KernelTestBase_v2)
    - get_backend_fpgapart(): Enable backend testing (from KernelTestBase_v2)
    - get_auto_backend_variants(): Backend classes for Brainsmith pipeline (auto-detect default)
    - get_tolerance_*(): Validation tolerances (from KernelTestBase_v2)

    Inherited from KernelTestBase_v2:
    - validate_against_golden() - GoldenValidator-based output validation
    - _auto_detect_backends() - Registry-based backend lookup
    - _specialize_to_backend_stage() - Stage 2→3 specialization with overrides

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

        Note:
            Return the CLASS, not an instance (no parentheses).
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

        Note:
            Return the CLASS, not an instance (no parentheses).
        """
        pass

    @abstractmethod
    def get_manual_backend_variants(self) -> List[Type]:
        """Backend variants for manual (FINN) pipeline.

        REQUIRED: Must override for all DualKernelTest_v2 subclasses.

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

        Note:
            Return backend CLASSES (e.g., [AddStreams_hls]), not instances.
        """
        pass

    def get_auto_backend_variants(self) -> Optional[List[Type]]:
        """Backend variants for auto (Brainsmith) pipeline.

        Override to specify Brainsmith backends explicitly, or leave as None
        to auto-detect from the registry (recommended for Brainsmith).

        Returns:
            List of backend classes to try in priority order, or None.
            None = auto-detect from Brainsmith registry (default, works well)

        Default:
            None (auto-detect from registry via inherited _auto_detect_backends())

        Example (explicit Brainsmith backend):
            def get_auto_backend_variants(self):
                from brainsmith.kernels.channelwise import ChannelwiseOp_hls
                return [ChannelwiseOp_hls]

        Note:
            Most subclasses should NOT override this (None is recommended).
            Auto-detection works well for Brainsmith because backends are
            registered in the Brainsmith registry.
        """
        return None

    # ========================================================================
    # Interface Adaptation - v2 methods not used by DualKernelTest
    # ========================================================================

    def get_kernel_inference_transform(self) -> Type[Transformation]:
        """Not used by DualKernelTest - use get_manual/auto_transform() instead.

        DualKernelTest runs TWO transforms (manual and auto), so a single
        get_kernel_inference_transform() doesn't make sense.

        Raises:
            NotImplementedError: Always (dual tests use separate manual/auto transforms)
        """
        raise NotImplementedError(
            "DualKernelTest does not use get_kernel_inference_transform(). "
            "Use get_manual_transform() and get_auto_transform() instead."
        )

    # ========================================================================
    # Additional Abstract Methods for Dual Testing
    # ========================================================================

    @abstractmethod
    def get_num_inputs(self) -> int:
        """Return number of inputs for the operation.

        Returns:
            Number of inputs

        Example:
            def get_num_inputs(self):
                return 2  # AddStreams has 2 inputs
        """
        pass

    @abstractmethod
    def get_num_outputs(self) -> int:
        """Return number of outputs for the operation.

        Returns:
            Number of outputs

        Example:
            def get_num_outputs(self):
                return 1  # AddStreams has 1 output
        """
        pass

    @abstractmethod
    def compute_golden_reference(
        self, inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Compute golden reference outputs for validation.

        Args:
            inputs: Dict mapping input names to numpy arrays

        Returns:
            Dict mapping output names to expected numpy arrays

        Example:
            def compute_golden_reference(self, inputs):
                return {"output": inputs["input0"] + inputs["input1"]}

        Note:
            This should implement the expected behavior using NumPy operations.
        """
        pass

    # ========================================================================
    # Pipeline Execution (uses Phase 1 PipelineRunner + inherited specialization)
    # ========================================================================

    def run_manual_pipeline(
        self,
        kernel_test_config: 'KernelTestConfig',
        to_backend: bool = False
    ) -> Tuple[HWCustomOp, ModelWrapper]:
        """Run manual (FINN) pipeline: ONNX → Base Kernel → Backend (optional).

        Uses PipelineRunner (Phase 1) for pipeline logic and inherited
        _specialize_to_backend_stage() for backend specialization (Phase 2).

        Args:
            kernel_test_config: Unified test configuration (v3.0, required)
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
        # Stage 1 → Stage 2: ONNX → QONNX → FINN Kernel
        runner = PipelineRunner()

        # Create model factory that wraps make_test_model() with input shapes
        def model_factory():
            input_shapes = self._get_input_shapes()
            model, input_names = self.make_test_model(input_shapes)
            # PipelineRunner expects (model, node_name), but we have (model, input_names list)
            # The runner will find the HW node automatically, so we pass None
            return model, None

        def configure_stage_2(op, m):
            # Apply imperative configuration (backward compatible)
            self.configure_parameters(op, m, stage=2)
            # Apply declarative configuration from fixture (v2.5)
            self.auto_configure_from_fixture(op, m, stage=2, config=kernel_test_config)

        op, model = runner.run(
            model_factory=model_factory,
            transform=self.get_manual_transform(),
            configure_fn=configure_stage_2,
            qonnx_annotations=self._get_input_datatypes(),
        )

        # Stage 2 → Stage 3: Base Kernel → Backend (use inherited method)
        if to_backend:
            op, model = self._specialize_to_backend_stage(
                op, model, kernel_test_config,
                backend_variants_override=self.get_manual_backend_variants()
            )
            # Configure backend-specific parameters
            self.configure_parameters(op, model, stage=3)
            # Apply declarative configuration from fixture (v2.5)
            self.auto_configure_from_fixture(op, model, stage=3, config=kernel_test_config)

        return op, model

    def run_auto_pipeline(
        self,
        kernel_test_config: 'KernelTestConfig',
        to_backend: bool = False
    ) -> Tuple[HWCustomOp, ModelWrapper]:
        """Run auto (Brainsmith) pipeline: ONNX → Base Kernel → Backend (optional).

        Uses PipelineRunner (Phase 1) for pipeline logic and inherited
        _specialize_to_backend_stage() for backend specialization (Phase 2).

        Args:
            kernel_test_config: Unified test configuration (v3.0, required)
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
        # Stage 1 → Stage 2: ONNX → QONNX → Brainsmith Kernel
        runner = PipelineRunner()

        # Create model factory that wraps make_test_model() with input shapes
        def model_factory():
            input_shapes = self._get_input_shapes()
            model, input_names = self.make_test_model(input_shapes)
            # PipelineRunner expects (model, node_name), but we have (model, input_names list)
            # The runner will find the HW node automatically, so we pass None
            return model, None

        def configure_stage_2(op, m):
            # Apply imperative configuration (backward compatible)
            self.configure_parameters(op, m, stage=2)
            # Apply declarative configuration from fixture (v2.5)
            self.auto_configure_from_fixture(op, m, stage=2, config=kernel_test_config)

        op, model = runner.run(
            model_factory=model_factory,
            transform=self.get_auto_transform(),
            configure_fn=configure_stage_2,
            qonnx_annotations=self._get_input_datatypes(),
        )

        # Stage 2 → Stage 3: Base Kernel → Backend (use inherited method)
        if to_backend:
            op, model = self._specialize_to_backend_stage(op, model, kernel_test_config)
            # Configure backend-specific parameters
            self.configure_parameters(op, model, stage=3)
            # Apply declarative configuration from fixture (v2.5)
            self.auto_configure_from_fixture(op, model, stage=3, config=kernel_test_config)

        return op, model

    # ========================================================================
    # Helper Methods for Attribute-Based Configuration
    # ========================================================================

    def _get_input_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Extract input shapes from test class attributes.

        Subclasses define shapes as attributes (e.g., batch=1, channels=64).
        This method should be overridden if needed for complex shape logic.

        Returns:
            Dict mapping input names to shapes

        Default:
            Returns empty dict (subclasses override as needed)
        """
        return {}

    def _get_input_datatypes(self) -> Dict[str, 'DataType']:
        """Extract input datatypes from test class attributes.

        Subclasses define datatypes as attributes (e.g., input_dtype=DataType["INT8"]).
        This method should be overridden if needed for multiple inputs.

        Returns:
            Dict mapping input names to DataTypes

        Default:
            Returns empty dict (subclasses override as needed)
        """
        return {}

    # ========================================================================
    # Core Parity Tests (7 tests) - Copied from v1
    # ========================================================================

    @pytest.mark.parity
    @pytest.mark.core
    @pytest.mark.dual_kernel
    def test_normal_shapes_parity(self, kernel_test_config: "KernelTestConfig"):
        """Test normal input/output shapes match between implementations."""
        manual_op, _ = self.run_manual_pipeline(kernel_test_config)
        auto_op, _ = self.run_auto_pipeline(kernel_test_config)

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
    def test_folded_shapes_parity(self, kernel_test_config: "KernelTestConfig"):
        """Test folded input/output shapes match between implementations."""
        manual_op, _ = self.run_manual_pipeline(kernel_test_config)
        auto_op, _ = self.run_auto_pipeline(kernel_test_config)

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
    def test_stream_widths_parity(self, kernel_test_config: "KernelTestConfig"):
        """Test input/output stream widths match between implementations."""
        manual_op, _ = self.run_manual_pipeline(kernel_test_config)
        auto_op, _ = self.run_auto_pipeline(kernel_test_config)

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
    def test_stream_widths_padded_parity(self, kernel_test_config: "KernelTestConfig"):
        """Test padded stream widths match (AXI alignment)."""
        manual_op, _ = self.run_manual_pipeline(kernel_test_config)
        auto_op, _ = self.run_auto_pipeline(kernel_test_config)

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
    def test_datatypes_parity(self, kernel_test_config: "KernelTestConfig"):
        """Test input/output datatypes match between implementations."""
        manual_op, _ = self.run_manual_pipeline(kernel_test_config)
        auto_op, _ = self.run_auto_pipeline(kernel_test_config)

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
    def test_datatype_inference_parity(self, kernel_test_config: "KernelTestConfig"):
        """Test datatype inference produces matching results.

        Note: This test compares datatypes only, not tensor names.
        Some kernels (e.g., DuplicateStreams) may rename tensors during
        infer_node_datatype(), so we compare datatypes by position, not name.
        """
        # Setup ops with fresh models
        manual_op, manual_model = self.run_manual_pipeline(kernel_test_config)
        auto_op, auto_model = self.run_auto_pipeline(kernel_test_config)

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
    def test_make_shape_compatible_op_parity(self, kernel_test_config: "KernelTestConfig"):
        """Test shape-compatible ops preserve output structure.

        Note: make_shape_compatible_op() returns an ONNX NodeProto (per FINN API),
        not a wrapped HWCustomOp. This is used for shape inference.
        """
        manual_op, manual_model = self.run_manual_pipeline(kernel_test_config)
        auto_op, auto_model = self.run_auto_pipeline(kernel_test_config)

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
    # Hardware Estimation Parity Tests (5 tests) - Copied from v1
    # ========================================================================

    @pytest.mark.parity
    @pytest.mark.hw_estimation
    @pytest.mark.dual_kernel
    def test_expected_cycles_parity(self, kernel_test_config: "KernelTestConfig"):
        """Test expected cycle counts match between implementations."""
        manual_op, _ = self.run_manual_pipeline(kernel_test_config)
        auto_op, _ = self.run_auto_pipeline(kernel_test_config)

        manual_cycles = manual_op.get_exp_cycles()
        auto_cycles = auto_op.get_exp_cycles()

        assert_values_match(manual_cycles, auto_cycles, "Expected cycles")

    @pytest.mark.parity
    @pytest.mark.hw_estimation
    @pytest.mark.dual_kernel
    def test_number_output_values_parity(self, kernel_test_config: "KernelTestConfig"):
        """Test number of output values match (for FIFO sizing)."""
        manual_op, _ = self.run_manual_pipeline(kernel_test_config)
        auto_op, _ = self.run_auto_pipeline(kernel_test_config)

        manual_count = manual_op.get_number_output_values()
        auto_count = auto_op.get_number_output_values()

        assert_values_match(manual_count, auto_count, "Number of output values")

    @pytest.mark.parity
    @pytest.mark.hw_estimation
    @pytest.mark.dual_kernel
    def test_resource_estimates_parity(self, kernel_test_config: "KernelTestConfig"):
        """Test resource estimates match between implementations."""
        manual_op, _ = self.run_manual_pipeline(kernel_test_config)
        auto_op, _ = self.run_auto_pipeline(kernel_test_config)

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
    def test_efficiency_metrics_parity(self, kernel_test_config: "KernelTestConfig"):
        """Test BRAM/URAM efficiency estimates match."""
        manual_op, _ = self.run_manual_pipeline(kernel_test_config)
        auto_op, _ = self.run_auto_pipeline(kernel_test_config)

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
    def test_operation_counts_parity(self, kernel_test_config: "KernelTestConfig"):
        """Test operation and parameter counts match."""
        manual_op, _ = self.run_manual_pipeline(kernel_test_config)
        auto_op, _ = self.run_auto_pipeline(kernel_test_config)

        if hasattr(manual_op, "get_op_and_param_counts") and hasattr(auto_op, "get_op_and_param_counts"):
            manual_counts = manual_op.get_op_and_param_counts()
            auto_counts = auto_op.get_op_and_param_counts()

            assert_values_match(manual_counts, auto_counts, "Operation and parameter counts")

    # ========================================================================
    # Golden Execution Tests (6 tests) - Uses inherited validate_against_golden()
    # ========================================================================

    @pytest.mark.golden
    @pytest.mark.dual_kernel
    def test_manual_python_vs_golden(self, kernel_test_config: "KernelTestConfig"):
        """Test manual (FINN) Python execution matches golden reference."""
        manual_op, manual_model = self.run_manual_pipeline(kernel_test_config)

        # Generate test inputs
        np.random.seed(42)
        inputs = make_execution_context_qonnx(manual_model, manual_op)

        # Compute golden reference
        golden_outputs = self.compute_golden_reference(inputs)

        # Execute manual via Python
        executor = PythonExecutor()
        actual_outputs = executor.execute(manual_op, manual_model, inputs)

        # Validate (uses inherited method from KernelTestBase_v2)
        tolerance = kernel_test_config.get_tolerance_python()
        self.validate_against_golden(
            actual_outputs, golden_outputs, "Manual Python execution", tolerance
        )

    @pytest.mark.golden
    @pytest.mark.dual_kernel
    def test_auto_python_vs_golden(self, kernel_test_config: "KernelTestConfig"):
        """Test auto (Brainsmith) Python execution matches golden reference."""
        auto_op, auto_model = self.run_auto_pipeline(kernel_test_config)

        # Generate test inputs
        np.random.seed(42)
        inputs = make_execution_context_qonnx(auto_model, auto_op)

        # Compute golden reference
        golden_outputs = self.compute_golden_reference(inputs)

        # Execute auto via Python
        executor = PythonExecutor()
        actual_outputs = executor.execute(auto_op, auto_model, inputs)

        # Validate (uses inherited method from KernelTestBase_v2)
        tolerance = kernel_test_config.get_tolerance_python()
        self.validate_against_golden(
            actual_outputs, golden_outputs, "Auto Python execution", tolerance
        )

    @pytest.mark.golden
    @pytest.mark.cppsim
    @pytest.mark.slow
    @pytest.mark.dual_kernel
    def test_manual_cppsim_vs_golden(self, kernel_test_config: "KernelTestConfig"):
        """Test manual (FINN) cppsim execution matches golden reference."""
        manual_op, manual_model = self.run_manual_pipeline(kernel_test_config, to_backend=True)

        # Generate test inputs
        np.random.seed(42)
        inputs = make_execution_context_qonnx(manual_model, manual_op)

        # Compute golden reference
        golden_outputs = self.compute_golden_reference(inputs)

        # Execute manual via cppsim
        executor = CppSimExecutor()
        actual_outputs = executor.execute(manual_op, manual_model, inputs)

        # Validate (uses inherited method from KernelTestBase_v2)
        tolerance = kernel_test_config.get_tolerance_cppsim()
        self.validate_against_golden(
            actual_outputs, golden_outputs, "Manual HLS cppsim", tolerance
        )

    @pytest.mark.golden
    @pytest.mark.cppsim
    @pytest.mark.slow
    @pytest.mark.dual_kernel
    def test_auto_cppsim_vs_golden(self, kernel_test_config: "KernelTestConfig"):
        """Test auto (Brainsmith) cppsim execution matches golden reference."""
        auto_op, auto_model = self.run_auto_pipeline(kernel_test_config, to_backend=True)

        # Generate test inputs
        np.random.seed(42)
        inputs = make_execution_context_qonnx(auto_model, auto_op)

        # Compute golden reference
        golden_outputs = self.compute_golden_reference(inputs)

        # Execute auto via cppsim
        executor = CppSimExecutor()
        actual_outputs = executor.execute(auto_op, auto_model, inputs)

        # Validate (uses inherited method from KernelTestBase_v2)
        tolerance = kernel_test_config.get_tolerance_cppsim()
        self.validate_against_golden(
            actual_outputs, golden_outputs, "Auto HLS cppsim", tolerance
        )

    @pytest.mark.golden
    @pytest.mark.rtlsim
    @pytest.mark.slow
    @pytest.mark.dual_kernel
    def test_manual_rtlsim_vs_golden(self, kernel_test_config: "KernelTestConfig"):
        """Test manual (FINN) rtlsim execution matches golden reference."""
        manual_op, manual_model = self.run_manual_pipeline(kernel_test_config, to_backend=True)

        # Generate test inputs
        np.random.seed(42)
        inputs = make_execution_context_qonnx(manual_model, manual_op)

        # Compute golden reference
        golden_outputs = self.compute_golden_reference(inputs)

        # Execute manual via rtlsim
        executor = RTLSimExecutor()
        actual_outputs = executor.execute(manual_op, manual_model, inputs)

        # Validate (uses inherited method from KernelTestBase_v2)
        tolerance = kernel_test_config.get_tolerance_rtlsim()
        self.validate_against_golden(
            actual_outputs, golden_outputs, "Manual RTL rtlsim", tolerance
        )

    @pytest.mark.golden
    @pytest.mark.rtlsim
    @pytest.mark.slow
    @pytest.mark.dual_kernel
    def test_auto_rtlsim_vs_golden(self, kernel_test_config: "KernelTestConfig"):
        """Test auto (Brainsmith) rtlsim execution matches golden reference."""
        auto_op, auto_model = self.run_auto_pipeline(kernel_test_config, to_backend=True)

        # Generate test inputs
        np.random.seed(42)
        inputs = make_execution_context_qonnx(auto_model, auto_op)

        # Compute golden reference
        golden_outputs = self.compute_golden_reference(inputs)

        # Execute auto via rtlsim
        executor = RTLSimExecutor()
        actual_outputs = executor.execute(auto_op, auto_model, inputs)

        # Validate (uses inherited method from KernelTestBase_v2)
        tolerance = kernel_test_config.get_tolerance_rtlsim()
        self.validate_against_golden(
            actual_outputs, golden_outputs, "Auto RTL rtlsim", tolerance
        )
