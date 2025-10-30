"""Base class for dual pipeline parity testing.

This module provides DualPipelineParityTest, a unified framework that combines:
1. Golden reference validation (absolute correctness)
2. Hardware parity validation (migration safety)

Philosophy:
-----------
Run the FULL IntegratedPipelineTest workflow for BOTH manual and auto implementations:
- Each validates against NumPy golden reference (proves correctness)
- Hardware specs compared between implementations (proves equivalence)

This gives you:
- Absolute correctness: Both implementations match ground truth
- Migration safety: Manual → Auto transition preserves behavior
- Comprehensive coverage: ~20 tests automatically

Architecture:
-------------
                Manual Pipeline              Auto Pipeline
                      │                           │
                      ├─ ONNX Transform          ├─ ONNX Transform
                      ├─ Shape Inference         ├─ Shape Inference
                      ├─ Datatype Inference      ├─ Datatype Inference
                      ├─ HLS Specialization      ├─ HLS Specialization
                      │                           │
                      v                           v
                ✓ vs Golden                 ✓ vs Golden
                      │                           │
                      └──── Compare Hardware ─────┘
                        (widths, shapes, cycles)

Subclass Pattern:
-----------------
    class TestMyKernelDualParity(DualPipelineParityTest):
        def make_test_model(self):
            # Create standard ONNX node (e.g., Add, not AddStreams)
            return model, node_name

        def get_manual_transform(self):
            return InferAddStreamsLayer  # FINN transform

        def get_auto_transform(self):
            return InferKernelList  # Brainsmith transform

        def get_kernel_class(self):
            return AddStreams  # For golden reference

Inherited Tests (~20 automatic):
--------------------------------
Golden Reference Tests (4):
- test_manual_python_execution_vs_golden()
- test_auto_python_execution_vs_golden()
- test_manual_cppsim_execution_vs_golden()
- test_auto_cppsim_execution_vs_golden()

Hardware Parity Tests (12):
- test_normal_shapes_parity()
- test_folded_shapes_parity()
- test_stream_widths_parity()
- test_stream_widths_padded_parity()
- test_datatypes_parity()
- test_datatype_inference_parity()
- test_expected_cycles_parity()
- test_number_output_values_parity()
- test_resource_estimates_parity()
- test_efficiency_metrics_parity()
- test_operation_counts_parity()
- test_make_shape_compatible_op_parity()

Integration Validation Tests (4):
- test_both_pipelines_create_hw_nodes()
- test_both_hw_nodes_have_same_type()
- test_both_specializations_succeed()
- test_golden_reference_properties()
"""

import logging
import numpy as np
import pytest
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, Type

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.custom_op.registry import getCustomOp
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from brainsmith.dataflow.kernel_op import KernelOp

# Reuse parity infrastructure
try:
    from tests.parity.assertions import (
        assert_shapes_match,
        assert_datatypes_match,
        assert_widths_match,
        assert_values_match,
        assert_arrays_close,
        assert_model_tensors_match,
    )
    from tests.parity.executors import CppSimExecutor, RTLSimExecutor
    from tests.parity.test_fixtures import make_execution_context
    from tests.parity.backend_helpers import setup_hls_backend_via_specialize
except ImportError:
    # Fallback for package-based imports
    from ..parity.assertions import (
        assert_shapes_match,
        assert_datatypes_match,
        assert_widths_match,
        assert_values_match,
        assert_arrays_close,
        assert_model_tensors_match,
    )
    from ..parity.executors import CppSimExecutor, RTLSimExecutor
    from ..parity.test_fixtures import make_execution_context
    from ..parity.backend_helpers import setup_hls_backend_via_specialize

logger = logging.getLogger(__name__)


class DualPipelineParityTest(ABC):
    """Base class for dual pipeline parity testing.

    Runs complete pipeline for BOTH manual and auto implementations,
    validating each against golden reference and comparing hardware specs.

    This is the UNIFIED testing framework that combines:
    - IntegratedPipelineTest (golden reference validation)
    - ParityTestBase (hardware equivalence validation)

    Best of both worlds for a small team.
    """

    # =========================================================================
    # Abstract Methods - Subclasses MUST implement
    # =========================================================================

    @abstractmethod
    def make_test_model(self) -> Tuple[ModelWrapper, str]:
        """Create standard ONNX model for testing.

        Create a standard ONNX node (NOT a hardware node). Both pipelines
        will transform it to hardware nodes via their respective transforms.

        Returns:
            (model, node_name): ModelWrapper and name of ONNX node

        Example:
            def make_test_model(self):
                # Create ONNX Add node (not AddStreams)
                model = create_add_model(shape=[1, 64])
                return model, "Add_0"
        """
        pass

    @abstractmethod
    def get_manual_transform(self) -> Type[Transformation]:
        """Return FINN's manual transform class.

        Returns:
            Transformation class (e.g., InferAddStreamsLayer)

        Example:
            def get_manual_transform(self):
                from finn.transformation.fpgadataflow.convert_to_hw_layers import InferAddStreamsLayer
                return InferAddStreamsLayer
        """
        pass

    @abstractmethod
    def get_auto_transform(self) -> Type[Transformation]:
        """Return Brainsmith's auto transform class.

        Returns:
            Transformation class (e.g., InferKernelList)

        Example:
            def get_auto_transform(self):
                from brainsmith.primitives.transforms.infer_kernel_list import InferKernelList
                return InferKernelList
        """
        pass

    @abstractmethod
    def get_kernel_class(self) -> Type[KernelOp]:
        """Return kernel class for golden reference.

        Returns:
            KernelOp class with compute_golden_reference() static method

        Example:
            def get_kernel_class(self):
                from brainsmith.kernels.addstreams import AddStreams
                return AddStreams
        """
        pass

    # =========================================================================
    # Optional Configuration Hooks
    # =========================================================================

    def get_num_inputs(self) -> int:
        """Number of inputs to test. Override if > 1."""
        return 1

    def get_num_outputs(self) -> int:
        """Number of outputs to test. Override if > 1."""
        return 1

    def configure_kernel_node(
        self, op: HWCustomOp, model: ModelWrapper, is_manual: bool
    ) -> None:
        """Configure kernel node after inference.

        Called after transform creates HW node, before specialization.
        Configure BOTH implementations identically for fair comparison.

        Args:
            op: Hardware kernel op instance
            model: ModelWrapper containing the op
            is_manual: True if manual implementation, False if auto

        Example:
            def configure_kernel_node(self, op, model, is_manual):
                op.set_nodeattr("PE", 8)  # Same for both
        """
        pass

    def get_golden_tolerance_python(self) -> Dict[str, float]:
        """Tolerance for Python execution vs golden reference."""
        return {"rtol": 1e-7, "atol": 1e-9}

    def get_golden_tolerance_cppsim(self) -> Dict[str, float]:
        """Tolerance for C++ simulation vs golden reference."""
        return {"rtol": 1e-5, "atol": 1e-6}

    # =========================================================================
    # Pipeline Execution - Core Infrastructure
    # =========================================================================

    def run_manual_pipeline(self) -> Tuple[HWCustomOp, ModelWrapper]:
        """Run complete inference pipeline for manual implementation.

        Pipeline stages:
        1. Create standard ONNX model
        2. InferShapes
        3. InferDataTypes
        4. Manual transform (creates manual HW node)
        5. Configure kernel node

        Returns:
            (op, model): Manual hardware op and model
        """
        return self._run_inference_pipeline(
            transform_class=self.get_manual_transform(),
            is_manual=True
        )

    def run_auto_pipeline(self) -> Tuple[HWCustomOp, ModelWrapper]:
        """Run complete inference pipeline for auto implementation.

        Pipeline stages:
        1. Create standard ONNX model
        2. InferShapes
        3. InferDataTypes
        4. Auto transform (creates auto HW node)
        5. Configure kernel node

        Returns:
            (op, model): Auto hardware op and model
        """
        return self._run_inference_pipeline(
            transform_class=self.get_auto_transform(),
            is_manual=False
        )

    def _run_inference_pipeline(
        self, transform_class: Type[Transformation], is_manual: bool
    ) -> Tuple[HWCustomOp, ModelWrapper]:
        """Generic pipeline runner.

        Args:
            transform_class: Transform to apply (manual or auto)
            is_manual: True if manual, False if auto

        Returns:
            (op, model): Hardware op and model
        """
        # Create base model
        model, expected_node_name = self.make_test_model()

        # Run standard ONNX inference
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        # Run kernel-specific inference
        transform = transform_class()
        model = model.transform(transform)

        # Find the hardware node
        hw_node = self._find_hw_node(model)
        if hw_node is None:
            available_nodes = [(n.name, n.op_type) for n in model.graph.node]
            impl_type = "Manual" if is_manual else "Auto"
            raise RuntimeError(
                f"{impl_type} transform failed to create hardware node.\n"
                f"Transform: {transform_class.__name__}\n"
                f"Available nodes: {available_nodes}"
            )

        # Get op instance
        op = getCustomOp(hw_node)

        # Initialize KernelOp design_point (if it's a KernelOp)
        if isinstance(op, KernelOp):
            op._ensure_ready(model)

        # Allow subclass configuration
        self.configure_kernel_node(op, model, is_manual)

        # Re-initialize after configuration (in case PE/SIMD changed)
        if isinstance(op, KernelOp):
            op._ensure_ready(model)

        return op, model

    def _find_hw_node(self, model: ModelWrapper) -> Optional[Any]:
        """Find hardware node in graph (after kernel inference)."""
        for node in model.graph.node:
            try:
                op = getCustomOp(node)
                if isinstance(op, HWCustomOp):
                    return node
            except:
                continue
        return None

    # =========================================================================
    # Specialization - HLS Backend Setup
    # =========================================================================

    def run_manual_hls_specialization(
        self, base_op: HWCustomOp, base_model: ModelWrapper
    ) -> Tuple[HWCustomOp, ModelWrapper]:
        """Specialize manual kernel to HLS backend."""
        hls_op, hls_model = setup_hls_backend_via_specialize(base_op, base_model)

        if isinstance(hls_op, KernelOp):
            hls_op._ensure_ready(hls_model)

        return hls_op, hls_model

    def run_auto_hls_specialization(
        self, base_op: HWCustomOp, base_model: ModelWrapper
    ) -> Tuple[HWCustomOp, ModelWrapper]:
        """Specialize auto kernel to HLS backend."""
        hls_op, hls_model = setup_hls_backend_via_specialize(base_op, base_model)

        if isinstance(hls_op, KernelOp):
            hls_op._ensure_ready(hls_model)

        return hls_op, hls_model

    # =========================================================================
    # Golden Reference Helpers
    # =========================================================================

    def compute_golden_reference(
        self, inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Compute golden reference using kernel's reference implementation.

        Maps ONNX tensor names to standard golden reference names:
        - input0, input1, ... (for inputs)
        - output, output1, ... (for outputs)
        """
        kernel_class = self.get_kernel_class()
        if not hasattr(kernel_class, "compute_golden_reference"):
            raise NotImplementedError(
                f"{kernel_class.__name__} does not implement compute_golden_reference()"
            )

        # Map ONNX tensor names to standard golden reference names
        # inputs dict uses actual ONNX tensor names, golden expects "input0", "input1", etc.
        input_list = list(inputs.items())
        golden_inputs = {}
        for i, (tensor_name, tensor_value) in enumerate(input_list):
            golden_inputs[f"input{i}"] = tensor_value

        # Compute golden reference with standard names
        golden_outputs = kernel_class.compute_golden_reference(golden_inputs)

        # Map golden output names back to ONNX tensor names
        # This is a simple 1:1 mapping for now
        return golden_outputs

    def validate_against_golden(
        self,
        actual_outputs: Dict[str, np.ndarray],
        golden_outputs: Dict[str, np.ndarray],
        impl_name: str,
        tolerance: Dict[str, float],
    ) -> None:
        """Validate outputs match golden reference.

        Maps golden output names ("output", "output1", ...) to actual ONNX tensor names.
        """
        # Golden outputs use standard names: "output", "output1", etc.
        # actual_outputs uses actual ONNX tensor names
        # Map by index
        actual_output_list = list(actual_outputs.items())
        golden_output_list = list(golden_outputs.items())

        if len(actual_output_list) != len(golden_output_list):
            raise AssertionError(
                f"{impl_name} output count mismatch.\n"
                f"Expected: {len(golden_output_list)} outputs\n"
                f"Actual: {len(actual_output_list)} outputs"
            )

        for i, ((actual_name, actual_array), (golden_name, golden_array)) in enumerate(
            zip(actual_output_list, golden_output_list)
        ):
            assert_arrays_close(
                actual_array,
                golden_array,
                f"{impl_name} output {i} ({actual_name}) vs golden ({golden_name})",
                rtol=tolerance["rtol"],
                atol=tolerance["atol"],
            )

    # =========================================================================
    # Execution Helpers
    # =========================================================================

    def execute_python(
        self, op: HWCustomOp, model: ModelWrapper, inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Execute kernel using Python simulation."""
        context = dict(inputs)

        # Pre-allocate output arrays with correct shape and dtype
        for i in range(self.get_num_outputs()):
            output_name = op.onnx_node.output[i]
            output_shape = model.get_tensor_shape(output_name)
            output_dtype = model.get_tensor_datatype(output_name)
            context[output_name] = np.zeros(output_shape, dtype=output_dtype.to_numpy_dt())

        op.execute_node(context, model.graph)

        outputs = {}
        for i in range(self.get_num_outputs()):
            output_name = op.onnx_node.output[i]
            outputs[output_name] = context[output_name]

        return outputs

    def execute_cppsim(
        self, op: HWCustomOp, model: ModelWrapper, inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Execute kernel using HLS C++ simulation."""
        executor = CppSimExecutor()
        context = dict(inputs)
        outputs = executor._prepare_and_execute(op, model, context, is_manual=False)
        return outputs

    # =========================================================================
    # GOLDEN REFERENCE VALIDATION TESTS
    # =========================================================================

    @pytest.mark.dual_pipeline
    @pytest.mark.golden
    def test_manual_python_execution_vs_golden(self):
        """Test manual implementation Python execution matches golden reference.

        Validates that FINN's manual implementation produces correct results.
        """
        # Run manual pipeline
        op, model = self.run_manual_pipeline()

        # Generate test inputs
        np.random.seed(42)
        inputs = make_execution_context(model, op)

        # Compute golden reference
        golden_outputs = self.compute_golden_reference(inputs)

        # Execute manual implementation
        actual_outputs = self.execute_python(op, model, inputs)

        # Validate
        tolerance = self.get_golden_tolerance_python()
        self.validate_against_golden(
            actual_outputs, golden_outputs, "Manual Python", tolerance
        )

    @pytest.mark.dual_pipeline
    @pytest.mark.golden
    def test_auto_python_execution_vs_golden(self):
        """Test auto implementation Python execution matches golden reference.

        Validates that Brainsmith's auto implementation produces correct results.
        """
        # Run auto pipeline
        op, model = self.run_auto_pipeline()

        # Generate test inputs
        np.random.seed(42)
        inputs = make_execution_context(model, op)

        # Compute golden reference
        golden_outputs = self.compute_golden_reference(inputs)

        # Execute auto implementation
        actual_outputs = self.execute_python(op, model, inputs)

        # Validate
        tolerance = self.get_golden_tolerance_python()
        self.validate_against_golden(
            actual_outputs, golden_outputs, "Auto Python", tolerance
        )

    @pytest.mark.dual_pipeline
    @pytest.mark.golden
    @pytest.mark.cppsim
    @pytest.mark.slow
    def test_manual_cppsim_execution_vs_golden(self):
        """Test manual HLS C++ simulation matches golden reference.

        Validates complete code generation pipeline for manual implementation.
        """
        # Run manual pipeline
        base_op, base_model = self.run_manual_pipeline()

        # Specialize to HLS
        hls_op, hls_model = self.run_manual_hls_specialization(base_op, base_model)

        # Generate test inputs
        np.random.seed(42)
        inputs = make_execution_context(hls_model, hls_op)

        # Compute golden reference
        golden_outputs = self.compute_golden_reference(inputs)

        # Execute via cppsim
        actual_outputs = self.execute_cppsim(hls_op, hls_model, inputs)

        # Validate
        tolerance = self.get_golden_tolerance_cppsim()
        self.validate_against_golden(
            actual_outputs, golden_outputs, "Manual HLS", tolerance
        )

    @pytest.mark.dual_pipeline
    @pytest.mark.golden
    @pytest.mark.cppsim
    @pytest.mark.slow
    def test_auto_cppsim_execution_vs_golden(self):
        """Test auto HLS C++ simulation matches golden reference.

        Validates complete code generation pipeline for auto implementation.
        """
        # Run auto pipeline
        base_op, base_model = self.run_auto_pipeline()

        # Specialize to HLS
        hls_op, hls_model = self.run_auto_hls_specialization(base_op, base_model)

        # Generate test inputs
        np.random.seed(42)
        inputs = make_execution_context(hls_model, hls_op)

        # Compute golden reference
        golden_outputs = self.compute_golden_reference(inputs)

        # Execute via cppsim
        actual_outputs = self.execute_cppsim(hls_op, hls_model, inputs)

        # Validate
        tolerance = self.get_golden_tolerance_cppsim()
        self.validate_against_golden(
            actual_outputs, golden_outputs, "Auto HLS", tolerance
        )

    # =========================================================================
    # HARDWARE PARITY TESTS
    # =========================================================================

    @pytest.mark.dual_pipeline
    @pytest.mark.parity
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

    @pytest.mark.dual_pipeline
    @pytest.mark.parity
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

    @pytest.mark.dual_pipeline
    @pytest.mark.parity
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

    @pytest.mark.dual_pipeline
    @pytest.mark.parity
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

    @pytest.mark.dual_pipeline
    @pytest.mark.parity
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

    @pytest.mark.dual_pipeline
    @pytest.mark.parity
    def test_datatype_inference_parity(self):
        """Test datatype inference produces matching results in models."""
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

        # Verify input datatypes in model
        for i in range(self.get_num_inputs()):
            input_name = manual_op.onnx_node.input[i]
            if not input_name:
                continue

            assert_model_tensors_match(
                manual_model, auto_model, input_name,
                f"After infer_node_datatype, input {i}"
            )

        # Verify output datatypes in model
        for i in range(self.get_num_outputs()):
            output_name = manual_op.onnx_node.output[i]

            assert_model_tensors_match(
                manual_model, auto_model, output_name,
                f"After infer_node_datatype, output {i}"
            )

    @pytest.mark.dual_pipeline
    @pytest.mark.parity
    def test_expected_cycles_parity(self):
        """Test expected cycle counts match between implementations."""
        manual_op, _ = self.run_manual_pipeline()
        auto_op, _ = self.run_auto_pipeline()

        manual_cycles = manual_op.get_exp_cycles()
        auto_cycles = auto_op.get_exp_cycles()

        assert_values_match(manual_cycles, auto_cycles, "Expected cycles")

    @pytest.mark.dual_pipeline
    @pytest.mark.parity
    def test_number_output_values_parity(self):
        """Test number of output values match (for FIFO sizing)."""
        manual_op, _ = self.run_manual_pipeline()
        auto_op, _ = self.run_auto_pipeline()

        manual_count = manual_op.get_number_output_values()
        auto_count = auto_op.get_number_output_values()

        assert_values_match(manual_count, auto_count, "Number of output values")

    @pytest.mark.dual_pipeline
    @pytest.mark.parity
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

    @pytest.mark.dual_pipeline
    @pytest.mark.parity
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

    @pytest.mark.dual_pipeline
    @pytest.mark.parity
    def test_operation_counts_parity(self):
        """Test operation and parameter counts match."""
        manual_op, _ = self.run_manual_pipeline()
        auto_op, _ = self.run_auto_pipeline()

        if hasattr(manual_op, "get_op_and_param_counts") and hasattr(auto_op, "get_op_and_param_counts"):
            manual_counts = manual_op.get_op_and_param_counts()
            auto_counts = auto_op.get_op_and_param_counts()

            assert_values_match(manual_counts, auto_counts, "Operation and parameter counts")

    @pytest.mark.dual_pipeline
    @pytest.mark.parity
    def test_make_shape_compatible_op_parity(self):
        """Test shape-compatible ops preserve output structure."""
        manual_op, manual_model = self.run_manual_pipeline()
        auto_op, auto_model = self.run_auto_pipeline()

        manual_compat_op = manual_op.make_shape_compatible_op(manual_model)
        auto_compat_op = auto_op.make_shape_compatible_op(auto_model)

        # Verify output count matches
        assert_values_match(
            len(manual_compat_op.output),
            len(auto_compat_op.output),
            "Shape-compatible op output count"
        )

        # Verify output names match
        for i in range(len(manual_compat_op.output)):
            assert_values_match(
                manual_compat_op.output[i],
                auto_compat_op.output[i],
                f"Shape-compatible op output {i} name"
            )

    # =========================================================================
    # INTEGRATION VALIDATION TESTS
    # =========================================================================

    @pytest.mark.dual_pipeline
    @pytest.mark.integration
    def test_both_pipelines_create_hw_nodes(self):
        """Test that both pipelines successfully create hardware nodes."""
        manual_op, manual_model = self.run_manual_pipeline()
        auto_op, auto_model = self.run_auto_pipeline()

        # Both should be HWCustomOp instances
        assert isinstance(manual_op, HWCustomOp), \
            f"Manual pipeline created {type(manual_op)}, expected HWCustomOp"
        assert isinstance(auto_op, HWCustomOp), \
            f"Auto pipeline created {type(auto_op)}, expected HWCustomOp"

    @pytest.mark.dual_pipeline
    @pytest.mark.integration
    def test_both_hw_nodes_have_same_type(self):
        """Test that both create the same base op_type (before specialization)."""
        manual_op, _ = self.run_manual_pipeline()
        auto_op, _ = self.run_auto_pipeline()

        # Extract base op_type (e.g., "AddStreams" from "AddStreams_hls")
        manual_base = manual_op.onnx_node.op_type.replace("_hls", "").replace("_rtl", "")
        auto_base = auto_op.onnx_node.op_type.replace("_hls", "").replace("_rtl", "")

        assert manual_base == auto_base, (
            f"Base op types differ:\n"
            f"  Manual: {manual_base}\n"
            f"  Auto:   {auto_base}"
        )

    @pytest.mark.dual_pipeline
    @pytest.mark.integration
    @pytest.mark.slow
    def test_both_specializations_succeed(self):
        """Test that both implementations specialize to HLS successfully."""
        # Run base pipelines
        manual_base_op, manual_base_model = self.run_manual_pipeline()
        auto_base_op, auto_base_model = self.run_auto_pipeline()

        # Specialize both
        manual_hls_op, manual_hls_model = self.run_manual_hls_specialization(
            manual_base_op, manual_base_model
        )
        auto_hls_op, auto_hls_model = self.run_auto_hls_specialization(
            auto_base_op, auto_base_model
        )

        # Both should succeed (no exceptions)
        assert manual_hls_op is not None, "Manual HLS specialization failed"
        assert auto_hls_op is not None, "Auto HLS specialization failed"

    @pytest.mark.dual_pipeline
    @pytest.mark.golden
    def test_golden_reference_properties(self):
        """Test that golden reference satisfies mathematical properties.

        This validates the golden reference implementation itself.
        Subclasses can override to test kernel-specific properties.
        """
        # Create test inputs
        manual_op, manual_model = self.run_manual_pipeline()

        np.random.seed(42)
        inputs = make_execution_context(manual_model, manual_op)

        # Map inputs to golden reference names
        input_list = list(inputs.items())
        golden_inputs = {}
        for i, (tensor_name, tensor_value) in enumerate(input_list):
            golden_inputs[f"input{i}"] = tensor_value

        # Compute golden reference
        golden_outputs = self.compute_golden_reference(inputs)

        # Validate via kernel class if it has validation method
        kernel_class = self.get_kernel_class()
        if hasattr(kernel_class, "validate_golden_properties"):
            kernel_class.validate_golden_properties(golden_inputs, golden_outputs)
