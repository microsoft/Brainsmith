"""Hardware estimation parity testing for resources and cycles.

This module provides HWEstimationParityTest, a focused framework for comparing
manual (FINN) vs auto (Brainsmith) implementations on hardware characteristics:
- Cycle counts
- Resource estimates (LUT, BRAM, URAM, DSP)
- Efficiency metrics
- Operation/parameter counts

Design Philosophy:
- Focus on hardware estimation, NOT execution correctness
- Compare manual vs auto implementations
- No golden reference needed (structural comparison only)
- Execution testing belongs in IntegratedPipelineTest

Usage:
    class TestMyKernelHWParity(HWEstimationParityTest):
        def make_test_model(self):
            # Create ONNX node
            return model, node_name

        def get_manual_transform(self):
            return InferMyKernelLayer  # FINN

        def get_auto_transform(self):
            return InferKernelList  # Brainsmith

        def get_num_inputs(self):
            return 2

        def get_num_outputs(self):
            return 1

Inherited Tests (5):
- test_expected_cycles_parity
- test_number_output_values_parity
- test_resource_estimates_parity
- test_efficiency_metrics_parity
- test_operation_counts_parity
"""

import pytest
from abc import ABC, abstractmethod
from typing import Tuple, Type

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp

# Import assertion helpers
from tests.parity.assertions import assert_values_match


class HWEstimationParityTest(ABC):
    """Base class for testing HW estimation parity between manual and auto kernels.

    Compares manual (FINN) vs auto (Brainsmith) implementations on:
    - Expected cycle counts
    - Output value counts (for FIFO sizing)
    - Resource estimates (LUT, BRAM, URAM, DSP)
    - Efficiency metrics (BRAM/URAM utilization)
    - Operation and parameter counts

    Does NOT test:
    - Execution correctness (use IntegratedPipelineTest)
    - Structural properties (use CoreParityTest)

    Total: 5 hardware estimation tests
    """

    # ========================================================================
    # Abstract Methods - Subclasses MUST implement
    # ========================================================================

    @abstractmethod
    def make_test_model(self) -> Tuple[ModelWrapper, str]:
        """Create test ONNX model.

        Returns:
            (model, node_name): Model with ONNX node and its name
        """
        pass

    @abstractmethod
    def get_manual_transform(self) -> Type[Transformation]:
        """Return FINN's manual transform for this kernel.

        Returns:
            Transform class (e.g., InferAddStreamsLayer)
        """
        pass

    @abstractmethod
    def get_auto_transform(self) -> Type[Transformation]:
        """Return Brainsmith's unified transform.

        Returns:
            Transform class (typically InferKernelList)
        """
        pass

    @abstractmethod
    def get_num_inputs(self) -> int:
        """Return number of inputs for this kernel.

        Returns:
            Number of inputs
        """
        pass

    @abstractmethod
    def get_num_outputs(self) -> int:
        """Return number of outputs for this kernel.

        Returns:
            Number of outputs
        """
        pass

    # ========================================================================
    # Optional Configuration Hooks
    # ========================================================================

    def configure_kernel_node(
        self, op: HWCustomOp, model: ModelWrapper, is_manual: bool
    ) -> None:
        """Configure kernel node before testing (optional).

        Override to set node attributes (PE, SIMD, etc.) for testing.

        IMPORTANT: If you change dimension parameters (PE, SIMD, etc.) on a KernelOp,
        you must call op._ensure_ready(model) afterwards to reconfigure the design point.

        Args:
            op: Kernel operator instance
            model: ModelWrapper containing the op
            is_manual: True if manual FINN implementation

        Example:
            def configure_kernel_node(self, op, model, is_manual):
                from brainsmith.dataflow.kernel_op import KernelOp

                op.set_nodeattr("PE", 8)

                # Reset design space after changing dimension parameters
                if isinstance(op, KernelOp):
                    op._ensure_ready(model)
        """
        pass

    # ========================================================================
    # Pipeline Execution Helpers
    # ========================================================================

    def run_manual_pipeline(self) -> Tuple[HWCustomOp, ModelWrapper]:
        """Run manual (FINN) pipeline: ONNX → HW node.

        Returns:
            (op, model): Manual operator and model
        """
        from qonnx.transformation.infer_shapes import InferShapes
        from qonnx.transformation.infer_datatypes import InferDataTypes
        from finn.util.basic import getHWCustomOp

        # Create model
        model, node_name = self.make_test_model()

        # Standard preprocessing
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        # Manual kernel inference
        transform_class = self.get_manual_transform()
        model = model.transform(transform_class())

        # Get hardware node
        hw_node = model.get_node_from_name(node_name)
        if hw_node is None:
            # Transform may rename node
            hw_node = model.graph.node[0]

        op = getHWCustomOp(hw_node, model)

        # Configure if needed
        self.configure_kernel_node(op, model, is_manual=True)

        return op, model

    def run_auto_pipeline(self) -> Tuple[HWCustomOp, ModelWrapper]:
        """Run auto (Brainsmith) pipeline: ONNX → HW node.

        Returns:
            (op, model): Auto operator and model
        """
        from qonnx.transformation.infer_shapes import InferShapes
        from qonnx.transformation.infer_datatypes import InferDataTypes
        from finn.util.basic import getHWCustomOp

        # Create model
        model, node_name = self.make_test_model()

        # Standard preprocessing
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        # Auto kernel inference
        transform_class = self.get_auto_transform()
        model = model.transform(transform_class())

        # Get hardware node
        hw_node = model.get_node_from_name(node_name)
        if hw_node is None:
            # Transform may rename node
            hw_node = model.graph.node[0]

        op = getHWCustomOp(hw_node, model)

        # Configure if needed
        self.configure_kernel_node(op, model, is_manual=False)

        return op, model

    # ========================================================================
    # Hardware Estimation Parity Tests (5 tests)
    # ========================================================================

    @pytest.mark.parity
    @pytest.mark.hw_estimation
    def test_expected_cycles_parity(self):
        """Test expected cycle counts match between implementations."""
        manual_op, _ = self.run_manual_pipeline()
        auto_op, _ = self.run_auto_pipeline()

        manual_cycles = manual_op.get_exp_cycles()
        auto_cycles = auto_op.get_exp_cycles()

        assert_values_match(manual_cycles, auto_cycles, "Expected cycles")

    @pytest.mark.parity
    @pytest.mark.hw_estimation
    def test_number_output_values_parity(self):
        """Test number of output values match (for FIFO sizing)."""
        manual_op, _ = self.run_manual_pipeline()
        auto_op, _ = self.run_auto_pipeline()

        manual_count = manual_op.get_number_output_values()
        auto_count = auto_op.get_number_output_values()

        assert_values_match(manual_count, auto_count, "Number of output values")

    @pytest.mark.parity
    @pytest.mark.hw_estimation
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
            fpgapart = "xczu3eg-sbva484-1-e"  # Default Zynq UltraScale+ part
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
    def test_operation_counts_parity(self):
        """Test operation and parameter counts match."""
        manual_op, _ = self.run_manual_pipeline()
        auto_op, _ = self.run_auto_pipeline()

        if hasattr(manual_op, "get_op_and_param_counts") and hasattr(auto_op, "get_op_and_param_counts"):
            manual_counts = manual_op.get_op_and_param_counts()
            auto_counts = auto_op.get_op_and_param_counts()

            assert_values_match(manual_counts, auto_counts, "Operation and parameter counts")
