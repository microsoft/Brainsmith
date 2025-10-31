"""Dual pipeline parity testing - convenience wrapper for comprehensive testing.

This module provides DualPipelineParityTest, a convenience wrapper that combines:
1. Core parity testing (CoreParityTest) - shapes, widths, datatypes
2. HW estimation parity (HWEstimationParityTest) - resources, cycles
3. Golden reference validation (GoldenReferenceMixin) - execution correctness

Architecture:
-----------
DualPipelineParityTest inherits from:
- CoreParityTest (7 tests) - structural parity
- HWEstimationParityTest (5 tests) - HW estimation parity
- GoldenReferenceMixin - golden reference validation helpers

And adds:
- 4 golden execution tests (manual/auto × Python/cppsim)

Total: 16 tests automatically inherited + 4 golden execution tests = 20 tests

Design Philosophy:
-----------------
This is a CONVENIENCE WRAPPER for testing legacy FINN kernels vs modern Brainsmith kernels.
- Use this when: You want comprehensive testing (all 20 tests)
- Use focused classes when: You only need specific test categories

Focused alternatives:
- CoreParityTest - Just test shapes/widths/datatypes (7 tests)
- HWEstimationParityTest - Just test resources/cycles (5 tests)
- IntegratedPipelineTest - Just test single kernel vs golden (6 tests)

Subclass Pattern:
----------------
    class TestMyKernelDualParity(DualPipelineParityTest):
        def make_test_model(self):
            # Create standard ONNX node (e.g., Add, not AddStreams)
            return model, node_name

        def get_manual_transform(self):
            return InferMyKernelLayer  # FINN

        def get_auto_transform(self):
            return InferKernelList  # Brainsmith

        def compute_golden_reference(self, inputs):
            # Test-owned golden reference
            return {"output": inputs["input0"] + inputs["input1"]}

        def get_num_inputs(self):
            return 2

        def get_num_outputs(self):
            return 1

Inherited Tests (20 total):
---------------------------
From CoreParityTest (7):
- test_normal_shapes_parity()
- test_folded_shapes_parity()
- test_stream_widths_parity()
- test_stream_widths_padded_parity()
- test_datatypes_parity()
- test_datatype_inference_parity()
- test_make_shape_compatible_op_parity()

From HWEstimationParityTest (5):
- test_expected_cycles_parity()
- test_number_output_values_parity()
- test_resource_estimates_parity()
- test_efficiency_metrics_parity()
- test_operation_counts_parity()

From this class (4):
- test_manual_python_execution_vs_golden()
- test_auto_python_execution_vs_golden()
- test_manual_cppsim_execution_vs_golden()
- test_auto_cppsim_execution_vs_golden()

Plus 4 for rtlsim (future):
- test_manual_rtlsim_execution_vs_golden()
- test_auto_rtlsim_execution_vs_golden()
"""

import logging
import numpy as np
import pytest
from abc import ABC
from typing import Dict

from qonnx.core.modelwrapper import ModelWrapper
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp

# Import focused base classes
from tests.parity.core_parity_test import CoreParityTest
from tests.parity.hw_estimation_parity_test import HWEstimationParityTest
from tests.common.golden_reference_mixin import GoldenReferenceMixin

# Import execution utilities
from tests.parity.executors import CppSimExecutor
from tests.parity.test_fixtures import make_execution_context
from tests.parity.backend_helpers import setup_hls_backend_via_specialize

logger = logging.getLogger(__name__)


class DualPipelineParityTest(CoreParityTest, HWEstimationParityTest, GoldenReferenceMixin, ABC):
    """Convenience wrapper for comprehensive dual pipeline testing.

    Combines:
    - Core parity (shapes, widths, datatypes) - 7 tests
    - HW estimation parity (resources, cycles) - 5 tests
    - Golden execution validation - 4 tests

    Total: 16 tests automatically + 4 golden tests = 20 comprehensive tests

    Use this for:
    - Testing legacy FINN vs modern Brainsmith kernels
    - Comprehensive validation (all test categories)
    - Migration safety (proves manual → auto equivalence)

    Use focused classes instead when:
    - You only need specific test categories
    - Testing a single kernel (use IntegratedPipelineTest)
    - You want faster test execution (run subset)

    Inherits From:
    - CoreParityTest: Structural parity tests (7)
    - HWEstimationParityTest: HW estimation parity tests (5)
    - GoldenReferenceMixin: Golden reference validation helpers

    Subclasses must implement:
    - make_test_model() - Create ONNX node
    - get_manual_transform() - FINN transform
    - get_auto_transform() - Brainsmith transform
    - compute_golden_reference() - Test-owned golden reference
    - get_num_inputs() - Number of inputs
    - get_num_outputs() - Number of outputs
    - configure_kernel_node() - Optional node configuration
    """

    # ========================================================================
    # Helper: Map ONNX names to golden standard names
    # ========================================================================

    def _map_inputs_to_golden_names(
        self, inputs: Dict[str, np.ndarray], num_inputs: int
    ) -> Dict[str, np.ndarray]:
        """Map ONNX tensor names to golden reference standard names.

        ONNX models use arbitrary names like "inp1", "inp2", "outp".
        Golden references expect standard names like "input0", "input1", "output".

        Args:
            inputs: Dict with ONNX tensor names
            num_inputs: Number of input tensors

        Returns:
            Dict with golden standard names ("input0", "input1", ...)
        """
        input_list = list(inputs.items())[:num_inputs]
        golden_inputs = {}
        for i, (_, tensor_value) in enumerate(input_list):
            golden_inputs[f"input{i}"] = tensor_value
        return golden_inputs

    # ========================================================================
    # Golden Execution Tests (4 tests)
    # ========================================================================

    @pytest.mark.dual_pipeline
    @pytest.mark.golden
    def test_manual_python_execution_vs_golden(self):
        """Test manual (FINN) Python execution matches golden reference."""
        # Run manual pipeline
        manual_op, manual_model = self.run_manual_pipeline()

        # Create random inputs
        np.random.seed(42)
        inputs = make_execution_context(manual_model, manual_op)

        # Execute Python
        manual_op.execute_node(inputs, manual_model.graph)

        # Get outputs
        actual_outputs = {}
        for i, output_name in enumerate(manual_op.onnx_node.output):
            actual_outputs[output_name] = inputs[output_name]

        # Map ONNX names to golden standard names
        golden_inputs = self._map_inputs_to_golden_names(inputs, self.get_num_inputs())

        # Compute golden reference
        golden_outputs = self.compute_golden_reference(golden_inputs)

        # Validate
        self.validate_against_golden(
            actual_outputs,
            golden_outputs,
            "Manual Python execution",
            self.get_golden_tolerance_python(),
        )

    @pytest.mark.dual_pipeline
    @pytest.mark.golden
    def test_auto_python_execution_vs_golden(self):
        """Test auto (Brainsmith) Python execution matches golden reference."""
        # Run auto pipeline
        auto_op, auto_model = self.run_auto_pipeline()

        # Create random inputs
        np.random.seed(42)
        inputs = make_execution_context(auto_model, auto_op)

        # Execute Python
        auto_op.execute_node(inputs, auto_model.graph)

        # Get outputs
        actual_outputs = {}
        for i, output_name in enumerate(auto_op.onnx_node.output):
            actual_outputs[output_name] = inputs[output_name]

        # Map ONNX names to golden standard names
        golden_inputs = self._map_inputs_to_golden_names(inputs, self.get_num_inputs())

        # Compute golden reference
        golden_outputs = self.compute_golden_reference(golden_inputs)

        # Validate
        self.validate_against_golden(
            actual_outputs,
            golden_outputs,
            "Auto Python execution",
            self.get_golden_tolerance_python(),
        )

    @pytest.mark.dual_pipeline
    @pytest.mark.golden
    @pytest.mark.slow
    @pytest.mark.cppsim
    def test_manual_cppsim_execution_vs_golden(self):
        """Test manual (FINN) cppsim execution matches golden reference."""
        # Run manual pipeline
        manual_op, manual_model = self.run_manual_pipeline()

        # Specialize for HLS backend
        manual_model = setup_hls_backend_via_specialize(manual_model, manual_op)

        # Create random inputs
        np.random.seed(42)
        inputs = make_execution_context(manual_model, manual_op)

        # Execute cppsim
        executor = CppSimExecutor(manual_op, manual_model)
        actual_outputs = executor.execute(inputs, num_inputs=self.get_num_inputs())

        # Map ONNX names to golden standard names
        golden_inputs = self._map_inputs_to_golden_names(inputs, self.get_num_inputs())

        # Compute golden reference
        golden_outputs = self.compute_golden_reference(golden_inputs)

        # Validate
        self.validate_against_golden(
            actual_outputs,
            golden_outputs,
            "Manual HLS cppsim",
            self.get_golden_tolerance_cppsim(),
        )

    @pytest.mark.dual_pipeline
    @pytest.mark.golden
    @pytest.mark.slow
    @pytest.mark.cppsim
    def test_auto_cppsim_execution_vs_golden(self):
        """Test auto (Brainsmith) cppsim execution matches golden reference."""
        # Run auto pipeline
        auto_op, auto_model = self.run_auto_pipeline()

        # Specialize for HLS backend
        auto_model = setup_hls_backend_via_specialize(auto_model, auto_op)

        # Create random inputs
        np.random.seed(42)
        inputs = make_execution_context(auto_model, auto_op)

        # Execute cppsim
        executor = CppSimExecutor(auto_op, auto_model)
        actual_outputs = executor.execute(inputs, num_inputs=self.get_num_inputs())

        # Map ONNX names to golden standard names
        golden_inputs = self._map_inputs_to_golden_names(inputs, self.get_num_inputs())

        # Compute golden reference
        golden_outputs = self.compute_golden_reference(golden_inputs)

        # Validate
        self.validate_against_golden(
            actual_outputs,
            golden_outputs,
            "Auto HLS cppsim",
            self.get_golden_tolerance_cppsim(),
        )

    # ========================================================================
    # Future: RTL simulation tests
    # ========================================================================
    # @pytest.mark.dual_pipeline
    # @pytest.mark.golden
    # @pytest.mark.slow
    # @pytest.mark.rtlsim
    # def test_manual_rtlsim_execution_vs_golden(self):
    #     """Test manual (FINN) RTL simulation matches golden reference."""
    #     pass
    #
    # @pytest.mark.dual_pipeline
    # @pytest.mark.golden
    # @pytest.mark.slow
    # @pytest.mark.rtlsim
    # def test_auto_rtlsim_execution_vs_golden(self):
    #     """Test auto (Brainsmith) RTL simulation matches golden reference."""
    #     pass
