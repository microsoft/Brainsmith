"""SingleKernelTest framework for testing one kernel vs golden reference.

This module provides SingleKernelTest, a composition-based test framework that
replaces IntegratedPipelineTest (722 lines → ~250 lines, 65% reduction).

Design Philosophy:
- Composition over inheritance: uses Phase 1 utilities (PipelineRunner, GoldenValidator, Executors)
- Single responsibility: tests ONE kernel implementation vs golden reference
- Test-owned golden: compute_golden_reference() is test logic, not production code
- Clear test names: "test_python_execution_vs_golden" is obvious

Replaces:
- IntegratedPipelineTest.run_inference_pipeline() → PipelineRunner.run()
- IntegratedPipelineTest.validate_against_golden() → GoldenValidator.validate()
- IntegratedPipelineTest.execute_*() → PythonExecutor/CppSimExecutor/RTLSimExecutor

Usage:
    from tests.frameworks.single_kernel_test import SingleKernelTest
    from brainsmith.primitives.transforms.infer_kernel_list import InferKernelList

    class TestMyKernel(SingleKernelTest):
        def make_test_model(self):
            # Create ONNX model
            return model, "Add_0"

        def get_kernel_inference_transform(self):
            return InferKernelList

        def compute_golden_reference(self, inputs):
            return {"output": inputs["input0"] + inputs["input1"]}

        def get_num_inputs(self):
            return 2

        def get_num_outputs(self):
            return 1

Inherited Tests (6):
- test_pipeline_creates_hw_node
- test_shapes_preserved_through_pipeline
- test_datatypes_preserved_through_pipeline
- test_python_execution_vs_golden
- test_cppsim_execution_vs_golden
- test_rtlsim_execution_vs_golden
"""

import pytest
import numpy as np
from abc import abstractmethod
from typing import Dict, Type, Tuple

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp

# Import Phase 1 utilities
from tests.common.pipeline import PipelineRunner
from tests.common.validator import GoldenValidator, TolerancePresets
from tests.common.executors import PythonExecutor, CppSimExecutor, RTLSimExecutor

# Import test fixtures
from tests.parity.test_fixtures import make_execution_context

# Import base config
from tests.frameworks.kernel_test_base import KernelTestConfig


class SingleKernelTest(KernelTestConfig):
    """Test one kernel implementation against golden reference.

    Subclasses implement:
    - make_test_model(): Create ONNX model (from KernelTestConfig)
    - get_num_inputs/outputs(): I/O counts (from KernelTestConfig)
    - get_kernel_inference_transform(): Returns transform class (e.g., InferKernelList)
    - compute_golden_reference(): Test-owned golden reference

    Provides 6 inherited tests:
    1. test_pipeline_creates_hw_node: Pipeline creates HW node
    2. test_shapes_preserved_through_pipeline: Shapes remain correct
    3. test_datatypes_preserved_through_pipeline: Datatypes remain correct
    4. test_python_execution_vs_golden: Python execution matches golden
    5. test_cppsim_execution_vs_golden: HLS C++ simulation matches golden
    6. test_rtlsim_execution_vs_golden: RTL simulation matches golden
    """

    # ========================================================================
    # Abstract Methods - Subclasses MUST implement (2 additional)
    # ========================================================================

    @abstractmethod
    def get_kernel_inference_transform(self) -> Type[Transformation]:
        """Return transform that converts ONNX node to hardware kernel.

        Returns:
            Transformation class (e.g., InferKernelList, InferLayerNorm)

        Example:
            def get_kernel_inference_transform(self):
                from brainsmith.primitives.transforms.infer_kernel_list import InferKernelList
                return InferKernelList
        """
        pass

    @abstractmethod
    def compute_golden_reference(
        self, inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Compute test-owned golden reference.

        Each test defines what "correct" means for its specific test case.
        This is TEST LOGIC, not production code!

        Args:
            inputs: Dict mapping input names → numpy arrays
                   Example: {"input0": arr1, "input1": arr2}

        Returns:
            Dict mapping output names → expected numpy arrays
            Example: {"output": expected_arr}

        Example:
            def compute_golden_reference(self, inputs):
                '''Element-wise addition.'''
                return {"output": inputs["input0"] + inputs["input1"]}
        """
        pass

    # ========================================================================
    # Pipeline Execution (uses Phase 1 PipelineRunner)
    # ========================================================================

    def run_inference_pipeline(self) -> Tuple[HWCustomOp, ModelWrapper]:
        """Run complete inference pipeline (ONNX → Hardware node).

        Uses PipelineRunner (Phase 1) instead of duplicating pipeline logic.

        Returns:
            (op, model): Hardware op instance and model

        Raises:
            RuntimeError: If pipeline fails to create HW node
        """
        runner = PipelineRunner()

        op, model = runner.run(
            model_factory=self.make_test_model,
            transform=self.get_kernel_inference_transform(),
            configure_fn=lambda op, model: self.configure_kernel_node(op, model)
        )

        return op, model

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
    # Test Suite (6 tests)
    # ========================================================================

    @pytest.mark.pipeline
    @pytest.mark.single_kernel
    def test_pipeline_creates_hw_node(self):
        """Validate that kernel inference creates hardware node.

        Pipeline:
        1. Create standard ONNX node
        2. Run inference (shapes, datatypes, kernel)
        3. Verify HW node was created
        4. Verify it's a HWCustomOp instance
        """
        op, model = self.run_inference_pipeline()

        # Verify op is HWCustomOp
        assert isinstance(
            op, HWCustomOp
        ), f"Kernel inference created {type(op)}, expected HWCustomOp"

        # Verify node exists in graph
        node_found = False
        for node in model.graph.node:
            if node.name == op.onnx_node.name:
                node_found = True
                break

        assert node_found, f"Hardware node '{op.onnx_node.name}' not found in graph"

    @pytest.mark.pipeline
    @pytest.mark.single_kernel
    def test_shapes_preserved_through_pipeline(self):
        """Validate tensor shapes remain correct through inference pipeline."""
        op, model = self.run_inference_pipeline()

        # Validate input shapes
        for i in range(self.get_num_inputs()):
            input_name = op.onnx_node.input[i]
            if not input_name:  # Optional input
                continue

            # Get shape from model
            model_shape = model.get_tensor_shape(input_name)
            op_shape = op.get_normal_input_shape(i)

            assert tuple(model_shape) == tuple(op_shape), (
                f"Input {i} shape mismatch:\n"
                f"  Model: {model_shape}\n"
                f"  Op:    {op_shape}"
            )

        # Validate output shapes
        for i in range(self.get_num_outputs()):
            output_name = op.onnx_node.output[i]

            # Get shape from model
            model_shape = model.get_tensor_shape(output_name)
            op_shape = op.get_normal_output_shape(i)

            assert tuple(model_shape) == tuple(op_shape), (
                f"Output {i} shape mismatch:\n"
                f"  Model: {model_shape}\n"
                f"  Op:    {op_shape}"
            )

    @pytest.mark.pipeline
    @pytest.mark.single_kernel
    def test_datatypes_preserved_through_pipeline(self):
        """Validate tensor datatypes remain correct through inference pipeline."""
        op, model = self.run_inference_pipeline()

        # Validate input datatypes
        for i in range(self.get_num_inputs()):
            input_name = op.onnx_node.input[i]
            if not input_name:  # Optional input
                continue

            # Get datatype from model
            model_dt = model.get_tensor_datatype(input_name)
            op_dt = op.get_input_datatype(i)

            assert model_dt == op_dt, (
                f"Input {i} datatype mismatch:\n"
                f"  Model: {model_dt}\n"
                f"  Op:    {op_dt}"
            )

        # Validate output datatypes
        for i in range(self.get_num_outputs()):
            output_name = op.onnx_node.output[i]

            # Get datatype from model
            model_dt = model.get_tensor_datatype(output_name)
            op_dt = op.get_output_datatype(i)

            assert model_dt == op_dt, (
                f"Output {i} datatype mismatch:\n"
                f"  Model: {model_dt}\n"
                f"  Op:    {op_dt}"
            )

    @pytest.mark.pipeline
    @pytest.mark.golden
    @pytest.mark.single_kernel
    def test_python_execution_vs_golden(self):
        """Test Python execution (execute_node) matches golden reference.

        Validates:
        1. Kernel inference creates correct HW node
        2. Python execution produces correct results
        3. Results match golden reference within tolerance
        """
        # Run pipeline
        op, model = self.run_inference_pipeline()

        # Generate test inputs (deterministic)
        np.random.seed(42)
        inputs = make_execution_context(model, op)

        # Compute golden reference
        golden_outputs = self.compute_golden_reference(inputs)

        # Execute via Python (Phase 1 PythonExecutor)
        executor = PythonExecutor()
        actual_outputs = executor.execute(op, model, inputs)

        # Validate against golden (Phase 1 GoldenValidator)
        tolerance = self.get_tolerance_python()
        self.validate_against_golden(
            actual_outputs, golden_outputs, "Python execution", tolerance
        )

    @pytest.mark.pipeline
    @pytest.mark.golden
    @pytest.mark.cppsim
    @pytest.mark.slow
    @pytest.mark.single_kernel
    def test_cppsim_execution_vs_golden(self):
        """Test HLS C++ simulation matches golden reference.

        Validates complete code generation pipeline:
        1. Kernel inference creates HW node
        2. C++ code generation succeeds
        3. Compilation succeeds
        4. Simulation produces correct results
        5. Results match golden reference within tolerance

        Requires:
            - VITIS_PATH environment variable
            - HLS backend (op inherits from HLSBackend)
        """
        # Run pipeline
        op, model = self.run_inference_pipeline()

        # Generate test inputs (deterministic)
        np.random.seed(42)
        inputs = make_execution_context(model, op)

        # Compute golden reference
        golden_outputs = self.compute_golden_reference(inputs)

        # Execute via cppsim (Phase 1 CppSimExecutor)
        executor = CppSimExecutor()
        actual_outputs = executor.execute(op, model, inputs)

        # Validate against golden (Phase 1 GoldenValidator)
        tolerance = self.get_tolerance_cppsim()
        self.validate_against_golden(
            actual_outputs, golden_outputs, "HLS simulation (cppsim)", tolerance
        )

    @pytest.mark.pipeline
    @pytest.mark.golden
    @pytest.mark.rtlsim
    @pytest.mark.slow
    @pytest.mark.single_kernel
    def test_rtlsim_execution_vs_golden(self):
        """Test RTL simulation matches golden reference.

        Validates complete HDL generation pipeline:
        1. Kernel inference creates HW node
        2. HDL generation succeeds (Verilog/VHDL)
        3. XSim compilation succeeds
        4. Simulation produces correct results
        5. Results match golden reference within tolerance

        For HLS backends: Synthesizes HLS → RTL using Vitis HLS first
        For RTL backends: Uses generated HDL directly

        Requires:
            - Vivado installation with XSim
            - RTL or HLS backend
        """
        # Run pipeline
        op, model = self.run_inference_pipeline()

        # Generate test inputs (deterministic)
        np.random.seed(42)
        inputs = make_execution_context(model, op)

        # Compute golden reference
        golden_outputs = self.compute_golden_reference(inputs)

        # Execute via rtlsim (Phase 1 RTLSimExecutor)
        executor = RTLSimExecutor()
        actual_outputs = executor.execute(op, model, inputs)

        # Validate against golden (Phase 1 GoldenValidator)
        tolerance = self.get_tolerance_rtlsim()
        self.validate_against_golden(
            actual_outputs, golden_outputs, "RTL simulation (rtlsim)", tolerance
        )
