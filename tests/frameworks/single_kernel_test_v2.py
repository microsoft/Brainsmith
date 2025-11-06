"""SingleKernelTest framework with fixture-based parameterization (v2.3).

This module provides SingleKernelTest, a fixture-driven test framework that replaces
the v1.0 hardcoded approach with pytest fixtures for maximum flexibility.

Design Philosophy (v2.3):
- Pytest fixtures control shapes and datatypes (NOT test methods)
- Tests define operations once, fixtures parameterize automatically
- **Stage 1 golden reference** - computed ONCE from ONNX model with annotations (v2.2)
- **Direct DataType annotations** - NO Quant nodes inserted (v2.3)
- Automatic test data generation (pre-quantized, using Phase 1 utilities)

Key Improvements (v2.3):
- **Direct annotations replace Quant nodes** - Simpler architecture, same semantics
- **Fixes rtlsim** - No Quant nodes to synthesize (v2.3)
- **Simpler graph structure** - Metadata (annotations) separate from operations (Quant nodes)
- **Pre-quantized test data** - Generated directly in target DataType range

Key Improvements (v2.2):
- **Golden reference computed from Stage 1 ONLY** - No more Stage 2/3 confusion
- **Shared golden fixture** - Computed once, used by all execution tests
- **Clear stage separation** - Stage 1 (golden) → Stage 2 (Python) → Stage 3 (backends)
- Fixes cppsim/rtlsim golden computation bug (was using backend model)

Key Improvements (v2.1):
- 70% less code per test (25 lines vs 80 lines)
- Zero TensorProto vs DataType confusion
- Zero Stage 1/Stage 2 confusion
- Zero manual golden reference implementation
- Maximum parameterization flexibility
- Deterministic test data with fixed seed (reproducible failures)

Usage:
    from tests.frameworks.single_kernel_test_v2 import SingleKernelTest

    # Define fixtures
    @pytest.fixture(params=[
        {"input": DataType["INT8"]},
        {"input": DataType["INT16"]},
    ])
    def input_datatypes(request):
        return request.param

    @pytest.fixture(params=[
        {"input": (1, 64)},
        {"input": (4, 128)},
    ])
    def input_shapes(request):
        return request.param

    # Test class - just operations
    class TestMyKernel(SingleKernelTest):
        def make_test_model(self, input_shapes):
            # Build model with shapes from fixture
            return model, ["input"]

        def get_kernel_inference_transform(self):
            return InferMyKernel

Result: 2 dtypes × 2 shapes = 4 tests automatically!

Inherited Tests (6 per configuration in v2.3):
Pipeline Validation (3 tests):
- test_pipeline_creates_hw_node
- test_shapes_preserved_through_pipeline
- test_datatypes_preserved_through_pipeline

Backend Execution (3 tests):
- test_python_execution_vs_golden (validates against Stage 1 golden)
- test_cppsim_execution_vs_golden (validates against Stage 1 golden)
- test_rtlsim_execution_vs_golden (validates against Stage 1 golden)

v2.3 Changes (Architectural):
- **Removed Quant node insertion** - Use direct DataType annotations instead
- **Removed 3 Quant validation tests** - No longer applicable (no Quant nodes)
- **Pre-quantized data generation** - Data generated directly in target DataType range
- **Fixes rtlsim failures** - No Quant nodes to synthesize, all tests now work
"""

from typing import Dict, Tuple

import numpy as np
import pytest
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

# Import base config
from tests.frameworks.kernel_test_base_v2 import KernelTestConfig

# Import backend specialization utilities
from tests.support.backend_utils import specialize_to_backend

# Import test executors
from tests.support.executors import CppSimExecutor, PythonExecutor, RTLSimExecutor

# Import Phase 1 utilities
from tests.support.pipeline import PipelineRunner
from tests.support.validator import GoldenValidator


class SingleKernelTest(KernelTestConfig):
    """Test one kernel with fixture-based parameterization.

    Subclasses implement (2 methods only!):
    - make_test_model(input_shapes): Create model with symbolic shapes
    - get_kernel_inference_transform(): Return transform class

    Pytest fixtures provide:
    - input_shapes: Dict[str, Tuple[int, ...]] from fixture
    - input_datatypes: Dict[str, DataType] from fixture

    Framework automatically:
    - Annotates inputs with QONNX DataTypes (NO Quant nodes inserted)
    - Generates test data (pre-quantized, using generate_test_data from Phase 1)
    - Computes golden reference (QONNX execution on Stage 1 model with annotations)
    - Validates Python/cppsim/rtlsim against golden

    Provides 6 inherited tests per configuration (v2.3):
    Pipeline Validation:
    1. test_pipeline_creates_hw_node: Pipeline creates HW node
    2. test_shapes_preserved_through_pipeline: Shapes remain correct
    3. test_datatypes_preserved_through_pipeline: Datatypes remain correct

    Backend Execution:
    4. test_python_execution_vs_golden: Python execution matches golden
    5. test_cppsim_execution_vs_golden: HLS C++ simulation matches golden
    6. test_rtlsim_execution_vs_golden: RTL simulation matches golden
    """

    # ========================================================================
    # Internal helpers - use fixtures and Phase 1 utilities
    # ========================================================================

    def _prepare_model_with_annotations(
        self, input_shapes: Dict[str, Tuple[int, ...]], input_datatypes: Dict[str, DataType]
    ) -> Tuple[ModelWrapper, str]:
        """Create model with QONNX DataType annotations (NO Quant nodes).

        This is an internal helper. Tests should not call directly.

        Instead of inserting Quant nodes (IntQuant/FloatQuant/BipolarQuant),
        this method directly annotates input tensors with QONNX DataTypes.
        This produces identical InferDataTypes results with simpler graph structure.

        Args:
            input_shapes: Dict from fixture, e.g., {"input": (1, 64)}
            input_datatypes: Dict from fixture, e.g., {"input": DataType["INT8"]}

        Returns:
            (model, target_node): Model with DataType annotations and target node name
        """
        # Step 1: Get simple model from test
        model, input_names = self.make_test_model(input_shapes)

        # Step 2: Annotate inputs with QONNX DataTypes (direct annotation, no Quant nodes)
        from tests.support.datatype_annotation import annotate_model_datatypes

        input_annotations = {name: input_datatypes[name] for name in input_names if name in input_datatypes}
        if input_annotations:
            annotate_model_datatypes(model, input_annotations, warn_unsupported=True)

        # Step 3: Annotate outputs (infer from model or use defaults)
        output_names = [out.name for out in model.graph.output]
        for out_name in output_names:
            if model.get_tensor_datatype(out_name) is None:
                # Default: same dtype as first input
                first_input = input_names[0]
                if first_input in input_datatypes:
                    model.set_tensor_datatype(out_name, input_datatypes[first_input])

        # Step 4: Find target node (first node in graph)
        if len(model.graph.node) == 0:
            raise RuntimeError("No nodes found in graph")

        target_node = model.graph.node[0].name

        return model, target_node

    def _generate_test_inputs(
        self,
        input_shapes: Dict[str, Tuple[int, ...]],
        input_datatypes: Dict[str, DataType],
    ) -> Dict[str, np.ndarray]:
        """Generate test data with correct shapes and datatypes.

        This is an internal helper. Tests should not call directly.

        Generates test data directly in the target DataType range (pre-quantized).
        Since we use direct annotations (no Quant nodes), input names remain unchanged.

        Args:
            input_shapes: Dict from fixture
            input_datatypes: Dict from fixture

        Returns:
            Dict mapping input names to test data arrays (pre-quantized)
        """
        from tests.support.data_generation import generate_test_data

        inputs = {}
        _, input_names = self.make_test_model(input_shapes)

        for name in input_names:
            if name not in input_datatypes:
                continue  # Skip if no datatype specified

            dtype = input_datatypes[name]
            shape = input_shapes[name]

            # Generate data with correct dtype and shape (using configured seed)
            # Data is pre-quantized (already in dtype's range)
            inputs[name] = generate_test_data(dtype, shape, seed=self.get_test_seed())

        return inputs

    def get_use_custom_golden_reference(self) -> bool:
        """Override to use custom golden reference instead of QONNX execution.

        Returns:
            bool: True to use compute_custom_golden_reference(), False for QONNX

        Default:
            False (use QONNX execution on quantized model)

        Override when:
            - Operation not supported by QONNX execution
            - Need specific golden reference behavior
            - Debugging quantization issues
        """
        return False

    def compute_custom_golden_reference(
        self, model: ModelWrapper, inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Override to provide custom golden reference implementation.

        Only called when get_use_custom_golden_reference() returns True.

        Args:
            model: ModelWrapper WITH DataType annotations (NO Quant nodes)
            inputs: Dict of input tensors (pre-quantized, input names as keys)

        Returns:
            Dict of output tensors

        Raises:
            NotImplementedError: Must override when using custom golden reference
        """
        raise NotImplementedError(
            "Must implement compute_custom_golden_reference() when "
            "get_use_custom_golden_reference() returns True"
        )

    def _compute_golden_reference(
        self,
        quant_model: ModelWrapper,
        inputs: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Compute golden reference using QONNX execution on annotated model.

        This is an internal helper. Tests should not call directly.

        Golden reference runs QONNX on the Stage 1 model WITH DataType annotations:
        - Model has QONNX DataType annotations on tensors (NO Quant nodes)
        - Executes ONNX operations with pre-quantized input data
        - Uses quantized values as golden reference (not floating-point)

        Args:
            quant_model: ModelWrapper WITH DataType annotations (NO Quant nodes)
            inputs: Test data (pre-quantized, input names as keys)

        Returns:
            Expected outputs from QONNX execution
        """
        # Check for custom golden reference
        if self.get_use_custom_golden_reference():
            return self.compute_custom_golden_reference(quant_model, inputs)

        # Default: Execute annotated model with QONNX
        from qonnx.core.onnx_exec import execute_onnx

        return execute_onnx(quant_model, inputs, return_full_exec_context=False)

    # ========================================================================
    # Pipeline execution
    # ========================================================================

    def run_inference_pipeline(
        self,
        input_shapes: Dict[str, Tuple[int, ...]],
        input_datatypes: Dict[str, DataType],
        to_backend: bool = False,
    ) -> Tuple[HWCustomOp, ModelWrapper]:
        """Run inference pipeline to Stage 2 (base kernel) or Stage 3 (backend).

        Pipeline stages:
            Stage 1: ONNX Node (Add, Mul, etc.)
            Stage 2: Base Kernel (AddStreams, no backend) ← to_backend=False
            Stage 3: Backend (AddStreams_hls, with HLSBackend) ← to_backend=True

        Args:
            input_shapes: Dict from fixture
            input_datatypes: Dict from fixture
            to_backend: If True, specialize to backend (Stage 3).
                       If False, return base kernel (Stage 2).

        Returns:
            (op, model): Hardware op instance and model
                        - Stage 2: Base kernel (e.g., AddStreams)
                        - Stage 3: Backend (e.g., AddStreams_hls)

        Raises:
            RuntimeError: If pipeline fails to create HW node
            pytest.skip: If to_backend=True but backend not configured
        """
        # Prepare model with QONNX annotations
        model, target_node = self._prepare_model_with_annotations(input_shapes, input_datatypes)

        # Stage 1 → Stage 2: ONNX → Base Kernel
        runner = PipelineRunner()
        op, model = runner.run(
            model_factory=lambda: (model, target_node),
            transform=self.get_kernel_inference_transform(),
            configure_fn=lambda op, m: self.configure_kernel_node(op, m),
        )

        # Stage 2 → Stage 3: Base Kernel → Backend (optional)
        if to_backend:
            fpgapart = self.get_backend_fpgapart()
            if fpgapart is None:
                pytest.skip(
                    "Backend specialization not configured. "
                    "Override get_backend_fpgapart() to enable backend testing."
                )

            backend_variants = self.get_backend_variants()
            if backend_variants is None:
                # Auto-detect from registry
                backend_variants = self._auto_detect_backends(op)

            op, model = specialize_to_backend(op, model, fpgapart, backend_variants)

            # Stage 3 configuration (backend-specific parameters)
            self.configure_backend_node(op, model)

        return op, model

    def _auto_detect_backends(self, op):
        """Auto-detect backend variants from Brainsmith registry.

        Args:
            op: HWCustomOp instance to find backends for

        Returns:
            List of backend classes

        Raises:
            pytest.skip: If no backends found
        """
        from brainsmith.registry import get_backend, list_backends_for_kernel

        backend_names = list_backends_for_kernel(op.onnx_node.op_type, language="hls")
        if not backend_names:
            pytest.skip(f"No HLS backend found for {op.onnx_node.op_type}")
        return [get_backend(name) for name in backend_names]

    # ========================================================================
    # Validation
    # ========================================================================

    def validate_against_golden(
        self,
        actual_outputs: Dict[str, np.ndarray],
        golden_outputs: Dict[str, np.ndarray],
        backend_name: str,
        tolerance: Dict[str, float],
    ) -> None:
        """Validate actual outputs match golden reference.

        Uses GoldenValidator (Phase 1) for consistent validation logic.

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
    # Pytest Fixtures - Stage 1 golden reference (v2.2)
    # ========================================================================

    @pytest.fixture(scope="function")
    def stage1_model(
        self,
        input_shapes: Dict[str, Tuple[int, ...]],
        input_datatypes: Dict[str, DataType]
    ) -> ModelWrapper:
        """Stage 1 model with QONNX annotations (before kernel inference).

        This is the ONNX model WITH DataType annotations (NO Quant nodes),
        but BEFORE any kernel inference transforms. This is the model we
        compute golden reference from.

        Args:
            input_shapes: From fixture
            input_datatypes: From fixture

        Returns:
            Stage 1 model (ONNX + annotations, no kernel inference, no Quant nodes)
        """
        model, _ = self._prepare_model_with_annotations(input_shapes, input_datatypes)

        # Run shape/datatype inference (required for QONNX execution)
        from qonnx.transformation.infer_shapes import InferShapes
        from qonnx.transformation.infer_datatypes import InferDataTypes

        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        return model

    @pytest.fixture(scope="function")
    def test_inputs(
        self,
        input_shapes: Dict[str, Tuple[int, ...]],
        input_datatypes: Dict[str, DataType]
    ) -> Dict[str, np.ndarray]:
        """Generate test inputs with deterministic seed.

        Args:
            input_shapes: From fixture
            input_datatypes: From fixture

        Returns:
            Dict mapping raw_* input names to test data arrays
        """
        return self._generate_test_inputs(input_shapes, input_datatypes)

    @pytest.fixture(scope="function")
    def golden_outputs(
        self,
        stage1_model: ModelWrapper,
        test_inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Golden reference computed from Stage 1 ONNX model.

        This is computed ONCE per test configuration and shared across all
        execution tests (Python, cppsim, rtlsim).

        Args:
            stage1_model: Stage 1 model fixture (ONNX + Quant)
            test_inputs: Test inputs fixture

        Returns:
            Expected outputs from QONNX execution on Stage 1 model
        """
        return self._compute_golden_reference(stage1_model, test_inputs)

    # ========================================================================
    # Test Suite (6 tests) - Parameterized via fixtures
    # ========================================================================

    @pytest.mark.pipeline
    @pytest.mark.single_kernel
    def test_pipeline_creates_hw_node(
        self, input_shapes: Dict[str, Tuple[int, ...]], input_datatypes: Dict[str, DataType]
    ):
        """Validate that kernel inference creates hardware node.

        Args:
            input_shapes: From fixture (parameterizes shapes)
            input_datatypes: From fixture (parameterizes dtypes)

        Pipeline:
        1. Create ONNX node with Quant nodes
        2. Run inference (shapes, datatypes, kernel)
        3. Verify HW node was created
        4. Verify it's a HWCustomOp instance
        """
        op, model = self.run_inference_pipeline(input_shapes, input_datatypes)

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
    def test_shapes_preserved_through_pipeline(
        self, input_shapes: Dict[str, Tuple[int, ...]], input_datatypes: Dict[str, DataType]
    ):
        """Validate tensor shapes remain correct through inference pipeline.

        Args:
            input_shapes: From fixture
            input_datatypes: From fixture
        """
        op, model = self.run_inference_pipeline(input_shapes, input_datatypes)

        # Get input names from test
        _, input_names = self.make_test_model(input_shapes)

        # Validate input shapes (check actual inputs after Quant)
        for input_name in input_names:
            if input_name not in input_datatypes:
                continue

            # After Quant insertion, check the Quant OUTPUT (which has original name)
            model_shape = model.get_tensor_shape(input_name)
            expected_shape = input_shapes[input_name]

            assert tuple(model_shape) == tuple(expected_shape), (
                f"Input '{input_name}' shape mismatch:\n"
                f"  Expected: {expected_shape}\n"
                f"  Got:      {model_shape}"
            )

        # Validate output shapes
        output_names = [out.name for out in model.graph.output]
        for i, output_name in enumerate(output_names):
            # Get shape from model
            model_shape = model.get_tensor_shape(output_name)
            op_shape = op.get_normal_output_shape(i)

            assert tuple(model_shape) == tuple(op_shape), (
                f"Output {i} ('{output_name}') shape mismatch:\n"
                f"  Model: {model_shape}\n"
                f"  Op:    {op_shape}"
            )

    @pytest.mark.pipeline
    @pytest.mark.single_kernel
    def test_datatypes_preserved_through_pipeline(
        self, input_shapes: Dict[str, Tuple[int, ...]], input_datatypes: Dict[str, DataType]
    ):
        """Validate tensor datatypes remain correct through inference pipeline.

        Args:
            input_shapes: From fixture
            input_datatypes: From fixture
        """
        op, model = self.run_inference_pipeline(input_shapes, input_datatypes)

        # Get input names from test
        _, input_names = self.make_test_model(input_shapes)

        # Validate input datatypes (check Quant outputs)
        for input_name in input_names:
            if input_name not in input_datatypes:
                continue

            # Check Quant output has correct datatype
            model_dt = model.get_tensor_datatype(input_name)
            expected_dt = input_datatypes[input_name]

            assert model_dt == expected_dt, (
                f"Input '{input_name}' datatype mismatch:\n"
                f"  Expected: {expected_dt}\n"
                f"  Got:      {model_dt}"
            )

        # Validate output datatypes
        output_names = [out.name for out in model.graph.output]
        for i, output_name in enumerate(output_names):
            # Get datatype from model
            model_dt = model.get_tensor_datatype(output_name)
            op_dt = op.get_output_datatype(i)

            assert model_dt == op_dt, (
                f"Output {i} ('{output_name}') datatype mismatch:\n"
                f"  Model: {model_dt}\n"
                f"  Op:    {op_dt}"
            )

    @pytest.mark.pipeline
    @pytest.mark.golden
    @pytest.mark.single_kernel
    def test_python_execution_vs_golden(
        self,
        input_shapes: Dict[str, Tuple[int, ...]],
        input_datatypes: Dict[str, DataType],
        test_inputs: Dict[str, np.ndarray],
        golden_outputs: Dict[str, np.ndarray]
    ):
        """Test Python execution (execute_node) matches QONNX golden reference.

        Args:
            input_shapes: From fixture (parameterizes shapes)
            input_datatypes: From fixture (parameterizes dtypes)
            test_inputs: From fixture (generated test data)
            golden_outputs: From fixture (Stage 1 golden reference)

        Validates:
        1. Kernel inference creates correct HW node
        2. Python execution produces correct results
        3. Results match Stage 1 QONNX golden reference (quantized) within tolerance
        """
        # Run pipeline to Stage 2 (base kernel)
        op, model = self.run_inference_pipeline(input_shapes, input_datatypes)

        # Execute via Python backend (uses Stage 2 model)
        executor = PythonExecutor()
        actual_outputs = executor.execute(op, model, test_inputs)

        # Validate against golden (from Stage 1 fixture)
        tolerance = self.get_tolerance_python()
        self.validate_against_golden(actual_outputs, golden_outputs, "Python execution", tolerance)

    @pytest.mark.pipeline
    @pytest.mark.golden
    @pytest.mark.cppsim
    @pytest.mark.slow
    @pytest.mark.single_kernel
    def test_cppsim_execution_vs_golden(
        self,
        input_shapes: Dict[str, Tuple[int, ...]],
        input_datatypes: Dict[str, DataType],
        test_inputs: Dict[str, np.ndarray],
        golden_outputs: Dict[str, np.ndarray]
    ):
        """Test HLS C++ simulation matches QONNX golden reference.

        Args:
            input_shapes: From fixture
            input_datatypes: From fixture
            test_inputs: From fixture (generated test data)
            golden_outputs: From fixture (Stage 1 golden reference)

        Validates complete code generation pipeline (3 stages):
        1. ONNX → Base Kernel (Stage 1 → 2)
        2. Base Kernel → HLS Backend (Stage 2 → 3)
        3. HLS Backend → C++ code → Compilation → Simulation

        Results must match Stage 1 QONNX golden reference (quantized) within tolerance.

        Requires:
            - VITIS_PATH environment variable
            - Backend configured (get_backend_fpgapart() returns FPGA part)
            - HLS backend available for kernel

        Skips:
            - If backend not configured (get_backend_fpgapart() returns None)
            - If VITIS_PATH not set (handled by CppSimExecutor)
        """
        # Run pipeline to Stage 3 (backend)
        op, model = self.run_inference_pipeline(input_shapes, input_datatypes, to_backend=True)

        # Execute via cppsim (uses Stage 3 backend model)
        executor = CppSimExecutor()
        actual_outputs = executor.execute(op, model, test_inputs)

        # Validate against golden (from Stage 1 fixture)
        tolerance = self.get_tolerance_cppsim()
        self.validate_against_golden(actual_outputs, golden_outputs, "HLS simulation (cppsim)", tolerance)

    @pytest.mark.pipeline
    @pytest.mark.golden
    @pytest.mark.rtlsim
    @pytest.mark.slow
    @pytest.mark.single_kernel
    def test_rtlsim_execution_vs_golden(
        self,
        input_shapes: Dict[str, Tuple[int, ...]],
        input_datatypes: Dict[str, DataType],
        test_inputs: Dict[str, np.ndarray],
        golden_outputs: Dict[str, np.ndarray]
    ):
        """Test RTL simulation matches QONNX golden reference.

        Args:
            input_shapes: From fixture
            input_datatypes: From fixture
            test_inputs: From fixture (generated test data)
            golden_outputs: From fixture (Stage 1 golden reference)

        Validates complete HDL generation pipeline (3 stages):
        1. ONNX → Base Kernel (Stage 1 → 2)
        2. Base Kernel → Backend (Stage 2 → 3, HLS or RTL)
        3. Backend → HDL → Synthesis → Simulation

        For HLS backends: Synthesizes HLS → RTL using Vitis HLS first
        For RTL backends: Uses generated HDL directly

        Results must match Stage 1 QONNX golden reference (quantized) within tolerance.

        Requires:
            - Vivado installation with XSim
            - Backend configured (get_backend_fpgapart() returns FPGA part)
            - Backend available for kernel (HLS or RTL)

        Skips:
            - If backend not configured (get_backend_fpgapart() returns None)
            - If Vivado not found (handled by RTLSimExecutor)
        """
        # Run pipeline to Stage 3 (backend)
        op, model = self.run_inference_pipeline(input_shapes, input_datatypes, to_backend=True)

        # Execute via rtlsim (uses Stage 3 backend model)
        executor = RTLSimExecutor()
        actual_outputs = executor.execute(op, model, test_inputs)

        # Validate against golden (from Stage 1 fixture)
        tolerance = self.get_tolerance_rtlsim()
        self.validate_against_golden(actual_outputs, golden_outputs, "RTL simulation (rtlsim)", tolerance)

