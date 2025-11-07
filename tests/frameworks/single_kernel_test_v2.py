"""SingleKernelTest framework with fixture-based parameterization (v2.3).

This module provides SingleKernelTest, a fixture-driven test framework that replaces
the v1.0 hardcoded approach with pytest fixtures for maximum flexibility.

Design Philosophy (v2.3):
- Pytest fixtures control shapes and datatypes (NOT test methods)
- Tests define operations once, fixtures parameterize automatically
- **Stage 1 golden reference** - computed ONCE from ONNX model with annotations (v2.2)
- **Direct DataType annotations** - NO Quant nodes inserted (v2.3)
- **Shared utilities** - Inherits from KernelTestBase_v2 (NEW in Phase 2)
- Automatic test data generation (pre-quantized, using Phase 1 utilities)

Inheritance Chain (v2.3):
    KernelTestConfig (abstract interface)
        ↓
    KernelTestBase_v2 (shared utilities)
        ↓
    SingleKernelTest (fixture-based testing) ← THIS CLASS

Inherited from KernelTestBase_v2:
- validate_against_golden() - GoldenValidator-based output validation
- _auto_detect_backends() - Registry-based backend lookup
- _specialize_to_backend_stage() - Stage 2→3 specialization with overrides

Key Improvements (v2.3):
- **Direct annotations replace Quant nodes** - Simpler architecture, same semantics
- **Fixes rtlsim** - No Quant nodes to synthesize (v2.3)
- **Simpler graph structure** - Metadata (annotations) separate from operations (Quant nodes)
- **Pre-quantized test data** - Generated directly in target DataType range
- **Shared utilities** - Zero code duplication with DualKernelTest_v2

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

# Import base utilities
from tests.frameworks.kernel_test_base_v2 import KernelTestBase_v2

# Import test executors
from tests.support.executors import CppSimExecutor, PythonExecutor, RTLSimExecutor

# Import Phase 1 utilities
from tests.support.pipeline import PipelineRunner


class SingleKernelTest(KernelTestBase_v2):
    """Test one kernel with fixture-based parameterization.

    Subclasses implement (2 methods only!):
    - make_test_model(kernel_test_config): Create Stage 1 ONNX model
    - get_kernel_op(): Return the kernel class to test

    get_kernel_inference_transform() has a default implementation using get_kernel_op().
    Override only if you need custom transform logic.

    Pytest fixtures provide:
    - kernel_test_config: Unified test configuration (v4.0+)

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
    # Abstract methods - must implement in subclass
    # ========================================================================

    def get_kernel_op(self):
        """Return the kernel class to test.

        Returns:
            Kernel class (e.g., ElementwiseBinaryOp, AddStreams)

        Example:
            def get_kernel_op(self):
                from brainsmith.kernels.elementwise_binary import ElementwiseBinaryOp
                return ElementwiseBinaryOp
        """
        raise NotImplementedError("Subclass must implement get_kernel_op()")

    def get_kernel_inference_transform(self):
        """Return the kernel inference transform.

        Default implementation uses InferKernels([get_kernel_op()]).
        Override only if you need custom transform logic.

        Returns:
            Transformation instance (not callable!)

        Example (using default):
            # No need to override - default uses get_kernel_op()

        Example (custom):
            def get_kernel_inference_transform(self):
                from my_package import CustomInferenceTransform
                return CustomInferenceTransform(special_config=True)
        """
        from brainsmith.primitives.transforms.infer_kernels import InferKernels
        return InferKernels([self.get_kernel_op()])

    # ========================================================================
    # Internal helpers - use fixtures and Phase 1 utilities
    # ========================================================================

    def _prepare_model_with_annotations(
        self, kernel_test_config: "KernelTestConfig"
    ) -> Tuple[ModelWrapper, str]:
        """Create model with QONNX DataType annotations (NO Quant nodes).

        This is an internal helper. Tests should not call directly.

        Instead of inserting Quant nodes (IntQuant/FloatQuant/BipolarQuant),
        this method directly annotates input tensors with QONNX DataTypes.
        This produces identical InferDataTypes results with simpler graph structure.

        Args:
            kernel_test_config: Unified test configuration (v3.0, required)
                Contains input_shapes, input_dtypes, operation, etc.

        Returns:
            (model, target_node): Model with DataType annotations and target node name
        """
        # Step 1: Get simple model from test
        model, input_names = self.make_test_model(kernel_test_config)

        # Step 2: Annotate inputs with QONNX DataTypes (direct annotation, no Quant nodes)
        from tests.fixtures.model_annotation import annotate_model_datatypes

        input_datatypes = kernel_test_config.input_dtypes
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
        kernel_test_config: "KernelTestConfig",
    ) -> Dict[str, np.ndarray]:
        """Generate test data with correct shapes and datatypes.

        This is an internal helper. Tests should not call directly.

        Generates test data directly in the target DataType range (pre-quantized).
        Since we use direct annotations (no Quant nodes), input names remain unchanged.

        Args:
            kernel_test_config: Unified test configuration (v3.0, required)
                Contains input_shapes, input_dtypes, etc.

        Returns:
            Dict mapping input names to test data arrays (pre-quantized)
        """
        from tests.fixtures.test_data import generate_test_data

        inputs = {}
        _, input_names = self.make_test_model(kernel_test_config)

        input_shapes = kernel_test_config.input_shapes
        input_datatypes = kernel_test_config.input_dtypes

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
    # Pipeline execution (REMOVED - use fixtures instead)
    # ========================================================================
    # run_inference_pipeline() was removed in v5.0
    # Use stage2_model and stage3_model fixtures instead


    # ========================================================================
    # Pytest Fixtures - v3.0 Extraction (Fixture Composition)
    # ========================================================================

    @pytest.fixture
    def input_shapes(self, kernel_test_config: "KernelTestConfig"):
        """Extract input_shapes from unified config (v3.0).

        This extraction fixture enables pytest fixture composition pattern,
        allowing tests to request input_shapes while framework provides them
        from kernel_test_config.

        Args:
            kernel_test_config: Unified test configuration (auto-injected by pytest)

        Returns:
            Dict mapping input names to shapes
        """
        return kernel_test_config.input_shapes

    @pytest.fixture
    def input_datatypes(self, kernel_test_config: "KernelTestConfig"):
        """Extract input_datatypes from unified config (v3.0).

        This extraction fixture enables pytest fixture composition pattern,
        allowing tests to request input_datatypes while framework provides them
        from kernel_test_config.

        Args:
            kernel_test_config: Unified test configuration (auto-injected by pytest)

        Returns:
            Dict mapping input names to DataTypes
        """
        return kernel_test_config.input_dtypes

    # ========================================================================
    # Pytest Fixtures - Stage 1 golden reference (v2.2)
    # ========================================================================

    @pytest.fixture(scope="function")
    def stage1_model(
        self, kernel_test_config: "KernelTestConfig", model_cache
    ) -> ModelWrapper:
        """Stage 1 model with QONNX annotations (before kernel inference).

        This is the ONNX model WITH DataType annotations (NO Quant nodes),
        but BEFORE any kernel inference transforms. This is the model we
        compute golden reference from.

        Uses session-scoped model_cache for computational reuse across tests
        with the same test_id.

        Args:
            kernel_test_config: Unified test configuration (auto-injected by pytest)
            model_cache: Session-scoped cache for model artifacts

        Returns:
            Stage 1 model (ONNX + annotations, no kernel inference, no Quant nodes)
        """

        def builder():
            model, _ = self._prepare_model_with_annotations(kernel_test_config)

            # Run shape/datatype inference (required for QONNX execution)
            from qonnx.transformation.infer_datatypes import InferDataTypes
            from qonnx.transformation.infer_shapes import InferShapes

            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())

            return model

        return model_cache.get_stage1_model(kernel_test_config.test_id, builder)

    @pytest.fixture(scope="function")
    def test_inputs(
        self, kernel_test_config: "KernelTestConfig", model_cache
    ) -> Dict[str, np.ndarray]:
        """Generate test inputs with deterministic seed (v3.0).

        Uses session-scoped model_cache for computational reuse across tests
        with the same test_id.

        Args:
            kernel_test_config: Unified test configuration (auto-injected by pytest)
                Contains input_shapes, input_dtypes for test data generation
            model_cache: Session-scoped cache for model artifacts

        Returns:
            Dict mapping input names to test data arrays (pre-quantized)
        """

        def builder():
            return self._generate_test_inputs(kernel_test_config)

        return model_cache.get_test_inputs(kernel_test_config.test_id, builder)

    @pytest.fixture(scope="function")
    def golden_outputs(
        self,
        kernel_test_config: "KernelTestConfig",
        stage1_model: ModelWrapper,
        test_inputs: Dict[str, np.ndarray],
        model_cache,
    ) -> Dict[str, np.ndarray]:
        """Golden reference computed from Stage 1 ONNX model.

        This is computed ONCE per test configuration and shared across all
        execution tests (Python, cppsim, rtlsim).

        Uses session-scoped model_cache for computational reuse across tests
        with the same test_id.

        Args:
            kernel_test_config: Unified test configuration (auto-injected by pytest)
            stage1_model: Stage 1 model fixture (ONNX + Quant)
            test_inputs: Test inputs fixture
            model_cache: Session-scoped cache for model artifacts

        Returns:
            Expected outputs from QONNX execution on Stage 1 model
        """

        def builder():
            return self._compute_golden_reference(stage1_model, test_inputs)

        return model_cache.get_golden_reference(kernel_test_config.test_id, builder)

    @pytest.fixture(scope="function")
    def stage2_model(
        self,
        kernel_test_config: "KernelTestConfig",
        stage1_model: ModelWrapper,
        model_cache,
    ) -> Tuple:
        """Stage 2 model (base kernel, no backend specialization).

        Runs kernel inference transform on Stage 1 model to produce base kernel
        (e.g., ONNX Add → AddStreams).

        Uses session-scoped model_cache for computational reuse across tests
        with the same test_id. Enables sharing between python/cppsim/rtlsim tests.

        Args:
            kernel_test_config: Unified test configuration
            stage1_model: Cached Stage 1 model fixture
            model_cache: Session-scoped cache for model artifacts

        Returns:
            (kernel_op, model) tuple for Python execution
        """

        def builder():
            # Reuse cached Stage 1 model
            model = stage1_model
            target_node = model.graph.node[0].name  # Assume first node

            # Stage 1 → Stage 2: ONNX → Base Kernel
            from tests.support.pipeline import PipelineRunner

            runner = PipelineRunner()

            def configure_stage_2(op, m):
                # Apply imperative configuration (backward compatible)
                self.configure_parameters(op, m, stage=2)
                # Apply declarative configuration from fixture (v3.0)
                self.auto_configure_from_fixture(
                    op, m, stage=2, config=kernel_test_config
                )

            op, model = runner.run(
                model_factory=lambda: (model, target_node),
                transform=self.get_kernel_inference_transform(),
                configure_fn=configure_stage_2,
            )

            return op, model

        return model_cache.get_stage2_model(kernel_test_config.test_id, builder)

    @pytest.fixture(scope="function")
    def stage3_model(
        self,
        kernel_test_config: "KernelTestConfig",
        stage2_model: Tuple,
        model_cache,
    ) -> Tuple:
        """Stage 3 model (backend-specialized kernel).

        Specializes Stage 2 base kernel to backend (e.g., AddStreams → AddStreams_hls).

        Uses session-scoped model_cache for computational reuse. Cache key includes
        fpgapart for platform-specific backend models.

        Skips test if fpgapart not configured in kernel_test_config.

        Args:
            kernel_test_config: Unified test configuration
            stage2_model: Cached Stage 2 model fixture
            model_cache: Session-scoped cache for model artifacts

        Returns:
            (backend_op, model) tuple for cppsim/rtlsim execution

        Raises:
            pytest.skip: If fpgapart not configured for this test
        """
        fpgapart = kernel_test_config.get_fpgapart()
        if fpgapart is None:
            pytest.skip("Backend testing skipped (no FPGA part configured for this test)")

        def builder():
            # Reuse cached Stage 2 model
            base_op, base_model = stage2_model

            # Stage 2 → Stage 3: Base Kernel → Backend
            op, model = self._specialize_to_backend_stage(
                base_op, base_model, kernel_test_config
            )

            # Configure backend-specific parameters
            self.configure_parameters(op, model, stage=3)
            # Apply declarative configuration from fixture (v3.0)
            self.auto_configure_from_fixture(op, model, stage=3, config=kernel_test_config)

            return op, model

        return model_cache.get_stage3_model(kernel_test_config.test_id, fpgapart, builder)

    # ========================================================================
    # Test Suite (6 tests) - Parameterized via fixtures
    # ========================================================================

    @pytest.mark.pipeline
    @pytest.mark.single_kernel
    def test_pipeline_creates_hw_node(
        self, kernel_test_config: "KernelTestConfig", stage2_model: Tuple
    ):
        """Validate that kernel inference creates hardware node.

        Args:
            kernel_test_config: Unified test configuration (v3.0, required)
            stage2_model: Cached Stage 2 model fixture

        Pipeline:
        1. Create ONNX node with Quant nodes
        2. Run inference (shapes, datatypes, kernel)
        3. Verify HW node was created
        4. Verify it's a HWCustomOp instance
        """
        op, model = stage2_model

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
        self, kernel_test_config: "KernelTestConfig", stage2_model: Tuple
    ):
        """Validate tensor shapes remain correct through inference pipeline.

        Args:
            kernel_test_config: Unified test configuration (v3.0, required)
            stage2_model: Cached Stage 2 model fixture
        """
        op, model = stage2_model

        # Extract config data for validation
        input_shapes = kernel_test_config.input_shapes
        input_datatypes = kernel_test_config.input_dtypes

        # Get input names from test
        _, input_names = self.make_test_model(kernel_test_config)

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
        self, kernel_test_config: "KernelTestConfig", stage2_model: Tuple
    ):
        """Validate tensor datatypes remain correct through inference pipeline.

        Args:
            kernel_test_config: Unified test configuration (v3.0, required)
            stage2_model: Cached Stage 2 model fixture
        """
        op, model = stage2_model

        # Extract config data for validation
        input_datatypes = kernel_test_config.input_dtypes

        # Get input names from test
        _, input_names = self.make_test_model(kernel_test_config)

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
        kernel_test_config: "KernelTestConfig",
        stage2_model: Tuple,
        test_inputs: Dict[str, np.ndarray],
        golden_outputs: Dict[str, np.ndarray],
    ):
        """Test Python execution (execute_node) matches QONNX golden reference.

        Uses cached Stage 2 model from fixture for computational reuse across
        test depths (python/cppsim/rtlsim).

        Args:
            kernel_test_config: Unified test configuration (v3.0, required)
            stage2_model: Cached Stage 2 model fixture (NEW: replaces run_inference_pipeline)
            test_inputs: From fixture (generated test data)
            golden_outputs: From fixture (Stage 1 golden reference)

        Validates:
        1. Kernel inference creates correct HW node
        2. Python execution produces correct results
        3. Results match Stage 1 QONNX golden reference (quantized) within tolerance
        """
        # Use cached Stage 2 model (reused across python/cppsim/rtlsim)
        op, model = stage2_model

        # Execute via Python backend (uses Stage 2 model)
        executor = PythonExecutor()
        actual_outputs = executor.execute(op, model, test_inputs)

        # Validate against golden (from Stage 1 fixture)
        tolerance = kernel_test_config.get_tolerance_python()
        self.validate_against_golden(
            actual_outputs, golden_outputs, "Python execution", tolerance
        )

    @pytest.mark.pipeline
    @pytest.mark.golden
    @pytest.mark.cppsim
    @pytest.mark.slow
    @pytest.mark.single_kernel
    def test_cppsim_execution_vs_golden(
        self,
        kernel_test_config: "KernelTestConfig",
        stage3_model: Tuple,
        test_inputs: Dict[str, np.ndarray],
        golden_outputs: Dict[str, np.ndarray],
    ):
        """Test HLS C++ simulation matches QONNX golden reference.

        Uses cached Stage 3 model from fixture for computational reuse.
        Stage 2 model is reused from test_python_execution_vs_golden.

        Args:
            kernel_test_config: Unified test configuration (v3.0, required)
            stage3_model: Cached Stage 3 model fixture (NEW: replaces run_inference_pipeline)
            test_inputs: From fixture (generated test data)
            golden_outputs: From fixture (Stage 1 golden reference)

        Validates complete code generation pipeline (3 stages):
        1. ONNX → Base Kernel (Stage 1 → 2) [cached from stage2_model]
        2. Base Kernel → HLS Backend (Stage 2 → 3) [cached here]
        3. HLS Backend → C++ code → Compilation → Simulation

        Results must match Stage 1 QONNX golden reference (quantized) within tolerance.

        Requires:
            - VITIS_PATH environment variable
            - Backend configured (get_backend_fpgapart() returns FPGA part)
            - HLS backend available for kernel

        Skips:
            - If backend not configured (handled by stage3_model fixture)
            - If VITIS_PATH not set (handled by CppSimExecutor)
        """
        # Use cached Stage 3 model (reused for rtlsim test)
        op, model = stage3_model

        # Execute via cppsim (uses Stage 3 backend model)
        executor = CppSimExecutor()
        actual_outputs = executor.execute(op, model, test_inputs)

        # Validate against golden (from Stage 1 fixture)
        tolerance = kernel_test_config.get_tolerance_cppsim()
        self.validate_against_golden(
            actual_outputs, golden_outputs, "HLS simulation (cppsim)", tolerance
        )

    @pytest.mark.pipeline
    @pytest.mark.golden
    @pytest.mark.rtlsim
    @pytest.mark.slow
    @pytest.mark.single_kernel
    def test_rtlsim_execution_vs_golden(
        self,
        kernel_test_config: "KernelTestConfig",
        stage3_model: Tuple,
        test_inputs: Dict[str, np.ndarray],
        golden_outputs: Dict[str, np.ndarray],
    ):
        """Test RTL simulation matches QONNX golden reference.

        Uses cached Stage 3 model from fixture for computational reuse.
        Stage 2 model reused from test_python_execution_vs_golden,
        Stage 3 model reused from test_cppsim_execution_vs_golden.

        Args:
            kernel_test_config: Unified test configuration (v3.0, required)
            stage3_model: Cached Stage 3 model fixture (NEW: replaces run_inference_pipeline)
            test_inputs: From fixture (generated test data)
            golden_outputs: From fixture (Stage 1 golden reference)

        Validates complete HDL generation pipeline (3 stages):
        1. ONNX → Base Kernel (Stage 1 → 2) [cached from stage2_model]
        2. Base Kernel → Backend (Stage 2 → 3) [cached from stage3_model]
        3. Backend → HDL → Synthesis → Simulation

        For HLS backends: Synthesizes HLS → RTL using Vitis HLS first
        For RTL backends: Uses generated HDL directly

        Results must match Stage 1 QONNX golden reference (quantized) within tolerance.

        Requires:
            - Vivado installation with XSim
            - Backend configured (handled by stage3_model fixture)
            - Backend available for kernel (HLS or RTL)

        Skips:
            - If backend not configured (handled by stage3_model fixture)
            - If Vivado not found (handled by RTLSimExecutor)
        """
        # Use cached Stage 3 model (same as cppsim test)
        op, model = stage3_model

        # Execute via rtlsim (uses Stage 3 backend model)
        executor = RTLSimExecutor()
        actual_outputs = executor.execute(op, model, test_inputs)

        # Validate against golden (from Stage 1 fixture)
        tolerance = kernel_test_config.get_tolerance_rtlsim()
        self.validate_against_golden(
            actual_outputs, golden_outputs, "RTL simulation (rtlsim)", tolerance
        )

