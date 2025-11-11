"""Test framework for single kernel implementation.

Provides fixture-based testing of one kernel implementation against golden reference.

Provides 6 inherited tests per configuration:
- 3 pipeline validation tests (HW node creation, shapes, datatypes)
- 3 execution tests (Python, cppsim, rtlsim) vs golden reference

Subclasses must implement:
- make_test_model(): Create ONNX model to test
- get_kernel_op(): Return kernel class to test

Framework handles: DataType annotations, test data generation, golden reference,
backend specialization, and validation.
"""


import numpy as np
import pytest
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from qonnx.core.modelwrapper import ModelWrapper

from brainsmith.primitives.transforms.infer_kernels import InferKernels
from tests.frameworks.kernel_test_base import KernelTestBase
from tests.support.pipeline import PipelineRunner


class KernelTest(KernelTestBase):
    """Test single kernel implementation against golden reference.

    Subclasses must implement:
    - make_test_model(kernel_test_config): Create ONNX model
    - get_kernel_op(): Return kernel class

    Provides 6 inherited tests: 3 pipeline validation + 3 execution (Python/cppsim/rtlsim)
    """

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
            return self._build_stage1_model(kernel_test_config)

        return model_cache.get_stage1_model(kernel_test_config.test_id, builder)

    @pytest.fixture(scope="function")
    def test_inputs(
        self, kernel_test_config: "KernelTestConfig", model_cache
    ) -> dict[str, np.ndarray]:
        """Generate test inputs with deterministic seed.

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
            return self._build_test_inputs(kernel_test_config)

        return model_cache.get_test_inputs(kernel_test_config.test_id, builder)

    @pytest.fixture(scope="function")
    def golden_outputs(
        self,
        kernel_test_config: "KernelTestConfig",
        stage1_model: ModelWrapper,
        test_inputs: dict[str, np.ndarray],
        model_cache,
    ) -> dict[str, np.ndarray]:
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
            return self._build_golden_outputs(stage1_model, test_inputs)

        return model_cache.get_golden_reference(kernel_test_config.test_id, builder)

    @pytest.fixture(scope="function")
    def stage2_model(
        self,
        kernel_test_config: "KernelTestConfig",
        stage1_model: ModelWrapper,
        model_cache,
    ) -> tuple:
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
            runner = PipelineRunner()

            def configure_stage_2(op, m):
                # Apply declarative configuration from fixture
                self.auto_configure_from_fixture(
                    op, m, stage=2, config=kernel_test_config
                )

            # Create kernel inference transform from kernel op
            transform = InferKernels([self.get_kernel_op()])

            op, model = runner.run(
                model_factory=lambda: (model, target_node),
                transform=transform,
                configure_fn=configure_stage_2,
            )

            return op, model

        return model_cache.get_stage2_model(kernel_test_config.test_id, builder)

    @pytest.fixture(scope="function")
    def stage3_model(
        self,
        kernel_test_config: "KernelTestConfig",
        stage2_model: tuple,
        model_cache,
    ) -> tuple:
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

        Note:
            Use pytest marks (@pytest.mark.cppsim, @pytest.mark.rtlsim) to control
            which backend tests run. Configure fpgapart in kernel_test_config.platform.
        """
        fpgapart = kernel_test_config.fpgapart

        def builder():
            # Reuse cached Stage 2 model
            base_op, base_model = stage2_model

            # Stage 2 → Stage 3: Base Kernel → Backend
            op, model = self.specialize_to_backend(
                base_op, base_model, kernel_test_config
            )

            # Apply declarative configuration from fixture
            self.auto_configure_from_fixture(op, model, stage=3, config=kernel_test_config)

            return op, model

        return model_cache.get_stage3_model(kernel_test_config.test_id, fpgapart, builder)

    @pytest.mark.pipeline
    @pytest.mark.kernel
    def test_pipeline_creates_hw_node(
        self, kernel_test_config: "KernelTestConfig", stage2_model: tuple
    ):
        """Validate that kernel inference creates hardware node.

        Args:
            kernel_test_config: Unified test configuration
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
    @pytest.mark.kernel
    def test_shapes_preserved_through_pipeline(
        self, kernel_test_config: "KernelTestConfig", stage2_model: tuple
    ):
        """Validate tensor shapes remain correct through inference pipeline.

        Args:
            kernel_test_config: Unified test configuration
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
    @pytest.mark.kernel
    def test_datatypes_preserved_through_pipeline(
        self, kernel_test_config: "KernelTestConfig", stage2_model: tuple
    ):
        """Validate tensor datatypes remain correct through inference pipeline.

        Args:
            kernel_test_config: Unified test configuration
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
    @pytest.mark.kernel
    def test_python_execution_vs_golden(
        self,
        kernel_test_config: "KernelTestConfig",
        stage2_model: tuple,
        test_inputs: dict[str, np.ndarray],
        golden_outputs: dict[str, np.ndarray],
    ):
        """Test Python execution matches QONNX golden reference."""
        # Delegate to shared utility
        self._execute_and_validate_golden(
            stage2_model, test_inputs, golden_outputs,
            "python", "Python execution", kernel_test_config
        )

    @pytest.mark.pipeline
    @pytest.mark.golden
    @pytest.mark.cppsim
    @pytest.mark.slow
    @pytest.mark.kernel
    def test_cppsim_execution_vs_golden(
        self,
        kernel_test_config: "KernelTestConfig",
        stage3_model: tuple,
        test_inputs: dict[str, np.ndarray],
        golden_outputs: dict[str, np.ndarray],
    ):
        """Test HLS C++ simulation matches QONNX golden reference."""
        # Delegate to shared utility
        self._execute_and_validate_golden(
            stage3_model, test_inputs, golden_outputs,
            "cppsim", "HLS simulation (cppsim)", kernel_test_config
        )

    @pytest.mark.pipeline
    @pytest.mark.golden
    @pytest.mark.rtlsim
    @pytest.mark.slow
    @pytest.mark.kernel
    def test_rtlsim_execution_vs_golden(
        self,
        kernel_test_config: "KernelTestConfig",
        stage3_model: tuple,
        test_inputs: dict[str, np.ndarray],
        golden_outputs: dict[str, np.ndarray],
    ):
        """Test RTL simulation matches QONNX golden reference."""
        # Delegate to shared utility
        self._execute_and_validate_golden(
            stage3_model, test_inputs, golden_outputs,
            "rtlsim", "RTL simulation (rtlsim)", kernel_test_config
        )

