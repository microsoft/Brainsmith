"""Test framework base class with compositional config (v4.0).

This module provides:
1. KernelTestBase_v2: Abstract base class for kernel tests with composition-based config

Design Philosophy:
- Composition over inheritance (config as parameter, not inheritance)
- Pytest fixtures control parameterization
- Tests define operations with symbolic shapes (concrete from fixtures)
- Direct DataType annotations (NO Quant nodes) in v2.3+
- Automatic test data generation with correct types
- Shared utilities for SingleKernelTest and DualKernelTest_v2

Architecture (v4.0):
    KernelTestBase_v2 (abstract base class + shared utilities)
        ↓ inherits
    ┌───────────────────┬──────────────────┐
    SingleKernelTest    DualKernelTest_v2
    (fixture-based)     (attribute-based)

    KernelTestConfig (dataclass from test_config.py)
        ↓ passed as parameter to methods
    Test methods receive config, not inherit from it

Usage:
    from tests.frameworks.kernel_test_base_v2 import KernelTestBase_v2
    from tests.frameworks.single_kernel_test_v2 import SingleKernelTest

    # Define fixtures in test file
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

    # Test class - just define operations
    class TestMyKernel(SingleKernelTest):
        def make_test_model(self, input_shapes):
            # Use shapes from fixture
            inp = helper.make_tensor_value_info(
                "input", TensorProto.FLOAT, input_shapes["input"]
            )
            # ... build graph
            return model, ["input"]

        def get_kernel_inference_transform(self):
            return InferMyKernel

Result: 2 dtypes × 2 shapes = 4 test configurations automatically!
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import pytest
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation

# Import backend specialization utilities
from tests.support.backend_utils import specialize_to_backend

# Import validation utilities
from tests.support.validator import GoldenValidator

# Import compositional config (v4.0)
from tests.frameworks.test_config import KernelTestConfig


class KernelTestBase_v2(ABC):
    """Abstract base class for kernel tests with compositional config (v4.0).

    This base class provides:
    1. Abstract methods that subclasses MUST implement
    2. Shared utilities for SingleKernelTest and DualKernelTest_v2
    3. Configuration hooks for parameterization

    Architecture (v4.0):
        KernelTestBase_v2 (abstract base class) ← THIS CLASS
            ↓ inherits
        ┌───────────────────┬──────────────────┐
        SingleKernelTest    DualKernelTest_v2

        KernelTestConfig (dataclass, passed as parameter)
            ↓ composition (not inheritance)
        Test methods receive config via pytest fixtures

    Shared Utilities:
    1. validate_against_golden(): GoldenValidator-based output validation
    2. _auto_detect_backends(): Registry-based backend auto-detection
    3. _specialize_to_backend_stage(): Stage 2→3 specialization with overrides
    4. auto_configure_from_fixture(): Auto-apply config parameters

    Abstract Methods (subclasses MUST implement):
    1. make_test_model(kernel_test_config): Create test model
    2. get_kernel_inference_transform(): Return transform class

    Optional Configuration Hooks:
    1. configure_kernel_node(op, model): Stage 2 configuration (deprecated)
    2. configure_backend_node(op, model): Stage 3 configuration (deprecated)
    3. configure_parameters(op, model, stage): Unified stage-aware configuration
    4. get_backend_variants(): Backend variant classes
    5. get_test_seed(): Random seed for test data
    """

    # ========================================================================
    # Abstract Methods - Subclasses MUST implement
    # ========================================================================

    @abstractmethod
    def make_test_model(
        self, kernel_test_config: KernelTestConfig
    ) -> Tuple[ModelWrapper, List[str]]:
        """Create test model from unified configuration.

        This method defines the operations to test WITHOUT DataType annotations.
        Framework automatically annotates tensors based on config.input_dtypes.

        Args:
            kernel_test_config: Unified test configuration (v4.0)
                Access via properties:
                - config.input_shapes: Dict[str, Tuple[int, ...]]
                - config.input_dtypes: Dict[str, DataType]
                - config.operation: str (for polymorphic models)
                - config.dse_dimensions: Dict[str, Any]

        Returns:
            (model, input_names):
                - model: ONNX model with operations (NO DataType annotations)
                - input_names: List of input tensor names to annotate

        Example (v4.0):
            def make_test_model(self, kernel_test_config):
                '''Create Add operation with shapes from config.'''
                import onnx.helper as helper
                from onnx import TensorProto

                # Extract from compositional config
                shapes = kernel_test_config.input_shapes

                # Create model with concrete shapes
                inp = helper.make_tensor_value_info(
                    "input", TensorProto.FLOAT, shapes["input"]
                )
                param = helper.make_tensor_value_info(
                    "param", TensorProto.FLOAT, shapes["param"]
                )
                out = helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, shapes["input"]
                )

                node = helper.make_node("Add", ["input", "param"], ["output"])
                graph = helper.make_graph([node], "test", [inp, param], [out])

                from qonnx.util.basic import qonnx_make_model
                model = ModelWrapper(qonnx_make_model(graph))

                return model, ["input", "param"]

        Note:
            - Use TensorProto.FLOAT for all inputs (annotations handle types)
            - Shapes are concrete (from config), not symbolic
            - Do NOT insert Quant nodes (v2.3+ uses direct annotations)
            - Do NOT set QONNX DataType annotations (framework handles)
        """
        pass

    @abstractmethod
    def get_kernel_inference_transform(self) -> Type[Transformation]:
        """Return transform class that converts ONNX node to hardware kernel.

        Returns:
            Transformation class (uninstantiated)

        Example:
            def get_kernel_inference_transform(self):
                from brainsmith.primitives.transforms.infer_kernels import InferKernelList
                return InferKernelList

        Note:
            Return the CLASS, not an instance (no parentheses).
        """
        pass

    # ========================================================================
    # Optional Configuration Hooks
    # ========================================================================

    def get_test_seed(self) -> int:
        """Return random seed for test data generation (optional).

        Returns:
            Random seed integer (default: 42)
        """
        return 42


    def get_backend_variants(self) -> Optional[List[Type]]:
        """Backend variant classes for specialization in priority order.

        Returns:
            List of backend classes to try in priority order.
            Default: None (auto-detect HLS backend based on kernel op_type)
        """
        return None

    # ========================================================================
    # Shared Utility 1: Golden Reference Validation
    # ========================================================================

    def validate_against_golden(
        self,
        actual_outputs: Dict[str, np.ndarray],
        golden_outputs: Dict[str, np.ndarray],
        backend_name: str,
        tolerance: Dict[str, float],
    ) -> None:
        """Validate actual outputs match golden reference.

        Uses GoldenValidator (Phase 1) for consistent validation logic
        across all test frameworks.

        Shared by:
        - SingleKernelTest: 3 execution tests (Python/cppsim/rtlsim)
        - DualKernelTest_v2: 6 execution tests (manual+auto × Python/cppsim/rtlsim)

        Args:
            actual_outputs: Outputs from backend execution
                           Dict mapping tensor names → numpy arrays
            golden_outputs: Expected outputs from golden reference
                           Dict mapping tensor names → numpy arrays
            backend_name: Name of backend for error messages
                         Examples: "Python execution", "HLS cppsim", "RTL rtlsim"
            tolerance: Dict with 'rtol' and 'atol' keys
                      Example: {"rtol": 1e-5, "atol": 1e-6}

        Raises:
            AssertionError: If outputs don't match within tolerance

        Example (SingleKernelTest):
            >>> def test_python_execution_vs_golden(self, test_inputs, golden_outputs):
            ...     op, model = self.run_inference_pipeline(...)
            ...     executor = PythonExecutor()
            ...     actual_outputs = executor.execute(op, model, test_inputs)
            ...     tolerance = self.get_tolerance_python()
            ...     self.validate_against_golden(actual_outputs, golden_outputs,
            ...                                   "Python execution", tolerance)

        Example (DualKernelTest_v2):
            >>> def test_manual_cppsim_vs_golden(self):
            ...     manual_op, manual_model = self.run_manual_pipeline(to_backend=True)
            ...     actual_outputs = cppsim_executor.execute(...)
            ...     tolerance = self.get_tolerance_cppsim()
            ...     self.validate_against_golden(actual_outputs, golden_outputs,
            ...                                   "Manual cppsim", tolerance)
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
    # Shared Utility 2: Backend Auto-Detection
    # ========================================================================

    def _auto_detect_backends(self, op: HWCustomOp) -> List[Type]:
        """Auto-detect backend variants from Brainsmith registry.

        Called when get_backend_variants() returns None (default).
        Looks up HLS backends registered for the kernel's op_type.

        Shared by:
        - SingleKernelTest: Auto-detect for single implementation
        - DualKernelTest_v2: Auto-detect for auto pipeline (Brainsmith)

        Args:
            op: HWCustomOp instance to find backends for

        Returns:
            List of backend classes

        Raises:
            pytest.skip: If no backends found

        Note:
            This uses Brainsmith's registry. For FINN backends in DualKernelTest,
            use get_manual_backend_variants() to specify explicitly.

        Example (SingleKernelTest):
            >>> # Auto-detect backend for AddStreams
            >>> op, model = self.run_inference_pipeline(...)  # Stage 2
            >>> backend_variants = self._auto_detect_backends(op)
            >>> # Returns: [AddStreams_hls]

        Example (DualKernelTest_v2 auto pipeline):
            >>> # Auto-detect Brainsmith backend
            >>> auto_op, auto_model = self.run_auto_pipeline()
            >>> backend_variants = self._auto_detect_backends(auto_op)
            >>> # Used automatically by _specialize_to_backend_stage()
        """
        from brainsmith.registry import get_backend, list_backends_for_kernel

        backend_names = list_backends_for_kernel(op.onnx_node.op_type, language="hls")
        if not backend_names:
            pytest.skip(f"No HLS backend found for {op.onnx_node.op_type}")

        return [get_backend(name) for name in backend_names]

    # ========================================================================
    # Shared Utility 3: Backend Specialization (Stage 2 → Stage 3)
    # ========================================================================

    def _specialize_to_backend_stage(
        self,
        op: HWCustomOp,
        model: ModelWrapper,
        kernel_test_config: "KernelTestConfig",
        backend_variants_override: Optional[List[Type]] = None,
    ) -> Tuple[HWCustomOp, ModelWrapper]:
        """Execute Stage 2 → Stage 3 backend specialization.

        Common logic for transforming base kernel (Stage 2) to backend (Stage 3):
        1. Check fpgapart configured (skip test if None)
        2. Get backend variants (use override, get_backend_variants(), or auto-detect)
        3. Specialize via specialize_to_backend()
        4. Call configure_backend_node() hook

        Used by:
        - SingleKernelTest.run_inference_pipeline(to_backend=True)
        - DualKernelTest_v2.run_manual_pipeline(to_backend=True)
        - DualKernelTest_v2.run_auto_pipeline(to_backend=True)

        Args:
            op: Base kernel operator instance (Stage 2)
                Example: AddStreams (no backend)
            model: Model containing the base kernel
            kernel_test_config: Unified test configuration (v3.0, required)
                Contains fpgapart for backend specialization
            backend_variants_override: Optional backend list (for manual pipeline)
                                      If None, uses priority:
                                      1. get_backend_variants() (if overridden)
                                      2. _auto_detect_backends() (registry lookup)

        Returns:
            (specialized_op, specialized_model): Backend operator and model (Stage 3)
            Example: (AddStreams_hls, model_with_backend)

        Raises:
            pytest.skip: If backend not configured (fpgapart is None)

        Example (SingleKernelTest):
            >>> # Stage 2 → Stage 3 (auto-detect backend)
            >>> def run_inference_pipeline(self, ..., kernel_test_config, to_backend=False):
            ...     # Stage 1 → Stage 2
            ...     op, model = runner.run(...)
            ...     # Stage 2 → Stage 3
            ...     if to_backend:
            ...         op, model = self._specialize_to_backend_stage(
            ...             op, model, kernel_test_config
            ...         )
            ...     return op, model

        Example (DualKernelTest_v2 manual pipeline):
            >>> # Stage 2 → Stage 3 (FINN backend override)
            >>> def run_manual_pipeline(self, kernel_test_config, to_backend=False):
            ...     # Stage 1 → Stage 2
            ...     manual_op, manual_model = runner.run(...)
            ...     # Stage 2 → Stage 3 (override with FINN backend)
            ...     if to_backend:
            ...         manual_op, manual_model = self._specialize_to_backend_stage(
            ...             manual_op, manual_model, kernel_test_config,
            ...             backend_variants_override=self.get_manual_backend_variants()
            ...         )
            ...     return manual_op, manual_model

        Example (DualKernelTest_v2 auto pipeline):
            >>> # Stage 2 → Stage 3 (auto-detect Brainsmith backend)
            >>> def run_auto_pipeline(self, kernel_test_config, to_backend=False):
            ...     # Stage 1 → Stage 2
            ...     auto_op, auto_model = runner.run(...)
            ...     # Stage 2 → Stage 3 (auto-detect)
            ...     if to_backend:
            ...         auto_op, auto_model = self._specialize_to_backend_stage(
            ...             auto_op, auto_model, kernel_test_config
            ...         )
            ...     return auto_op, auto_model
        """
        fpgapart = kernel_test_config.get_fpgapart()
        if fpgapart is None:
            pytest.skip(
                "Backend testing skipped (no FPGA part configured for this test)"
            )

        # Determine backend variants (priority: override > get_backend_variants > auto-detect)
        backend_variants = backend_variants_override
        if backend_variants is None:
            backend_variants = self.get_backend_variants()
        if backend_variants is None:
            backend_variants = self._auto_detect_backends(op)

        # Specialize to backend (Stage 2 → Stage 3)
        op, model = specialize_to_backend(op, model, fpgapart, backend_variants)

        return op, model

    # ========================================================================
    # Shared Utility 4: Stage-Aware Parameter Configuration (v2.4)
    # ========================================================================

    def get_stage(self, op) -> int:
        """Detect pipeline stage from backend attribute.

        Args:
            op: Operator instance (HWCustomOp or KernelOp)

        Returns:
            2: Stage 2 (kernel, unspecialized - backend="fpgadataflow")
            3: Stage 3 (backend, specialized to HLS or RTL)
        """
        backend = op.get_nodeattr("backend")
        return 3 if backend in ("hls", "rtl") else 2

    def is_brainsmith(self, op) -> bool:
        """Check if node is Brainsmith KernelOp.

        Args:
            op: Operator instance

        Returns:
            True if op has kernel_schema attribute (Brainsmith KernelOp)
            False if FINN HWCustomOp only
        """
        return hasattr(op, 'kernel_schema')

    def configure_parameters(self, op, model: ModelWrapper, stage: int):
        """Configure node parameters at any stage (NEW HOOK in v2.4).

        Called twice during pipeline:
          1. After kernel inference (stage=2)
          2. After backend specialization (stage=3, if enabled)

        Args:
            op: Operator instance (KernelOp or HWCustomOp)
            model: ModelWrapper
            stage: Pipeline stage (2=kernel, 3=backend)

        Use native APIs directly:
            • op.set_nodeattr(name, value) - Structural/runtime params
            • op.design_point.with_dimension() - Brainsmith DSE (multi-param)

        Detection helpers:
            • self.is_brainsmith(op) - Check if Brainsmith KernelOp
            • self.get_stage(op) - Get current stage (2 or 3)

        Example:
            def configure_parameters(self, op, model, stage):
                if stage == 2:
                    # Stage 2: Structural params
                    op.set_nodeattr("input0Datatype", "INT8")

                elif stage == 3:
                    # Stage 3: DSE params
                    if self.is_brainsmith(op):
                        # Multiple params: chain for efficiency
                        point = op.design_point \\
                            .with_dimension("PE", 32) \\
                            .with_dimension("SIMD", 16)
                        op.apply_design_point(point)
                    else:
                        # FINN
                        op.set_nodeattr("PE", 32)
        """
        pass

    # ========================================================================
    # Shared Utility 5: Auto-Configuration from Fixture (v2.5)
    # ========================================================================

    def auto_configure_from_fixture(
        self,
        op,
        model: ModelWrapper,
        stage: int,
        config: KernelTestConfig
    ) -> None:
        """Auto-apply configuration from unified fixture (v2.5).

        This method enables declarative test configuration via the KernelTestConfig
        dataclass. Instead of writing imperative configure_parameters() code, test
        parameters are specified in pytest fixtures and auto-applied by the framework.

        Args:
            op: Operator instance (KernelOp or HWCustomOp)
            model: ModelWrapper
            stage: Pipeline stage (2=kernel, 3=backend)
            config: KernelTestConfig with declarative parameters

        Stage-Aware Application:
            Stage 2 (Kernel):
                - input_streams: Applied via with_input_stream() (semantic API)
                - output_streams: Applied via with_output_stream() (semantic API)
                - dse_dimensions: Generic DSE params (SIMD, etc.)

            Stage 3 (Backend):
                - All Stage 2 params still available
                - Backend-specific params in dse_dimensions (mem_mode, ram_style)

        API Selection:
            - Stream dimensions: with_input_stream() / with_output_stream()
              * Kernel-agnostic (param name auto-resolved from interface)
              * More semantic and future-proof
            - Other dimensions: with_dimension()
              * Generic API for any DSE parameter

        Tolerances and Build Config:
            - Not applied here (used by test execution methods)
            - Access via config.get_tolerance(), config.fpgapart, etc.

        Example:
            # In test class
            @pytest.fixture(params=[
                KernelTestConfig(
                    operation="Add",
                    input_shapes={"input": (1, 64), "param": (1, 64)},
                    input_dtypes={"input": DataType["INT8"], "param": DataType["INT8"]},
                    input_streams={0: 8},       # PE=8 (semantic API, Stage 2+)
                    dse_dimensions={"SIMD": 16}, # Generic dimension
                    fpgapart="xc7z020clg400-1",
                )
            ])
            def kernel_test_config(request):
                return request.param

            class TestMyKernel(SingleKernelTest):
                def make_test_model(self, kernel_test_config):
                    # Framework auto-applies configuration!
                    pass

        Backward Compatibility:
            - If config is None, does nothing (v2.4 tests still work)
            - configure_parameters() still called for imperative overrides
        """
        if config is None:
            return  # Backward compatibility: v2.4 tests don't use config

        # FINN nodes: use direct nodeattr setting
        if not self.is_brainsmith(op):
            self._apply_finn_config(op, config, stage)
            return

        # Brainsmith nodes: use semantic DSE API
        op._ensure_ready(model)
        point = op.design_point

        # Apply stream dimensions (semantic API) - available at Stage 2+
        if stage >= 2:
            # Input stream parallelism (e.g., PE)
            if config.input_streams:
                for idx, value in config.input_streams.items():
                    point = point.with_input_stream(idx, value)

            # Output stream parallelism
            if config.output_streams:
                for idx, value in config.output_streams.items():
                    point = point.with_output_stream(idx, value)

        # Apply other DSE dimensions (generic API)
        if config.dse_dimensions:
            for name, value in config.dse_dimensions.items():
                point = point.with_dimension(name, value)

        # Apply the configured design point
        op.apply_design_point(point)

    def _apply_finn_config(self, op, config: KernelTestConfig, stage: int):
        """Apply configuration to FINN HWCustomOp node.

        FINN nodes use direct nodeattr mutation instead of design points.

        Args:
            op: FINN HWCustomOp instance
            config: KernelTestConfig
            stage: Pipeline stage
        """
        # Stream dimensions - FINN typically uses PE/SIMD directly
        if stage >= 2:
            if config.input_streams:
                # FINN convention: first input stream → PE
                if 0 in config.input_streams:
                    op.set_nodeattr("PE", config.input_streams[0])

            if config.output_streams:
                # FINN convention: first output stream → output PE
                if 0 in config.output_streams:
                    # Some FINN nodes have separate output parallelism
                    try:
                        op.set_nodeattr("output_PE", config.output_streams[0])
                    except (AttributeError, KeyError):
                        pass  # Not all FINN nodes have output_PE

        # Other DSE dimensions
        if config.dse_dimensions:
            for name, value in config.dse_dimensions.items():
                try:
                    op.set_nodeattr(name, value)
                except (AttributeError, KeyError):
                    # Skip params that don't exist on this node
                    pass
