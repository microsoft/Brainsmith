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
    # Stage 2: Kernel Inference (v5.0 - Flexible API)
    # ========================================================================

    def get_kernel_op(self) -> Type:
        """Return kernel operator class for default inference (optional).

        Optional method used by simple Brainsmith test cases.
        Not required if overriding infer_kernel() directly.

        Returns:
            Kernel operator class (e.g., ElementwiseBinary, AddStreams)

        Example:
            def get_kernel_op(self):
                from brainsmith.kernels.elementwise_binary import ElementwiseBinary
                return ElementwiseBinary

        Raises:
            NotImplementedError: If not overridden and default inference used

        Note:
            - Not abstract (optional)
            - Used by simple cases with single kernel class
            - Override infer_kernel() for custom inference logic
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_kernel_op() "
            "or override infer_kernel() with custom inference logic"
        )

    def infer_kernel(
        self,
        model: ModelWrapper,
        target_node: str,
    ) -> Tuple[HWCustomOp, ModelWrapper]:
        """Execute Stage 1 → Stage 2 kernel inference (v5.0).

        Default implementation uses get_kernel_inference_transform() for
        backward compatibility with existing tests.

        Override for custom inference logic (multiple transforms, conditional, etc.).

        Args:
            model: Stage 1 model (ONNX nodes)
            target_node: Name of target ONNX node to transform

        Returns:
            (op, model): Kernel operator instance and transformed model

        Default behavior:
            1. Call get_kernel_inference_transform() to get transform class
            2. Apply transform
            3. Find and return transformed node

        Override for:
            - Multiple transforms: Apply several transforms in sequence
            - Conditional logic: Different transforms based on config
            - Transform arguments: Transforms with custom params
            - FINN inference: Use FINN-specific transforms

        Example (default - uses get_kernel_inference_transform):
            # Uses default implementation automatically
            def get_kernel_inference_transform(self):
                return InferKernelList

        Example (override - multiple transforms):
            def infer_kernel(self, model, target_node):
                # Apply inference
                transform = self.get_kernel_inference_transform()
                model = model.transform(transform())

                # Apply additional transforms
                model = model.transform(SomeOtherTransform())

                # Find and return node
                op = self._find_hw_node(model, target_node)
                return op, model

        Example (override - FINN manual):
            def infer_kernel(self, model, target_node):
                from finn.transformation.fpgadataflow.infer_addstreams import InferAddStreams
                model = model.transform(InferAddStreams())

                # Find FINN node by op_type
                op = self._find_hw_node(model, target_node, expected_type="AddStreams")
                return op, model
        """
        # Get transform class from subclass (backward compatible)
        transform_class = self.get_kernel_inference_transform()

        # Apply inference transform
        model = model.transform(transform_class())

        # Find transformed node
        op = self._find_hw_node(model, target_node)

        return op, model

    # ========================================================================
    # Shared Helper Methods (Extracted from SingleKernelTest/KernelParityTest)
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
        input_annotations = {
            name: input_datatypes[name] for name in input_names if name in input_datatypes
        }
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
        self, kernel_test_config: "KernelTestConfig"
    ) -> Dict[str, np.ndarray]:
        """Generate test data with correct shapes and datatypes.

        This is an internal helper. Tests should not call directly.

        Args:
            kernel_test_config: Unified test configuration

        Returns:
            Dict mapping input names to test data arrays
        """
        from tests.fixtures.test_data import generate_test_data

        inputs = {}
        _, input_names = self.make_test_model(kernel_test_config)

        input_shapes = kernel_test_config.input_shapes
        input_datatypes = kernel_test_config.input_dtypes

        for name in input_names:
            if name not in input_datatypes:
                continue

            dtype = input_datatypes[name]
            shape = input_shapes[name]

            inputs[name] = generate_test_data(dtype, shape, seed=self.get_test_seed())

        return inputs

    def _find_hw_node(
        self,
        model: ModelWrapper,
        target_node: str,
        expected_type=None,
    ) -> HWCustomOp:
        """Find HW node after kernel inference (v5.0 helper).

        Helper method to locate the transformed kernel operator.

        Args:
            model: Model after inference transform
            target_node: Original ONNX node name
            expected_type: Expected kernel class or op_type string
                          If class: checks isinstance()
                          If str: checks op.onnx_node.op_type
                          If None: returns any HWCustomOp/KernelOp found

        Returns:
            Kernel operator instance

        Raises:
            AssertionError: If node not found or wrong type

        Example:
            op = self._find_hw_node(model, "Add_0", ElementwiseBinary)
            op = self._find_hw_node(model, "Add_0", expected_type="AddStreams")
        """
        # Get ONNX node from model
        onnx_node = model.get_node_from_name(target_node)

        # Wrap with custom op class
        from qonnx.custom_op.registry import getCustomOp
        op = getCustomOp(onnx_node)

        # Verify it's a HW node
        assert isinstance(op, HWCustomOp), (
            f"Node {target_node} is not a hardware operator (found {type(op).__name__})"
        )

        # Type checking if expected_type provided
        if expected_type is not None:
            if isinstance(expected_type, type):
                # Class check
                assert isinstance(op, expected_type), (
                    f"Node {target_node} is {type(op).__name__}, expected {expected_type.__name__}"
                )
            elif isinstance(expected_type, str):
                # op_type string check
                assert op.onnx_node.op_type == expected_type, (
                    f"Node {target_node} has op_type {op.onnx_node.op_type}, expected {expected_type}"
                )

        return op

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

    def _execute_and_validate_golden(
        self,
        stage_model: Tuple[HWCustomOp, ModelWrapper],
        test_inputs: Dict[str, np.ndarray],
        golden_outputs: Dict[str, np.ndarray],
        execution_mode: str,
        backend_label: str,
        config: KernelTestConfig,
    ) -> None:
        """Execute kernel and validate against golden reference (v5.0).

        Common logic shared by:
        - SingleKernelTest: 3 golden tests (python/cppsim/rtlsim)
        - KernelParityTest: 6 golden tests (kernel_a + kernel_b × 3 modes)

        This method eliminates 74% code duplication (180 lines → 47 lines).

        Args:
            stage_model: (op, model) tuple from fixture
            test_inputs: Test data
            golden_outputs: Expected outputs
            execution_mode: "python", "cppsim", or "rtlsim"
            backend_label: Human-readable name for error messages
                          Examples: "Python execution", "Kernel A cppsim", "FINN rtlsim"
            config: Test configuration with tolerances

        Raises:
            AssertionError: If outputs don't match golden within tolerance
            ValueError: If invalid execution_mode

        Example (SingleKernelTest):
            def test_python_execution_vs_golden(self, ...):
                self._execute_and_validate_golden(
                    stage2_model, test_inputs, golden_outputs,
                    "python", "Python execution", kernel_test_config
                )

        Example (KernelParityTest):
            def test_kernel_a_cppsim_vs_golden(self, ...):
                self._execute_and_validate_golden(
                    stage3_model_a, test_inputs, golden_outputs,
                    "cppsim", "Kernel A cppsim", kernel_test_config
                )
        """
        from tests.support.executors import PythonExecutor, CppSimExecutor, RTLSimExecutor

        op, model = stage_model

        # Select executor based on mode
        if execution_mode == "python":
            executor = PythonExecutor()
        elif execution_mode == "cppsim":
            executor = CppSimExecutor()
        elif execution_mode == "rtlsim":
            executor = RTLSimExecutor()
        else:
            raise ValueError(f"Invalid execution_mode: {execution_mode}")

        # Execute kernel
        actual_outputs = executor.execute(op, model, test_inputs)

        # Get tolerance based on execution mode
        if execution_mode == "python":
            tolerance = config.get_tolerance_python()
        elif execution_mode == "cppsim":
            tolerance = config.get_tolerance_cppsim()
        elif execution_mode == "rtlsim":
            tolerance = config.get_tolerance_rtlsim()
        else:
            raise ValueError(f"Invalid execution_mode: {execution_mode}")

        # Validate against golden
        self.validate_against_golden(
            actual_outputs,
            golden_outputs,
            backend_label,
            tolerance
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
    # Shared Utility 3: Backend Specialization (Stage 2 → Stage 3) - v5.0
    # ========================================================================

    def specialize_to_backend(
        self,
        op: HWCustomOp,
        model: ModelWrapper,
        config: KernelTestConfig,
        backend_variants_override: Optional[List[Type]] = None,
    ) -> Tuple[HWCustomOp, ModelWrapper]:
        """Execute Stage 2 → Stage 3 backend specialization (v5.0 - public).

        Default implementation: auto-detect backends from registry.
        Override for custom specialization logic.

        Common logic for transforming base kernel (Stage 2) to backend (Stage 3):
        1. Check fpgapart configured (skip test if None)
        2. Get backend variants (use override, get_backend_variants(), or auto-detect)
        3. Specialize via support utility
        4. Return specialized op and model

        Used by:
        - SingleKernelTest stage3_model fixture
        - KernelParityTest stage3_model_a/b fixtures
        - Can be overridden for custom specialization

        Args:
            op: Base kernel operator instance (Stage 2)
                Example: AddStreams (no backend)
            model: Model containing the base kernel
            config: Test configuration (contains fpgapart)
            backend_variants_override: Optional backend list (for parity testing)
                                      If None, uses priority:
                                      1. get_backend_variants() (if overridden)
                                      2. _auto_detect_backends() (registry lookup)

        Returns:
            (specialized_op, specialized_model): Backend operator and model (Stage 3)
            Example: (AddStreams_hls, model_with_backend)

        Note:
            Use pytest marks (@pytest.mark.cppsim, @pytest.mark.rtlsim) to control
            which backend tests run. Configure fpgapart in kernel_test_config.platform.
            If fpgapart is None, specialization will fail with a clear error.

        Example (default - auto-detect):
            # Uses default implementation automatically
            def get_backend_variants(self):
                return None  # Auto-detect

        Example (override - explicit backends):
            def get_backend_variants(self):
                return [MyKernel_hls]

        Example (override - custom specialization):
            def specialize_to_backend(self, op, model, config, backend_variants_override=None):
                # Custom pre-processing
                self._prepare_for_backend(op, model)

                # Use default specialization
                op, model = super().specialize_to_backend(op, model, config, backend_variants_override)

                # Custom post-processing
                self._finalize_backend(op, model)

                return op, model

        Backward compatibility:
            Old code using _specialize_to_backend_stage() will continue to work
            via alias maintained below.
        """
        fpgapart = config.get_fpgapart()
        if fpgapart is None:
            raise ValueError(
                "Backend specialization requires fpgapart to be configured. "
                "Set platform=PlatformConfig(fpgapart='...') in your kernel_test_config."
            )

        # Determine backend variants (priority: override > get_backend_variants > auto-detect)
        backend_variants = backend_variants_override
        if backend_variants is None:
            backend_variants = self.get_backend_variants()
        if backend_variants is None:
            backend_variants = self._auto_detect_backends(op)

        # Specialize to backend (Stage 2 → Stage 3)
        from tests.support.backend_utils import specialize_to_backend as do_specialize
        op, model = do_specialize(op, model, fpgapart, backend_variants)

        return op, model

    def _specialize_to_backend_stage(
        self,
        op: HWCustomOp,
        model: ModelWrapper,
        kernel_test_config: "KernelTestConfig",
        backend_variants_override: Optional[List[Type]] = None,
    ) -> Tuple[HWCustomOp, ModelWrapper]:
        """Backward compatibility alias for specialize_to_backend().

        DEPRECATED: Use specialize_to_backend() instead.
        This method maintained for backward compatibility with existing code.
        """
        return self.specialize_to_backend(op, model, kernel_test_config, backend_variants_override)

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
