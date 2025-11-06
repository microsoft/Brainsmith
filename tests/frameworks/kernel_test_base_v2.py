"""Test configuration and shared utilities for v2.3 frameworks.

This module provides:
1. KernelTestConfig: Minimal abstract base class for fixture-based kernel tests (v2.0+)
2. KernelTestBase_v2: Shared utilities for v2.3 frameworks (NEW in Phase 2)

Design Philosophy:
- Pytest fixtures control parameterization (NOT test methods)
- Tests define operations with symbolic shapes (concrete from fixtures)
- Direct DataType annotations (NO Quant nodes) in v2.3
- Automatic test data generation with correct types
- Shared utilities extracted for DualKernelTest_v2 reuse

Architecture (v2.3):
    KernelTestConfig (abstract interface)
        ↓
    KernelTestBase_v2 (shared utilities) ← NEW
        ↓
    ┌───────────────────┬──────────────────┐
    SingleKernelTest    DualKernelTest_v2
    (fixture-based)     (attribute-based)

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


class KernelTestConfig(ABC):
    """Minimal configuration interface for fixture-based kernel tests (v2.0+).

    Subclasses implement:
    - make_test_model(input_shapes): Create model with symbolic shapes
    - get_kernel_inference_transform(): Return transform class

    Optional overrides:
    - configure_kernel_node(op, model): Configure PE, SIMD, etc.
    - get_tolerance_*(): Validation tolerances
    - get_backend_fpgapart(): Enable backend testing

    Pytest fixtures provide:
    - input_shapes: Dict[str, Tuple[int, ...]] (concrete shapes)
    - input_datatypes: Dict[str, DataType] (concrete types)

    Framework handles:
    - Automatic DataType annotations (v2.3: direct annotations, no Quant nodes)
    - Automatic test data generation (using Phase 1 utilities)
    - Automatic golden reference (QONNX execution on annotated model)
    """

    # ========================================================================
    # Abstract Methods - Subclasses MUST implement (2 only!)
    # ========================================================================

    @abstractmethod
    def make_test_model(
        self, input_shapes: Dict[str, Tuple[int, ...]]
    ) -> Tuple[ModelWrapper, List[str]]:
        """Create test model with concrete shapes from fixture.

        This method defines the operations to test WITHOUT DataType annotations.
        Framework automatically annotates tensors based on input_datatypes fixture.

        Args:
            input_shapes: Dict mapping input names to concrete shapes from fixture.
                         Example: {"input": (1, 64), "param": (64,)}

        Returns:
            (model, input_names):
                - model: ONNX model with operations (NO DataType annotations)
                - input_names: List of input tensor names to annotate

        Example:
            def make_test_model(self, input_shapes):
                '''Create Add operation with shapes from fixture.'''
                import onnx.helper as helper
                from onnx import TensorProto

                # Use concrete shapes from fixture
                inp = helper.make_tensor_value_info(
                    "input", TensorProto.FLOAT, input_shapes["input"]
                )
                param = helper.make_tensor_value_info(
                    "param", TensorProto.FLOAT, input_shapes["param"]
                )
                out = helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, input_shapes["input"]
                )

                node = helper.make_node("Add", ["input", "param"], ["output"])
                graph = helper.make_graph([node], "test", [inp, param], [out])

                from qonnx.util.basic import qonnx_make_model
                model = ModelWrapper(qonnx_make_model(graph))

                # Framework annotates these inputs automatically
                return model, ["input", "param"]

        Note:
            - Use TensorProto.FLOAT for all inputs (annotations handle types)
            - Shapes are concrete (from fixture), not symbolic
            - Do NOT insert Quant nodes (v2.3 uses direct annotations)
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

    def configure_kernel_node(self, op: HWCustomOp, model: ModelWrapper) -> None:
        """Configure kernel node after inference (Stage 2, optional).

        Override to set dimension parameters (PE, SIMD, folding factors, etc.).

        Args:
            op: Base kernel operator instance (e.g., AddStreams, MVAU)
            model: Model containing the operator

        Default:
            No configuration (empty implementation)
        """
        pass

    def configure_backend_node(self, op: HWCustomOp, model: ModelWrapper) -> None:
        """Configure backend node after specialization (Stage 3, optional).

        Override to set backend-specific parameters (memory mode, RTL pragmas, etc.).

        Args:
            op: Backend operator instance (e.g., AddStreams_hls, LayerNorm_rtl)
            model: Model containing the operator

        Default:
            No configuration (empty implementation)
        """
        pass

    def get_test_seed(self) -> int:
        """Return random seed for test data generation (optional).

        Returns:
            Random seed integer (default: 42)
        """
        return 42

    def get_tolerance_python(self) -> Dict[str, float]:
        """Tolerance for Python execution vs golden reference.

        Returns:
            Dict with 'rtol' and 'atol' keys for np.allclose()

        Default:
            rtol=1e-7, atol=1e-9 (very tight)
        """
        return {"rtol": 1e-7, "atol": 1e-9}

    def get_tolerance_cppsim(self) -> Dict[str, float]:
        """Tolerance for C++ simulation vs golden reference.

        Returns:
            Dict with 'rtol' and 'atol' keys for np.allclose()

        Default:
            rtol=1e-5, atol=1e-6 (moderate)
        """
        return {"rtol": 1e-5, "atol": 1e-6}

    def get_tolerance_rtlsim(self) -> Dict[str, float]:
        """Tolerance for RTL simulation vs golden reference.

        Returns:
            Dict with 'rtol' and 'atol' keys for np.allclose()

        Default:
            Same as cppsim
        """
        return self.get_tolerance_cppsim()

    def get_backend_fpgapart(self) -> Optional[str]:
        """FPGA part for backend specialization (optional).

        Override to enable backend specialization (Stage 2 → Stage 3).

        Returns:
            str: FPGA part string (e.g., "xc7z020clg400-1")
            None: Backend specialization disabled (default)
        """
        return None

    def get_backend_variants(self) -> Optional[List[Type]]:
        """Backend variant classes for specialization in priority order.

        Returns:
            List of backend classes to try in priority order.
            Default: None (auto-detect HLS backend based on kernel op_type)
        """
        return None


class KernelTestBase_v2(KernelTestConfig):
    """Shared utilities for v2.3 test frameworks.

    This intermediate base class provides common utilities shared by both
    SingleKernelTest and DualKernelTest_v2:

    1. validate_against_golden(): GoldenValidator-based output validation
    2. _auto_detect_backends(): Registry-based backend auto-detection
    3. _specialize_to_backend_stage(): Stage 2→3 specialization with overrides

    Inheritance Chain:
        KernelTestConfig (abstract interface)
            ↓ inherits
        KernelTestBase_v2 (shared utilities) ← THIS CLASS
            ↓ inherits
        SingleKernelTest / DualKernelTest_v2 (framework-specific)
    """

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
            backend_variants_override: Optional backend list (for manual pipeline)
                                      If None, uses priority:
                                      1. get_backend_variants() (if overridden)
                                      2. _auto_detect_backends() (registry lookup)

        Returns:
            (specialized_op, specialized_model): Backend operator and model (Stage 3)
            Example: (AddStreams_hls, model_with_backend)

        Raises:
            pytest.skip: If backend not configured (get_backend_fpgapart() returns None)

        Example (SingleKernelTest):
            >>> # Stage 2 → Stage 3 (auto-detect backend)
            >>> def run_inference_pipeline(self, ..., to_backend=False):
            ...     # Stage 1 → Stage 2
            ...     op, model = runner.run(...)
            ...     # Stage 2 → Stage 3
            ...     if to_backend:
            ...         op, model = self._specialize_to_backend_stage(op, model)
            ...     return op, model

        Example (DualKernelTest_v2 manual pipeline):
            >>> # Stage 2 → Stage 3 (FINN backend override)
            >>> def run_manual_pipeline(self, to_backend=False):
            ...     # Stage 1 → Stage 2
            ...     manual_op, manual_model = runner.run(...)
            ...     # Stage 2 → Stage 3 (override with FINN backend)
            ...     if to_backend:
            ...         manual_op, manual_model = self._specialize_to_backend_stage(
            ...             manual_op, manual_model,
            ...             backend_variants_override=self.get_manual_backend_variants()
            ...         )
            ...     return manual_op, manual_model

        Example (DualKernelTest_v2 auto pipeline):
            >>> # Stage 2 → Stage 3 (auto-detect Brainsmith backend)
            >>> def run_auto_pipeline(self, to_backend=False):
            ...     # Stage 1 → Stage 2
            ...     auto_op, auto_model = runner.run(...)
            ...     # Stage 2 → Stage 3 (auto-detect)
            ...     if to_backend:
            ...         auto_op, auto_model = self._specialize_to_backend_stage(
            ...             auto_op, auto_model
            ...         )
            ...     return auto_op, auto_model
        """
        fpgapart = self.get_backend_fpgapart()
        if fpgapart is None:
            pytest.skip(
                "Backend specialization not configured. "
                "Override get_backend_fpgapart() to enable backend testing."
            )

        # Determine backend variants (priority: override > get_backend_variants > auto-detect)
        backend_variants = backend_variants_override
        if backend_variants is None:
            backend_variants = self.get_backend_variants()
        if backend_variants is None:
            backend_variants = self._auto_detect_backends(op)

        # Specialize to backend (Stage 2 → Stage 3)
        op, model = specialize_to_backend(op, model, fpgapart, backend_variants)

        # Stage 3 configuration hook (backend-specific parameters)
        # Example: op.set_nodeattr("mem_mode", "internal_decoupled")
        self.configure_backend_node(op, model)

        return op, model
