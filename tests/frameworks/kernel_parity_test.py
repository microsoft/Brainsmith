"""KernelParityTest framework for comparing two kernel implementations (v5.0).

This module provides KernelParityTest, a fixture-driven test framework for comparing
two kernel implementations (e.g., FINN vs Brainsmith, version A vs B, platform A vs B).

Design Philosophy (v5.0):
- Generalizes DualKernelTest_v2 "manual" vs "auto" to "kernel_a" vs "kernel_b"
- Leverages Phase 1 v5.0 shared utilities (74% code reduction in golden tests)
- Per-kernel methods enable 90/10 configuration rule (90% shared, 10% custom)
- Consistent override pattern across all 4 pipeline stages
- Maximum code reuse from SingleKernelTest and KernelTestBase_v2

Inheritance Chain (v5.0):
    KernelTestConfig (abstract interface)
        ↓
    KernelTestBase_v2 (v5.0 shared utilities)
        ↓
    KernelParityTest (dual-kernel parity testing) ← THIS CLASS

Inherited from KernelTestBase_v2 (v5.0):
- infer_kernel() - Default implementation for kernel_b
- specialize_to_backend() - Default backend specialization
- _execute_and_validate_golden() - Shared golden validation (74% reduction)
- _find_hw_node() - Helper to locate transformed nodes
- auto_configure_from_fixture() - Declarative configuration
- validate_against_golden() - GoldenValidator-based validation

Test Coverage (18 tests):
1. Core Parity (7 tests): Shapes, datatypes, stream widths
2. HW Estimation Parity (5 tests): Cycles, resources, efficiency
3. Golden Execution (6 tests): kernel_a/b × python/cppsim/rtlsim

Usage Example:

    @pytest.fixture(params=[
        KernelTestConfig(
            operation="Add",
            input_shapes={"input": (1, 64), "param": (1, 64)},
            input_dtypes={"input": DataType["INT8"], "param": DataType["INT8"]},
            fpgapart="xc7z020clg400-1",
        )
    ])
    def kernel_test_config(request):
        return request.param

    class TestAddParity(KernelParityTest):
        # ========================================================================
        # Shared Model
        # ========================================================================

        def make_test_model(self, kernel_test_config):
            '''Create ONNX Add model (shared by both kernels).'''
            # Build ONNX Add node
            return model, ["input", "param"]

        # ========================================================================
        # Kernel A (FINN AddStreams)
        # ========================================================================

        def infer_kernel_a(self, model, target_node):
            '''FINN AddStreams inference.'''
            from finn.transformation.fpgadataflow.infer_addstreams import InferAddStreamsLayer
            model = model.transform(InferAddStreamsLayer())
            op = self._find_hw_node(model, target_node, expected_type="AddStreams")
            return op, model

        def get_backend_variants_a(self):
            '''FINN AddStreams HLS backend.'''
            from finn.custom_op.fpgadataflow.hls.addstreams_hls import AddStreams_hls
            return [AddStreams_hls]

        def configure_kernel_a(self, op, model, stage, config):
            '''Configure FINN-specific parameters.'''
            super().configure_kernel_a(op, model, stage, config)
            if stage == 2:
                op.set_nodeattr("Func", config.operation.upper())  # FINN uses uppercase

        # ========================================================================
        # Kernel B (Brainsmith ElementwiseBinary) - uses defaults
        # ========================================================================

        def get_kernel_op(self):
            '''Brainsmith kernel for default inference.'''
            from brainsmith.kernels.elementwise_binary import ElementwiseBinary
            return ElementwiseBinary

        # infer_kernel_b() - uses default (calls self.infer_kernel())
        # get_backend_variants_b() - uses default (auto-detect)
        # configure_kernel_b() - uses default (auto_configure_from_fixture)

        # ========================================================================
        # Test Structure Information
        # ========================================================================

        def get_num_inputs(self):
            return 2

        def get_num_outputs(self):
            return 1

Result: 18 tests automatically run comparing FINN vs Brainsmith implementations!

v5.0 Changes:
- Uses Phase 1 _execute_and_validate_golden() for all golden tests
- Public specialize_to_backend_a/b() methods (v3.0 design)
- Consistent override pattern across all pipeline stages
- Per-kernel configuration hooks (90/10 rule)
"""

from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import pytest
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

# Import base utilities
from tests.frameworks.kernel_test_base_v2 import KernelTestBase_v2

# Import Phase 1 utilities
from tests.support.pipeline import PipelineRunner


class KernelParityTest(KernelTestBase_v2):
    """Test two kernel implementations for parity (v5.0).

    This class provides a comprehensive test framework for comparing two kernel
    implementations (e.g., FINN vs Brainsmith, manual vs auto, v1 vs v2).

    Abstract Methods (MUST implement - 5 total):
    1. make_test_model() - Create shared ONNX model
    2. infer_kernel_a() - Kernel A inference (e.g., FINN)
    3. get_backend_variants_a() - Kernel A backends (explicit)
    4. get_num_inputs() - Number of inputs for validation
    5. get_num_outputs() - Number of outputs for validation

    Optional Overrides (with defaults - 4 total):
    1. infer_kernel_b() - Default: self.infer_kernel() (Brainsmith)
    2. get_backend_variants_b() - Default: None (auto-detect)
    3. configure_kernel_a() - Default: auto_configure_from_fixture()
    4. configure_kernel_b() - Default: auto_configure_from_fixture()

    Backend Specialization (v5.0 - 2 methods):
    1. specialize_to_backend_a() - Kernel A backend specialization
    2. specialize_to_backend_b() - Kernel B backend specialization

    Test Suite (18 tests):
    - 7 Core parity tests (shapes, datatypes, streams)
    - 5 HW estimation parity tests (cycles, resources)
    - 6 Golden execution tests (kernel_a/b × python/cppsim/rtlsim)

    Inherited Utilities (from Phase 1):
    - _execute_and_validate_golden() - Eliminates 74% duplication
    - infer_kernel() - Default Brainsmith inference
    - specialize_to_backend() - Public backend specialization
    - _find_hw_node() - Node location helper
    - auto_configure_from_fixture() - Declarative config
    """

    # ========================================================================
    # Abstract Methods - Subclasses MUST implement (5 total)
    # ========================================================================

    @abstractmethod
    def infer_kernel_a(
        self,
        model: ModelWrapper,
        target_node: str,
    ) -> Tuple[HWCustomOp, ModelWrapper]:
        """Execute Stage 1 → Stage 2 kernel inference for Kernel A.

        Typically used for FINN inference or non-Brainsmith kernels that don't
        work with the default infer_kernel() implementation.

        Args:
            model: Stage 1 model (ONNX nodes with DataType annotations)
            target_node: Name of target ONNX node to transform

        Returns:
            (op, model): Kernel A operator instance and transformed model

        Example (FINN AddStreams):
            def infer_kernel_a(self, model, target_node):
                from finn.transformation.fpgadataflow.infer_addstreams import InferAddStreamsLayer
                model = model.transform(InferAddStreamsLayer())
                op = self._find_hw_node(model, target_node, expected_type="AddStreams")
                return op, model

        Example (Brainsmith with custom logic):
            def infer_kernel_a(self, model, target_node):
                # Multi-step inference
                model = model.transform(Transform1())
                model = model.transform(Transform2())
                op = self._find_hw_node(model, target_node, expected_type=MyKernel)
                return op, model
        """
        pass

    @abstractmethod
    def get_backend_variants_a(self) -> List[Type]:
        """Return backend variants for Kernel A in priority order.

        Required because FINN backends are not auto-detectable from registry.
        Must specify explicitly.

        Returns:
            List of backend classes to try in priority order

        Example (FINN):
            def get_backend_variants_a(self):
                from finn.custom_op.fpgadataflow.hls.addstreams_hls import AddStreams_hls
                return [AddStreams_hls]

        Example (Brainsmith explicit):
            def get_backend_variants_a(self):
                from brainsmith.kernels.elementwise_binary import ElementwiseBinaryOp_hls
                return [ElementwiseBinaryOp_hls]
        """
        pass

    @abstractmethod
    def get_num_inputs(self) -> int:
        """Return number of inputs for validation.

        Used by parity tests to validate shapes/datatypes have correct count.

        Returns:
            Number of inputs to kernel

        Example:
            def get_num_inputs(self):
                return 2  # Binary operation has 2 inputs
        """
        pass

    @abstractmethod
    def get_num_outputs(self) -> int:
        """Return number of outputs for validation.

        Used by parity tests to validate shapes/datatypes have correct count.

        Returns:
            Number of outputs from kernel

        Example:
            def get_num_outputs(self):
                return 1  # Most operations have 1 output
        """
        pass

    # ========================================================================
    # Reference-Based API (v6.0) - New Asymmetric Design
    # ========================================================================

    def infer_kernel_reference(
        self,
        model: ModelWrapper,
        target_node: str,
    ) -> Tuple[HWCustomOp, ModelWrapper]:
        """Execute kernel inference for reference implementation (v6.0).

        New asymmetric API. Primary implementation uses inherited infer_kernel(),
        reference implementation uses this method.

        Default: Delegates to infer_kernel_a() for backward compatibility.
        Override: Implement reference-specific inference logic.

        Args:
            model: Stage 1 model
            target_node: Target node name

        Returns:
            (op, model): Reference kernel and model

        Example:
            def infer_kernel_reference(self, model, target_node):
                # FINN inference
                from finn.transformation.fpgadataflow.convert_to_hw_layers import (
                    InferElementwiseBinaryOperation,
                )
                model = model.transform(InferElementwiseBinaryOperation())
                nodes = model.get_nodes_by_op_type("ElementwiseAdd")
                from qonnx.custom_op.registry import getCustomOp
                return getCustomOp(nodes[0]), model
        """
        # Default: delegate to old API for compatibility
        if hasattr(self, "infer_kernel_a"):
            return self.infer_kernel_a(model, target_node)
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__} must implement infer_kernel_reference()"
            )

    def get_backend_variants_reference(self) -> List[Type]:
        """Return backend variants for reference implementation (v6.0).

        New asymmetric API. Primary uses inherited get_backend_variants(),
        reference uses this method.

        Default: Delegates to get_backend_variants_a() for backward compatibility.
        Override: Return reference backend classes explicitly.

        Returns:
            List of backend classes

        Example:
            def get_backend_variants_reference(self):
                from finn.custom_op.fpgadataflow.hls.elementwise_binary_hls import (
                    ElementwiseAdd_hls,
                )
                return [ElementwiseAdd_hls]
        """
        # Default: delegate to old API for compatibility
        if hasattr(self, "get_backend_variants_a"):
            return self.get_backend_variants_a()
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__} must implement get_backend_variants_reference()"
            )

    def configure_kernel_reference(
        self,
        op: HWCustomOp,
        model: ModelWrapper,
        stage: int,
        config: "KernelTestConfig",
    ):
        """Configure reference kernel parameters (v6.0).

        New asymmetric API. Primary uses inherited configure_kernel(),
        reference uses this method.

        Default: Delegates to configure_kernel_a() for backward compatibility.
        Override: Implement reference-specific configuration.

        Args:
            op: Reference operator
            model: Model
            stage: Pipeline stage (2=kernel, 3=backend)
            config: Test configuration
        """
        # Default: delegate to old API for compatibility
        if hasattr(self, "configure_kernel_a"):
            return self.configure_kernel_a(op, model, stage, config)
        else:
            self.auto_configure_from_fixture(op, model, stage, config)

    def specialize_to_backend_reference(
        self,
        op: HWCustomOp,
        model: ModelWrapper,
        config: "KernelTestConfig",
    ) -> Tuple[HWCustomOp, ModelWrapper]:
        """Specialize reference to backend (v6.0).

        New asymmetric API. Uses explicit backend variants (no method swapping!).

        Default: Uses get_backend_variants_reference() and shared logic.
        Override: Custom specialization for reference implementation.

        Args:
            op: Reference operator (Stage 2)
            model: Model
            config: Configuration

        Returns:
            (backend_op, model): Specialized reference and model
        """
        import pytest

        fpgapart = config.get_fpgapart()
        if fpgapart is None:
            pytest.skip("Backend testing skipped (no FPGA part configured)")

        backend_variants = self.get_backend_variants_reference()

        from tests.support.backend_utils import specialize_to_backend

        return specialize_to_backend(op, model, fpgapart, backend_variants)

    # ========================================================================
    # Optional Overrides - Kernel B Inference (with default)
    # ========================================================================

    def infer_kernel_b(
        self,
        model: ModelWrapper,
        target_node: str,
    ) -> Tuple[HWCustomOp, ModelWrapper]:
        """Execute Stage 1 → Stage 2 kernel inference for Kernel B.

        Default: Uses inherited infer_kernel() (works for Brainsmith kernels).
        Override: Custom inference for Kernel B.

        Args:
            model: Stage 1 model (ONNX nodes with DataType annotations)
            target_node: Name of target ONNX node to transform

        Returns:
            (op, model): Kernel B operator instance and transformed model

        Example (default - Brainsmith):
            # Just implement get_kernel_op()
            def get_kernel_op(self):
                from brainsmith.kernels.elementwise_binary import ElementwiseBinary
                return ElementwiseBinary
            # This method uses default implementation

        Example (override - custom):
            def infer_kernel_b(self, model, target_node):
                # Custom multi-step inference
                model = model.transform(CustomTransform())
                op = self._find_hw_node(model, target_node)
                return op, model
        """
        # Default: Use inherited infer_kernel() (works for Brainsmith)
        return self.infer_kernel(model, target_node)

    # ========================================================================
    # Optional Overrides - Backend Variants (with defaults)
    # ========================================================================

    def get_backend_variants_b(self) -> Optional[List[Type]]:
        """Return backend variants for Kernel B in priority order.

        Optional: None = auto-detect from registry (works for Brainsmith).
        Override: Explicit backend classes.

        Returns:
            List of backend classes, or None for auto-detect

        Example (default - auto-detect):
            # Don't implement - uses default None
            # Auto-detects ElementwiseBinaryOp_hls from registry

        Example (override - explicit):
            def get_backend_variants_b(self):
                from brainsmith.kernels.elementwise_binary import ElementwiseBinaryOp_hls
                return [ElementwiseBinaryOp_hls]
        """
        return None  # Auto-detect from registry

    # ========================================================================
    # Optional Overrides - Per-Kernel Configuration (with defaults)
    # ========================================================================

    def configure_kernel_a(
        self,
        op: HWCustomOp,
        model: ModelWrapper,
        stage: int,
        config: "KernelTestConfig",
    ):
        """Configure Kernel A parameters at any stage.

        Default: Apply shared configuration from fixture via auto_configure_from_fixture().
        Override: Add Kernel A-specific parameters (90/10 rule - override for 10%).

        Called twice:
        1. After kernel inference (stage=2)
        2. After backend specialization (stage=3, if backends configured)

        Args:
            op: Kernel A operator (KernelOp or HWCustomOp)
            model: ModelWrapper containing the kernel
            stage: Pipeline stage (2=kernel, 3=backend)
            config: Test configuration with shared parameters

        Example (default - 100% shared config):
            # Don't implement - uses default
            # Applies config.input_streams, config.dse_dimensions automatically

        Example (override - 90% shared + 10% custom):
            def configure_kernel_a(self, op, model, stage, config):
                # Apply shared config (90%)
                super().configure_kernel_a(op, model, stage, config)

                # Add FINN-specific params (10%)
                if stage == 2:
                    op.set_nodeattr("Func", config.operation.upper())  # FINN uses uppercase
        """
        # Default: Apply shared configuration from fixture
        self.auto_configure_from_fixture(op, model, stage, config)

    def configure_kernel_b(
        self,
        op: HWCustomOp,
        model: ModelWrapper,
        stage: int,
        config: "KernelTestConfig",
    ):
        """Configure Kernel B parameters at any stage.

        Default: Apply shared configuration from fixture via auto_configure_from_fixture().
        Override: Add Kernel B-specific parameters (90/10 rule).

        Called twice:
        1. After kernel inference (stage=2)
        2. After backend specialization (stage=3, if backends configured)

        Args:
            op: Kernel B operator (KernelOp or HWCustomOp)
            model: ModelWrapper containing the kernel
            stage: Pipeline stage (2=kernel, 3=backend)
            config: Test configuration with shared parameters

        Example (default - 100% shared config):
            # Don't implement - uses default
            # Applies config.input_streams, config.dse_dimensions automatically

        Example (override - 90% shared + 10% custom):
            def configure_kernel_b(self, op, model, stage, config):
                # Apply shared config (90%)
                super().configure_kernel_b(op, model, stage, config)

                # Add Brainsmith-specific params (10%)
                if stage == 2:
                    op.set_nodeattr("custom_param", value)
        """
        # Default: Apply shared configuration from fixture
        self.auto_configure_from_fixture(op, model, stage, config)

    # ========================================================================
    # Backend Specialization - Per-Kernel Methods (v5.0)
    # ========================================================================

    def specialize_to_backend_a(
        self,
        op: HWCustomOp,
        model: ModelWrapper,
        config: "KernelTestConfig",
    ) -> Tuple[HWCustomOp, ModelWrapper]:
        """Specialize Kernel A to backend (Stage 2 → Stage 3).

        Default: Uses inherited specialize_to_backend() with get_backend_variants_a().
        Override: Custom backend specialization for Kernel A.

        Args:
            op: Kernel A operator (Stage 2)
            model: Model containing Kernel A
            config: Test configuration

        Returns:
            (backend_op, model): Kernel A backend and model

        Example (default - FINN backends):
            # Just implement get_backend_variants_a()
            def get_backend_variants_a(self):
                return [AddStreams_hls]
            # This method uses default implementation

        Example (override - conditional):
            def specialize_to_backend_a(self, op, model, config):
                # FINN-specific pre-specialization
                self._prepare_finn_backend(op, model)

                # Use default specialization
                op, model = super().specialize_to_backend_a(op, model, config)

                # FINN-specific post-specialization
                self._finalize_finn_backend(op, model)

                return op, model
        """
        # Temporarily swap get_backend_variants to use Kernel A's variants
        original_get_variants = self.get_backend_variants
        self.get_backend_variants = self.get_backend_variants_a

        try:
            # Use inherited specialization logic
            return self.specialize_to_backend(op, model, config)
        finally:
            # Restore original method
            self.get_backend_variants = original_get_variants

    def specialize_to_backend_b(
        self,
        op: HWCustomOp,
        model: ModelWrapper,
        config: "KernelTestConfig",
    ) -> Tuple[HWCustomOp, ModelWrapper]:
        """Specialize Kernel B to backend (Stage 2 → Stage 3).

        Default: Uses inherited specialize_to_backend() with get_backend_variants_b().
        Override: Custom backend specialization for Kernel B.

        Args:
            op: Kernel B operator (Stage 2)
            model: Model containing Kernel B
            config: Test configuration

        Returns:
            (backend_op, model): Kernel B backend and model

        Example (default - auto-detect):
            # Just return None from get_backend_variants_b()
            def get_backend_variants_b(self):
                return None  # Auto-detect
            # This method uses default implementation

        Example (override - multi-attempt):
            def specialize_to_backend_b(self, op, model, config):
                # Try HLS first
                try:
                    backend_variants = [MyKernel_hls]
                    from tests.support.backend_utils import specialize_to_backend
                    return specialize_to_backend(op, model, config.get_fpgapart(), backend_variants)
                except Exception as e:
                    # Fallback to RTL
                    backend_variants = [MyKernel_rtl]
                    return specialize_to_backend(op, model, config.get_fpgapart(), backend_variants)
        """
        # Temporarily swap get_backend_variants to use Kernel B's variants
        original_get_variants = self.get_backend_variants
        self.get_backend_variants = self.get_backend_variants_b

        try:
            # Use inherited specialization logic
            return self.specialize_to_backend(op, model, config)
        finally:
            # Restore original method
            self.get_backend_variants = original_get_variants

    # ========================================================================
    # Helper Methods (copied from SingleKernelTest)
    # ========================================================================

    # Note: _prepare_model_with_annotations and _generate_test_inputs moved to
    # KernelTestBase_v2 to eliminate duplication with SingleKernelTest (v6.0)

    def _compute_golden_reference(
        self, quant_model: ModelWrapper, inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Compute golden reference using test's compute_golden_reference().

        Args:
            quant_model: ModelWrapper WITH DataType annotations
            inputs: Test data

        Returns:
            Expected outputs
        """
        # Call the abstract method that test must implement
        return self.compute_golden_reference(inputs)

    # ========================================================================
    # Pytest Fixtures - Shared (reuse from SingleKernelTest pattern)
    # ========================================================================

    @pytest.fixture(scope="function")
    def stage1_model(
        self, kernel_test_config: "KernelTestConfig", model_cache
    ) -> ModelWrapper:
        """Stage 1 model with QONNX annotations (shared by both kernels).

        Same as SingleKernelTest.stage1_model - creates ONNX model before kernel inference.

        Args:
            kernel_test_config: Unified test configuration
            model_cache: Session-scoped cache for model artifacts

        Returns:
            Stage 1 model (ONNX + annotations, no kernel inference)
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
        """Generate test inputs (shared by both kernels).

        Same as SingleKernelTest.test_inputs - generates test data.

        Args:
            kernel_test_config: Unified test configuration
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
        """Golden reference from Stage 1 ONNX (shared by both kernels).

        Same as SingleKernelTest.golden_outputs - computes golden reference once.

        Args:
            kernel_test_config: Unified test configuration
            stage1_model: Stage 1 model fixture
            test_inputs: Test inputs fixture
            model_cache: Session-scoped cache

        Returns:
            Expected outputs from QONNX execution
        """

        def builder():
            return self._compute_golden_reference(stage1_model, test_inputs)

        return model_cache.get_golden_reference(kernel_test_config.test_id, builder)

    # ========================================================================
    # Pytest Fixtures - Kernel A Pipeline
    # ========================================================================

    @pytest.fixture(scope="function")
    def stage2_model_a(
        self,
        kernel_test_config: "KernelTestConfig",
        stage1_model: ModelWrapper,
        model_cache,
    ) -> Tuple:
        """Stage 2 Kernel A model (base kernel, no backend).

        Runs infer_kernel_a() to transform ONNX → Kernel A.

        Args:
            kernel_test_config: Unified test configuration
            stage1_model: Cached Stage 1 model
            model_cache: Session-scoped cache

        Returns:
            (kernel_a_op, model) tuple
        """

        def builder():
            # Reuse cached Stage 1 model
            model = stage1_model
            target_node = model.graph.node[0].name  # Assume first node

            # Stage 1 → Stage 2: ONNX → Kernel A
            op, model = self.infer_kernel_a(model, target_node)

            # Configure Kernel A (Stage 2)
            self.configure_kernel_a(op, model, stage=2, config=kernel_test_config)

            return op, model

        cache_key = f"{kernel_test_config.test_id}_kernel_a"
        return model_cache.get_stage2_model(cache_key, builder)

    @pytest.fixture(scope="function")
    def stage2_model_b(
        self,
        kernel_test_config: "KernelTestConfig",
        stage1_model: ModelWrapper,
        model_cache,
    ) -> Tuple:
        """Stage 2 Kernel B model (base kernel, no backend).

        Runs infer_kernel_b() to transform ONNX → Kernel B.

        Args:
            kernel_test_config: Unified test configuration
            stage1_model: Cached Stage 1 model
            model_cache: Session-scoped cache

        Returns:
            (kernel_b_op, model) tuple
        """

        def builder():
            # Reuse cached Stage 1 model
            model = stage1_model
            target_node = model.graph.node[0].name  # Assume first node

            # Stage 1 → Stage 2: ONNX → Kernel B
            op, model = self.infer_kernel_b(model, target_node)

            # Configure Kernel B (Stage 2)
            self.configure_kernel_b(op, model, stage=2, config=kernel_test_config)

            return op, model

        cache_key = f"{kernel_test_config.test_id}_kernel_b"
        return model_cache.get_stage2_model(cache_key, builder)

    @pytest.fixture(scope="function")
    def stage3_model_a(
        self,
        kernel_test_config: "KernelTestConfig",
        stage2_model_a: Tuple,
        model_cache,
    ) -> Tuple:
        """Stage 3 Kernel A model (backend-specialized).

        Specializes Kernel A to backend via specialize_to_backend_a().

        Args:
            kernel_test_config: Unified test configuration
            stage2_model_a: Cached Stage 2 Kernel A model
            model_cache: Session-scoped cache

        Returns:
            (kernel_a_backend_op, model) tuple

        Note:
            Use pytest marks (@pytest.mark.cppsim, @pytest.mark.rtlsim) to control
            which backend tests run. Configure fpgapart in kernel_test_config.platform.
        """
        fpgapart = kernel_test_config.get_fpgapart()

        def builder():
            # Reuse cached Stage 2 Kernel A
            base_op, base_model = stage2_model_a

            # Stage 2 → Stage 3: Base Kernel A → Backend A
            op, model = self.specialize_to_backend_a(base_op, base_model, kernel_test_config)

            # Configure Backend A (Stage 3)
            self.configure_kernel_a(op, model, stage=3, config=kernel_test_config)

            return op, model

        cache_key = f"{kernel_test_config.test_id}_kernel_a"
        return model_cache.get_stage3_model(cache_key, fpgapart, builder)

    @pytest.fixture(scope="function")
    def stage3_model_b(
        self,
        kernel_test_config: "KernelTestConfig",
        stage2_model_b: Tuple,
        model_cache,
    ) -> Tuple:
        """Stage 3 Kernel B model (backend-specialized).

        Specializes Kernel B to backend via specialize_to_backend_b().

        Args:
            kernel_test_config: Unified test configuration
            stage2_model_b: Cached Stage 2 Kernel B model
            model_cache: Session-scoped cache

        Returns:
            (kernel_b_backend_op, model) tuple

        Note:
            Use pytest marks (@pytest.mark.cppsim, @pytest.mark.rtlsim) to control
            which backend tests run. Configure fpgapart in kernel_test_config.platform.
        """
        fpgapart = kernel_test_config.get_fpgapart()

        def builder():
            # Reuse cached Stage 2 Kernel B
            base_op, base_model = stage2_model_b

            # Stage 2 → Stage 3: Base Kernel B → Backend B
            op, model = self.specialize_to_backend_b(base_op, base_model, kernel_test_config)

            # Configure Backend B (Stage 3)
            self.configure_kernel_b(op, model, stage=3, config=kernel_test_config)

            return op, model

        cache_key = f"{kernel_test_config.test_id}_kernel_b"
        return model_cache.get_stage3_model(cache_key, fpgapart, builder)

    # ========================================================================
    # Reference-Based Fixtures (v6.0) - New Naming Convention
    # ========================================================================

    @pytest.fixture(scope="function")
    def stage2_model(
        self,
        kernel_test_config: "KernelTestConfig",
        stage1_model: ModelWrapper,
        model_cache,
    ) -> Tuple:
        """Stage 2 primary model (v6.0 - unqualified).

        Uses inherited infer_kernel() from base class.
        For compatibility, delegates to stage2_model_b.

        Args:
            kernel_test_config: Test configuration
            stage1_model: Annotated model
            model_cache: Cache

        Returns:
            (op, model): Primary kernel and model
        """
        return self.stage2_model_b(kernel_test_config, stage1_model, model_cache)

    @pytest.fixture(scope="function")
    def stage2_model_reference(
        self,
        kernel_test_config: "KernelTestConfig",
        stage1_model: ModelWrapper,
        model_cache,
    ) -> Tuple:
        """Stage 2 reference model (v6.0).

        Uses infer_kernel_reference() method.

        Args:
            kernel_test_config: Test configuration
            stage1_model: Annotated model
            model_cache: Cache

        Returns:
            (op, model): Reference kernel and model
        """

        def builder():
            model = stage1_model
            target_node = model.graph.node[0].name

            op, model = self.infer_kernel_reference(model, target_node)
            self.configure_kernel_reference(
                op, model, stage=2, config=kernel_test_config
            )

            return op, model

        cache_key = f"{kernel_test_config.test_id}_reference"
        return model_cache.get_stage2_model(cache_key, builder)

    @pytest.fixture(scope="function")
    def stage3_model(
        self,
        kernel_test_config: "KernelTestConfig",
        stage2_model: Tuple,
        model_cache,
    ) -> Tuple:
        """Stage 3 primary model (v6.0 - unqualified).

        Uses inherited specialize_to_backend() from base class.

        Args:
            kernel_test_config: Test configuration
            stage2_model: Stage 2 primary model
            model_cache: Cache

        Returns:
            (op, model): Primary backend and model
        """
        fpgapart = kernel_test_config.get_fpgapart()

        def builder():
            base_op, base_model = stage2_model
            op, model = self.specialize_to_backend(
                base_op, base_model, kernel_test_config
            )

            # Configure backend
            self.auto_configure_from_fixture(op, model, stage=3, config=kernel_test_config)

            return op, model

        return model_cache.get_stage3_model(kernel_test_config.test_id, fpgapart, builder)

    @pytest.fixture(scope="function")
    def stage3_model_reference(
        self,
        kernel_test_config: "KernelTestConfig",
        stage2_model_reference: Tuple,
        model_cache,
    ) -> Tuple:
        """Stage 3 reference model (v6.0).

        Uses specialize_to_backend_reference() method.

        Args:
            kernel_test_config: Test configuration
            stage2_model_reference: Stage 2 reference model
            model_cache: Cache

        Returns:
            (op, model): Reference backend and model
        """
        fpgapart = kernel_test_config.get_fpgapart()

        def builder():
            base_op, base_model = stage2_model_reference
            op, model = self.specialize_to_backend_reference(
                base_op, base_model, kernel_test_config
            )
            self.configure_kernel_reference(
                op, model, stage=3, config=kernel_test_config
            )
            return op, model

        cache_key = f"{kernel_test_config.test_id}_reference"
        return model_cache.get_stage3_model(cache_key, fpgapart, builder)

    # ========================================================================
    # Golden Execution Tests (6 tests) - Uses Phase 1 _execute_and_validate_golden()
    # ========================================================================

    @pytest.mark.golden
    @pytest.mark.kernel_parity
    def test_kernel_a_python_vs_golden(
        self, kernel_test_config, stage2_model_a, test_inputs, golden_outputs
    ):
        """Test Kernel A Python execution matches golden reference."""
        self._execute_and_validate_golden(
            stage2_model_a, test_inputs, golden_outputs,
            "python", "Kernel A Python execution", kernel_test_config
        )

    @pytest.mark.golden
    @pytest.mark.kernel_parity
    def test_kernel_b_python_vs_golden(
        self, kernel_test_config, stage2_model_b, test_inputs, golden_outputs
    ):
        """Test Kernel B Python execution matches golden reference."""
        self._execute_and_validate_golden(
            stage2_model_b, test_inputs, golden_outputs,
            "python", "Kernel B Python execution", kernel_test_config
        )

    @pytest.mark.golden
    @pytest.mark.cppsim
    @pytest.mark.slow
    @pytest.mark.kernel_parity
    def test_kernel_a_cppsim_vs_golden(
        self, kernel_test_config, stage3_model_a, test_inputs, golden_outputs
    ):
        """Test Kernel A cppsim execution matches golden reference."""
        self._execute_and_validate_golden(
            stage3_model_a, test_inputs, golden_outputs,
            "cppsim", "Kernel A HLS cppsim", kernel_test_config
        )

    @pytest.mark.golden
    @pytest.mark.cppsim
    @pytest.mark.slow
    @pytest.mark.kernel_parity
    def test_kernel_b_cppsim_vs_golden(
        self, kernel_test_config, stage3_model_b, test_inputs, golden_outputs
    ):
        """Test Kernel B cppsim execution matches golden reference."""
        self._execute_and_validate_golden(
            stage3_model_b, test_inputs, golden_outputs,
            "cppsim", "Kernel B HLS cppsim", kernel_test_config
        )

    @pytest.mark.golden
    @pytest.mark.rtlsim
    @pytest.mark.slow
    @pytest.mark.kernel_parity
    def test_kernel_a_rtlsim_vs_golden(
        self, kernel_test_config, stage3_model_a, test_inputs, golden_outputs
    ):
        """Test Kernel A rtlsim execution matches golden reference."""
        self._execute_and_validate_golden(
            stage3_model_a, test_inputs, golden_outputs,
            "rtlsim", "Kernel A RTL rtlsim", kernel_test_config
        )

    @pytest.mark.golden
    @pytest.mark.rtlsim
    @pytest.mark.slow
    @pytest.mark.kernel_parity
    def test_kernel_b_rtlsim_vs_golden(
        self, kernel_test_config, stage3_model_b, test_inputs, golden_outputs
    ):
        """Test Kernel B rtlsim execution matches golden reference."""
        self._execute_and_validate_golden(
            stage3_model_b, test_inputs, golden_outputs,
            "rtlsim", "Kernel B RTL rtlsim", kernel_test_config
        )

    # ========================================================================
    # Core Parity Tests (7 tests) - Structural comparison at Stage 2
    # ========================================================================

    @pytest.mark.parity
    @pytest.mark.core
    @pytest.mark.kernel_parity
    def test_normal_shapes_parity(self, kernel_test_config, stage2_model_a, stage2_model_b):
        """Test normal input/output shapes match between implementations."""
        op_a, _ = stage2_model_a
        op_b, _ = stage2_model_b

        # Input shapes
        for i in range(self.get_num_inputs()):
            shape_a = op_a.get_normal_input_shape(i)
            shape_b = op_b.get_normal_input_shape(i)
            from tests.support.assertions import assert_shapes_match
            assert_shapes_match(shape_a, shape_b, i, "normal input")

        # Output shapes
        for i in range(self.get_num_outputs()):
            shape_a = op_a.get_normal_output_shape(i)
            shape_b = op_b.get_normal_output_shape(i)
            from tests.support.assertions import assert_shapes_match
            assert_shapes_match(shape_a, shape_b, i, "normal output")

    @pytest.mark.parity
    @pytest.mark.core
    @pytest.mark.kernel_parity
    def test_folded_shapes_parity(self, kernel_test_config, stage2_model_a, stage2_model_b):
        """Test folded input/output shapes match between implementations."""
        op_a, _ = stage2_model_a
        op_b, _ = stage2_model_b

        # Input shapes
        for i in range(self.get_num_inputs()):
            shape_a = op_a.get_folded_input_shape(i)
            shape_b = op_b.get_folded_input_shape(i)
            from tests.support.assertions import assert_shapes_match
            assert_shapes_match(shape_a, shape_b, i, "folded input")

        # Output shapes
        for i in range(self.get_num_outputs()):
            shape_a = op_a.get_folded_output_shape(i)
            shape_b = op_b.get_folded_output_shape(i)
            from tests.support.assertions import assert_shapes_match
            assert_shapes_match(shape_a, shape_b, i, "folded output")

    @pytest.mark.parity
    @pytest.mark.core
    @pytest.mark.kernel_parity
    def test_stream_widths_parity(self, kernel_test_config, stage2_model_a, stage2_model_b):
        """Test input/output stream widths match between implementations."""
        op_a, _ = stage2_model_a
        op_b, _ = stage2_model_b

        # Input stream widths
        for i in range(self.get_num_inputs()):
            width_a = op_a.get_instream_width(i)
            width_b = op_b.get_instream_width(i)
            from tests.support.assertions import assert_widths_match
            assert_widths_match(width_a, width_b, i, "Input")

        # Output stream widths
        for i in range(self.get_num_outputs()):
            width_a = op_a.get_outstream_width(i)
            width_b = op_b.get_outstream_width(i)
            from tests.support.assertions import assert_widths_match
            assert_widths_match(width_a, width_b, i, "Output")

    @pytest.mark.parity
    @pytest.mark.core
    @pytest.mark.kernel_parity
    def test_stream_widths_padded_parity(self, kernel_test_config, stage2_model_a, stage2_model_b):
        """Test padded stream widths match (AXI alignment)."""
        op_a, _ = stage2_model_a
        op_b, _ = stage2_model_b

        # Input stream widths padded
        for i in range(self.get_num_inputs()):
            width_a = op_a.get_instream_width_padded(i)
            width_b = op_b.get_instream_width_padded(i)

            def format_width(w):
                return f"{w} bits (padded)"
            from tests.support.assertions import assert_values_match
            assert_values_match(
                width_a, width_b, f"Input {i} stream width", format_width
            )

        # Output stream widths padded
        for i in range(self.get_num_outputs()):
            width_a = op_a.get_outstream_width_padded(i)
            width_b = op_b.get_outstream_width_padded(i)

            def format_width(w):
                return f"{w} bits (padded)"
            from tests.support.assertions import assert_values_match
            assert_values_match(
                width_a, width_b, f"Output {i} stream width", format_width
            )

    @pytest.mark.parity
    @pytest.mark.core
    @pytest.mark.kernel_parity
    def test_datatypes_parity(self, kernel_test_config, stage2_model_a, stage2_model_b):
        """Test input/output datatypes match between implementations."""
        op_a, _ = stage2_model_a
        op_b, _ = stage2_model_b

        # Input datatypes
        for i in range(self.get_num_inputs()):
            dt_a = op_a.get_input_datatype(i)
            dt_b = op_b.get_input_datatype(i)
            from tests.support.assertions import assert_datatypes_match
            assert_datatypes_match(dt_a, dt_b, i, "Input")

        # Output datatypes
        for i in range(self.get_num_outputs()):
            dt_a = op_a.get_output_datatype(i)
            dt_b = op_b.get_output_datatype(i)
            from tests.support.assertions import assert_datatypes_match
            assert_datatypes_match(dt_a, dt_b, i, "Output")

    @pytest.mark.parity
    @pytest.mark.core
    @pytest.mark.kernel_parity
    def test_datatype_inference_parity(self, kernel_test_config, stage2_model_a, stage2_model_b):
        """Test datatype inference produces matching results.

        Note: This test compares datatypes only, not tensor names.
        Some kernels may rename tensors during infer_node_datatype(),
        so we compare datatypes by position, not name.
        """
        op_a, model_a = stage2_model_a
        op_b, model_b = stage2_model_b

        # Run datatype inference
        model_a_out = op_a.infer_node_datatype(model_a)
        model_b_out = op_b.infer_node_datatype(model_b)

        # Use returned model if provided
        if model_a_out is not None:
            model_a = model_a_out
        if model_b_out is not None:
            model_b = model_b_out

        # Verify input datatypes (compare by position, not name)
        for i in range(self.get_num_inputs()):
            input_name_a = op_a.onnx_node.input[i]
            input_name_b = op_b.onnx_node.input[i]

            if not input_name_a or not input_name_b:
                continue

            # Get datatypes from each model using its own tensor names
            dt_a = model_a.get_tensor_datatype(input_name_a)
            dt_b = model_b.get_tensor_datatype(input_name_b)

            # Compare datatypes (names may differ)
            from tests.support.assertions import assert_datatypes_match
            assert_datatypes_match(
                dt_a, dt_b, i,
                f"After infer_node_datatype, input"
            )

        # Verify output datatypes (compare by position, not name)
        for i in range(self.get_num_outputs()):
            output_name_a = op_a.onnx_node.output[i]
            output_name_b = op_b.onnx_node.output[i]

            # Get datatypes from each model using its own tensor names
            dt_a = model_a.get_tensor_datatype(output_name_a)
            dt_b = model_b.get_tensor_datatype(output_name_b)

            # Compare datatypes (names may differ)
            from tests.support.assertions import assert_datatypes_match
            assert_datatypes_match(
                dt_a, dt_b, i,
                f"After infer_node_datatype, output"
            )

    @pytest.mark.parity
    @pytest.mark.core
    @pytest.mark.kernel_parity
    def test_make_shape_compatible_op_parity(self, kernel_test_config, stage2_model_a, stage2_model_b):
        """Test shape-compatible ops preserve output structure.

        Note: make_shape_compatible_op() returns an ONNX NodeProto (per FINN API),
        not a wrapped HWCustomOp. This is used for shape inference.
        """
        op_a, model_a = stage2_model_a
        op_b, model_b = stage2_model_b

        # Returns ONNX NodeProto for shape inference
        compat_node_a = op_a.make_shape_compatible_op(model_a)
        compat_node_b = op_b.make_shape_compatible_op(model_b)

        # Verify output count matches (NodeProto.output is a list of output names)
        assert len(compat_node_a.output) == len(compat_node_b.output), (
            f"Shape-compatible op output count mismatch: "
            f"kernel_a={len(compat_node_a.output)}, "
            f"kernel_b={len(compat_node_b.output)}"
        )

        # Verify output names match (both should use same output names as original op)
        for i in range(self.get_num_outputs()):
            output_name_a = compat_node_a.output[i] if i < len(compat_node_a.output) else None
            output_name_b = compat_node_b.output[i] if i < len(compat_node_b.output) else None

            assert output_name_a == op_a.onnx_node.output[i], (
                f"Kernel A shape-compatible op output {i} name mismatch"
            )
            assert output_name_b == op_b.onnx_node.output[i], (
                f"Kernel B shape-compatible op output {i} name mismatch"
            )

    # ========================================================================
    # Hardware Estimation Parity Tests (5 tests)
    # ========================================================================

    @pytest.mark.parity
    @pytest.mark.hw_estimation
    @pytest.mark.kernel_parity
    def test_expected_cycles_parity(self, kernel_test_config, stage2_model_a, stage2_model_b):
        """Test expected cycle counts match between implementations."""
        op_a, _ = stage2_model_a
        op_b, _ = stage2_model_b

        cycles_a = op_a.get_exp_cycles()
        cycles_b = op_b.get_exp_cycles()

        from tests.support.assertions import assert_values_match
        assert_values_match(cycles_a, cycles_b, "Expected cycles")

    @pytest.mark.parity
    @pytest.mark.hw_estimation
    @pytest.mark.kernel_parity
    def test_number_output_values_parity(self, kernel_test_config, stage2_model_a, stage2_model_b):
        """Test number of output values match (for FIFO sizing)."""
        op_a, _ = stage2_model_a
        op_b, _ = stage2_model_b

        count_a = op_a.get_number_output_values()
        count_b = op_b.get_number_output_values()

        from tests.support.assertions import assert_values_match
        assert_values_match(count_a, count_b, "Number of output values")

    @pytest.mark.parity
    @pytest.mark.hw_estimation
    @pytest.mark.kernel_parity
    def test_resource_estimates_parity(self, kernel_test_config, stage2_model_a, stage2_model_b):
        """Test resource estimates match between implementations."""
        op_a, _ = stage2_model_a
        op_b, _ = stage2_model_b

        # LUT estimation
        if hasattr(op_a, "lut_estimation") and hasattr(op_b, "lut_estimation"):
            luts_a = op_a.lut_estimation()
            luts_b = op_b.lut_estimation()

            def format_lut(count):
                return f"{count:,} LUTs"
            from tests.support.assertions import assert_values_match
            assert_values_match(luts_a, luts_b, "LUT estimation", format_lut)

        # DSP estimation (requires fpgapart parameter per FINN API)
        if hasattr(op_a, "dsp_estimation") and hasattr(op_b, "dsp_estimation"):
            # Use default fpgapart for estimation comparison
            from tests.support.constants import PARITY_DEFAULT_FPGA_PART_HLS
            fpgapart = PARITY_DEFAULT_FPGA_PART_HLS
            dsps_a = op_a.dsp_estimation(fpgapart)
            dsps_b = op_b.dsp_estimation(fpgapart)

            def format_dsp(count):
                return f"{count:,} DSPs"
            from tests.support.assertions import assert_values_match
            assert_values_match(dsps_a, dsps_b, "DSP estimation", format_dsp)

        # BRAM estimation
        if hasattr(op_a, "bram_estimation") and hasattr(op_b, "bram_estimation"):
            brams_a = op_a.bram_estimation()
            brams_b = op_b.bram_estimation()

            def format_bram(count):
                return f"{count:,} BRAMs"
            from tests.support.assertions import assert_values_match
            assert_values_match(brams_a, brams_b, "BRAM estimation", format_bram)

        # URAM estimation
        if hasattr(op_a, "uram_estimation") and hasattr(op_b, "uram_estimation"):
            urams_a = op_a.uram_estimation()
            urams_b = op_b.uram_estimation()

            def format_uram(count):
                return f"{count:,} URAMs"
            from tests.support.assertions import assert_values_match
            assert_values_match(urams_a, urams_b, "URAM estimation", format_uram)

    @pytest.mark.parity
    @pytest.mark.hw_estimation
    @pytest.mark.kernel_parity
    def test_efficiency_metrics_parity(self, kernel_test_config, stage2_model_a, stage2_model_b):
        """Test BRAM/URAM efficiency estimates match."""
        op_a, _ = stage2_model_a
        op_b, _ = stage2_model_b

        # BRAM efficiency
        if hasattr(op_a, "bram_efficiency_estimation") and hasattr(op_b, "bram_efficiency_estimation"):
            eff_a = op_a.bram_efficiency_estimation()
            eff_b = op_b.bram_efficiency_estimation()

            def format_efficiency(eff):
                return f"{eff:.4f} ({eff*100:.2f}%)"
            from tests.support.assertions import assert_values_match
            assert_values_match(eff_a, eff_b, "BRAM efficiency", format_efficiency)

        # URAM efficiency
        if hasattr(op_a, "uram_efficiency_estimation") and hasattr(op_b, "uram_efficiency_estimation"):
            eff_a = op_a.uram_efficiency_estimation()
            eff_b = op_b.uram_efficiency_estimation()

            def format_efficiency(eff):
                return f"{eff:.4f} ({eff*100:.2f}%)"
            from tests.support.assertions import assert_values_match
            assert_values_match(eff_a, eff_b, "URAM efficiency", format_efficiency)

    @pytest.mark.parity
    @pytest.mark.hw_estimation
    @pytest.mark.kernel_parity
    def test_operation_counts_parity(self, kernel_test_config, stage2_model_a, stage2_model_b):
        """Test operation and parameter counts match."""
        op_a, _ = stage2_model_a
        op_b, _ = stage2_model_b

        if hasattr(op_a, "get_op_and_param_counts") and hasattr(op_b, "get_op_and_param_counts"):
            counts_a = op_a.get_op_and_param_counts()
            counts_b = op_b.get_op_and_param_counts()

            from tests.support.assertions import assert_values_match
            assert_values_match(counts_a, counts_b, "Operation and parameter counts")
