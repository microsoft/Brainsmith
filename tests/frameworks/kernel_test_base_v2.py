"""Minimal test configuration interface with fixture-based parameterization (v2.0).

This module provides KernelTestConfig, a minimal abstract base class for kernel tests
that uses pytest fixtures for parameterization instead of hardcoded values.

Design Philosophy (v2.0):
- Pytest fixtures control parameterization (NOT test methods)
- Tests define operations with symbolic shapes (concrete from fixtures)
- ORT (ONNX Runtime) as golden reference (NOT NumPy)
- Automatic Quant node insertion based on fixture datatypes
- Automatic test data generation with correct types

Key Changes from v1.0:
- Removed get_qonnx_annotations() - replaced by input_datatypes fixture
- Removed get_num_inputs/outputs() - inferred from model
- Removed compute_golden_reference() - ORT on test_model automatically
- Added make_test_model(input_shapes) - symbolic shapes from fixture

Usage:
    from tests.frameworks.kernel_test_base_v2 import KernelTestConfig
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
            return model, ["input"]  # Quant inserted automatically

        def get_kernel_inference_transform(self):
            return InferMyKernel

Result: 2 dtypes × 2 shapes = 4 test configurations automatically!
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Type

import numpy as np
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation


class KernelTestConfig(ABC):
    """Minimal configuration interface for fixture-based kernel tests.

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
    - Automatic Quant node insertion (based on input_datatypes)
    - Automatic test data generation (using Phase 1 utilities)
    - Automatic golden reference (ORT on test_model)
    """

    # ========================================================================
    # Abstract Methods - Subclasses MUST implement (2 only!)
    # ========================================================================

    @abstractmethod
    def make_test_model(
        self, input_shapes: Dict[str, Tuple[int, ...]]
    ) -> Tuple[ModelWrapper, List[str]]:
        """Create test model with concrete shapes from fixture.

        This method defines the operations to test WITHOUT Quant nodes.
        Framework automatically inserts Quant nodes before specified inputs
        based on input_datatypes fixture.

        Args:
            input_shapes: Dict mapping input names to concrete shapes from fixture.
                         Example: {"input": (1, 64), "param": (64,)}

        Returns:
            (model, input_names):
                - model: ONNX model with operations (NO Quant nodes)
                - input_names: List of input tensor names to insert Quant before

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

                # Framework inserts Quant before these inputs automatically
                return model, ["input", "param"]

        Note:
            - Use TensorProto.FLOAT for all inputs (Quant nodes handle types)
            - Shapes are concrete (from fixture), not symbolic
            - Do NOT insert Quant nodes manually
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

        This is called AFTER kernel inference (InferKernels) transforms ONNX nodes
        into base kernels, but BEFORE backend specialization. The node at this point
        is a base kernel (e.g., AddStreams, MVAU) without backend-specific code.

        Timing:
        - Stage 1 → Stage 2: InferKernels transforms ONNX → Base kernel
        - Stage 2: configure_kernel_node() sets dimension parameters ← YOU ARE HERE
        - Stage 2 → Stage 3: SpecializeKernels transforms Base kernel → Backend
        - Stage 3: configure_backend_node() sets backend parameters

        IMPORTANT: If you change dimension parameters (PE, SIMD) on a KernelOp,
        you MUST call op._ensure_ready(model) afterwards to reconfigure the
        design space and design point.

        Args:
            op: Base kernel operator instance (e.g., AddStreams, MVAU)
            model: Model containing the operator

        Example:
            def configure_kernel_node(self, op, model):
                from brainsmith.dataflow.kernel_op import KernelOp

                # Set dimension parameters
                op.set_nodeattr("PE", 8)
                op.set_nodeattr("SIMD", 16)

                # Reconfigure design point (KernelOp only)
                if isinstance(op, KernelOp):
                    op._ensure_ready(model)

        Note:
            - Called for ALL tests (Python, cppsim, rtlsim)
            - Configuration is automatically preserved during backend specialization
            - Default implementation does nothing (no configuration)

        See Also:
            configure_backend_node(): Stage 3 configuration (backend parameters)
        """
        pass

    def configure_backend_node(
        self,
        op: HWCustomOp,
        model: ModelWrapper
    ) -> None:
        """Configure backend node after specialization (Stage 3, optional).

        Override to set backend-specific parameters (memory mode, RTL pragmas, etc.).

        This is called AFTER SpecializeKernels/SpecializeLayers transforms the base
        kernel (Stage 2) into a backend implementation (Stage 3). The backend node
        has op_type like "KernelName_hls" or "KernelName_rtl" and inherits from
        HLSBackend or RTLBackend.

        Timing:
        - Stage 2: configure_kernel_node() configures base kernel (PE, SIMD, etc.)
        - Stage 2 → Stage 3: SpecializeKernels/SpecializeLayers transforms kernel
        - Stage 3: configure_backend_node() configures backend (mem_mode, ram_style, etc.)

        Args:
            op: Backend operator instance (e.g., AddStreams_hls, LayerNorm_rtl)
            model: Model containing the operator

        Example:
            def configure_backend_node(self, op, model):
                # HLS memory configuration
                op.set_nodeattr("mem_mode", "internal_decoupled")
                op.set_nodeattr("ram_style", "ultra")

                # Resource type selection
                op.set_nodeattr("resType", "dsp")

                # RTL-specific pragmas (if applicable)
                if hasattr(op, 'set_pipeline_pragma'):
                    op.set_pipeline_pragma(True)

        Note:
            - Only called when running backend tests (cppsim/rtlsim)
            - NOT called for Python execution tests (Stage 2 only)
            - Stage 2 configuration (PE, SIMD) is automatically preserved
            - Default implementation does nothing (no backend config)

        See Also:
            configure_kernel_node(): Stage 2 configuration (dimension parameters)
        """
        pass

    def get_test_seed(self) -> int:
        """Return random seed for test data generation (optional).

        Provides deterministic random seed for reproducible test failures.
        Override to customize seed per test class or configuration.

        Returns:
            Random seed integer (default: 42)

        Example:
            def get_test_seed(self):
                return 12345  # Custom seed for this test

        Note:
            - Default seed is 42 for deterministic behavior
            - Can be overridden by pytest --seed CLI option
            - Same seed → Same random data → Reproducible failures
            - Used by make_execution_context_* functions

        CLI usage:
            pytest --seed=12345    # Override seed for all tests
            pytest                 # Use default (42) or per-test override
        """
        return 42

    def get_tolerance_python(self) -> Dict[str, float]:
        """Tolerance for Python execution vs golden reference.

        Python execution is the most accurate (NumPy floating-point),
        so tolerances can be very tight.

        Returns:
            Dict with 'rtol' and 'atol' keys for np.allclose()

        Default:
            rtol=1e-7, atol=1e-9 (very tight)

        Override for kernels with lower numerical precision:
            def get_tolerance_python(self):
                return {"rtol": 1e-5, "atol": 1e-6}  # Approximate ops
        """
        return {"rtol": 1e-7, "atol": 1e-9}

    def get_tolerance_cppsim(self) -> Dict[str, float]:
        """Tolerance for C++ simulation vs golden reference.

        HLS C++ simulation uses fixed-point arithmetic which introduces
        rounding errors, so tolerances are looser than Python.

        Returns:
            Dict with 'rtol' and 'atol' keys for np.allclose()

        Default:
            rtol=1e-5, atol=1e-6 (moderate)

        Override for very approximate operations:
            def get_tolerance_cppsim(self):
                return {"rtol": 1e-3, "atol": 1e-4}  # Softmax, etc.
        """
        return {"rtol": 1e-5, "atol": 1e-6}

    def get_tolerance_rtlsim(self) -> Dict[str, float]:
        """Tolerance for RTL simulation vs golden reference.

        RTL simulation has same precision as cppsim (same fixed-point design),
        so default to same tolerance.

        Returns:
            Dict with 'rtol' and 'atol' keys for np.allclose()

        Default:
            Same as cppsim

        Override if RTL has different precision:
            def get_tolerance_rtlsim(self):
                return {"rtol": 1e-4, "atol": 1e-5}
        """
        return self.get_tolerance_cppsim()

    def get_backend_fpgapart(self) -> str:
        """FPGA part for backend specialization (optional).

        Override to enable backend specialization (Stage 2 → Stage 3).
        When specified, tests can specialize base kernels (e.g., AddStreams)
        to backend variants (e.g., AddStreams_hls) for cppsim/rtlsim execution.

        Returns:
            str: FPGA part string (e.g., "xc7z020clg400-1")
            None: Backend specialization disabled (default)

        Pipeline stages:
            Stage 1: ONNX Node (Add, Mul, etc.)
            Stage 2: Base Kernel (AddStreams, no backend)
            Stage 3: Backend (AddStreams_hls, with HLSBackend) ← Enabled by this hook

        Default:
            None (backend specialization disabled)

        Override to enable backend testing:
            def get_backend_fpgapart(self):
                return "xc7z020clg400-1"  # Enable HLS backend

        Note:
            When None, cppsim/rtlsim tests will be skipped (base kernels
            don't have backend inheritance).
        """
        return None

    def get_backend_variants(self) -> list:
        """Backend variant classes for specialization in priority order.

        Returns:
            List of backend classes to try in priority order.
            Default: None (auto-detect HLS backend based on kernel op_type)

        Override to specify explicit backend classes:
            def get_backend_variants(self):
                from brainsmith.kernels.elementwise_binary import ElementwiseBinaryOp_hls
                return [ElementwiseBinaryOp_hls]

        Override for priority ordering (e.g., prefer RTL, fallback to HLS):
            def get_backend_variants(self):
                from brainsmith.kernels.mvau import MVAU_rtl, MVAU_hls
                return [MVAU_rtl, MVAU_hls]

        Default:
            None (automatically looks up HLS backend from registry)
        """
        return None

    # ========================================================================
    # Optional Golden Reference Hooks (v2.1)
    # ========================================================================

    def get_use_custom_golden_reference(self) -> bool:
        """Use custom golden reference instead of QONNX execution (v2.1).

        Override to provide custom golden reference implementation instead of
        the default QONNX execution on quantized model.

        Returns:
            bool: True to use compute_custom_golden_reference(), False for QONNX

        Default:
            False (use QONNX execution on quantized model - validates Quant nodes)

        Override when:
            - Operation not supported by QONNX execution
            - Need specific golden reference behavior (e.g., floating-point only)
            - Debugging quantization issues

        Note:
            When True, you MUST implement compute_custom_golden_reference().
        """
        return False

    def compute_custom_golden_reference(
        self, model: ModelWrapper, inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Provide custom golden reference implementation (v2.1).

        Only called when get_use_custom_golden_reference() returns True.

        Args:
            model: ModelWrapper WITH Quant nodes inserted
            inputs: Dict of input tensors (with raw_* names after Quant insertion)

        Returns:
            Dict of output tensors

        Raises:
            NotImplementedError: Must override when using custom golden reference

        Example:
            def compute_custom_golden_reference(self, model, inputs):
                # Use ORT on float model (debugging)
                float_model, _ = self.make_test_model(self._current_input_shapes)

                # Convert raw_* inputs back to original names
                ort_inputs = {}
                for name, data in inputs.items():
                    if name.startswith("raw_"):
                        ort_inputs[name[4:]] = data  # Remove "raw_" prefix
                    else:
                        ort_inputs[name] = data

                # Execute with ONNX Runtime
                import onnxruntime as rt
                sess = rt.InferenceSession(float_model.model.SerializeToString())
                return sess.run(None, ort_inputs)
        """
        raise NotImplementedError(
            "Must implement compute_custom_golden_reference() when "
            "get_use_custom_golden_reference() returns True"
        )
