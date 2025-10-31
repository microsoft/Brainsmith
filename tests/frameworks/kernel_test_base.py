"""Minimal shared interface for kernel test frameworks.

This module provides KernelTestConfig, a minimal abstract base class that both
SingleKernelTest and DualKernelTest inherit from.

Design Philosophy:
- Minimal abstract methods (3 only): make_test_model, get_num_inputs, get_num_outputs
- Optional configuration hooks (no forced implementation)
- NO pipeline/validation/execution logic (composition, not inheritance)
- Clear separation: config only, no behavior

This eliminates the "abstract method stutter" problem where the same 5+ abstract
methods were redeclared across CoreParityTest, HWEstimationParityTest, and
IntegratedPipelineTest.

Usage:
    from tests.frameworks.kernel_test_base import KernelTestConfig

    class MyTest(KernelTestConfig):
        def make_test_model(self):
            return model, "node_name"

        def get_num_inputs(self):
            return 2

        def get_num_outputs(self):
            return 1

        # Optional: configure_kernel_node, tolerance methods
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

from qonnx.core.modelwrapper import ModelWrapper
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class KernelTestConfig(ABC):
    """Minimal configuration interface for kernel tests.

    Provides ONLY abstract methods for test-specific configuration.
    Does NOT provide pipeline/validation/execution logic - subclasses
    use composition (PipelineRunner, GoldenValidator, Executors).

    Abstract Methods (3):
    - make_test_model(): Create ONNX model
    - get_num_inputs(): Number of input tensors
    - get_num_outputs(): Number of output tensors

    Optional Hooks:
    - configure_kernel_node(): Configure PE, SIMD, etc.
    - get_tolerance_*(): Tolerances for golden reference validation
    """

    # ========================================================================
    # Model Creation - Required
    # ========================================================================

    @abstractmethod
    def make_test_model(self) -> Tuple[ModelWrapper, str]:
        """Create standard ONNX model for testing.

        Returns:
            (model, node_name): ModelWrapper and name of ONNX node

        Example:
            def make_test_model(self):
                import onnx.helper as helper
                from onnx import TensorProto

                inp0 = helper.make_tensor_value_info("input0", TensorProto.FLOAT, [1, 64])
                inp1 = helper.make_tensor_value_info("input1", TensorProto.FLOAT, [1, 64])
                out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 64])

                node = helper.make_node("Add", ["input0", "input1"], ["output"])
                graph = helper.make_graph([node], "test_add", [inp0, inp1], [out])
                model = helper.make_model(graph)

                return ModelWrapper(model), "Add_0"
        """
        pass

    # ========================================================================
    # I/O Configuration - Required
    # ========================================================================

    @abstractmethod
    def get_num_inputs(self) -> int:
        """Return number of input tensors.

        Used for iterating over inputs in test assertions.

        Returns:
            Number of inputs (typically 1 or 2)

        Example:
            def get_num_inputs(self):
                return 2  # Binary operation
        """
        pass

    @abstractmethod
    def get_num_outputs(self) -> int:
        """Return number of output tensors.

        Used for iterating over outputs in test assertions.

        Returns:
            Number of outputs (typically 1)

        Example:
            def get_num_outputs(self):
                return 1  # Single output
        """
        pass

    # ========================================================================
    # Optional Configuration Hooks
    # ========================================================================

    def configure_kernel_node(
        self,
        op: HWCustomOp,
        model: ModelWrapper
    ) -> None:
        """Configure kernel node after inference (optional).

        Override to set non-default parameters (PE, SIMD, etc.).

        IMPORTANT: If you change dimension parameters (PE, SIMD) on a KernelOp,
        you MUST call op._ensure_ready(model) afterwards to reconfigure the
        design space and design point.

        Args:
            op: Hardware operator instance
            model: Model containing the operator

        Example:
            def configure_kernel_node(self, op, model):
                from brainsmith.dataflow.kernel_op import KernelOp

                # Set parameters
                op.set_nodeattr("PE", 8)
                op.set_nodeattr("SIMD", 16)

                # Reconfigure design point (KernelOp only)
                if isinstance(op, KernelOp):
                    op._ensure_ready(model)
        """
        pass

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

    # ========================================================================
    # Backend Configuration Hooks (Optional)
    # ========================================================================

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

    def get_backend_type(self) -> str:
        """Backend type for specialization.

        Returns:
            str: "hls" or "rtl"

        Default:
            "hls" (HLS backend)

        Override for RTL backends:
            def get_backend_type(self):
                return "rtl"
        """
        return "hls"
