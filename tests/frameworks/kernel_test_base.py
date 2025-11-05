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
from typing import Dict, Tuple, Optional, List

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
import qonnx.core.data_layout as DataLayout
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class KernelTestConfig(ABC):
    """Minimal configuration interface for kernel tests.

    Provides ONLY abstract methods for test-specific configuration.
    Does NOT provide pipeline/validation/execution logic - subclasses
    use composition (PipelineRunner, GoldenValidator, Executors).

    Abstract Methods (4):
    - make_onnx_model(): Create pure ONNX model (Stage 1)
    - get_qonnx_annotations(): QONNX DataType annotations (Stage 1→2)
    - get_num_inputs(): Number of input tensors
    - get_num_outputs(): Number of output tensors

    Optional Hooks:
    - get_qonnx_layouts(): DataLayout annotations (optional)
    - configure_kernel_node(): Configure PE, SIMD, etc.
    - get_tolerance_*(): Tolerances for golden reference validation
    - get_dtype_sweep(): Parameterized dtype testing (optional)
    - get_shape_sweep(): Parameterized shape testing (optional)
    """

    # ========================================================================
    # Model Creation - Required
    # ========================================================================

    @abstractmethod
    def make_onnx_model(self) -> Tuple[ModelWrapper, str]:
        """Create pure ONNX model with standard TensorProto types (Stage 1).

        This method creates the ONNX representation WITHOUT QONNX annotations.
        QONNX DataType annotations are added separately via get_qonnx_annotations().

        Stage separation:
        - Stage 1 (ONNX): make_onnx_model() → TensorProto types
        - Stage 2 (QONNX): get_qonnx_annotations() → DataType annotations

        Returns:
            (model, node_name): Pure ONNX model and target node name

        Example:
            def make_onnx_model(self):
                import onnx.helper as helper
                from onnx import TensorProto

                # Use standard ONNX TensorProto types
                inp0 = helper.make_tensor_value_info("input0", TensorProto.FLOAT, [1, 64])
                inp1 = helper.make_tensor_value_info("input1", TensorProto.FLOAT, [1, 64])
                out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 64])

                node = helper.make_node("Add", ["input0", "input1"], ["output"])
                graph = helper.make_graph([node], "test_add", [inp0, inp1], [out])
                model = helper.make_model(graph)

                # NO QONNX annotations here!
                return ModelWrapper(model), "Add_0"

        Note:
            - Use TensorProto.FLOAT for quantized integers (FINN convention)
            - Use TensorProto.INT8 for true integer types (if ONNX Runtime supports)
            - Do NOT call model.set_tensor_datatype() here
            - Do NOT add DataLayout annotations here
        """
        pass

    @abstractmethod
    def get_qonnx_annotations(self) -> Dict[str, DataType]:
        """Return QONNX DataType annotations for tensors.

        Maps tensor names to QONNX DataTypes for Stage 1 → Stage 2 transition.
        These annotations enable FINN/Brainsmith semantic interpretation.

        Returns:
            Dict mapping tensor names → QONNX DataTypes

        Example:
            def get_qonnx_annotations(self):
                return {
                    "input0": DataType["INT8"],
                    "input1": DataType["INT8"],
                    "output": DataType["INT16"]  # Accumulation requires wider type
                }

        Note:
            - Must cover all inputs and outputs
            - Should match expected kernel behavior
            - Used by InferDataTypes to propagate semantics
        """
        pass

    def get_qonnx_layouts(self) -> Dict[str, DataLayout]:
        """Return QONNX DataLayout annotations for tensors (optional).

        Maps tensor names to QONNX DataLayouts (NCHW, NHWC, etc.).
        Only needed for spatial operations (convolution, pooling, etc.).

        Returns:
            Dict mapping tensor names → QONNX DataLayouts
            Default: {} (no layout annotations)

        Example:
            def get_qonnx_layouts(self):
                return {
                    "input": DataLayout.NHWC,
                    "output": DataLayout.NHWC
                }
        """
        return {}

    # ========================================================================
    # Parameterization Hooks (Optional)
    # ========================================================================

    def get_dtype_sweep(self) -> Optional[List[Dict[str, DataType]]]:
        """Return dtype combinations for parameterized testing (optional).

        Enables testing same operation with multiple dtype configurations.
        Each dict in the list represents one test configuration.

        Returns:
            List of annotation dicts for parameterization
            Default: None (no parameterization)

        Example:
            def get_dtype_sweep(self):
                return [
                    # INT8 × INT8
                    {"input": DataType["INT8"], "param": DataType["INT8"], "output": DataType["INT16"]},
                    # FLOAT32 × FLOAT32
                    {"input": DataType["FLOAT32"], "param": DataType["FLOAT32"], "output": DataType["FLOAT32"]},
                    # Mixed precision
                    {"input": DataType["INT8"], "param": DataType["FLOAT32"], "output": DataType["FLOAT32"]},
                ]

        Note:
            - Framework will run test for each configuration
            - Useful for comprehensive dtype validation
            - Can significantly increase test count
        """
        return None

    def get_shape_sweep(self) -> Optional[List[Dict[str, Tuple[int, ...]]]]:
        """Return shape combinations for parameterized testing (optional).

        Enables testing same operation with multiple shape configurations.
        Each dict in the list represents one test configuration.

        Returns:
            List of shape dicts for parameterization
            Default: None (no parameterization)

        Example:
            def get_shape_sweep(self):
                return [
                    {"input": (1, 64), "param": (64,), "output": (1, 64)},
                    {"input": (1, 8, 8, 64), "param": (64,), "output": (1, 8, 8, 64)},
                    {"input": (4, 16, 16, 32), "param": (32,), "output": (4, 16, 16, 32)},
                ]

        Note:
            - Broadcasting must be handled by make_onnx_model()
            - Framework will run test for each configuration
            - Can be combined with dtype_sweep for comprehensive testing
        """
        return None

    def configure_for_dtype_config(self, config: Dict[str, DataType]) -> None:
        """Apply dtype configuration for parameterized testing.

        Called automatically by pytest fixtures before each parameterized test.
        Override this to apply dtype configuration to instance attributes.

        Args:
            config: Dict mapping tensor names to DataTypes from get_dtype_sweep()

        Example:
            def configure_for_dtype_config(self, config: Dict[str, DataType]):
                '''Apply dtype configuration to instance attributes.'''
                self.input_dtype = config["input"]
                self.param_dtype = config["param"]
                self.output_dtype = config["output"]

        Note:
            - This method is called BEFORE make_onnx_model() and get_qonnx_annotations()
            - Update instance attributes that affect model creation
            - Default implementation does nothing (override in subclass if needed)
        """
        pass

    def configure_for_shape_config(self, config: Dict[str, Tuple[int, ...]]) -> None:
        """Apply shape configuration for parameterized testing.

        Called automatically by pytest fixtures before each parameterized test.
        Override this to apply shape configuration to instance attributes.

        Args:
            config: Dict mapping tensor names to shapes from get_shape_sweep()

        Example:
            def configure_for_shape_config(self, config: Dict[str, Tuple]):
                '''Apply shape configuration to instance attributes.'''
                # Extract dimensions from shape tuples
                input_shape = config["input"]
                if len(input_shape) == 4:
                    self.batch, self.height, self.width, self.channels = input_shape
                elif len(input_shape) == 2:
                    self.batch, self.channels = input_shape
                    self.height = 1
                    self.width = 1

        Note:
            - This method is called BEFORE make_onnx_model() and get_qonnx_annotations()
            - Update instance attributes that affect model creation
            - Handle different shape tuple lengths appropriately
            - Default implementation does nothing (override in subclass if needed)
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
