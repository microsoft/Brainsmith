"""Example: SingleKernelTest with backend specialization enabled.

This test demonstrates the complete 3-stage pipeline:
    Stage 1: ONNX Node (Add)
    Stage 2: Base Kernel (AddStreams)
    Stage 3: Backend (AddStreams_hls with HLSBackend inheritance)

The key difference from test_addstreams_integration.py is ONE METHOD:
    def get_backend_fpgapart(self):
        return "xc7z020clg400-1"  # Enable backend specialization

This enables cppsim/rtlsim tests that were previously skipped.

Test Coverage:
- Pipeline validation (ONNX Add → AddStreams → AddStreams_hls)
- Shape/datatype preservation through ALL stages
- Python execution vs golden (Stage 2: base kernel)
- HLS C++ simulation vs golden (Stage 3: backend)
- RTL simulation vs golden (Stage 3: backend)

Example Usage:
    # Run fast tests only (no backend execution)
    pytest tests/pipeline/test_addstreams_backend_example.py -v -m "not slow"

    # Run cppsim test (requires VITIS_PATH)
    pytest tests/pipeline/test_addstreams_backend_example.py::TestAddStreamsBackendExample::test_cppsim_execution_vs_golden -v -s

    # Run ALL tests including rtlsim (requires Vivado)
    pytest tests/pipeline/test_addstreams_backend_example.py -v
"""

import numpy as np
import pytest
from onnx import helper, TensorProto
from typing import Dict, Tuple, Type

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.transformation.base import Transformation
from qonnx.util.basic import qonnx_make_model

from brainsmith.kernels.addstreams import AddStreams
from brainsmith.primitives.transforms.infer_kernel_list import InferKernelList
from brainsmith.dataflow.kernel_op import KernelOp
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp

from tests.frameworks.single_kernel_test import SingleKernelTest


class TestAddStreamsBackendExample(SingleKernelTest):
    """Example: Complete 3-stage pipeline with backend specialization.

    Demonstrates how to enable backend testing (cppsim/rtlsim) by overriding
    a single method: get_backend_fpgapart().

    Pipeline Stages:
        Stage 1: ONNX Add node
                    ↓ InferKernelList
        Stage 2: AddStreams (base kernel, no backend)
                    ↓ SpecializeLayers (enabled by get_backend_fpgapart())
        Stage 3: AddStreams_hls (backend with HLSBackend inheritance)
                    ↓
                 cppsim/rtlsim execution

    Inherited Tests (6):
    1. test_pipeline_creates_hw_node ← Stage 2
    2. test_shapes_preserved_through_pipeline ← Stage 2
    3. test_datatypes_preserved_through_pipeline ← Stage 2
    4. test_python_execution_vs_golden ← Stage 2 (base kernel)
    5. test_cppsim_execution_vs_golden ← Stage 3 (backend) ✨ ENABLED
    6. test_rtlsim_execution_vs_golden ← Stage 3 (backend) ✨ ENABLED

    The last two tests (cppsim/rtlsim) require backend specialization,
    which is enabled by get_backend_fpgapart() returning an FPGA part.
    """

    # ================================================================
    # Test Configuration
    # ================================================================

    def get_test_shape(self) -> Tuple[int, ...]:
        """Shape for test inputs (NHWC format)."""
        return (1, 64)  # Batch=1, Channels=64

    def get_test_datatype(self) -> DataType:
        """Datatype for test inputs."""
        return DataType["INT8"]

    # ================================================================
    # Required Abstract Methods (4 total)
    # ================================================================

    def make_test_model(self) -> Tuple[ModelWrapper, str]:
        """Create ONNX model with Add node.

        Creates standard ONNX Add operator (Stage 1).
        Will be transformed to AddStreams kernel (Stage 2).

        Returns:
            (model, node_name): ModelWrapper and ONNX node name
        """
        shape = self.get_test_shape()
        dtype = self.get_test_datatype()

        # Create ONNX Add node
        inp0 = helper.make_tensor_value_info("inp0", TensorProto.FLOAT, shape)
        inp1 = helper.make_tensor_value_info("inp1", TensorProto.FLOAT, shape)
        outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, shape)

        add_node = helper.make_node(
            "Add",
            inputs=["inp0", "inp1"],
            outputs=["outp"],
            name="Add_0"
        )

        graph = helper.make_graph(
            nodes=[add_node],
            name="addstreams_backend_test",
            inputs=[inp0, inp1],
            outputs=[outp]
        )

        model = qonnx_make_model(graph, producer_name="addstreams-backend-example")
        model_wrapper = ModelWrapper(model)

        # Set datatypes
        model_wrapper.set_tensor_datatype("inp0", dtype)
        model_wrapper.set_tensor_datatype("inp1", dtype)
        model_wrapper.set_tensor_datatype("outp", dtype)

        return model_wrapper, "Add_0"

    def get_kernel_inference_transform(self) -> Type[Transformation]:
        """Transform that creates AddStreams kernel (Stage 1 → Stage 2)."""
        return InferKernelList

    def compute_golden_reference(
        self, inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Compute golden reference (element-wise addition).

        This is TEST LOGIC, not production code!
        Defines what "correct" means for this test.

        Args:
            inputs: Dict with "inp0" and "inp1" arrays

        Returns:
            Dict with "outp" array (element-wise sum)
        """
        return {"outp": inputs["inp0"] + inputs["inp1"]}

    def get_num_inputs(self) -> int:
        """AddStreams has 2 inputs."""
        return 2

    def get_num_outputs(self) -> int:
        """AddStreams has 1 output."""
        return 1

    # ================================================================
    # Optional Configuration Hooks
    # ================================================================

    def configure_kernel_node(
        self,
        op: HWCustomOp,
        model: ModelWrapper
    ) -> None:
        """Configure AddStreams kernel after inference.

        Set PE (processing elements) for parallelism.
        """
        op.set_nodeattr("PE", 8)

        # Reconfigure design point (KernelOp only)
        if isinstance(op, KernelOp):
            op._ensure_ready(model)

    # ================================================================
    # Backend Configuration (KEY FEATURE!)
    # ================================================================

    def get_backend_fpgapart(self) -> str:
        """Enable backend specialization (Stage 2 → Stage 3).

        ✨ THIS IS THE KEY METHOD THAT ENABLES BACKEND TESTING ✨

        By returning an FPGA part, we enable:
        1. SpecializeLayers transform (AddStreams → AddStreams_hls)
        2. Backend inheritance (HLSBackend mixin)
        3. cppsim/rtlsim execution (tests no longer skip)

        Without this method (or if it returns None):
        - Tests stop at Stage 2 (base kernel)
        - cppsim/rtlsim tests are skipped
        - Only Python execution works

        With this method:
        - Tests proceed to Stage 3 (backend)
        - cppsim/rtlsim tests execute
        - Complete hardware validation pipeline

        Returns:
            str: FPGA part for Xilinx Zynq-7000 (common dev board)
        """
        return "xc7z020clg400-1"

    def get_backend_type(self) -> str:
        """Backend type (hls or rtl).

        Default is "hls" (HLS backend with C++ code generation).
        Override to "rtl" for RTL backends with direct HDL generation.

        Returns:
            str: "hls" (default)
        """
        return "hls"

    # ================================================================
    # Additional Test Methods (Optional)
    # ================================================================

    @pytest.mark.pipeline
    @pytest.mark.single_kernel
    def test_backend_has_correct_inheritance(self):
        """Verify backend has HLSBackend inheritance.

        This test validates that backend specialization worked correctly.
        The specialized backend should inherit from HLSBackend, which is
        what enables cppsim/rtlsim execution.

        Pipeline:
            Stage 2: AddStreams (NOT HLSBackend)
            Stage 3: AddStreams_hls (IS HLSBackend) ← This test
        """
        from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend

        # Get Stage 3 (backend)
        backend_op, backend_model = self.run_inference_pipeline(to_backend=True)

        # Verify backend inheritance
        assert isinstance(backend_op, HLSBackend), (
            f"Backend operator should inherit from HLSBackend.\n"
            f"Got: {type(backend_op)}\n"
            f"Op type: {backend_op.onnx_node.op_type}"
        )

        # Verify op_type suffix
        assert backend_op.onnx_node.op_type.endswith("_hls"), (
            f"Backend op_type should have '_hls' suffix.\n"
            f"Got: {backend_op.onnx_node.op_type}"
        )

    @pytest.mark.pipeline
    @pytest.mark.single_kernel
    def test_base_kernel_vs_backend_comparison(self):
        """Compare base kernel (Stage 2) vs backend (Stage 3).

        This test demonstrates the difference between the two stages:
        - Stage 2: Base kernel (AddStreams)
        - Stage 3: Backend (AddStreams_hls)

        Both should preserve configuration (PE, SIMD, etc.).
        """
        from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend

        # Get Stage 2 (base kernel)
        base_op, base_model = self.run_inference_pipeline(to_backend=False)

        # Get Stage 3 (backend)
        backend_op, backend_model = self.run_inference_pipeline(to_backend=True)

        # Verify base kernel is NOT backend
        assert not isinstance(base_op, HLSBackend), (
            "Base kernel should NOT have HLSBackend inheritance"
        )

        # Verify backend IS backend
        assert isinstance(backend_op, HLSBackend), (
            "Backend should have HLSBackend inheritance"
        )

        # Verify op_type transformation
        assert base_op.onnx_node.op_type == "AddStreams"
        assert backend_op.onnx_node.op_type == "AddStreams_hls"

        # Verify configuration preserved
        base_pe = base_op.get_nodeattr("PE")
        backend_pe = backend_op.get_nodeattr("PE")
        assert base_pe == backend_pe == 8, (
            "PE configuration should be preserved through specialization"
        )


# ============================================================================
# Usage Examples
# ============================================================================

if __name__ == "__main__":
    """
    Run this test file to see the complete 3-stage pipeline in action.

    Fast tests (Stage 2 only):
        pytest tests/pipeline/test_addstreams_backend_example.py -v -m "not slow"

    Stage 3 tests (backend):
        # cppsim (requires VITIS_PATH)
        pytest tests/pipeline/test_addstreams_backend_example.py::TestAddStreamsBackendExample::test_cppsim_execution_vs_golden -v -s

        # rtlsim (requires Vivado)
        pytest tests/pipeline/test_addstreams_backend_example.py::TestAddStreamsBackendExample::test_rtlsim_execution_vs_golden -v -s

    All tests:
        pytest tests/pipeline/test_addstreams_backend_example.py -v

    Custom tests:
        # Just backend inheritance check
        pytest tests/pipeline/test_addstreams_backend_example.py::TestAddStreamsBackendExample::test_backend_has_correct_inheritance -v

        # Stage 2 vs Stage 3 comparison
        pytest tests/pipeline/test_addstreams_backend_example.py::TestAddStreamsBackendExample::test_base_kernel_vs_backend_comparison -v
    """
    pytest.main([__file__, "-v", "-s"])
