# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""DuplicateStreams parity tests with backend support (DualKernelTest).

This test demonstrates Stage 6: enabling backend testing in production kernels
using the new DualKernelTest framework.

Migration from old framework:
- OLD: tests/parity/base_parity_test.py (ParityTestBase)
- NEW: tests/frameworks/dual_kernel_test.py (DualKernelTest)

Benefits of new framework:
- Unified test infrastructure (composition pattern)
- 3-stage pipeline coverage (ONNX → Base → Backend)
- Backend testing via simple configuration hook
- 20 inherited tests automatically
- Clear separation of concerns

This test suite validates:
- FINN DuplicateStreams vs Brainsmith DuplicateStreams (manual vs auto)
- Both implementations at Stage 2 (base kernel - Python)
- Both implementations at Stage 3 (backend - cppsim/rtlsim)
- Complete functional parity across all execution modes
"""

import numpy as np
import pytest
from onnx import helper, TensorProto
from typing import Dict, Tuple, Type

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.transformation.base import Transformation

from finn.transformation.fpgadataflow.convert_to_hw_layers import InferDuplicateStreamsLayer
from brainsmith.primitives.transforms.infer_kernel_list import InferKernelList

from tests.frameworks.dual_kernel_test import DualKernelTest


class TestDuplicateStreamsDualBackend(DualKernelTest):
    """Parity: FINN DuplicateStreams vs Brainsmith DuplicateStreams (with backend).

    Validates manual (FINN) vs auto (Brainsmith) parity for DuplicateStreams
    across all 3 pipeline stages:
    - Stage 1: ONNX Add node
    - Stage 2: Base DuplicateStreams kernel (Python execution)
    - Stage 3: DuplicateStreams_hls backend (cppsim/rtlsim execution)

    Test Coverage (20 inherited tests):
    - 7 core parity tests (shapes, widths, datatypes at Stage 2)
    - 5 HW estimation tests (cycles, resources at Stage 2)
    - 8 golden execution tests:
      * 2 Python tests (Stage 2: manual/auto vs golden)
      * 3 cppsim tests (Stage 3: manual/auto vs golden + parity)
      * 2 rtlsim tests (Stage 3: manual/auto vs golden)
      * 1 Python parity test (Stage 2: manual vs auto)

    Kernel Properties:
    - Function: Stream duplication (1 input → N outputs)
    - Category: Routing (no computation)
    - Outputs: Variable count (2 in this test)
    - Parallelism: PE=8 (8-way channel parallelism)

    Example Configuration:
    - Shape: [1, 8, 8, 64] (batch=1, H=8, W=8, C=64)
    - Datatype: INT8
    - PE: 8
    - NumOutputStreams: 2
    """

    # ========================================================================
    # Required Configuration (from KernelTestConfig)
    # ========================================================================

    def make_test_model(self) -> Tuple[ModelWrapper, str]:
        """Create ONNX model with Identity node (will become DuplicateStreams).

        DuplicateStreams is inferred from Identity nodes with multiple outputs.

        Returns:
            (model, node_name): ModelWrapper and name of Identity node
        """
        # Create Identity node with 2 outputs (will become DuplicateStreams)
        # FINN/Brainsmith infer DuplicateStreams from Identity with fanout
        node = helper.make_node(
            "Identity",
            inputs=["inp"],
            outputs=["out0", "out1"],
            name="Identity_dup"
        )

        shape = [1, 8, 8, 64]  # NHWC: batch=1, H=8, W=8, C=64
        inp_vi = helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape)
        out0_vi = helper.make_tensor_value_info("out0", TensorProto.FLOAT, shape)
        out1_vi = helper.make_tensor_value_info("out1", TensorProto.FLOAT, shape)

        graph = helper.make_graph(
            [node],
            "test_duplicate_streams",
            [inp_vi],
            [out0_vi, out1_vi]
        )

        model = ModelWrapper(helper.make_model(graph))

        # Set datatypes
        dtype = DataType["INT8"]
        model.set_tensor_datatype("inp", dtype)
        model.set_tensor_datatype("out0", dtype)
        model.set_tensor_datatype("out1", dtype)

        return model, "Identity_dup"

    def get_manual_transform(self) -> Type[Transformation]:
        """Return FINN's InferDuplicateStreamsLayer transform."""
        return InferDuplicateStreamsLayer

    def get_auto_transform(self) -> Type[Transformation]:
        """Return Brainsmith's InferKernelList transform."""
        return InferKernelList

    def compute_golden_reference(
        self, inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Compute golden reference: output0 = output1 = input (fanout).

        DuplicateStreams duplicates input to all outputs.

        Args:
            inputs: Dict with "inp" key

        Returns:
            Dict with "out0" and "out1" keys (both identical to input)
        """
        inp = inputs["inp"]
        return {
            "out0": inp,  # Duplicate to output 0
            "out1": inp,  # Duplicate to output 1
        }

    def get_num_inputs(self) -> int:
        """DuplicateStreams has 1 input."""
        return 1

    def get_num_outputs(self) -> int:
        """DuplicateStreams has 2 outputs (in this test configuration)."""
        return 2

    # ========================================================================
    # Backend Configuration (NEW - Stage 6)
    # ========================================================================

    def get_backend_fpgapart(self) -> str:
        """Enable backend testing by returning FPGA part.

        This enables Stage 2 → Stage 3 specialization:
        - DuplicateStreams → DuplicateStreams_hls (with HLSBackend inheritance)
        - Enables cppsim/rtlsim execution tests

        Returns:
            FPGA part string for Zynq-7020
        """
        return "xc7z020clg400-1"

    # ========================================================================
    # Optional Configuration
    # ========================================================================

    def configure_kernel_node(self, op, model: ModelWrapper) -> None:
        """Configure DuplicateStreams with PE=8.

        Args:
            op: HWCustomOp instance (DuplicateStreams)
            model: ModelWrapper

        Note: This is called for BOTH manual and auto pipelines.
        """
        op.set_nodeattr("PE", 8)


# =============================================================================
# Validation Meta-Tests
# =============================================================================

@pytest.mark.validation
def test_backend_enabled():
    """Verify backend is enabled for DuplicateStreams test."""
    test_instance = TestDuplicateStreamsDualBackend()
    fpgapart = test_instance.get_backend_fpgapart()

    assert fpgapart is not None, "Backend should be enabled"
    assert fpgapart == "xc7z020clg400-1", f"Expected xc7z020clg400-1, got {fpgapart}"


@pytest.mark.validation
def test_dual_kernel_test_count():
    """Verify DualKernelTest provides 20 tests."""
    import inspect

    # Get all test methods
    test_methods = [
        name for name, method in inspect.getmembers(
            TestDuplicateStreamsDualBackend,
            inspect.isfunction
        )
        if name.startswith("test_")
    ]

    assert len(test_methods) == 20, (
        f"DualKernelTest should provide 20 tests, found {len(test_methods)}: {test_methods}"
    )


@pytest.mark.validation
def test_multiple_outputs_handled():
    """Verify framework handles multiple outputs correctly."""
    test_instance = TestDuplicateStreamsDualBackend()

    assert test_instance.get_num_outputs() == 2, "Should have 2 outputs"

    # Verify golden reference produces both outputs
    inputs = {"inp": np.array([[[[1, 2, 3, 4]]]])}
    golden = test_instance.compute_golden_reference(inputs)

    assert "out0" in golden, "Golden should have out0"
    assert "out1" in golden, "Golden should have out1"
    assert np.array_equal(golden["out0"], inputs["inp"]), "out0 should equal input"
    assert np.array_equal(golden["out1"], inputs["inp"]), "out1 should equal input"
