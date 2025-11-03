"""Validation test for DualKernelTest with backend specialization enabled.

This module validates Stage 5 (DualKernelTest backend support) by running
AddStreams tests with backend specialization enabled (Stage 2 → Stage 3).

Purpose:
- Validate DualKernelTest backend support works correctly
- Ensure cppsim tests specialize to Stage 3 (backend)
- Ensure rtlsim tests specialize to Stage 3 (backend)
- Ensure Python tests remain at Stage 2 (base kernel)

Validation Strategy:
1. Enable backend via get_backend_fpgapart()
2. Run all 20 tests
3. Verify cppsim/rtlsim tests reach Stage 3 (backend)
4. Verify Python tests remain at Stage 2 (base kernel)

Usage:
    # Run validation test (Python only, fast)
    pytest tests/frameworks/test_addstreams_dual_backend.py -v -m "not slow"

    # Run all tests including backend (slow)
    pytest tests/frameworks/test_addstreams_dual_backend.py -v --run-slow

    # Run only cppsim tests
    pytest tests/frameworks/test_addstreams_dual_backend.py -v -m "cppsim" --run-slow

    # Run only rtlsim tests
    pytest tests/frameworks/test_addstreams_dual_backend.py -v -m "rtlsim" --run-slow
"""

import numpy as np
import pytest
from onnx import helper, TensorProto
from typing import Dict, Tuple, Type

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.transformation.base import Transformation

from brainsmith.primitives.transforms.infer_kernel_list import InferKernelList
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferAddStreamsLayer
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp

from tests.frameworks.dual_kernel_test import DualKernelTest


# =============================================================================
# Shared Model Creation
# =============================================================================

def make_addstreams_model(
    shape: Tuple[int, ...] = (1, 64),
    dtype: DataType = DataType["INT8"]
) -> Tuple[ModelWrapper, str]:
    """Create ONNX model with Add node for AddStreams testing.

    Args:
        shape: Input tensor shape (NHWC format)
        dtype: QONNX DataType for inputs

    Returns:
        (model, node_name): ModelWrapper and name of Add node
    """
    # Convert DataType to ONNX TensorProto type
    # For integer types, use FLOAT as container (FINN convention)
    onnx_dtype = TensorProto.FLOAT

    # Create inputs
    input0 = helper.make_tensor_value_info("input0", onnx_dtype, shape)
    input1 = helper.make_tensor_value_info("input1", onnx_dtype, shape)

    # Create output
    output = helper.make_tensor_value_info("output", onnx_dtype, shape)

    # Create Add node
    add_node = helper.make_node(
        "Add", ["input0", "input1"], ["output"], name="Add_test"
    )

    # Create graph
    graph = helper.make_graph(
        [add_node], "test_addstreams", [input0, input1], [output]
    )

    # Create model
    model = helper.make_model(graph)
    model_w = ModelWrapper(model)

    # Set datatypes (FINN convention: store in model annotations)
    model_w.set_tensor_datatype("input0", dtype)
    model_w.set_tensor_datatype("input1", dtype)
    # Output datatype will be inferred (INT8 + INT8 → INT9)

    return model_w, "Add_test"


# =============================================================================
# DualKernelTest with Backend Support
# =============================================================================

class TestAddStreamsDualBackend(DualKernelTest):
    """Validate DualKernelTest with backend specialization enabled.

    This test validates Stage 5 implementation by enabling backend testing
    via get_backend_fpgapart(). It should:
    - Run all 20 tests successfully
    - Python tests use Stage 2 (base kernel)
    - cppsim tests use Stage 3 (backend with HLSBackend)
    - rtlsim tests use Stage 3 (backend with RTLBackend)

    Expected test count: 20 tests
    - 7 core parity tests (Stage 2)
    - 5 HW estimation tests (Stage 2)
    - 8 golden execution tests:
      * 2 Python tests (Stage 2: manual/auto vs golden)
      * 3 cppsim tests (Stage 3: manual/auto vs golden + parity)
      * 2 rtlsim tests (Stage 3: manual/auto vs golden)
      * 1 Python parity test (Stage 2: manual vs auto)
    """

    # ========================================================================
    # Required Configuration
    # ========================================================================

    def make_test_model(self) -> Tuple[ModelWrapper, str]:
        """Create ONNX Add node model."""
        return make_addstreams_model(shape=(1, 64), dtype=DataType["INT8"])

    def get_manual_transform(self) -> Type[Transformation]:
        """Return InferAddStreamsLayer (manual FINN transform)."""
        return InferAddStreamsLayer

    def get_auto_transform(self) -> Type[Transformation]:
        """Return InferKernelList (auto Brainsmith transform)."""
        return InferKernelList

    def get_manual_backend_variants(self):
        """Return FINN AddStreams backend for manual pipeline.

        Manual pipeline uses InferAddStreamsLayer (FINN transform) which
        creates nodes requiring FINN backends.
        """
        from finn.custom_op.fpgadataflow.hls.addstreams_hls import AddStreams_hls
        return [AddStreams_hls]

    def compute_golden_reference(
        self, inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Element-wise addition: output = input0 + input1."""
        return {"output": inputs["input0"] + inputs["input1"]}

    def get_num_inputs(self) -> int:
        """AddStreams has 2 inputs."""
        return 2

    def get_num_outputs(self) -> int:
        """AddStreams has 1 output."""
        return 1

    # ========================================================================
    # Backend Configuration (NEW - Stage 5)
    # ========================================================================

    def get_backend_fpgapart(self) -> str:
        """Enable backend testing by returning FPGA part.

        This enables Stage 2 → Stage 3 specialization:
        - AddStreams → AddStreams_hls (with HLSBackend inheritance)
        - Enables cppsim/rtlsim execution tests
        """
        return "xc7z020clg400-1"

    # ========================================================================
    # Optional Configuration
    # ========================================================================

    def configure_kernel_node(self, op: HWCustomOp, model: ModelWrapper) -> None:
        """Configure AddStreams with PE=4 for parallelism testing."""
        op.set_nodeattr("PE", 4)


# =============================================================================
# Validation Meta-Tests
# =============================================================================

@pytest.mark.validation
def test_backend_enabled():
    """Verify backend is enabled for this test."""
    test_instance = TestAddStreamsDualBackend()
    fpgapart = test_instance.get_backend_fpgapart()

    assert fpgapart is not None, "Backend should be enabled"
    assert fpgapart == "xc7z020clg400-1", f"Expected xc7z020clg400-1, got {fpgapart}"


@pytest.mark.validation
def test_backend_type_default():
    """Verify default backend type is 'hls'."""
    test_instance = TestAddStreamsDualBackend()
    backend_type = test_instance.get_backend_type()

    assert backend_type == "hls", f"Expected 'hls', got '{backend_type}'"


@pytest.mark.validation
def test_dual_kernel_test_count():
    """Verify DualKernelTest still provides 20 tests."""
    import inspect

    # Get all test methods
    test_methods = [
        name for name, method in inspect.getmembers(TestAddStreamsDualBackend, inspect.isfunction)
        if name.startswith("test_")
    ]

    assert len(test_methods) == 20, (
        f"DualKernelTest should provide 20 tests, found {len(test_methods)}: {test_methods}"
    )
