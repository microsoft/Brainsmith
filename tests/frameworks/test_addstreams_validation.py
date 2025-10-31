"""Validation tests for new frameworks using AddStreams.

This module validates Phase 2 frameworks by running AddStreams tests using both
SingleKernelTest and DualKernelTest, then comparing results with old frameworks.

Purpose:
- Validate SingleKernelTest produces correct results
- Validate DualKernelTest produces correct results
- Ensure new frameworks match old framework behavior

Validation Strategy:
1. Run new frameworks on AddStreams
2. Compare test counts (6 for Single, 20 for Dual)
3. Compare execution results with old frameworks

Usage:
    # Run validation tests
    pytest tests/frameworks/test_addstreams_validation.py -v

    # Compare with old frameworks
    pytest tests/pipeline/test_addstreams_integration.py -v
    pytest tests/dual_pipeline/test_addstreams_v2.py -v
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

from tests.frameworks.single_kernel_test import SingleKernelTest
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
# SingleKernelTest Validation
# =============================================================================

class TestAddStreamsSingle(SingleKernelTest):
    """Validate SingleKernelTest framework using AddStreams.

    This test should produce identical results to TestAddStreamsIntegration
    from tests/pipeline/test_addstreams_integration.py.

    Expected test count: 6 tests
    - test_pipeline_creates_hw_node
    - test_shapes_preserved_through_pipeline
    - test_datatypes_preserved_through_pipeline
    - test_python_execution_vs_golden
    - test_cppsim_execution_vs_golden
    - test_rtlsim_execution_vs_golden
    """

    # ========================================================================
    # Required Configuration
    # ========================================================================

    def make_test_model(self) -> Tuple[ModelWrapper, str]:
        """Create ONNX Add node model."""
        return make_addstreams_model(shape=(1, 64), dtype=DataType["INT8"])

    def get_kernel_inference_transform(self) -> Type[Transformation]:
        """Return InferKernelList (auto Brainsmith transform)."""
        return InferKernelList

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
    # Optional Configuration
    # ========================================================================

    def configure_kernel_node(self, op: HWCustomOp, model: ModelWrapper) -> None:
        """Configure AddStreams with PE=4 for parallelism testing."""
        op.set_nodeattr("PE", 4)


# =============================================================================
# DualKernelTest Validation
# =============================================================================

class TestAddStreamsDual(DualKernelTest):
    """Validate DualKernelTest framework using AddStreams.

    This test should produce parity results for manual (FINN) vs auto (Brainsmith)
    implementations, plus golden reference validation for both.

    Expected test count: 20 tests
    - 7 core parity tests (shapes, widths, datatypes)
    - 5 HW estimation tests (cycles, resources)
    - 8 golden execution tests (manual/auto vs golden + parity)
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
    # Optional Configuration
    # ========================================================================

    def configure_kernel_node(self, op: HWCustomOp, model: ModelWrapper) -> None:
        """Configure AddStreams with PE=4 for parallelism testing.

        Note: This is called for BOTH manual and auto pipelines.
        The old frameworks had separate configurations, but unified
        configuration is simpler and covers 99% of use cases.
        """
        op.set_nodeattr("PE", 4)


# =============================================================================
# Validation Meta-Tests
# =============================================================================

@pytest.mark.validation
def test_single_kernel_test_count():
    """Verify SingleKernelTest provides 6 tests."""
    import inspect

    # Get all test methods
    test_methods = [
        name for name, method in inspect.getmembers(TestAddStreamsSingle, inspect.isfunction)
        if name.startswith("test_")
    ]

    assert len(test_methods) == 6, (
        f"SingleKernelTest should provide 6 tests, found {len(test_methods)}: {test_methods}"
    )

    # Verify expected test names
    expected_tests = {
        "test_pipeline_creates_hw_node",
        "test_shapes_preserved_through_pipeline",
        "test_datatypes_preserved_through_pipeline",
        "test_python_execution_vs_golden",
        "test_cppsim_execution_vs_golden",
        "test_rtlsim_execution_vs_golden",
    }

    actual_tests = set(test_methods)

    assert actual_tests == expected_tests, (
        f"Test names mismatch.\n"
        f"Expected: {expected_tests}\n"
        f"Actual: {actual_tests}\n"
        f"Missing: {expected_tests - actual_tests}\n"
        f"Extra: {actual_tests - expected_tests}"
    )


@pytest.mark.validation
def test_dual_kernel_test_count():
    """Verify DualKernelTest provides 20 tests."""
    import inspect

    # Get all test methods
    test_methods = [
        name for name, method in inspect.getmembers(TestAddStreamsDual, inspect.isfunction)
        if name.startswith("test_")
    ]

    assert len(test_methods) == 20, (
        f"DualKernelTest should provide 20 tests, found {len(test_methods)}: {test_methods}"
    )

    # Verify test categories
    parity_tests = [name for name in test_methods if "parity" in name]
    golden_tests = [name for name in test_methods if "golden" in name or ("manual" in name or "auto" in name)]

    # Should have 7 core parity + 5 HW parity = 12 parity tests
    # Should have 8 golden/execution tests
    assert len(parity_tests) >= 12, f"Expected ≥12 parity tests, found {len(parity_tests)}"
    assert len(golden_tests) >= 8, f"Expected ≥8 golden tests, found {len(golden_tests)}"
