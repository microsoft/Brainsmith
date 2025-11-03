# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Shuffle parity tests: LegacyShuffle vs modern Shuffle (DualKernelTest).

This test suite validates that the legacy imperative Shuffle implementation
(LegacyShuffle with stored shapes) produces identical results to the modern
schema-driven implementation (Shuffle with extracted shapes).

Key Architectural Differences Being Tested:
- Shape storage: Legacy stores out_shape nodeattr vs modern extracts from design_point
- SIMD naming: Legacy lowercase "SIMD" vs modern uppercase "SIMD"
- Transform method: InferShuffle (external) vs Shuffle.infer_from() (built-in)
- Schema-driven: Legacy imperative vs modern declarative

Both should produce:
- Identical structural properties (shapes, widths, datatypes)
- Identical hardware estimates (cycles, resources)
- Identical execution results (Python, cppsim, rtlsim)

Test Pattern:
- Input: INT8[1, 56, 56, 128] in NHWC format
- Transpose: perm=[0, 2, 1, 3] (swap height and width dimensions)
- Output: INT8[1, 56, 56, 128] (dimensions swapped)
- SIMD=16 for 8-way folding (128 channels / 16 = 8 folds)

Test Coverage (20 inherited tests):
- 7 core parity tests (shapes, widths, datatypes)
- 5 HW estimation tests (cycles, resources)
- 8 golden execution tests (Python + cppsim + rtlsim)

Usage:
    # Run all Shuffle parity tests
    pytest tests/parity/test_shuffle_parity.py -v

    # Run only backend tests (skip Python)
    pytest tests/parity/test_shuffle_parity.py -m "cppsim or rtlsim" -v

    # Skip slow tests
    pytest tests/parity/test_shuffle_parity.py -m "not slow" -v
"""

import pytest
import numpy as np
from onnx import helper, TensorProto
from typing import Dict, Tuple, Type

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.transformation.base import Transformation
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model
import qonnx.core.data_layout as DataLayout

from brainsmith.kernels.shuffle.infer_shuffle import InferShuffle
from brainsmith.primitives.transforms.infer_kernel_list import InferKernelList

from tests.frameworks.dual_kernel_test import DualKernelTest


class TestShuffleParity(DualKernelTest):
    """Parity: LegacyShuffle (imperative) vs Shuffle (schema-driven).

    Tests tensor rearrangement (transpose) across full 3-stage pipeline:
    - Stage 1: ONNX Transpose node with perm=[0,2,1,3]
    - Stage 2: Shuffle base kernel (LegacyShuffle vs Shuffle)
    - Stage 3: Shuffle_hls backend (LegacyShuffle_hls vs Shuffle_hls)

    Test Coverage (20 tests):
    - 7 core parity tests (shapes, widths, datatypes)
    - 5 HW estimation tests (cycles, resources)
    - 8 golden execution tests (Python + cppsim + rtlsim)

    Architectural Validation:
    - Shape extraction: Stored nodeattrs vs design_point
    - SIMD naming: Lowercase vs uppercase (both accept via get_nodeattr)
    - Loop coefficients: Both use identical computation
    - HLS generation: Both use same input_gen template

    Expected: All 20 tests pass, proving implementations are equivalent.
    """

    # ========================================================================
    # Required Configuration (from KernelTestConfig)
    # ========================================================================

    def make_test_model(self) -> Tuple[ModelWrapper, str]:
        """Create ONNX model with Transpose operation.

        Pattern:
        - Input: INT8[1, 56, 56, 128] (NHWC format)
        - Transpose: perm=[0, 2, 1, 3] (swap H and W)
        - Output: INT8[1, 56, 56, 128] (H and W swapped)

        Returns:
            (model, node_name): ModelWrapper and name of Transpose node
        """
        # Configuration
        batch = 1
        h = 56
        w = 56
        ch = 128
        shape = [batch, h, w, ch]  # NHWC format
        node_name = "Transpose_shuffle_test"

        # Datatype (pass-through for Shuffle)
        idt = DataType["INT8"]
        odt = DataType["INT8"]

        # Permutation: swap height and width dimensions
        # [N, H, W, C] → [N, W, H, C]
        perm = [0, 2, 1, 3]

        # Create input/output tensor info
        inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)
        outp = helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)

        # Create ONNX Transpose node
        node = helper.make_node(
            "Transpose",
            ["input"],
            ["output"],
            name=node_name,
            perm=perm,
        )

        # Build graph and model
        graph = helper.make_graph(
            nodes=[node],
            name="test_shuffle",
            inputs=[inp],
            outputs=[outp],
        )
        model = ModelWrapper(qonnx_make_model(graph, producer_name="shuffle-parity-test"))

        # Set datatypes
        model.set_tensor_datatype("input", idt)
        model.set_tensor_datatype("output", odt)

        # Set data layout (required for FINN inference transforms)
        model.set_tensor_layout("input", DataLayout.NHWC)
        model.set_tensor_layout("output", DataLayout.NHWC)

        return model, node_name

    def get_manual_transform(self) -> Type[Transformation]:
        """Return legacy InferShuffle transform.

        Returns:
            InferShuffle transform class (creates LegacyShuffle nodes)
        """
        return InferShuffle

    def get_auto_transform(self) -> Type[Transformation]:
        """Return Brainsmith's unified InferKernelList transform.

        Returns:
            InferKernelList transform class (creates Shuffle nodes)
        """
        return InferKernelList

    def get_manual_backend_variants(self):
        """Return LegacyShuffle backend for manual pipeline.

        Manual pipeline uses InferShuffle (legacy transform) which creates
        LegacyShuffle nodes requiring LegacyShuffleHLS backend.
        """
        from brainsmith.kernels.shuffle.legacy_shuffle_hls import LegacyShuffleHLS
        return [LegacyShuffleHLS]

    def compute_golden_reference(
        self, inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Compute golden reference for transpose operation.

        Args:
            inputs: Dict with "input" key

        Returns:
            Dict with "output" key containing transposed result
        """
        # Get input data
        data = inputs["input"]

        # Apply transpose with perm=[0, 2, 1, 3]
        # This swaps height and width dimensions
        result = np.transpose(data, axes=[0, 2, 1, 3])

        return {"output": result}

    def get_num_inputs(self) -> int:
        """Shuffle has 1 input."""
        return 1

    def get_num_outputs(self) -> int:
        """Shuffle has 1 output."""
        return 1

    # ========================================================================
    # Backend Configuration
    # ========================================================================

    def get_backend_fpgapart(self) -> str:
        """Enable backend testing.

        Returns:
            FPGA part string for Zynq-7020
        """
        return "xc7z020clg400-1"

    # ========================================================================
    # Optional Configuration
    # ========================================================================

    def configure_kernel_node(self, op, model: ModelWrapper) -> None:
        """Configure Shuffle with SIMD=16.

        Args:
            op: HWCustomOp instance (LegacyShuffle or Shuffle)
            model: ModelWrapper

        Note: This is called for BOTH manual and auto pipelines.

        SIMD=16 gives 128 channels / 16 = 8-way folding.
        Both implementations should accept SIMD parameter despite
        internal case differences (legacy lowercase vs modern uppercase).
        """
        # Set SIMD for testing (128 channels / 16 = 8-way folding)
        op.set_nodeattr("SIMD", 16)

        # Set preferred implementation style for backend
        op.set_nodeattr("preferred_impl_style", "hls")


# =============================================================================
# Validation Meta-Tests
# =============================================================================


@pytest.mark.validation
def test_transforms_exist():
    """Verify both transforms are importable."""
    assert InferShuffle is not None, "InferShuffle should be importable"
    assert InferKernelList is not None, "InferKernelList should be importable"


@pytest.mark.validation
def test_backend_enabled():
    """Verify backend is enabled for Shuffle parity tests."""
    test_instance = TestShuffleParity()
    fpgapart = test_instance.get_backend_fpgapart()

    assert fpgapart is not None, "Backend should be enabled"
    assert fpgapart == "xc7z020clg400-1", f"Expected xc7z020clg400-1, got {fpgapart}"


@pytest.mark.validation
def test_test_count_correct():
    """Verify test class has correct number of tests.

    Should have 20 inherited tests from DualKernelTest.
    """
    import inspect

    # Get all test methods
    test_methods = [
        name
        for name, method in inspect.getmembers(TestShuffleParity, inspect.isfunction)
        if name.startswith("test_")
    ]

    assert len(test_methods) == 20, (
        f"TestShuffleParity should have 20 tests, "
        f"found {len(test_methods)}: {test_methods}"
    )


@pytest.mark.validation
def test_permutation_is_valid():
    """Verify test permutation [0,2,1,3] is valid for 4D tensor."""
    perm = [0, 2, 1, 3]

    # Valid permutation contains each index 0..3 exactly once
    expected_indices = set(range(4))
    actual_indices = set(perm)

    assert actual_indices == expected_indices, (
        f"Permutation {perm} is invalid. "
        f"Missing: {expected_indices - actual_indices}, "
        f"Extra: {actual_indices - expected_indices}"
    )


@pytest.mark.validation
def test_golden_reference_correct():
    """Verify golden reference computation is correct."""
    test_instance = TestShuffleParity()

    # Create test input: [1, 2, 3, 4] shape with known pattern
    # Use small dimensions for easy verification
    input_data = np.arange(1 * 2 * 3 * 4).reshape(1, 2, 3, 4)

    # Compute golden reference
    golden = test_instance.compute_golden_reference({"input": input_data})

    # Verify shape is correct (H and W swapped)
    assert golden["output"].shape == (1, 3, 2, 4), (
        f"Golden reference shape should be (1, 3, 2, 4), "
        f"got {golden['output'].shape}"
    )

    # Verify values are correctly transposed
    expected = np.transpose(input_data, axes=[0, 2, 1, 3])
    np.testing.assert_array_equal(golden["output"], expected)
