"""Parity tests for Softmax: manual HWSoftmax vs AutoSoftmax.

NOTE: The manual Softmax implementation (HWSoftmax) has been removed and replaced
in-place with the unified Softmax implementation. This parity test file is retained
as a reference example showing how to perform parity testing when both manual and
auto implementations exist.

IMPORTANT: This test file cannot run anymore since manual HWSoftmax has been removed.
All tests in this file are skipped. This file serves only as documentation/reference.

This module originally validated equivalence between:
- Manual implementation: HWSoftmax (from hwsoftmax.py via InferHWSoftmax) [REMOVED]
- Auto implementation: AutoSoftmax (from auto_softmax.py via InferAutoSoftmax) [NOW: Softmax]

Tests verify that both implementations produce identical results for:
- Shape methods (normal, folded)
- Stream widths
- Datatypes
- Expected cycles
- Execution (numerical correctness)

Key Feature:
- Uses actual Infer transforms (InferHWSoftmax, InferAutoSoftmax) to create ops
- Tests the real production workflow: ONNX Softmax → HWSoftmax/AutoSoftmax
"""

import pytest

# Skip entire module - manual Softmax implementation has been removed
pytestmark = pytest.mark.skip(
    reason="Manual HWSoftmax removed - unified Softmax implementation now in place. "
           "This file is retained as a reference example only."
)

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.custom_op.registry import getCustomOp

from tests.parity import ParityTestBase
from tests.fixtures.model_utils import create_softmax_model
# Old imports (no longer valid - kept for reference):
# from brainsmith.kernels.softmax import HWSoftmax, AutoSoftmax, InferHWSoftmax, InferAutoSoftmax
# New imports:
# from brainsmith.kernels.softmax import Softmax, SoftmaxHLS, InferSoftmax

# Dummy placeholders to prevent NameError in class definitions
HWSoftmax = None
AutoSoftmax = None
InferHWSoftmax = None
InferAutoSoftmax = None


class TestSoftmaxParity(ParityTestBase):
    """Parity tests for Softmax kernel implementations via Infer transforms.

    Validates equivalence between manual HWSoftmax and AutoSoftmax
    implementations using the ParityTestBase framework.

    This test uses the actual production workflow:
    1. Create standard ONNX Softmax node
    2. Apply InferHWSoftmax transform → HWSoftmax node
    3. Apply InferAutoSoftmax transform → AutoSoftmax node
    4. Compare the resulting operations

    Test configuration:
    - Input shape: [1, 128, 768] (batch × seq_len × channels)
    - Output shape: [1, 128, 768] (shape-preserving)
    - SIMD: 16 (parallelization factor)
    - Input datatype: INT8
    - Output datatype: FLOAT32 (Softmax always outputs probabilities)

    Known Limitations:
    - Manual HWSoftmax doesn't implement get_exp_cycles(), so that test is skipped
    """

    @pytest.mark.parity
    @pytest.mark.skip(reason="Manual HWSoftmax doesn't implement get_exp_cycles()")
    def test_exp_cycles_parity(self):
        """Skip exp_cycles test - manual implementation doesn't have this method."""
        pass

    @property
    def manual_op_class(self):
        """Manual HWSoftmax implementation."""
        return HWSoftmax

    @property
    def auto_op_class(self):
        """Auto AutoSoftmax implementation."""
        return AutoSoftmax

    def make_test_model(self):
        """Create ONNX model with standard Softmax node.

        Returns base ONNX Softmax model that will be transformed by
        InferHWSoftmax or InferAutoSoftmax.

        Returns:
            Tuple of (ModelWrapper, node_name)
        """
        # Create ONNX model with standard Softmax operation
        # Shape: [1, 128, 768] = batch × seq_len × channels
        model_proto = create_softmax_model(
            batch_size=1,
            seq_len=128,
            channels=768,
            input_dtype="INT8",
            output_dtype="FLOAT32"
        )

        # Wrap in ModelWrapper
        model = ModelWrapper(model_proto)

        # Set tensor datatypes (required for transforms)
        model.set_tensor_datatype("input", DataType["INT8"])
        model.set_tensor_datatype("output", DataType["FLOAT32"])

        # Get the Softmax node name
        softmax_node = model.graph.node[0]
        node_name = softmax_node.name if softmax_node.name else "Softmax_0"

        return model, node_name

    def get_manual_transform(self):
        """Return the manual Infer transform class."""
        return InferHWSoftmax

    def get_auto_transform(self):
        """Return the auto Infer transform class."""
        return InferAutoSoftmax

    def configure_test_op(self, op, model, is_auto):
        """Configure op for testing - override SIMD to 16."""
        op.set_nodeattr("SIMD", 16)
        if is_auto:
            # Auto ops need state refresh after SIMD change
            op.refresh_df_model(model)


# Additional test: Verify Softmax numerical correctness with custom tolerance
class TestSoftmaxNumericalParity(TestSoftmaxParity):
    """Extended parity tests with custom numerical tolerances.

    Softmax involves exponentials and division, which can introduce
    floating-point precision differences between implementations.
    This test class uses relaxed tolerances appropriate for Softmax.

    Inherits setup_manual_op() and setup_auto_op() from parent class,
    so also uses InferHWSoftmax and InferAutoSoftmax transforms.
    """

    @pytest.mark.parity
    @pytest.mark.skip(reason="Manual HWSoftmax doesn't implement get_exp_cycles()")
    def test_exp_cycles_parity(self):
        """Skip exp_cycles test - manual implementation doesn't have this method."""
        pass

    @pytest.mark.parity
    def test_execute_node_parity_relaxed(self):
        """Test Python execution with relaxed tolerance for Softmax.

        Softmax computation involves:
        1. Max extraction (exact)
        2. Exponentiation (floating-point)
        3. Sum reduction (accumulation error)
        4. Division (floating-point)

        These operations can introduce small numerical differences
        between manual and auto implementations due to:
        - Different computation order
        - Fixed-point accumulation in HLS vs floating-point in Python
        - Rounding in intermediate steps

        Uses relaxed tolerance: rtol=1e-4, atol=1e-6
        """
        import numpy as np

        manual_op, manual_model = self.setup_manual_op()
        auto_op, auto_model = self.setup_auto_op()

        # Create execution context with same random inputs
        np.random.seed(42)  # Deterministic for reproducibility
        manual_context = self._make_execution_context(manual_model, manual_op)

        np.random.seed(42)  # Same seed for auto
        auto_context = self._make_execution_context(auto_model, auto_op)

        # Execute both
        manual_op.execute_node(manual_context, manual_model.graph)
        auto_op.execute_node(auto_context, auto_model.graph)

        # Compare outputs with relaxed tolerance
        manual_output = manual_context[manual_op.onnx_node.output[0]]
        auto_output = auto_context[auto_op.onnx_node.output[0]]

        # Verify output is valid probability distribution (sum to 1, all positive)
        assert np.all(manual_output >= 0), "Manual output has negative probabilities"
        assert np.all(auto_output >= 0), "Auto output has negative probabilities"

        # Check sum along channel dimension (should be ~1.0 for each position)
        manual_sums = np.sum(manual_output, axis=-1)
        auto_sums = np.sum(auto_output, axis=-1)
        np.testing.assert_allclose(
            manual_sums, np.ones_like(manual_sums), rtol=1e-5, atol=1e-6,
            err_msg="Manual output probabilities don't sum to 1"
        )
        np.testing.assert_allclose(
            auto_sums, np.ones_like(auto_sums), rtol=1e-5, atol=1e-6,
            err_msg="Auto output probabilities don't sum to 1"
        )

        # Compare manual vs auto with relaxed tolerance
        np.testing.assert_allclose(
            manual_output,
            auto_output,
            rtol=1e-4,  # Relaxed relative tolerance for Softmax
            atol=1e-6,  # Absolute tolerance
            err_msg="Output execution results differ beyond acceptable tolerance"
        )
