"""Dual pipeline parity tests for AddStreams kernel.

This demonstrates the new DualPipelineParityTest framework, combining:
1. Golden reference validation (NumPy ground truth)
2. Hardware parity validation (manual vs auto equivalence)

Test Coverage:
--------------
Automatically inherits ~20 tests:
- 4 golden reference tests (manual Python, auto Python, manual cppsim, auto cppsim)
- 12 hardware parity tests (shapes, widths, datatypes, cycles, resources)
- 4 integration tests (pipeline validation, specialization)

Plus AddStreams-specific tests:
- Overflow prevention (INT8 + INT8 → INT9)
- Mathematical properties (commutativity)

Usage:
------
    # Run all tests
    pytest tests/dual_pipeline/test_addstreams_dual_parity.py -v

    # Run only golden reference tests
    pytest tests/dual_pipeline/test_addstreams_dual_parity.py -v -m golden

    # Run only parity tests
    pytest tests/dual_pipeline/test_addstreams_dual_parity.py -v -m parity

    # Run fast tests (skip slow cppsim)
    pytest tests/dual_pipeline/test_addstreams_dual_parity.py -v -m "not slow"
"""

import pytest
import numpy as np
from typing import Tuple, Type

from onnx import helper, TensorProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.transformation.base import Transformation
from qonnx.util.basic import qonnx_make_model
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from brainsmith.dataflow.kernel_op import KernelOp

# Import transforms
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferAddStreamsLayer
from brainsmith.primitives.transforms.infer_kernel_list import InferKernelList

# Import kernel for golden reference
from brainsmith.kernels.addstreams import AddStreams

# Import dual pipeline framework
from tests.dual_pipeline import DualPipelineParityTest


class TestAddStreamsDualParity(DualPipelineParityTest):
    """Comprehensive dual pipeline parity testing for AddStreams.

    This single test class provides:
    - Manual (FINN) vs NumPy golden reference validation
    - Auto (Brainsmith) vs NumPy golden reference validation
    - Manual vs Auto hardware parity validation

    Total: ~22 tests automatically inherited + 2 AddStreams-specific
    """

    # =========================================================================
    # Required Configuration
    # =========================================================================

    def make_test_model(self) -> Tuple[ModelWrapper, str]:
        """Create ONNX Add node for AddStreams inference.

        Creates standard ONNX Add node (not AddStreams).
        Both manual and auto pipelines will transform it.

        Returns:
            (model, node_name): Model with Add node and its name
        """
        # Test configuration: 4D NHWC layout
        ch = 64  # Channels (must be divisible by PE)
        h = w = 56  # Spatial dimensions
        batch = 1

        # Create ONNX graph with Add node
        inp1 = helper.make_tensor_value_info(
            "inp1", TensorProto.FLOAT, [batch, h, w, ch]
        )
        inp2 = helper.make_tensor_value_info(
            "inp2", TensorProto.FLOAT, [batch, h, w, ch]
        )
        outp = helper.make_tensor_value_info(
            "outp", TensorProto.FLOAT, [batch, h, w, ch]
        )

        add_node = helper.make_node(
            "Add",
            inputs=["inp1", "inp2"],
            outputs=["outp"],
            name="Add_test"
        )

        graph = helper.make_graph(
            nodes=[add_node],
            name="addstreams_dual_test",
            inputs=[inp1, inp2],
            outputs=[outp]
        )

        model = qonnx_make_model(graph, producer_name="addstreams-dual-parity-test")
        model = ModelWrapper(model)

        # Set integer datatypes (required for AddStreams)
        model.set_tensor_datatype("inp1", DataType["INT8"])
        model.set_tensor_datatype("inp2", DataType["INT8"])
        model.set_tensor_datatype("outp", DataType["INT8"])

        return model, "Add_test"

    def get_manual_transform(self) -> Type[Transformation]:
        """Return FINN's InferAddStreamsLayer transform."""
        return InferAddStreamsLayer

    def get_auto_transform(self) -> Type[Transformation]:
        """Return Brainsmith's unified InferKernelList transform."""
        return InferKernelList

    def get_kernel_class(self) -> Type[KernelOp]:
        """Return AddStreams class for golden reference."""
        return AddStreams

    def get_num_inputs(self) -> int:
        """AddStreams has 2 inputs."""
        return 2

    def get_num_outputs(self) -> int:
        """AddStreams has 1 output."""
        return 1

    def configure_kernel_node(
        self, op: HWCustomOp, model: ModelWrapper, is_manual: bool
    ) -> None:
        """Configure AddStreams node identically for both implementations.

        Args:
            op: AddStreams operator instance
            model: ModelWrapper containing the op
            is_manual: True if manual FINN implementation
        """
        # Set PE for testing (64 channels / 8 = 8-way folding)
        # Configure BOTH implementations identically for fair comparison
        op.set_nodeattr("PE", 8)

    # =========================================================================
    # AddStreams-Specific Tests
    # =========================================================================

    @pytest.mark.dual_pipeline
    @pytest.mark.parity
    def test_overflow_prevention_both_implementations(self):
        """Both implementations must widen INT8 + INT8 → INT9.

        This is critical business logic: AddStreams must prevent overflow
        by widening the output datatype.

        INT8 range: [-128, 127]
        INT8 + INT8 worst case: -128 + (-128) = -256, 127 + 127 = 254
        Requires INT9 range: [-256, 255]
        """
        manual_op, _ = self.run_manual_pipeline()
        auto_op, _ = self.run_auto_pipeline()

        # Both must calculate INT9 for overflow prevention
        manual_output_dt = manual_op.get_output_datatype(0)
        auto_output_dt = auto_op.get_output_datatype(0)

        assert manual_output_dt == DataType["INT9"], (
            f"Manual implementation should widen to INT9, got {manual_output_dt.name}"
        )
        assert auto_output_dt == DataType["INT9"], (
            f"Auto implementation should widen to INT9, got {auto_output_dt.name}"
        )

    @pytest.mark.dual_pipeline
    @pytest.mark.golden
    def test_commutativity_both_implementations(self):
        """Both implementations must satisfy a + b == b + a.

        This validates the mathematical property of addition for both
        manual and auto implementations.
        """
        # Setup both pipelines
        manual_op, manual_model = self.run_manual_pipeline()
        auto_op, auto_model = self.run_auto_pipeline()

        # Create test inputs: (a, b)
        np.random.seed(42)
        shape = manual_op.get_normal_input_shape(0)
        input_a = np.random.randint(-128, 128, shape).astype(np.float32)
        input_b = np.random.randint(-128, 128, shape).astype(np.float32)

        # Test manual implementation: a + b == b + a
        output_name = manual_op.onnx_node.output[0]
        output_shape = manual_model.get_tensor_shape(output_name)
        output_dtype = manual_model.get_tensor_datatype(output_name)

        context_manual_ab = {
            manual_op.onnx_node.input[0]: input_a,
            manual_op.onnx_node.input[1]: input_b,
            output_name: np.zeros(output_shape, dtype=output_dtype.to_numpy_dt()),
        }
        context_manual_ba = {
            manual_op.onnx_node.input[0]: input_b,
            manual_op.onnx_node.input[1]: input_a,
            output_name: np.zeros(output_shape, dtype=output_dtype.to_numpy_dt()),
        }

        manual_op.execute_node(context_manual_ab, manual_model.graph)
        manual_op.execute_node(context_manual_ba, manual_model.graph)

        manual_ab = context_manual_ab[output_name]
        manual_ba = context_manual_ba[output_name]

        np.testing.assert_array_equal(
            manual_ab, manual_ba,
            err_msg="Manual implementation: a + b != b + a (not commutative)"
        )

        # Test auto implementation: a + b == b + a
        output_name = auto_op.onnx_node.output[0]
        output_shape = auto_model.get_tensor_shape(output_name)
        output_dtype = auto_model.get_tensor_datatype(output_name)

        context_auto_ab = {
            auto_op.onnx_node.input[0]: input_a,
            auto_op.onnx_node.input[1]: input_b,
            output_name: np.zeros(output_shape, dtype=output_dtype.to_numpy_dt()),
        }
        context_auto_ba = {
            auto_op.onnx_node.input[0]: input_b,
            auto_op.onnx_node.input[1]: input_a,
            output_name: np.zeros(output_shape, dtype=output_dtype.to_numpy_dt()),
        }

        auto_op.execute_node(context_auto_ab, auto_model.graph)
        auto_op.execute_node(context_auto_ba, auto_model.graph)

        auto_ab = context_auto_ab[output_name]
        auto_ba = context_auto_ba[output_name]

        np.testing.assert_array_equal(
            auto_ab, auto_ba,
            err_msg="Auto implementation: a + b != b + a (not commutative)"
        )


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
