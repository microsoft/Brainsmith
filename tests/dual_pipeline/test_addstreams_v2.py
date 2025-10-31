"""Test new DualPipelineParityTest architecture with AddStreams.

This demonstrates the new modular architecture where:
- DualPipelineParityTest inherits from CoreParityTest + HWEstimationParityTest
- Total: 14 inherited tests + 2 AddStreams-specific tests = 16 tests

Test Coverage:
- 7 core parity tests (shapes, widths, datatypes)
- 5 HW estimation tests (resources, cycles)
- 2 golden execution tests (Python: manual/auto)
- 2 AddStreams-specific tests (overflow prevention, commutativity)

Usage:
------
    # Run all tests
    pytest tests/dual_pipeline/test_addstreams_v2.py -v

    # Run only parity tests
    pytest tests/dual_pipeline/test_addstreams_v2.py -v -m parity

    # Run only golden tests
    pytest tests/dual_pipeline/test_addstreams_v2.py -v -m golden

    # Run fast tests (skip slow cppsim)
    pytest tests/dual_pipeline/test_addstreams_v2.py -v -m "not slow"
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

# Import transforms
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferAddStreamsLayer
from brainsmith.primitives.transforms.infer_kernel_list import InferKernelList

# Import NEW dual pipeline framework
from tests.dual_pipeline.dual_pipeline_parity_test_v2 import DualPipelineParityTest


class TestAddStreamsV2(DualPipelineParityTest):
    """Test AddStreams using new modular architecture.

    Inherits 16 tests:
    - 7 core parity tests (shapes, widths, datatypes)
    - 5 HW estimation tests (resources, cycles)
    - 4 golden execution tests (manual/auto × Python/cppsim)
    """

    # =========================================================================
    # Required Configuration
    # =========================================================================

    def make_test_model(self) -> Tuple[ModelWrapper, str]:
        """Create ONNX Add node for AddStreams inference."""
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
            name="addstreams_v2_test",
            inputs=[inp1, inp2],
            outputs=[outp]
        )

        model = qonnx_make_model(graph, producer_name="addstreams-v2-test")
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

    def get_num_inputs(self) -> int:
        """AddStreams has 2 inputs."""
        return 2

    def get_num_outputs(self) -> int:
        """AddStreams has 1 output."""
        return 1

    def configure_kernel_node(
        self, op: HWCustomOp, model: ModelWrapper, is_manual: bool
    ) -> None:
        """Configure AddStreams node identically for both implementations."""
        from brainsmith.dataflow.kernel_op import KernelOp

        # Set PE for testing (64 channels / 8 = 8-way folding)
        op.set_nodeattr("PE", 8)

        # Reset design space after configuration (PE changed)
        if isinstance(op, KernelOp):
            op._ensure_ready(model)

    # =========================================================================
    # Test-Owned Golden Reference
    # =========================================================================

    def compute_golden_reference(self, inputs: dict) -> dict:
        """NumPy golden reference for AddStreams - test-owned!

        This is test logic, not kernel logic. The test defines what "correct"
        means for element-wise addition.

        Args:
            inputs: Dict with "input0" and "input1" numpy arrays

        Returns:
            Dict with "output" = input0 + input1
        """
        return {"output": inputs["input0"] + inputs["input1"]}

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
