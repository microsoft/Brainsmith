"""Test parity between Brainsmith ElementwiseBinary and FINN ElementwiseAdd (v6.0).

This test demonstrates the KernelParityTest framework by comparing two implementations:
- Primary (Brainsmith): InferKernels → ElementwiseBinaryOp → ElementwiseBinaryOp_hls
- Reference (FINN): InferElementwiseBinaryOperation → ElementwiseAdd → ElementwiseAdd_hls

Provides 18 inherited tests:
- 6 golden execution tests (primary/reference × python/cppsim/rtlsim)
- 7 core parity tests (shapes, widths, datatypes)
- 5 HW estimation tests (cycles, resources, efficiency)
"""

import pytest
import numpy as np
import onnx.helper as helper
from onnx import TensorProto

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from tests.frameworks.kernel_parity_test import KernelParityTest
from tests.frameworks.test_config import (
    KernelTestConfig,
    ModelStructure,
    PlatformConfig,
)


class TestAddParity(KernelParityTest):
    """Test parity between Brainsmith ElementwiseBinary (primary) and FINN ElementwiseAdd (reference)."""

    # ========================================================================
    # Test Configuration Fixture
    # ========================================================================

    @pytest.fixture(
        params=[
            KernelTestConfig(
                test_id="add_int8_baseline",
                model=ModelStructure(
                    operation="Add",
                    input_shapes={"input0": (1, 64), "input1": (1, 64)},
                    input_dtypes={"input0": DataType["INT8"], "input1": DataType["INT8"]},
                ),
                platform=PlatformConfig(fpgapart="xc7z020clg400-1"),
            ),
        ]
    )
    def kernel_test_config(self, request):
        """Provide test configurations via fixture.

        Test execution controlled by pytest marks:
        - Python only: pytest test_add_parity.py -m "not slow" -v
        - Skip backend: pytest test_add_parity.py -m "not cppsim and not rtlsim" -v
        - Only cppsim: pytest test_add_parity.py -m "cppsim" -v
        - Only rtlsim: pytest test_add_parity.py -m "rtlsim" -v
        - All tests: pytest test_add_parity.py -v
        """
        return request.param

    # ========================================================================
    # Shared Model Creation
    # ========================================================================

    def make_test_model(self, kernel_test_config):
        """Create ONNX model with Add node.

        Args:
            kernel_test_config: Configuration containing input shapes and datatypes

        Returns:
            Tuple of (model, input_names) - input names for DataType annotation
        """
        # Get shapes from config
        shape0 = list(kernel_test_config.input_shapes["input0"])
        shape1 = list(kernel_test_config.input_shapes["input1"])

        # Create tensor infos with FLOAT container (QONNX convention)
        inp0 = helper.make_tensor_value_info("input0", TensorProto.FLOAT, shape0)
        inp1 = helper.make_tensor_value_info("input1", TensorProto.FLOAT, shape1)
        out = helper.make_tensor_value_info("output", TensorProto.FLOAT, shape0)

        # Create Add ONNX node
        node = helper.make_node(
            "Add",
            ["input0", "input1"],
            ["output"],
            name="Add_0"
        )

        # Create graph and model
        graph = helper.make_graph(
            [node],
            "test_elementwise_add_parity",
            [inp0, inp1],
            [out]
        )

        model = ModelWrapper(qonnx_make_model(graph))

        # Return model and input names for DataType annotation
        return model, ["input0", "input1"]

    # ========================================================================
    # Reference Implementation: FINN ElementwiseAdd
    # ========================================================================

    def infer_kernel_reference(self, model, target_node):
        """Infer reference kernel using FINN InferElementwiseBinaryOperation.

        Applies dtype optimization transforms to match Brainsmith's built-in optimization.
        FINN has dtype optimization as separate transforms, while Brainsmith has it
        built into the schema. We apply FINN's minimize transforms here to achieve parity.

        Args:
            model: Stage 1 ONNX model
            target_node: Name of the original Add node (before transformation)

        Returns:
            Tuple of (op, model) where op is ElementwiseAdd instance
        """
        from finn.transformation.fpgadataflow.convert_to_hw_layers import (
            InferElementwiseBinaryOperation,
        )
        from finn.transformation.fpgadataflow.minimize_accumulator_width import (
            MinimizeAccumulatorWidth,
        )
        from finn.transformation.fpgadataflow.minimize_weight_bit_width import (
            MinimizeWeightBitWidth,
        )
        from finn.transformation.streamline.round_thresholds import (
            RoundAndClipThresholds,
        )
        from qonnx.transformation.infer_datatypes import InferDataTypes

        # Apply FINN transform
        # NOTE: FINN creates a new node with op_type "ElementwiseAdd" but doesn't
        # preserve the original node name, so we need to find it by op_type
        model = model.transform(InferElementwiseBinaryOperation())

        # Apply FINN dtype optimization transforms (matches Brainsmith's built-in optimization)
        # These transforms are from FINN's step_minimize_bit_width
        model = model.transform(MinimizeWeightBitWidth())
        model = model.transform(MinimizeAccumulatorWidth())
        model = model.transform(RoundAndClipThresholds())
        model = model.transform(InferDataTypes())

        # Find the ElementwiseAdd node by op_type (FINN doesn't preserve node name)
        nodes_by_op_type = model.get_nodes_by_op_type("ElementwiseAdd")
        assert len(nodes_by_op_type) == 1, (
            f"Expected exactly 1 ElementwiseAdd node, found {len(nodes_by_op_type)}"
        )

        # Get the ONNX node and wrap with custom op
        onnx_node = nodes_by_op_type[0]
        from qonnx.custom_op.registry import getCustomOp
        op = getCustomOp(onnx_node)

        return op, model

    def get_backend_variants_reference(self):
        """Return FINN backend variants for ElementwiseAdd.

        Returns:
            List of backend classes [ElementwiseAdd_hls]
        """
        from finn.custom_op.fpgadataflow.hls.elementwise_binary_hls import (
            ElementwiseAdd_hls,
        )
        return [ElementwiseAdd_hls]


    # No configure_kernel_reference() override needed - uses default auto_configure_from_fixture()

    # ========================================================================
    # Primary Implementation: Brainsmith ElementwiseBinary (uses defaults)
    # ========================================================================

    def get_kernel_inference_transform(self):
        """Return Brainsmith InferKernels transform for primary implementation.

        Required by base class. The inherited infer_kernel() implementation
        calls this method to get the kernel inference transform.

        Returns:
            InferKernels transform configured for ElementwiseBinaryOp
        """
        from brainsmith.primitives.transforms.infer_kernels import InferKernels
        from brainsmith.kernels.elementwise_binary import ElementwiseBinaryOp

        # Return lambda that creates configured InferKernels instance
        return lambda: InferKernels([ElementwiseBinaryOp])

    # Primary implementation uses inherited defaults from KernelTestBase_v2:
    # - infer_kernel() (via get_kernel_inference_transform() above)
    # - get_backend_variants() (auto-detect from registry)
    # - configure_kernel() (auto_configure_from_fixture)

    # ========================================================================
    # Test Structure Information
    # ========================================================================

    def get_num_inputs(self):
        """ElementwiseBinary Add has 2 inputs."""
        return 2

    def get_num_outputs(self):
        """ElementwiseBinary Add has 1 output."""
        return 1

    # ========================================================================
    # Golden Reference
    # ========================================================================

    def compute_golden_reference(self, inputs):
        """Compute expected output using NumPy element-wise addition.

        Args:
            inputs: Dict mapping input names to numpy arrays
                   {"input0": ndarray, "input1": ndarray}

        Returns:
            Dict mapping output names to expected numpy arrays
            {"output": ndarray}
        """
        return {"output": inputs["input0"] + inputs["input1"]}
