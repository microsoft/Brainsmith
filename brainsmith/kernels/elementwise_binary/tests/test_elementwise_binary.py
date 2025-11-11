"""Test base class for ElementwiseBinary operations.

This module provides ElementwiseBinaryTestBase, the shared test class
for all elementwise binary operations (Add, Sub, Mul, Div).

Architecture:
- Test base class lives with kernel code (here)
- Test cases live in tests/kernels/elementwise_binary/
- Shared test case definitions in test_cases.py

Usage:
    from brainsmith.kernels.elementwise_binary.tests import ElementwiseBinaryTestBase

    class TestAddValidation(ElementwiseBinaryTestBase):
        # Automatically inherits make_test_model() and get_kernel_op()
        # Just need to provide kernel_test_config fixture
        pass
"""

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model
from tests.frameworks.kernel_test import KernelTest
from tests.frameworks.test_config import KernelTestConfig


class ElementwiseBinaryTestBase(KernelTest):
    """Shared test base for ElementwiseBinary operations.

    Provides common implementation for:
    - make_test_model(): Build Stage 1 ONNX model for binary operations
    - get_kernel_op(): Return ElementwiseBinaryOp kernel class

    Supports both equal-shape and broadcasting scenarios.

    Subclasses just need to provide kernel_test_config fixture with
    the operation name (Add, Sub, Mul, Div).
    """

    def make_test_model(
        self, kernel_test_config: KernelTestConfig
    ) -> tuple[ModelWrapper, list[str]]:
        """Build Stage 1 ONNX model for elementwise binary operation.

        Supports both equal-shape and broadcasting scenarios.
        Output shape is computed via np.broadcast_shapes().

        Args:
            kernel_test_config: Test configuration with operation, shapes, dtypes

        Returns:
            (model, input_names) tuple
        """
        operation = kernel_test_config.model.operation
        input_shapes = kernel_test_config.model.input_shapes

        # Create input tensor infos with lhs/rhs naming
        lhs = helper.make_tensor_value_info("lhs", TensorProto.FLOAT, input_shapes["lhs"])
        rhs = helper.make_tensor_value_info("rhs", TensorProto.FLOAT, input_shapes["rhs"])

        # Compute output shape (handles broadcasting)
        output_shape = tuple(np.broadcast_shapes(input_shapes["lhs"], input_shapes["rhs"]))
        out = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)

        # Create node and graph
        node = helper.make_node(operation, ["lhs", "rhs"], ["output"], name=f"{operation}_0")
        graph = helper.make_graph([node], f"test_{operation.lower()}", [lhs, rhs], [out])
        model = ModelWrapper(qonnx_make_model(graph))

        return model, ["lhs", "rhs"]

    def get_kernel_op(self):
        """Return ElementwiseBinaryOp kernel class.

        Returns:
            ElementwiseBinaryOp kernel class
        """
        from brainsmith.kernels.elementwise_binary import ElementwiseBinaryOp

        return ElementwiseBinaryOp
