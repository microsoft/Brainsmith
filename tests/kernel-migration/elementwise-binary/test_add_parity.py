"""Parity test for ElementwiseBinary Add operation.

Compares FINN's ElementwiseAdd (manual) vs Brainsmith's ElementwiseBinaryOp (auto)
for the Add operation.

This is the pilot implementation to establish the pattern before extending to
all 17 elementwise binary operations.
"""

import pytest
import numpy as np
import onnx.helper as helper
from onnx import TensorProto

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from tests.frameworks.dual_kernel_test_v2 import DualKernelTest_v2


class TestElementwiseBinaryAdd_Parity(DualKernelTest_v2):
    """Test Add operation parity between FINN and Brainsmith.

    Pipeline comparison:
    - Manual (FINN): Add (ONNX) → ElementwiseAdd → ElementwiseAdd_hls
    - Auto (Brainsmith): Add (ONNX) → ElementwiseBinaryOp → ElementwiseBinaryOp_hls

    Provides 18 inherited tests:
    - 7 core parity tests (shapes, widths, datatypes)
    - 5 HW estimation tests (cycles, resources)
    - 6 golden execution tests (2 Python + 4 backend if enabled)
    """

    # ========================================================================
    # Test Configuration (Attribute-based)
    # ========================================================================

    batch = 1
    channels = 64
    input_dtype = DataType["INT8"]

    # ========================================================================
    # Model Creation (v2.3 interface with direct DataType annotations)
    # ========================================================================

    def make_test_model(self, input_shapes):
        """Create Add operation test model.

        Args:
            input_shapes: Dict with input shapes (not used - we use class attributes)

        Returns:
            (model, input_names): Model with Add ONNX node and list of inputs to annotate
        """
        # Use shapes from class attributes
        shape = [self.batch, self.channels]

        # Create inputs with FLOAT container type (QONNX convention)
        inp0 = helper.make_tensor_value_info("input0", TensorProto.FLOAT, shape)
        inp1 = helper.make_tensor_value_info("input1", TensorProto.FLOAT, shape)
        out = helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)

        # Create Add ONNX node
        node = helper.make_node(
            "Add",
            ["input0", "input1"],
            ["output"],
            name="Add_0"
        )

        graph = helper.make_graph(
            [node],
            "test_elementwise_binary_add",
            [inp0, inp1],
            [out]
        )

        model = ModelWrapper(qonnx_make_model(graph))

        # Return model and input names for DataType annotation
        # Framework will annotate these with datatypes from _get_input_datatypes()
        return model, ["input0", "input1"]

    # ========================================================================
    # Input/Output Configuration
    # ========================================================================

    def _get_input_shapes(self):
        """Return input shapes from class attributes."""
        shape = (self.batch, self.channels)
        return {
            "input0": shape,
            "input1": shape,
        }

    def _get_input_datatypes(self):
        """Return input datatypes for QONNX annotation."""
        return {
            "input0": self.input_dtype,
            "input1": self.input_dtype,
        }

    def get_num_inputs(self):
        """ElementwiseBinary Add has 2 inputs."""
        return 2

    def get_num_outputs(self):
        """ElementwiseBinary Add has 1 output."""
        return 1

    # ========================================================================
    # Transform Configuration (Manual vs Auto)
    # ========================================================================

    def get_manual_transform(self):
        """Return FINN's manual ElementwiseBinary transform.

        FINN uses a unified InferElementwiseBinaryOperation transform that
        auto-detects the operation type from the ONNX node's op_type.

        Returns:
            InferElementwiseBinaryOperation class
        """
        from finn.transformation.fpgadataflow.convert_to_hw_layers import (
            InferElementwiseBinaryOperation,
        )
        return InferElementwiseBinaryOperation

    def get_auto_transform(self):
        """Return Brainsmith's unified kernel inference transform.

        Brainsmith uses InferKernels with ElementwiseBinaryOp, which creates
        a polymorphic kernel that handles all binary operations via the 'func'
        parameter.

        Returns:
            Callable that creates InferKernels([ElementwiseBinaryOp])
        """
        from brainsmith.primitives.transforms.infer_kernels import InferKernels
        from brainsmith.kernels.elementwise_binary import ElementwiseBinaryOp

        # Return a lambda that creates configured InferKernels instance
        return lambda: InferKernels([ElementwiseBinaryOp])

    def get_manual_backend_variants(self):
        """Return FINN backend for manual pipeline.

        REQUIRED: Must use FINN's ElementwiseAdd_hls backend because the manual
        pipeline creates nodes with FINN-specific attributes (e.g., "Func" vs "func").

        Returns:
            List containing ElementwiseAdd_hls class
        """
        from finn.custom_op.fpgadataflow.hls.elementwise_binary_hls import (
            ElementwiseAdd_hls,
        )
        return [ElementwiseAdd_hls]

    # get_auto_backend_variants() not needed - auto-detects from Brainsmith registry

    # ========================================================================
    # Backend Testing (Optional)
    # ========================================================================

    # Uncomment to enable cppsim/rtlsim tests (adds 4 tests: 2 cppsim + 2 rtlsim)
    # def get_backend_fpgapart(self):
    #     return "xc7z020clg400-1"

    # ========================================================================
    # Golden Reference
    # ========================================================================

    def compute_golden_reference(self, inputs):
        """Compute expected output using NumPy.

        Args:
            inputs: Dict mapping input names to numpy arrays
                   {"input0": ndarray, "input1": ndarray}

        Returns:
            Dict mapping output names to expected numpy arrays
            {"output": ndarray}
        """
        return {"output": inputs["input0"] + inputs["input1"]}
