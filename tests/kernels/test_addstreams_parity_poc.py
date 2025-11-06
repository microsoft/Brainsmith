"""Proof of concept for DualKernelTest_v2 framework.

This test validates that the DualKernelTest_v2 framework works correctly
by testing AddStreams kernel parity between FINN (manual) and Brainsmith (auto).
"""

import onnx.helper as helper
import numpy as np
from onnx import TensorProto

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from tests.frameworks.dual_kernel_test_v2 import DualKernelTest_v2


class TestAddStreamsParity_POC(DualKernelTest_v2):
    """Minimal AddStreams parity test for framework validation.

    Tests:
    - 12 parity tests (7 core + 5 HW estimation)
    - 6 golden execution tests (2 Python + 4 backend, backend tests skipped by default)

    Total: 18 tests inherited from DualKernelTest_v2
    """

    # ========================================================================
    # Test Configuration (Attribute-based)
    # ========================================================================

    batch = 1
    channels = 64
    input_dtype = DataType["INT8"]

    # ========================================================================
    # Model Creation (v2.3 interface)
    # ========================================================================

    def make_test_model(self, input_shapes):
        """Create AddStreams test model with v2.3 direct annotations.

        Args:
            input_shapes: Dict with input shapes (not used, we use class attributes)

        Returns:
            (model, input_names): Model with Add operation and list of inputs
        """
        # Use shapes from class attributes (attribute-based configuration)
        shape = [self.batch, self.channels]

        # Create inputs
        inp0 = helper.make_tensor_value_info("input0", TensorProto.FLOAT, shape)
        inp1 = helper.make_tensor_value_info("input1", TensorProto.FLOAT, shape)
        out = helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)

        # Create Add node
        node = helper.make_node("Add", ["input0", "input1"], ["output"], name="Add_0")
        graph = helper.make_graph([node], "test_add", [inp0, inp1], [out])

        model = ModelWrapper(qonnx_make_model(graph))

        # Return model and input names for annotation
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
        """Return input datatypes for annotation."""
        return {
            "input0": self.input_dtype,
            "input1": self.input_dtype,
        }

    def get_num_inputs(self):
        """AddStreams has 2 inputs."""
        return 2

    def get_num_outputs(self):
        """AddStreams has 1 output."""
        return 1

    # ========================================================================
    # Transform Configuration
    # ========================================================================

    def get_manual_transform(self):
        """Return FINN's manual AddStreams transform."""
        from finn.transformation.fpgadataflow.convert_to_hw_layers import (
            InferAddStreamsLayer,
        )

        return InferAddStreamsLayer

    def get_auto_transform(self):
        """Return Brainsmith's unified kernel transform."""
        from brainsmith.primitives.transforms.infer_kernels import InferKernels
        from brainsmith.kernels.addstreams import AddStreams

        # Return a lambda that creates InferKernels([AddStreams])
        return lambda: InferKernels([AddStreams])

    def get_manual_backend_variants(self):
        """Return FINN backend for manual pipeline."""
        from finn.custom_op.fpgadataflow.hls.addstreams_hls import AddStreams_hls

        return [AddStreams_hls]

    # Optional: Enable backend testing by overriding
    # def get_backend_fpgapart(self):
    #     return "xc7z020clg400-1"

    # ========================================================================
    # Golden Reference
    # ========================================================================

    def compute_golden_reference(self, inputs):
        """Compute expected output (simple addition)."""
        return {"output": inputs["input0"] + inputs["input1"]}
