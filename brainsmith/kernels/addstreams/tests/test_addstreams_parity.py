# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Parity tests for AddStreams kernel.

Comprehensive testing between FINN's manual AddStreams implementation and
Brainsmith's auto-generated implementation using KernelOp framework.

Test Coverage:
- 25 base tests (shapes, datatypes, execution, resources)
- 7 HLS code generation tests (new)
- Total: 32 tests

Usage:
    # Run all tests
    pytest brainsmith/kernels/addstreams/tests/test_addstreams_parity.py -v

    # Run only code generation tests
    pytest brainsmith/kernels/addstreams/tests/test_addstreams_parity.py -m hls -v

    # Run fast (skip slow tests)
    pytest brainsmith/kernels/addstreams/tests/test_addstreams_parity.py -m "not slow" -v
"""

import pytest
from typing import Tuple, Type

from onnx import helper, TensorProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.util.basic import qonnx_make_model  # Leverage FINN utility
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.base import Transformation
from qonnx.custom_op.registry import getCustomOp
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.basic import getHWCustomOp  # Leverage FINN utility
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferAddStreamsLayer

# Import Brainsmith transform
from brainsmith.transforms.infer_kernel_list import InferKernelList

# Import parity test framework
from tests.parity.base_parity_test import ParityTestBase
from tests.parity.hls_codegen_parity import HLSCodegenParityMixin


class TestAddStreamsHLSParity(ParityTestBase, HLSCodegenParityMixin):
    """Comprehensive parity tests for AddStreams HLS implementation.

    Tests both FINN's manual implementation and Brainsmith's auto-generated
    implementation using the new HLS code generation validation mixin.

    Test Count: 25 (base) + 7 (codegen) = 32 total
    """

    # =========================================================================
    # Required Properties
    # =========================================================================

    @property
    def manual_op_class(self) -> Type[HWCustomOp]:
        """FINN's manual AddStreams_hls implementation."""
        from finn.custom_op.fpgadataflow.hls.addstreams_hls import AddStreams_hls
        return AddStreams_hls

    @property
    def auto_op_class(self) -> Type[HWCustomOp]:
        """Brainsmith's auto-generated AddStreams_hls implementation."""
        from brainsmith.kernels.addstreams.addstreams_hls import AddStreams_hls
        return AddStreams_hls

    # =========================================================================
    # Test Configuration
    # =========================================================================

    def make_test_model(self) -> Tuple[ModelWrapper, str]:
        """Create ONNX Add node for AddStreams inference.

        Creates standard ONNX Add node that will be transformed to AddStreams.
        Uses NHWC layout with integer datatypes.

        Returns:
            (model, node_name): Model with Add node and its name
        """
        # Test configuration
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
            name="addstreams_test",
            inputs=[inp1, inp2],
            outputs=[outp]
        )

        model = qonnx_make_model(graph, producer_name="addstreams-parity-test")
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

    def configure_test_op(self, op: HWCustomOp, model: ModelWrapper, is_auto: bool) -> None:
        """Configure AddStreams op for testing.

        Sets PE parallelization for testing. Base class handles specialization separately.

        Args:
            op: AddStreams operator instance
            model: ModelWrapper containing the op
            is_auto: True if auto implementation
        """
        # Set PE for testing (64 channels / 8 = 8-way folding)
        op.set_nodeattr("PE", 8)

    def _specialize_and_get_op(self, model: ModelWrapper, node_name_prefix: str) -> Tuple[HWCustomOp, ModelWrapper]:
        """Specialize model and return HLS operator instance.

        Helper to handle SpecializeLayers transform and get the _hls node.

        Args:
            model: Model with AddStreams node
            node_name_prefix: Expected node name prefix (e.g., "AddStreams")

        Returns:
            (op, model) tuple with specialized operator
        """
        # Specialize to HLS backend
        fpgapart = "xc7z020clg400-1"
        model = model.transform(SpecializeLayers(fpgapart))

        # Find the specialized node (AddStreams → AddStreams_hls)
        specialized_node = None
        for node in model.graph.node:
            if "AddStreams_hls" in node.op_type:
                specialized_node = node
                break

        if specialized_node is None:
            available = [n.op_type for n in model.graph.node]
            raise RuntimeError(
                f"SpecializeLayers failed to create AddStreams_hls node. "
                f"Available: {available}"
            )

        # Get specialized op instance
        op = getHWCustomOp(specialized_node, model)
        return op, model

    def setup_manual_op(self) -> Tuple[HWCustomOp, ModelWrapper]:
        """Setup manual AddStreams_hls with specialization.

        FINN workflow: InferAddStreamsLayer → AddStreams → SpecializeLayers → AddStreams_hls
        """
        # Get the transform that creates AddStreams (not _hls yet)
        transform_class = self.get_manual_transform()

        # Use base class helper but specify "AddStreams" as target (pre-specialization)
        op, model = self._setup_via_transform(
            transform_class,
            "AddStreams",  # Transform creates this
            is_auto=False
        )

        # Now specialize to get AddStreams_hls
        return self._specialize_and_get_op(model, "AddStreams")

    def setup_auto_op(self) -> Tuple[HWCustomOp, ModelWrapper]:
        """Setup auto AddStreams_hls with specialization.

        Brainsmith workflow: InferKernelList → AddStreams → SpecializeLayers → AddStreams_hls
        """
        # Get the transform that creates AddStreams (not _hls yet)
        transform_class = self.get_auto_transform()

        # Use base class helper but specify "AddStreams" as target (pre-specialization)
        op, model = self._setup_via_transform(
            transform_class,
            "AddStreams",  # Transform creates this
            is_auto=True
        )

        # Now specialize to get AddStreams_hls
        return self._specialize_and_get_op(model, "AddStreams")

    def get_num_inputs(self) -> int:
        """AddStreams has 2 inputs."""
        return 2

    def get_num_outputs(self) -> int:
        """AddStreams has 1 output."""
        return 1


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
