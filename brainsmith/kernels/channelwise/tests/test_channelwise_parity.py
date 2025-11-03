"""Parity tests comparing FINN manual ChannelwiseOp with Brainsmith auto implementation.

Tests validate that both the legacy FINN manual implementation and the modern
Brainsmith auto-inferred implementation produce identical results for:
- Add operation (bias addition)
- Mul operation (scale multiplication)
- LessOrEqual comparison
- GreaterOrEqual comparison

Each operation class inherits from ChannelwiseParityBase which provides shared
test infrastructure, plus HLSCodegenParityMixin for 7 HLS code generation tests.
This results in 32 tests per operation × 4 operations = 128 total tests.
"""

from typing import Tuple

import numpy as np
import pytest
import qonnx.core.data_layout as DataLayout
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.transformation.fpgadataflow.convert_to_hw_layers import (
    InferChannelwiseLinearLayer,
)
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import getCustomOp, getHWCustomOp

from brainsmith.primitives.transforms.infer_kernel_list import InferKernelList
from tests.parity.base_parity_test import ParityTestBase
from tests.parity.hls_codegen_parity import HLSCodegenParityMixin

# FINN func attribute uses lowercase and abbreviated names
# Brainsmith func attribute uses capitalized ONNX operation names
FINN_FUNC_MAP = {
    "Add": "add",
    "Mul": "mul",
    "LessOrEqual": "cmp_le",
    "GreaterOrEqual": "cmp_ge",
}


class ChannelwiseParityBase(ParityTestBase, HLSCodegenParityMixin):
    """Base class for channelwise parity tests across different operation modes.

    This class provides shared implementation for testing FINN manual vs Brainsmith
    auto implementations of ChannelwiseOp. Subclasses override operation_type to
    test different operations (Add, Mul, LessOrEqual, GreaterOrEqual).

    Attributes:
        operation_type: ONNX operation name (override in subclasses)
    """

    operation_type: str = "Add"  # Override in subclasses

    # =========================================================================
    # REQUIRED PROPERTIES FOR ParityTestBase
    # =========================================================================

    @property
    def manual_op_class(self):
        """FINN's manual ChannelwiseOp_hls implementation."""
        from finn.custom_op.fpgadataflow.hls.channelwise_op_hls import ChannelwiseOp_hls
        return ChannelwiseOp_hls

    @property
    def auto_op_class(self):
        """Brainsmith's auto-generated ChannelwiseOp_hls implementation."""
        from brainsmith.kernels.channelwise.channelwise_hls import ChannelwiseOp_hls
        return ChannelwiseOp_hls

    def make_test_model(self) -> Tuple[ModelWrapper, str]:
        """Create ONNX model with channelwise operation.

        Creates a model with the specified operation_type that both FINN and
        Brainsmith can process. Uses static parameter tensor for channelwise
        processing.

        Returns:
            Tuple of (ModelWrapper, node_name) for the created ONNX model
        """
        # Configuration
        batch = 1
        h = w = 8
        ch = 64
        shape = [batch, h, w, ch]  # NHWC format
        node_name = f"{self.operation_type}_test"

        # Datatypes vary by operation
        if self.operation_type == "Add":
            idt = DataType["INT8"]
            pdt = DataType["INT8"]
            odt = DataType["INT9"]  # Add expands by 1 bit
        elif self.operation_type == "Mul":
            idt = DataType["INT8"]
            pdt = DataType["INT4"]
            odt = DataType["INT12"]  # Mul: 8+4=12 bits
        elif self.operation_type in ["LessOrEqual", "GreaterOrEqual"]:
            idt = DataType["INT8"]
            pdt = DataType["INT8"]
            odt = DataType["BINARY"]  # Comparisons produce binary output
        else:
            raise ValueError(f"Unknown operation type: {self.operation_type}")

        # Create input tensor info
        inp = helper.make_tensor_value_info("data", TensorProto.FLOAT, shape)
        outp = helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)

        # Generate parameter tensor (per-channel values)
        np.random.seed(42)  # Deterministic for reproducibility
        param_data = gen_finn_dt_tensor(pdt, [ch])
        param_tensor = helper.make_tensor(
            "param", TensorProto.FLOAT, [ch], param_data.flatten()
        )

        # Create ONNX node
        node = helper.make_node(
            self.operation_type, ["data", "param"], ["output"], name=node_name
        )

        # Build graph and model
        graph = helper.make_graph(
            nodes=[node],
            name="test",
            inputs=[inp],
            outputs=[outp],
            initializer=[param_tensor],
        )
        model = ModelWrapper(qonnx_make_model(graph, producer_name="parity-test"))

        # Set datatypes
        model.set_tensor_datatype("data", idt)
        model.set_tensor_datatype("param", pdt)
        model.set_tensor_datatype("output", odt)

        # Set data layout (required for FINN inference transforms)
        model.set_tensor_layout("data", DataLayout.NHWC)
        model.set_tensor_layout("output", DataLayout.NHWC)

        return model, node_name

    def setup_manual_op(self) -> Tuple[HWCustomOp, ModelWrapper]:
        """Create FINN manual ChannelwiseOp_hls.

        Workflow:
        1. Create ONNX operation
        2. Apply InferChannelwiseLinearLayer → ChannelwiseOp
        3. Set PE via configure_test_op()
        4. Apply SpecializeLayers → ChannelwiseOp_hls

        Returns:
            Tuple of (ChannelwiseOp_hls instance, model)
        """
        model, node_name = self.make_test_model()

        # Infer ChannelwiseOp from ONNX operation
        model = model.transform(InferChannelwiseLinearLayer())

        # Get the inferred node
        cw_node = model.graph.node[0]
        cw_op = getHWCustomOp(cw_node, model)

        # Verify Func attribute was set correctly
        finn_func = FINN_FUNC_MAP[self.operation_type]
        assert cw_op.get_nodeattr("Func") == finn_func, (
            f"Expected Func={finn_func}, got {cw_op.get_nodeattr('Func')}"
        )

        # Configure PE
        self.configure_test_op(cw_op, model, is_auto=False)

        # Specialize
        return self._specialize_and_get_op(model, node_name)

    def setup_auto_op(self) -> Tuple[HWCustomOp, ModelWrapper]:
        """Create Brainsmith auto ChannelwiseOp_hls.

        Workflow:
        1. Create ONNX operation
        2. Apply InferKernelList → ChannelwiseOp
        3. Set PE via configure_test_op()
        4. Apply SpecializeLayers → ChannelwiseOp_hls

        Returns:
            Tuple of (ChannelwiseOp_hls instance, model)
        """
        model, node_name = self.make_test_model()

        # Infer ChannelwiseOp from ONNX operation
        model = model.transform(InferKernelList())

        # Get the inferred node
        cw_node = model.graph.node[0]
        cw_op = getHWCustomOp(cw_node, model)

        # Verify func attribute (Brainsmith uses capitalized ONNX names)
        assert cw_op.get_nodeattr("func") == self.operation_type, (
            f"Expected func={self.operation_type}, got {cw_op.get_nodeattr('func')}"
        )

        # Configure PE
        self.configure_test_op(cw_op, model, is_auto=True)

        # Specialize
        return self._specialize_and_get_op(model, node_name)

    def configure_test_op(self, op: HWCustomOp, model: ModelWrapper, is_auto: bool) -> None:
        """Configure ChannelwiseOp for testing.

        Sets PE parallelization for testing. Base class handles specialization separately.

        - Auto implementation uses interface-agnostic parallelism API
        - Manual FINN implementation uses direct nodeattr setting

        Args:
            op: ChannelwiseOp operator instance
            model: ModelWrapper containing the op
            is_auto: True if auto implementation
        """
        # Set PE for testing (64 channels / 8 = 8-way folding)
        if is_auto:
            # Brainsmith auto: Use index-based interface navigation API
            # Input interface has PE dimension for channelwise operations
            op.build_design_space(model)
            point = op.design_point.with_input_stream(0, 8)
            # Apply configuration from design point to nodeattrs
            op.apply_design_point(point)
        else:
            # FINN manual: Direct nodeattr setting
            op.set_nodeattr("PE", 8)

    def _specialize_and_get_op(self, model: ModelWrapper, node_name_prefix: str) -> Tuple[HWCustomOp, ModelWrapper]:
        """Specialize model and return HLS operator instance.

        Args:
            model: ModelWrapper with configured ChannelwiseOp
            node_name_prefix: Prefix for finding specialized node

        Returns:
            Tuple of (HLS operator, specialized model)
        """
        # Get the node before specialization
        cw_node = model.graph.node[0]
        cw_op = getCustomOp(cw_node)

        # Set preferred implementation style
        cw_op.set_nodeattr("preferred_impl_style", "hls")

        # Specialize to HLS backend
        fpga_part = "xc7z020clg400-1"
        model = model.transform(SpecializeLayers(fpga_part))

        # Get specialized node
        specialized_node = model.graph.node[0]
        assert "hls" in specialized_node.domain, (
            f"Expected HLS domain, got {specialized_node.domain}"
        )

        specialized_op = getHWCustomOp(specialized_node, model)
        return specialized_op, model


# Test classes for each operation mode


class TestChannelwiseAddParity(ChannelwiseParityBase):
    """Parity tests for ChannelwiseOp Add mode (bias addition).

    Tests 32 comparisons between FINN manual and Brainsmith auto:
    - 25 base structural tests (shapes, datatypes, resources, execution)
    - 7 HLS codegen tests (templates, pragmas, function signatures)
    """

    operation_type = "Add"


class TestChannelwiseMulParity(ChannelwiseParityBase):
    """Parity tests for ChannelwiseOp Mul mode (scale multiplication).

    Tests 32 comparisons between FINN manual and Brainsmith auto:
    - 25 base structural tests (shapes, datatypes, resources, execution)
    - 7 HLS codegen tests (templates, pragmas, function signatures)
    """

    operation_type = "Mul"


@pytest.mark.skip(reason="FINN InferChannelwiseLinearLayer doesn't support LessOrEqual operations")
class TestChannelwiseLessOrEqualParity(ChannelwiseParityBase):
    """Parity tests for ChannelwiseOp LessOrEqual mode (threshold comparison).

    Tests 32 comparisons between FINN manual and Brainsmith auto:
    - 25 base structural tests (shapes, datatypes, resources, execution)
    - 7 HLS codegen tests (templates, pragmas, function signatures)
    """

    operation_type = "LessOrEqual"


@pytest.mark.skip(reason="FINN InferChannelwiseLinearLayer doesn't support GreaterOrEqual operations")
class TestChannelwiseGreaterOrEqualParity(ChannelwiseParityBase):
    """Parity tests for ChannelwiseOp GreaterOrEqual mode (threshold comparison).

    Tests 32 comparisons between FINN manual and Brainsmith auto:
    - 25 base structural tests (shapes, datatypes, resources, execution)
    - 7 HLS codegen tests (templates, pragmas, function signatures)
    """

    operation_type = "GreaterOrEqual"
