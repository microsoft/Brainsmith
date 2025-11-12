# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for SpecializeKernels transform."""

import numpy as np
import pytest
import qonnx.core.data_layout as DataLayout
from finn.util.basic import getHWCustomOp
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

from brainsmith.primitives.transforms.infer_kernels import InferKernels
from brainsmith.primitives.transforms.specialize_kernels import SpecializeKernels


def make_channelwise_model():
    """Create test model with ChannelwiseOp (Add operation)."""
    batch, h, w, ch = 1, 8, 8, 64
    shape = [batch, h, w, ch]
    idt = DataType["INT8"]
    pdt = DataType["INT8"]
    odt = DataType["INT9"]

    inp = helper.make_tensor_value_info("data", TensorProto.FLOAT, shape)
    outp = helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)

    np.random.seed(42)
    param_data = gen_finn_dt_tensor(pdt, [ch])
    param_tensor = helper.make_tensor("param", TensorProto.FLOAT, [ch], param_data.flatten())

    node = helper.make_node("Add", ["data", "param"], ["output"], name="Add_test")

    graph = helper.make_graph(
        nodes=[node],
        name="test",
        inputs=[inp],
        outputs=[outp],
        initializer=[param_tensor],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="test"))

    model.set_tensor_datatype("data", idt)
    model.set_tensor_datatype("param", pdt)
    model.set_tensor_datatype("output", odt)
    model.set_tensor_layout("data", DataLayout.NHWC)
    model.set_tensor_layout("output", DataLayout.NHWC)

    return model


class TestBrainsmithNodeStableDomain:
    """Test that Brainsmith nodes keep stable domain."""

    def test_domain_unchanged(self):
        """Brainsmith nodes should not have domain mutation."""
        model = make_channelwise_model()

        # Infer to ChannelwiseOp (Brainsmith kernel)
        model = model.transform(InferKernels())
        node = model.graph.node[0]

        assert (
            node.domain == "brainsmith.kernels"
        ), f"Expected domain 'brainsmith.kernels', got '{node.domain}'"

        # Specialize
        model = model.transform(SpecializeKernels("xc7z020clg400-1"))
        node = model.graph.node[0]

        # Domain should be unchanged
        assert (
            node.domain == "brainsmith.kernels"
        ), f"Domain mutated! Expected 'brainsmith.kernels', got '{node.domain}'"

    def test_optype_unchanged(self):
        """Brainsmith nodes should not have op_type mutation."""
        model = make_channelwise_model()

        # Infer to ChannelwiseOp
        model = model.transform(InferKernels())
        node = model.graph.node[0]

        assert (
            node.op_type == "ChannelwiseOp"
        ), f"Expected op_type 'ChannelwiseOp', got '{node.op_type}'"

        # Specialize
        model = model.transform(SpecializeKernels("xc7z020clg400-1"))
        node = model.graph.node[0]

        # Op type should be unchanged (not ChannelwiseOp_hls)
        assert (
            node.op_type == "ChannelwiseOp"
        ), f"Op type mutated! Expected 'ChannelwiseOp', got '{node.op_type}'"

    def test_implementation_attribute_set(self):
        """Brainsmith nodes should have implementation attribute set."""
        model = make_channelwise_model()

        # Infer to ChannelwiseOp
        model = model.transform(InferKernels())

        # Specialize
        model = model.transform(SpecializeKernels("xc7z020clg400-1"))

        node = model.graph.node[0]
        op = getHWCustomOp(node, model)

        # Should have implementation attribute
        impl = op.get_nodeattr("implementation")
        assert impl in {
            "vitis_hls",
            "vivado_hls",
            "verilog",
            "systemverilog",
        }, f"Invalid implementation '{impl}'"

    def test_backend_attribute_set(self):
        """Brainsmith nodes should have backend='fpgadataflow' for FINN compatibility."""
        model = make_channelwise_model()

        # Infer to ChannelwiseOp
        model = model.transform(InferKernels())

        # Specialize
        model = model.transform(SpecializeKernels("xc7z020clg400-1"))

        node = model.graph.node[0]
        op = getHWCustomOp(node, model)

        # Should have backend attribute for FINN compatibility
        backend = op.get_nodeattr("backend")
        assert backend == "fpgadataflow", f"Expected backend='fpgadataflow', got '{backend}'"

    def test_preferred_impl_style_hls(self):
        """Respect preferred_impl_style='hls'."""
        model = make_channelwise_model()

        # Infer to ChannelwiseOp
        model = model.transform(InferKernels())

        node = model.graph.node[0]
        op = getHWCustomOp(node, model)
        op.set_nodeattr("preferred_impl_style", "hls")

        # Specialize
        model = model.transform(SpecializeKernels("xc7z020clg400-1"))

        node = model.graph.node[0]
        op = getHWCustomOp(node, model)

        impl = op.get_nodeattr("implementation")
        assert "hls" in impl.lower(), f"Expected HLS implementation, got '{impl}'"

    def test_preferred_impl_style_rtl(self):
        """Respect preferred_impl_style='rtl'."""
        model = make_channelwise_model()

        # Infer to ChannelwiseOp
        model = model.transform(InferKernels())

        node = model.graph.node[0]
        op = getHWCustomOp(node, model)
        op.set_nodeattr("preferred_impl_style", "rtl")

        # Specialize
        model = model.transform(SpecializeKernels("xc7z020clg400-1"))

        node = model.graph.node[0]
        op = getHWCustomOp(node, model)

        impl = op.get_nodeattr("implementation")
        assert impl in {"verilog", "systemverilog"}, f"Expected RTL implementation, got '{impl}'"


class TestFINNNodeBackwardCompatibility:
    """Test that FINN nodes still get domain mutation (backward compatibility)."""

    @pytest.mark.skip(reason="FINN node test requires FINN ChannelwiseOp setup")
    def test_finn_domain_mutation(self):
        """FINN nodes should get domain mutation for backward compatibility."""
        # This test would require creating a FINN-style ChannelwiseOp node
        # (domain='finn.custom_op.fpgadataflow')
        # Skipping for now as it requires FINN inference transforms
        pass

    @pytest.mark.skip(reason="FINN node test requires FINN ChannelwiseOp setup")
    def test_finn_optype_mutation(self):
        """FINN nodes should get op_type mutation for backward compatibility."""
        # Similar to above - requires FINN-style node setup
        pass


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_already_specialized_node_unchanged(self):
        """Already specialized nodes should not be re-specialized."""
        model = make_channelwise_model()

        # Infer to ChannelwiseOp
        model = model.transform(InferKernels())

        # Specialize once
        model = model.transform(SpecializeKernels("xc7z020clg400-1"))

        node = model.graph.node[0]
        op = getHWCustomOp(node, model)
        first_impl = op.get_nodeattr("implementation")

        # Specialize again (should be idempotent)
        model = model.transform(SpecializeKernels("xc7z020clg400-1"))

        node = model.graph.node[0]
        op = getHWCustomOp(node, model)
        second_impl = op.get_nodeattr("implementation")

        assert (
            first_impl == second_impl
        ), f"Re-specialization changed implementation: {first_impl} → {second_impl}"

    def test_non_hw_nodes_unchanged(self):
        """Non-hardware nodes should be left unchanged."""
        # Create model with non-HW node (regular ONNX Relu)
        inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 10])
        outp = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 10])
        node = helper.make_node("Relu", ["x"], ["y"], name="relu")

        graph = helper.make_graph(
            nodes=[node],
            name="test",
            inputs=[inp],
            outputs=[outp],
        )
        model = ModelWrapper(qonnx_make_model(graph, producer_name="test"))

        # Specialize (should do nothing)
        model = model.transform(SpecializeKernels("xc7z020clg400-1"))

        # Node should be unchanged
        node = model.graph.node[0]
        assert node.op_type == "Relu"
        assert node.domain == ""
