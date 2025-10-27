# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for ChannelwiseOp kernel with inference support."""

import pytest
import numpy as np
from onnx import helper, TensorProto, NodeProto

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model, get_by_name

from brainsmith.kernels.channelwise.channelwise import (
    ChannelwiseOp,
    CHANNELWISE_SCHEMA,
)
from brainsmith.transforms.infer_kernel_list import InferKernelList
from brainsmith.core.plugins import get_kernel
from brainsmith.dataflow import KernelOp

# Import global test helpers
from tests.fixtures.kernel_test_helpers import make_parametric_op_model


def get_func_attr(node: NodeProto) -> str:
    """Extract 'func' attribute from node."""
    return get_by_name(node.attribute, "func").s.decode()


class TestChannelwiseSchema:
    """Test suite for ChannelwiseOp schema structure."""

    def test_schema_structure(self):
        """Verify ChannelwiseOp schema defines 2 inputs and 1 output."""
        assert CHANNELWISE_SCHEMA.name == "ChannelwiseOp"
        assert len(CHANNELWISE_SCHEMA.inputs) == 2
        assert len(CHANNELWISE_SCHEMA.outputs) == 1
        assert CHANNELWISE_SCHEMA.inputs[0].name == "input"
        assert CHANNELWISE_SCHEMA.inputs[1].name == "parameters"
        assert CHANNELWISE_SCHEMA.outputs[0].name == "output"

    def test_input_schemas(self):
        """Verify input requires NHWC layout and parameters have no tiling."""
        input_schema = CHANNELWISE_SCHEMA.inputs[0]
        assert input_schema.name == "input"
        assert input_schema.required_layout == "NHWC"
        assert "PE" in input_schema.stream_tiling

        param_schema = CHANNELWISE_SCHEMA.inputs[1]
        assert param_schema.name == "parameters"
        assert param_schema.block_tiling == []
        assert param_schema.stream_tiling == []

    def test_output_schema(self):
        """Verify output requires NHWC layout."""
        output_schema = CHANNELWISE_SCHEMA.outputs[0]
        assert output_schema.name == "output"
        assert output_schema.required_layout == "NHWC"

    def test_kernel_params(self):
        """Verify func kernel parameter and ram_style DSE dimension are defined."""
        params = CHANNELWISE_SCHEMA.kernel_params
        assert "func" in params
        assert "ram_style" in CHANNELWISE_SCHEMA.dse_dimensions

    def test_func_modes(self):
        """Verify all channelwise operation modes are supported."""
        _name, _type, _desc, valid_values = CHANNELWISE_SCHEMA.kernel_params["func"]
        assert valid_values == {"Add", "Mul", "LessOrEqual", "GreaterOrEqual"}

    def test_constraints_present(self):
        """Verify schema has structural constraints defined."""
        assert len(CHANNELWISE_SCHEMA.constraints) > 0

    def test_dse_dimensions(self):
        """Verify ram_style DSE dimension with distributed/block values."""
        assert "ram_style" in CHANNELWISE_SCHEMA.dse_dimensions
        dim = CHANNELWISE_SCHEMA.dse_dimensions["ram_style"]
        assert dim.name == "ram_style"
        assert dim.values == {"distributed", "block"}
        assert dim.default == "distributed"


class TestChannelwiseOp:
    """Test suite for ChannelwiseOp class."""

    def test_build_schema(self):
        """Verify build_schema returns the canonical schema."""
        node = helper.make_node(
            "ChannelwiseOp",
            inputs=["data", "params"],
            outputs=["output"],
            domain="brainsmith.kernels",
        )

        schema = ChannelwiseOp.build_schema(node, None)
        assert schema is CHANNELWISE_SCHEMA

    def test_kernel_registration(self):
        """Verify ChannelwiseOp is registered in kernel plugin system."""
        assert issubclass(ChannelwiseOp, KernelOp)
        kernel_cls = get_kernel("ChannelwiseOp")
        assert kernel_cls is ChannelwiseOp


class TestChannelwiseInference:
    """Test suite for ONNX → ChannelwiseOp inference."""

    def test_can_infer_from_add(self):
        """Verify can_infer_from detects Add with static parameter."""
        model, node = make_parametric_op_model("Add", param_input="bias")
        assert ChannelwiseOp.can_infer_from(node, model)

    def test_can_infer_from_wrong_optype(self):
        """Verify can_infer_from rejects unsupported op types."""
        model, _node = make_parametric_op_model("Add")
        relu_node = helper.make_node("Relu", ["data"], ["output"])
        assert not ChannelwiseOp.can_infer_from(relu_node, model)

    def test_can_infer_from_both_dynamic(self):
        """Verify can_infer_from requires one static parameter."""
        shape = [1, 8, 8, 64]
        data1 = helper.make_tensor_value_info("data1", TensorProto.FLOAT, shape)
        data2 = helper.make_tensor_value_info("data2", TensorProto.FLOAT, shape)
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)

        node = helper.make_node("Add", ["data1", "data2"], ["output"])
        graph = helper.make_graph([node], "test", inputs=[data1, data2], outputs=[output])

        model_w = ModelWrapper(qonnx_make_model(graph))
        assert not ChannelwiseOp.can_infer_from(node, model_w)

    def test_can_infer_from_swapped_inputs(self):
        """Verify can_infer_from handles swapped input order."""
        model, node = make_parametric_op_model("Add", param_input="bias", input_order="static_first")
        assert ChannelwiseOp.can_infer_from(node, model)

    def test_infer_from_add(self):
        """Verify infer_from creates ChannelwiseOp with canonical input order."""
        model, add_node = make_parametric_op_model("Add", param_input="bias")

        result = ChannelwiseOp.infer_from(add_node, model, insert_index=0)

        assert result.nodes_to_remove == [add_node]
        assert len(result.nodes_to_insert) == 1

        hw_node = result.nodes_to_insert[0]
        assert hw_node.op_type == "ChannelwiseOp"
        assert hw_node.domain == "brainsmith.kernels"
        assert list(hw_node.input) == ["data", "bias"]  # Preserves param name from input
        assert get_func_attr(hw_node) == "Add"

    def test_transform_integration(self):
        """Verify InferKernelList replaces Add with ChannelwiseOp."""
        model, _node = make_parametric_op_model("Add", param_input="bias")

        model_transformed = model.transform(InferKernelList())

        node_types = [n.op_type for n in model_transformed.graph.node]
        assert node_types == ["ChannelwiseOp"]

        cw_node = model_transformed.graph.node[0]
        assert get_func_attr(cw_node) == "Add"

    def test_can_infer_from_less_or_equal(self):
        """Verify can_infer_from detects LessOrEqual with static parameter."""
        model, node = make_parametric_op_model("LessOrEqual", param_input="threshold")
        assert ChannelwiseOp.can_infer_from(node, model)

    def test_can_infer_from_greater_or_equal(self):
        """Verify can_infer_from detects GreaterOrEqual with static parameter."""
        model, node = make_parametric_op_model("GreaterOrEqual", param_input="threshold")
        assert ChannelwiseOp.can_infer_from(node, model)

    def test_infer_from_comparison(self):
        """Verify infer_from creates ChannelwiseOp with comparison func."""
        model, le_node = make_parametric_op_model("LessOrEqual", param_input="threshold")

        result = ChannelwiseOp.infer_from(le_node, model, insert_index=0)

        assert len(result.nodes_to_insert) == 1
        hw_node = result.nodes_to_insert[0]

        assert hw_node.op_type == "ChannelwiseOp"
        assert hw_node.domain == "brainsmith.kernels"
        assert get_func_attr(hw_node) == "LessOrEqual"

    def test_transform_integration_comparison(self):
        """Verify InferKernelList replaces LessOrEqual with ChannelwiseOp."""
        model, _node = make_parametric_op_model("LessOrEqual", param_input="threshold")

        model_transformed = model.transform(InferKernelList())

        node_types = [n.op_type for n in model_transformed.graph.node]
        assert node_types == ["ChannelwiseOp"]

        cw_node = model_transformed.graph.node[0]
        assert get_func_attr(cw_node) == "LessOrEqual"


class TestChannelwiseExecution:
    """Test suite for ChannelwiseOp execution (hardware only)."""

    @pytest.mark.skip(
        reason="ChannelwiseOp has no Python execution mode. "
               "This kernel is HLS/RTL-only and verified through "
               "hardware simulation and synthesis. "
               "See brainsmith/kernels/channelwise/channelwise_hls.py"
    )
    def test_execution_not_supported(self):
        """ChannelwiseOp execution tests require hardware simulation."""
        pass
