############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
############################################################################

"""Integration tests for normalize_layouts build step."""

import pytest
import qonnx.core.data_layout as DataLayout
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from brainsmith.steps.normalize_layouts import normalize_dataflow_layouts_step


def make_test_nchw_model():
    """Create a simple test model with NCHW layout."""
    in0 = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [1, 64, 224, 224])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 64, 224, 224])

    relu_node = helper.make_node("Relu", ["in0"], ["out"], name="Relu_0")
    graph = helper.make_graph([relu_node], "test_nchw", [in0], [out])
    model = helper.make_model(graph, producer_name="test")
    model = ModelWrapper(model)

    # Set INT8 datatype and NCHW layout
    model.set_tensor_datatype("in0", DataType["INT8"])
    model.set_tensor_datatype("out", DataType["INT8"])
    model.set_tensor_layout("in0", DataLayout.NCHW)
    model.set_tensor_layout("out", DataLayout.NCHW)

    return model


def test_normalize_layouts_step_execution():
    """Test that the step can be invoked and transforms the model."""
    # Create test model
    model = make_test_nchw_model()

    # Verify initial state
    assert model.get_tensor_layout("in0") == DataLayout.NCHW
    assert model.get_tensor_layout("out") == DataLayout.NCHW
    initial_node_count = len(model.graph.node)

    # Mock config object (step doesn't use it)
    class MockConfig:
        pass

    cfg = MockConfig()

    # Execute the step directly
    transformed_model = normalize_dataflow_layouts_step(model, cfg)

    # Verify transformation occurred
    assert transformed_model is not None
    assert (
        len(transformed_model.graph.node) > initial_node_count
    ), "Should have added Transpose nodes"

    # Find Transpose nodes
    transpose_nodes = [n for n in transformed_model.graph.node if n.op_type == "Transpose"]
    assert len(transpose_nodes) >= 2, "Should have input and output Transposes"

    # Verify output layout is preserved (NCHW due to reverse Transpose)
    assert transformed_model.get_tensor_layout("out") == DataLayout.NCHW


def test_normalize_layouts_step_preserves_datatypes():
    """Test that the step preserves tensor datatypes."""
    model = make_test_nchw_model()

    # Get original datatypes
    orig_in_dtype = model.get_tensor_datatype("in0")
    orig_out_dtype = model.get_tensor_datatype("out")

    # Execute step directly
    class MockConfig:
        pass

    transformed_model = normalize_dataflow_layouts_step(model, MockConfig())

    # Verify datatypes preserved
    assert transformed_model.get_tensor_datatype("in0") == orig_in_dtype
    assert transformed_model.get_tensor_datatype("out") == orig_out_dtype


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
