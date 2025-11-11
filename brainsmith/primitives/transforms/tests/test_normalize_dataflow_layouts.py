############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
############################################################################

"""Tests for NormalizeDataflowLayouts transformation."""

import pytest
import qonnx.core.data_layout as DataLayout
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from brainsmith.primitives.transforms.normalize_dataflow_layouts import NormalizeDataflowLayouts


def make_nchw_model():
    """Create a simple ONNX model with NCHW layout tensors.

    Returns:
        ModelWrapper with NCHW tensors that need conversion
    """
    # Create input/output tensors in NCHW format
    in0 = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [1, 64, 224, 224])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 64, 224, 224])

    # Create a simple ReLU node
    relu_node = helper.make_node("Relu", ["in0"], ["out"], name="Relu_0")

    # Create graph
    graph = helper.make_graph(
        [relu_node],
        "test_nchw",
        [in0],
        [out]
    )

    # Create model
    model = helper.make_model(graph, producer_name="test")
    model = ModelWrapper(model)

    # Set datatypes (integer for dataflow)
    model.set_tensor_datatype("in0", DataType["INT8"])
    model.set_tensor_datatype("out", DataType["INT8"])

    # Set NCHW layout
    model.set_tensor_layout("in0", DataLayout.NCHW)
    model.set_tensor_layout("out", DataLayout.NCHW)

    return model


def make_mixed_layout_model():
    """Create a model with mixed NCHW and NHWC tensors.

    Returns:
        ModelWrapper with both NCHW and NHWC tensors
    """
    # Create tensors with different layouts
    in_nchw = helper.make_tensor_value_info("in_nchw", TensorProto.FLOAT, [1, 64, 224, 224])
    in_nhwc = helper.make_tensor_value_info("in_nhwc", TensorProto.FLOAT, [1, 224, 224, 64])

    # Intermediate tensors
    relu_out = helper.make_tensor_value_info("relu_out", TensorProto.FLOAT, [1, 64, 224, 224])
    add_out = helper.make_tensor_value_info("add_out", TensorProto.FLOAT, [1, 224, 224, 64])

    # Create nodes
    relu_node = helper.make_node("Relu", ["in_nchw"], ["relu_out"], name="Relu_0")
    # Note: This is a contrived example for testing - real models wouldn't have this mismatch
    add_node = helper.make_node("Add", ["in_nhwc", "in_nhwc"], ["add_out"], name="Add_0")

    # Create graph
    graph = helper.make_graph(
        [relu_node, add_node],
        "test_mixed",
        [in_nchw, in_nhwc],
        [relu_out, add_out]
    )

    # Create model
    model = helper.make_model(graph, producer_name="test")
    model = ModelWrapper(model)

    # Set datatypes
    model.set_tensor_datatype("in_nchw", DataType["INT8"])
    model.set_tensor_datatype("in_nhwc", DataType["INT8"])
    model.set_tensor_datatype("relu_out", DataType["INT8"])
    model.set_tensor_datatype("add_out", DataType["INT8"])

    # Set layouts
    model.set_tensor_layout("in_nchw", DataLayout.NCHW)
    model.set_tensor_layout("relu_out", DataLayout.NCHW)
    model.set_tensor_layout("in_nhwc", DataLayout.NHWC)
    model.set_tensor_layout("add_out", DataLayout.NHWC)

    return model


def make_multi_stage_nchw_model():
    """Create a model with multiple NCHW operations in sequence.

    Returns:
        ModelWrapper with chain of NCHW operations
    """
    # Create tensors
    in0 = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [1, 64, 224, 224])
    relu1_out = helper.make_tensor_value_info("relu1_out", TensorProto.FLOAT, [1, 64, 224, 224])
    relu2_out = helper.make_tensor_value_info("relu2_out", TensorProto.FLOAT, [1, 64, 224, 224])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 64, 224, 224])

    # Create nodes
    relu1_node = helper.make_node("Relu", ["in0"], ["relu1_out"], name="Relu_1")
    relu2_node = helper.make_node("Relu", ["relu1_out"], ["relu2_out"], name="Relu_2")
    relu3_node = helper.make_node("Relu", ["relu2_out"], ["out"], name="Relu_3")

    # Create graph
    graph = helper.make_graph(
        [relu1_node, relu2_node, relu3_node],
        "test_multi_stage",
        [in0],
        [out]
    )

    # Create model
    model = helper.make_model(graph, producer_name="test")
    model = ModelWrapper(model)

    # Set datatypes
    for tensor_name in ["in0", "relu1_out", "relu2_out", "out"]:
        model.set_tensor_datatype(tensor_name, DataType["INT8"])
        model.set_tensor_layout(tensor_name, DataLayout.NCHW)

    return model


def test_normalize_basic_nchw():
    """Test basic NCHW to NHWC conversion."""
    model = make_nchw_model()

    # Verify initial state
    assert model.get_tensor_layout("in0") == DataLayout.NCHW
    assert model.get_tensor_layout("out") == DataLayout.NCHW
    initial_node_count = len(model.graph.node)
    assert initial_node_count == 1  # Just the ReLU

    # Apply transformation
    transform = NormalizeDataflowLayouts()
    model, modified = transform.apply(model)

    # Verify transformation occurred
    assert modified, "Graph should be modified"

    # Should have added Transpose nodes
    # Expected structure:
    # Input(NCHW) -> Transpose -> ReLU -> Transpose -> Output(NCHW preserved)
    assert len(model.graph.node) > initial_node_count, "Should have added Transpose nodes"

    # Find Transpose nodes
    transpose_nodes = [n for n in model.graph.node if n.op_type == "Transpose"]
    assert len(transpose_nodes) >= 2, "Should have at least input and output Transposes"

    # Verify output layout is preserved (should still be NCHW due to reverse Transpose)
    assert model.get_tensor_layout("out") == DataLayout.NCHW, "Output layout should be preserved"


def test_normalize_already_nhwc():
    """Test that NHWC models are not modified."""
    # Create a model that's already in NHWC
    in0 = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [1, 224, 224, 64])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 224, 224, 64])

    relu_node = helper.make_node("Relu", ["in0"], ["out"], name="Relu_0")

    graph = helper.make_graph([relu_node], "test_nhwc", [in0], [out])
    model = helper.make_model(graph, producer_name="test")
    model = ModelWrapper(model)

    model.set_tensor_datatype("in0", DataType["INT8"])
    model.set_tensor_datatype("out", DataType["INT8"])
    model.set_tensor_layout("in0", DataLayout.NHWC)
    model.set_tensor_layout("out", DataLayout.NHWC)

    initial_node_count = len(model.graph.node)

    # Apply transformation
    transform = NormalizeDataflowLayouts()
    model, modified = transform.apply(model)

    # Should not be modified since already NHWC
    assert not modified, "NHWC model should not be modified"
    assert len(model.graph.node) == initial_node_count, "No nodes should be added"


def test_normalize_mixed_layouts():
    """Test model with mixed NCHW and NHWC tensors."""
    model = make_mixed_layout_model()

    # Verify initial state
    assert model.get_tensor_layout("in_nchw") == DataLayout.NCHW
    assert model.get_tensor_layout("in_nhwc") == DataLayout.NHWC

    # Apply transformation
    transform = NormalizeDataflowLayouts()
    model, modified = transform.apply(model)

    # Should be modified due to NCHW tensors
    assert modified, "Graph should be modified"

    # Find Transpose nodes
    transpose_nodes = [n for n in model.graph.node if n.op_type == "Transpose"]
    assert len(transpose_nodes) > 0, "Should have added Transpose nodes for NCHW tensors"


def test_normalize_multi_stage():
    """Test multi-stage NCHW model."""
    model = make_multi_stage_nchw_model()

    initial_node_count = len(model.graph.node)

    # Apply transformation
    transform = NormalizeDataflowLayouts()
    model, modified = transform.apply(model)

    # Verify transformation
    assert modified, "Graph should be modified"
    assert len(model.graph.node) > initial_node_count, "Should have added Transpose nodes"

    # Output should still be NCHW (preserved)
    assert model.get_tensor_layout("out") == DataLayout.NCHW


def test_normalize_preserves_shapes():
    """Test that transformation preserves tensor shapes correctly."""
    model = make_nchw_model()

    # Get original shapes
    orig_in_shape = model.get_tensor_shape("in0")
    orig_out_shape = model.get_tensor_shape("out")

    # Apply transformation
    transform = NormalizeDataflowLayouts()
    model, modified = transform.apply(model)

    # Verify shapes are preserved at graph boundaries
    # Input and output should maintain their original shapes
    new_in_shape = model.get_tensor_shape("in0")
    new_out_shape = model.get_tensor_shape("out")

    assert new_in_shape == orig_in_shape, "Input shape should be preserved"
    assert new_out_shape == orig_out_shape, "Output shape should be preserved"


def test_normalize_preserves_datatypes():
    """Test that transformation preserves datatypes."""
    model = make_nchw_model()

    # Get original datatypes
    orig_in_dtype = model.get_tensor_datatype("in0")
    orig_out_dtype = model.get_tensor_datatype("out")

    # Apply transformation
    transform = NormalizeDataflowLayouts()
    model, modified = transform.apply(model)

    # Verify datatypes are preserved
    new_in_dtype = model.get_tensor_datatype("in0")
    new_out_dtype = model.get_tensor_datatype("out")

    assert new_in_dtype == orig_in_dtype, "Input datatype should be preserved"
    assert new_out_dtype == orig_out_dtype, "Output datatype should be preserved"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
