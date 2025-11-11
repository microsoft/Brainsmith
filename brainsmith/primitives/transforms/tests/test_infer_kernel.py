############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
############################################################################

"""Tests for InferKernel transform."""

import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from brainsmith.primitives.transforms.infer_kernel import InferKernel


def make_simple_add_model():
    """Create a simple ONNX model with an Add node.

    Returns:
        ModelWrapper with Add node (compatible with AddStreams inference)
    """
    # Create input tensors
    in0 = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [1, 224, 224, 64])
    in1 = helper.make_tensor_value_info("in1", TensorProto.FLOAT, [1, 224, 224, 64])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 224, 224, 64])

    # Create Add node
    add_node = helper.make_node("Add", ["in0", "in1"], ["out"], name="Add_0")

    # Create graph
    graph = helper.make_graph(
        [add_node],
        "test_add",
        [in0, in1],
        [out]
    )

    # Create model
    model = helper.make_model(graph, producer_name="test")
    model = ModelWrapper(model)

    # Set datatypes (integer required for AddStreams)
    model.set_tensor_datatype("in0", DataType["INT8"])
    model.set_tensor_datatype("in1", DataType["INT8"])
    model.set_tensor_datatype("out", DataType["INT8"])

    # Set layout
    import qonnx.core.data_layout as DataLayout
    model.set_tensor_layout("in0", DataLayout.NHWC)
    model.set_tensor_layout("in1", DataLayout.NHWC)
    model.set_tensor_layout("out", DataLayout.NHWC)

    return model


def test_infer_kernel_basic():
    """Test InferKernel with AddStreams kernel."""
    from brainsmith.kernels.addstreams import AddStreams

    # Create test model
    model = make_simple_add_model()

    # Verify initial state
    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == "Add"

    # Apply InferKernel
    transform = InferKernel(AddStreams)
    model, modified = transform.apply(model)

    # Verify transformation
    assert modified, "Graph should be modified"
    assert len(model.graph.node) == 1, "Should have one node (AddStreams)"
    assert model.graph.node[0].op_type == "AddStreams", "Should be AddStreams node"
    assert model.graph.node[0].domain == "brainsmith.kernels"


def test_infer_kernel_invalid_class():
    """Test InferKernel with non-KernelOp class."""
    from finn.custom_op.fpgadataflow.addstreams import AddStreams as FinnAddStreams

    # Finn's AddStreams is HWCustomOp, not KernelOp
    with pytest.raises(ValueError, match="InferKernel requires a KernelOp subclass"):
        InferKernel(FinnAddStreams)


def test_infer_kernel_no_matching_nodes():
    """Test InferKernel when no nodes match."""
    from brainsmith.kernels.addstreams import AddStreams

    # Create model with MatMul (doesn't match AddStreams pattern)
    in0 = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [1, 128])
    in1 = helper.make_tensor_value_info("in1", TensorProto.FLOAT, [128, 64])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 64])

    matmul_node = helper.make_node("MatMul", ["in0", "in1"], ["out"])
    graph = helper.make_graph([matmul_node], "test", [in0, in1], [out])
    model = ModelWrapper(helper.make_model(graph))

    # Apply transform
    transform = InferKernel(AddStreams)
    model, modified = transform.apply(model)

    # Verify no changes
    assert not modified, "Graph should not be modified"
    assert model.graph.node[0].op_type == "MatMul"


def test_infer_kernel_statistics():
    """Test that InferKernel logs statistics correctly."""
    import logging
    from io import StringIO

    from brainsmith.kernels.addstreams import AddStreams

    # Create model with two Add nodes
    in0 = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [1, 64])
    in1 = helper.make_tensor_value_info("in1", TensorProto.FLOAT, [1, 64])
    helper.make_tensor_value_info("mid", TensorProto.FLOAT, [1, 64])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 64])

    add1 = helper.make_node("Add", ["in0", "in1"], ["mid"], name="Add_1")
    add2 = helper.make_node("Add", ["mid", "in1"], ["out"], name="Add_2")

    graph = helper.make_graph([add1, add2], "test", [in0, in1], [out])
    model = ModelWrapper(helper.make_model(graph))

    # Set integer datatypes
    for tensor in ["in0", "in1", "mid", "out"]:
        model.set_tensor_datatype(tensor, DataType["INT8"])

    # Set layout
    import qonnx.core.data_layout as DataLayout
    for tensor in ["in0", "in1", "mid", "out"]:
        model.set_tensor_layout(tensor, DataLayout.NHWC)

    # Capture logs
    logger = logging.getLogger("brainsmith.primitives.transforms.infer_kernel")
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Apply transform
    transform = InferKernel(AddStreams)
    model, modified = transform.apply(model)

    # Check logs (at least 1 node should be processed)
    log_output = stream.getvalue()
    assert "processed" in log_output
    assert "converted" in log_output
    # Note: Might process 1 or 2 nodes depending on graph state after first conversion

    # Clean up
    logger.removeHandler(handler)


def test_infer_kernel_error_handling():
    """Test that InferKernel handles errors gracefully."""
    from brainsmith.kernels.addstreams import AddStreams

    # Create model with Add node but missing datatype info
    # This should cause can_infer_from to fail or infer_from to raise
    in0 = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [1, 64])
    in1 = helper.make_tensor_value_info("in1", TensorProto.FLOAT, [1, 64])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 64])

    add_node = helper.make_node("Add", ["in0", "in1"], ["out"])
    graph = helper.make_graph([add_node], "test", [in0, in1], [out])
    model = ModelWrapper(helper.make_model(graph))

    # Don't set datatypes - this may cause issues

    # Apply transform (should not crash, just skip)
    transform = InferKernel(AddStreams)
    model, modified = transform.apply(model)

    # Transform should complete (might not modify graph if validation fails)
    # The important thing is it doesn't crash
    assert model is not None
