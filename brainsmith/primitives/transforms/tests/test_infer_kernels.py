############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
############################################################################

"""Tests for InferKernels meta-transform."""

import pytest
import numpy as np
from onnx import helper, TensorProto

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType

from brainsmith.primitives.transforms.infer_kernels import InferKernels


def make_mixed_model():
    """Create ONNX model with multiple operations for testing.

    Returns:
        ModelWrapper with Add and MatMul nodes
    """
    # Inputs
    in0 = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [1, 64])
    in1 = helper.make_tensor_value_info("in1", TensorProto.FLOAT, [1, 64])
    weights = helper.make_tensor_value_info("weights", TensorProto.FLOAT, [64, 32])
    mid = helper.make_tensor_value_info("mid", TensorProto.FLOAT, [1, 64])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 32])

    # Create nodes
    add_node = helper.make_node("Add", ["in0", "in1"], ["mid"], name="Add_0")
    matmul_node = helper.make_node("MatMul", ["mid", "weights"], ["out"], name="MatMul_0")

    # Create graph
    graph = helper.make_graph(
        [add_node, matmul_node],
        "test_mixed",
        [in0, in1, weights],
        [out]
    )

    model = ModelWrapper(helper.make_model(graph))

    # Set datatypes
    for tensor in ["in0", "in1", "mid"]:
        model.set_tensor_datatype(tensor, DataType["INT8"])
    model.set_tensor_datatype("weights", DataType["INT8"])
    model.set_tensor_datatype("out", DataType["INT8"])

    # Set layout
    import qonnx.core.data_layout as DataLayout
    for tensor in ["in0", "in1", "mid", "out"]:
        model.set_tensor_layout(tensor, DataLayout.NHWC)

    # Initialize weights
    weights_data = np.random.randint(-128, 127, size=(64, 32), dtype=np.int8)
    model.set_initializer("weights", weights_data)

    return model


def test_infer_kernels_single_kernelop():
    """Test InferKernels with single KernelOp class."""
    from brainsmith.kernels.addstreams import AddStreams

    # Create simple Add model
    in0 = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [1, 64])
    in1 = helper.make_tensor_value_info("in1", TensorProto.FLOAT, [1, 64])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 64])

    add_node = helper.make_node("Add", ["in0", "in1"], ["out"])
    graph = helper.make_graph([add_node], "test", [in0, in1], [out])
    model = ModelWrapper(helper.make_model(graph))

    # Set integer datatypes
    for tensor in ["in0", "in1", "out"]:
        model.set_tensor_datatype(tensor, DataType["INT8"])
        import qonnx.core.data_layout as DataLayout
        model.set_tensor_layout(tensor, DataLayout.NHWC)

    # Apply InferKernels with single kernel
    transform = InferKernels([AddStreams])
    model, modified = transform.apply(model)

    assert modified, "Graph should be modified"
    assert model.graph.node[0].op_type == "AddStreams"


def test_infer_kernels_backward_compatible():
    """Test InferKernels with None (backward compatible mode)."""
    # Create simple Add model
    in0 = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [1, 64])
    in1 = helper.make_tensor_value_info("in1", TensorProto.FLOAT, [1, 64])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 64])

    add_node = helper.make_node("Add", ["in0", "in1"], ["out"])
    graph = helper.make_graph([add_node], "test", [in0, in1], [out])
    model = ModelWrapper(helper.make_model(graph))

    # Set integer datatypes
    for tensor in ["in0", "in1", "out"]:
        model.set_tensor_datatype(tensor, DataType["INT8"])
        import qonnx.core.data_layout as DataLayout
        model.set_tensor_layout(tensor, DataLayout.NHWC)

    # Apply InferKernels with None (should infer all registered KernelOp kernels)
    transform = InferKernels()
    model, modified = transform.apply(model)

    # Should convert Add to AddStreams (if AddStreams is registered)
    assert modified or model.graph.node[0].op_type == "Add"


def test_infer_kernels_filters_infrastructure():
    """Test that InferKernels skips infrastructure kernels in backward compatible mode."""
    from brainsmith.registry import kernel, reset_registry, discover_components
    from brainsmith.registry import get_component_metadata
    from brainsmith.dataflow import KernelOp

    # Create a mock infrastructure kernel for testing
    @kernel(name="TestInfraKernel", is_infrastructure=True)
    class TestInfraKernel(KernelOp):
        """Mock infrastructure kernel for testing."""
        @classmethod
        def can_infer_from(cls, node, model):
            return node.op_type == "TestOp"

    # Verify it's marked as infrastructure (registered as 'custom' source)
    metadata = get_component_metadata("custom:TestInfraKernel", "kernel")
    assert metadata.is_infrastructure is True

    # Create a simple model with TestOp node
    in0 = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [1, 64])
    out0 = helper.make_tensor_value_info("out0", TensorProto.FLOAT, [1, 64])
    node = helper.make_node("TestOp", ["in0"], ["out0"])
    graph = helper.make_graph([node], "test", [in0], [out0])
    onnx_model = helper.make_model(graph, producer_name="test")
    model = ModelWrapper(onnx_model)

    # Apply InferKernels in backward compatible mode (kernel_classes=None)
    transform = InferKernels()
    model, modified = transform.apply(model)

    # Infrastructure kernel should be skipped, so node should remain as TestOp
    assert model.graph.node[0].op_type == "TestOp", \
        "Infrastructure kernel should be skipped by InferKernels"
    # Since no computational kernel matches TestOp, modified should be False
    assert modified is False


def test_infer_kernels_legacy_transform():
    """Test InferKernels with legacy HWCustomOp class."""
    # This test requires FINN transforms to be registered
    try:
        from finn.custom_op.fpgadataflow.addstreams import AddStreams as FinnAddStreams
    except ImportError:
        pytest.skip("FINN not available")

    # Create Add model
    in0 = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [1, 64])
    in1 = helper.make_tensor_value_info("in1", TensorProto.FLOAT, [1, 64])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 64])

    add_node = helper.make_node("Add", ["in0", "in1"], ["out"])
    graph = helper.make_graph([add_node], "test", [in0, in1], [out])
    model = ModelWrapper(helper.make_model(graph))

    # Set integer datatypes
    for tensor in ["in0", "in1", "out"]:
        model.set_tensor_datatype(tensor, DataType["INT8"])
        import qonnx.core.data_layout as DataLayout
        model.set_tensor_layout(tensor, DataLayout.NHWC)

    # Apply InferKernels with FINN HWCustomOp class
    # Should lookup InferAddStreamsLayer via metadata
    transform = InferKernels([FinnAddStreams])
    model, modified = transform.apply(model)

    # Might be modified (depends on whether transform is registered)
    # The important thing is it doesn't crash
    assert model is not None


def test_infer_kernels_mixed_types():
    """Test InferKernels with mix of KernelOp and HWCustomOp classes."""
    from brainsmith.kernels.addstreams import AddStreams

    # Try to import FINN MVAU
    try:
        from finn.custom_op.fpgadataflow.matrixvectoractivation import MVAU
    except ImportError:
        pytest.skip("FINN not available")

    # Create model with both Add and MatMul
    model = make_mixed_model()

    # Apply InferKernels with mixed list
    # AddStreams is KernelOp, MVAU is HWCustomOp
    transform = InferKernels([AddStreams, MVAU])
    model, modified = transform.apply(model)

    # At least one should be converted
    assert model is not None


def test_infer_kernels_empty_list():
    """Test InferKernels with empty list."""
    in0 = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [1, 64])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 64])

    identity = helper.make_node("Identity", ["in0"], ["out"])
    graph = helper.make_graph([identity], "test", [in0], [out])
    model = ModelWrapper(helper.make_model(graph))

    # Apply with empty list
    transform = InferKernels([])
    model, modified = transform.apply(model)

    assert not modified, "Should not modify with empty list"
    assert model.graph.node[0].op_type == "Identity"


def test_infer_kernels_error_handling():
    """Test that InferKernels handles errors gracefully."""
    from brainsmith.kernels.addstreams import AddStreams

    # Create model with malformed Add node (missing datatype)
    in0 = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [1, 64])
    in1 = helper.make_tensor_value_info("in1", TensorProto.FLOAT, [1, 64])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 64])

    add_node = helper.make_node("Add", ["in0", "in1"], ["out"])
    graph = helper.make_graph([add_node], "test", [in0, in1], [out])
    model = ModelWrapper(helper.make_model(graph))

    # Apply transform without setting datatypes
    # Should handle gracefully (warn but not crash)
    transform = InferKernels([AddStreams])
    model, modified = transform.apply(model)

    assert model is not None


def test_infer_kernels_logging():
    """Test that InferKernels logs appropriately."""
    from brainsmith.kernels.addstreams import AddStreams
    import logging
    from io import StringIO

    # Create Add model
    in0 = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [1, 64])
    in1 = helper.make_tensor_value_info("in1", TensorProto.FLOAT, [1, 64])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 64])

    add_node = helper.make_node("Add", ["in0", "in1"], ["out"])
    graph = helper.make_graph([add_node], "test", [in0, in1], [out])
    model = ModelWrapper(helper.make_model(graph))

    # Set datatypes
    for tensor in ["in0", "in1", "out"]:
        model.set_tensor_datatype(tensor, DataType["INT8"])
        import qonnx.core.data_layout as DataLayout
        model.set_tensor_layout(tensor, DataLayout.NHWC)

    # Capture logs
    logger = logging.getLogger("brainsmith.primitives.transforms.infer_kernels")
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Apply transform
    transform = InferKernels([AddStreams])
    model, modified = transform.apply(model)

    # Check logs mention the kernel
    log_output = stream.getvalue()
    assert "AddStreams" in log_output
    assert "KernelOp" in log_output

    # Clean up
    logger.removeHandler(handler)
