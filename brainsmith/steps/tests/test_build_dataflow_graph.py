############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
############################################################################

"""Tests for build_dataflow_graph step."""

import tempfile
from types import SimpleNamespace

import pytest
import qonnx.core.data_layout as DataLayout
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from brainsmith.steps.build_dataflow_graph import build_dataflow_graph


def create_add_model():
    """Create simple ONNX model with Add node for testing.

    Returns:
        ModelWrapper with Add node (configured for AddStreams inference)
    """
    in0 = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [1, 64])
    in1 = helper.make_tensor_value_info("in1", TensorProto.FLOAT, [1, 64])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 64])

    add_node = helper.make_node("Add", ["in0", "in1"], ["out"], name="add_0")
    graph = helper.make_graph([add_node], "test_add", [in0, in1], [out])
    model = ModelWrapper(helper.make_model(graph))

    # Set integer datatypes and NHWC layout (required for AddStreams inference)
    for tensor in ["in0", "in1", "out"]:
        model.set_tensor_datatype(tensor, DataType["INT8"])
        model.set_tensor_layout(tensor, DataLayout.NHWC)

    return model


def test_build_dataflow_graph_basic():
    """Test infer_kernels step with single kernel (AddStreams)."""
    model = create_add_model()

    # Create mock config
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = SimpleNamespace(kernel_selections=[("AddStreams", "hls")], output_dir=tmpdir)

        # Run step
        result_model = build_dataflow_graph(model, cfg)

        # Verify Add node was converted to AddStreams
        node_types = [n.op_type for n in result_model.graph.node]
        assert "Add" not in node_types, "Add node should be removed"
        assert "AddStreams" in node_types, "AddStreams node should be created"


def test_build_dataflow_graph_multiple_kernels():
    """Test infer_kernels step with multiple kernel selections."""
    model = create_add_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = SimpleNamespace(
            kernel_selections=[
                ("AddStreams", "hls"),
                ("Thresholding", "rtl"),  # Won't match anything in this model
            ],
            output_dir=tmpdir,
        )

        result_model = build_dataflow_graph(model, cfg)

        # AddStreams should be applied
        node_types = [n.op_type for n in result_model.graph.node]
        assert "AddStreams" in node_types


def test_build_dataflow_graph_no_selections():
    """Test infer_kernels step with kernel_selections = None."""
    model = create_add_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = SimpleNamespace(kernel_selections=None, output_dir=tmpdir)

        # Should return model unchanged
        result_model = build_dataflow_graph(model, cfg)

        # Model should be unchanged (Add node still present)
        node_types = [n.op_type for n in result_model.graph.node]
        assert "Add" in node_types


def test_build_dataflow_graph_missing_attribute():
    """Test infer_kernels step with missing kernel_selections attribute."""
    model = create_add_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = SimpleNamespace(
            output_dir=tmpdir
            # No kernel_selections attribute
        )

        # Should handle gracefully and return unchanged model
        result_model = build_dataflow_graph(model, cfg)

        # Model should be unchanged
        node_types = [n.op_type for n in result_model.graph.node]
        assert "Add" in node_types


def test_build_dataflow_graph_invalid_kernel_name():
    """Test infer_kernels step with non-existent kernel name."""
    model = create_add_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = SimpleNamespace(
            kernel_selections=[
                ("NonExistentKernel", "hls"),
                ("AddStreams", "hls"),  # This one should still work
            ],
            output_dir=tmpdir,
        )

        # Should continue without crashing
        result_model = build_dataflow_graph(model, cfg)

        # AddStreams should still be applied despite invalid kernel
        node_types = [n.op_type for n in result_model.graph.node]
        assert "AddStreams" in node_types


def test_build_dataflow_graph_empty_selections():
    """Test infer_kernels step with empty kernel_selections list."""
    model = create_add_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = SimpleNamespace(kernel_selections=[], output_dir=tmpdir)

        # Should return model unchanged
        result_model = build_dataflow_graph(model, cfg)

        # Model should be unchanged
        node_types = [n.op_type for n in result_model.graph.node]
        assert "Add" in node_types


def test_build_dataflow_graph_uses_infer_kernels():
    """Test that step uses InferKernels transform internally."""
    model = create_add_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = SimpleNamespace(kernel_selections=[("AddStreams", "hls")], output_dir=tmpdir)

        # This should use InferKernels internally
        result_model = build_dataflow_graph(model, cfg)

        # Verify conversion happened (proving InferKernels worked)
        node_types = [n.op_type for n in result_model.graph.node]
        assert "AddStreams" in node_types

        # Verify the node has correct domain (added by InferKernels/InferKernel)
        addstreams_node = [n for n in result_model.graph.node if n.op_type == "AddStreams"][0]
        assert addstreams_node.domain == "brainsmith.kernels"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
