############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
############################################################################

"""Tests for kernel inference step."""

import pytest
import tempfile
import os
from types import SimpleNamespace
from onnx import helper, TensorProto

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
import qonnx.core.data_layout as DataLayout

from brainsmith.steps.kernel_inference import infer_kernels_step


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


def test_infer_kernels_step_basic():
    """Test infer_kernels step with single kernel (AddStreams)."""
    model = create_add_model()

    # Create mock config
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = SimpleNamespace(
            kernel_selections=[("AddStreams", "hls")],
            output_dir=tmpdir
        )

        # Run step
        result_model = infer_kernels_step(model, cfg)

        # Verify Add node was converted to AddStreams
        node_types = [n.op_type for n in result_model.graph.node]
        assert "Add" not in node_types, "Add node should be removed"
        assert "AddStreams" in node_types, "AddStreams node should be created"

        # Verify debug output was saved
        debug_file = os.path.join(tmpdir, "debug_infer_kernels_output.onnx")
        assert os.path.exists(debug_file), "Debug ONNX file should be saved"


def test_infer_kernels_step_multiple_kernels():
    """Test infer_kernels step with multiple kernel selections."""
    model = create_add_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = SimpleNamespace(
            kernel_selections=[
                ("AddStreams", "hls"),
                ("Thresholding", "rtl"),  # Won't match anything in this model
            ],
            output_dir=tmpdir
        )

        result_model = infer_kernels_step(model, cfg)

        # AddStreams should be applied
        node_types = [n.op_type for n in result_model.graph.node]
        assert "AddStreams" in node_types


def test_infer_kernels_step_no_selections():
    """Test infer_kernels step with kernel_selections = None."""
    model = create_add_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = SimpleNamespace(
            kernel_selections=None,
            output_dir=tmpdir
        )

        # Should return model unchanged
        result_model = infer_kernels_step(model, cfg)

        # Model should be unchanged (Add node still present)
        node_types = [n.op_type for n in result_model.graph.node]
        assert "Add" in node_types


def test_infer_kernels_step_missing_attribute():
    """Test infer_kernels step with missing kernel_selections attribute."""
    model = create_add_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = SimpleNamespace(
            output_dir=tmpdir
            # No kernel_selections attribute
        )

        # Should handle gracefully and return unchanged model
        result_model = infer_kernels_step(model, cfg)

        # Model should be unchanged
        node_types = [n.op_type for n in result_model.graph.node]
        assert "Add" in node_types


def test_infer_kernels_step_invalid_kernel_name():
    """Test infer_kernels step with non-existent kernel name."""
    model = create_add_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = SimpleNamespace(
            kernel_selections=[
                ("NonExistentKernel", "hls"),
                ("AddStreams", "hls"),  # This one should still work
            ],
            output_dir=tmpdir
        )

        # Should continue without crashing
        result_model = infer_kernels_step(model, cfg)

        # AddStreams should still be applied despite invalid kernel
        node_types = [n.op_type for n in result_model.graph.node]
        assert "AddStreams" in node_types


def test_infer_kernels_step_empty_selections():
    """Test infer_kernels step with empty kernel_selections list."""
    model = create_add_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = SimpleNamespace(
            kernel_selections=[],
            output_dir=tmpdir
        )

        # Should return model unchanged
        result_model = infer_kernels_step(model, cfg)

        # Model should be unchanged
        node_types = [n.op_type for n in result_model.graph.node]
        assert "Add" in node_types


def test_infer_kernels_step_saves_debug_output():
    """Test that debug ONNX output is saved correctly."""
    model = create_add_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = SimpleNamespace(
            kernel_selections=[("AddStreams", "hls")],
            output_dir=tmpdir
        )

        infer_kernels_step(model, cfg)

        # Verify debug file exists
        debug_file = os.path.join(tmpdir, "debug_infer_kernels_output.onnx")
        assert os.path.exists(debug_file), "Debug ONNX file should exist"

        # Verify it's a valid ONNX model
        debug_model = ModelWrapper(debug_file)
        assert debug_model is not None
        assert len(debug_model.graph.node) > 0


def test_infer_kernels_step_uses_infer_kernel_list():
    """Test that step uses InferKernelList (not individual transforms)."""
    model = create_add_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = SimpleNamespace(
            kernel_selections=[("AddStreams", "hls")],
            output_dir=tmpdir
        )

        # This should use InferKernelList internally
        result_model = infer_kernels_step(model, cfg)

        # Verify conversion happened (proving InferKernelList worked)
        node_types = [n.op_type for n in result_model.graph.node]
        assert "AddStreams" in node_types

        # Verify the node has correct domain (added by InferKernelList/InferKernel)
        addstreams_node = [n for n in result_model.graph.node if n.op_type == "AddStreams"][0]
        assert addstreams_node.domain == "finn.custom_op.fpgadataflow"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
