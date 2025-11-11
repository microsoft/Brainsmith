############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
############################################################################

"""Tests for InsertInfrastructureKernels transform."""

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from brainsmith.kernels.duplicate_streams import DuplicateStreams
from brainsmith.primitives.transforms import InsertInfrastructureKernels
from brainsmith.primitives.transforms.insert_infrastructure_kernels import (
    INFRASTRUCTURE_TRANSFORM_MAP,
    _register_infrastructure_transform,
)


def make_fanout_model():
    """Create a simple model with tensor fanout for testing."""
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 64])
    outp0 = helper.make_tensor_value_info("outp0", TensorProto.FLOAT, [1, 64])
    outp1 = helper.make_tensor_value_info("outp1", TensorProto.FLOAT, [1, 64])

    # Intermediate tensor with fanout
    tensor_x = helper.make_tensor_value_info("tensor_x", TensorProto.FLOAT, [1, 64])

    # Create fanout: inp → Relu → tensor_x → Add (consumer 0)
    #                                       → Relu (consumer 1)
    relu_prod = helper.make_node("Relu", ["inp"], ["tensor_x"], name="Relu_prod")
    add = helper.make_node("Add", ["tensor_x", "bias"], ["outp0"], name="Add_0")
    relu_cons = helper.make_node("Relu", ["tensor_x"], ["outp1"], name="Relu_cons")

    graph = helper.make_graph(
        [relu_prod, add, relu_cons],
        "test_fanout",
        [inp],
        [outp0, outp1],
        value_info=[tensor_x]  # Include intermediate tensor
    )

    model = helper.make_model(graph, producer_name="test")
    model = ModelWrapper(model)

    # Set tensor shapes and datatypes
    for tensor in ["inp", "tensor_x", "outp0", "outp1"]:
        model.set_tensor_shape(tensor, [1, 64])
        model.set_tensor_datatype(tensor, DataType["INT8"])

    # Initialize bias for Add operation
    bias_data = np.random.randn(1, 64).astype(np.float32)
    model.set_initializer("bias", bias_data)

    return model


def make_simple_model():
    """Create a simple model without fanout."""
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 64])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 64])

    # Simple chain: inp → Relu → outp (no fanout)
    relu = helper.make_node("Relu", ["inp"], ["outp"], name="Relu_0")

    graph = helper.make_graph([relu], "test_simple", [inp], [outp])

    model = helper.make_model(graph, producer_name="test")
    model = ModelWrapper(model)

    # Set tensor shapes
    for tensor in ["inp", "outp"]:
        model.set_tensor_shape(tensor, [1, 64])

    return model


class TestInsertInfrastructureKernels:
    """Test suite for InsertInfrastructureKernels transform."""

    def test_insert_duplicate_streams(self):
        """Test inserting DuplicateStreams infrastructure kernel."""
        model = make_fanout_model()

        # Verify no DuplicateStreams initially
        assert not any(n.op_type == "DuplicateStreams" for n in model.graph.node)

        # Apply transform
        transform = InsertInfrastructureKernels([DuplicateStreams])
        model, modified = transform.apply(model)

        # Should have modified graph and inserted DuplicateStreams
        assert modified
        assert any(n.op_type == "DuplicateStreams" for n in model.graph.node)

    def test_empty_kernel_list(self):
        """Test with empty kernel list (no-op)."""
        model = make_fanout_model()

        # Apply transform with empty list
        transform = InsertInfrastructureKernels([])
        model, modified = transform.apply(model)

        # Should not modify graph
        assert not modified
        assert not any(n.op_type == "DuplicateStreams" for n in model.graph.node)

    def test_no_insertion_needed(self):
        """Test when infrastructure kernel not needed (no fanout)."""
        model = make_simple_model()

        # Apply transform
        transform = InsertInfrastructureKernels([DuplicateStreams])
        model, modified = transform.apply(model)

        # Should not modify graph (no fanout to handle)
        assert not modified
        assert not any(n.op_type == "DuplicateStreams" for n in model.graph.node)

    def test_unmapped_kernel_warning(self, caplog):
        """Test warning when kernel has no registered transform."""
        model = make_simple_model()

        # Create a fake kernel class with no mapping
        class FakeInfrastructureKernel:
            pass

        # Apply transform with unmapped kernel
        transform = InsertInfrastructureKernels([FakeInfrastructureKernel])
        model, modified = transform.apply(model)

        # Should not modify graph
        assert not modified

        # Should log warning
        assert "No insertion transform registered" in caplog.text
        assert "FakeInfrastructureKernel" in caplog.text

    def test_multiple_infrastructure_kernels(self):
        """Test with multiple infrastructure kernels (when more are available)."""
        model = make_fanout_model()

        # Currently only DuplicateStreams is registered
        # This test verifies the transform handles multiple kernels in principle
        transform = InsertInfrastructureKernels([DuplicateStreams])
        model, modified = transform.apply(model)

        assert modified
        # Verify DuplicateStreams was inserted
        dup_nodes = [n for n in model.graph.node if n.op_type == "DuplicateStreams"]
        assert len(dup_nodes) >= 1

    def test_idempotent_application(self):
        """Test that applying twice doesn't duplicate infrastructure."""
        model = make_fanout_model()

        # First application
        transform = InsertInfrastructureKernels([DuplicateStreams])
        model, modified1 = transform.apply(model)
        assert modified1

        dup_count_1 = len([n for n in model.graph.node if n.op_type == "DuplicateStreams"])

        # Second application
        model, modified2 = transform.apply(model)

        # Should not modify again (DuplicateStreams already inserted)
        assert not modified2

        dup_count_2 = len([n for n in model.graph.node if n.op_type == "DuplicateStreams"])

        # Count should not increase
        assert dup_count_1 == dup_count_2

    def test_transform_registration(self):
        """Test the transform registration mechanism."""
        from brainsmith.primitives.transforms.insert_duplicate_streams import InsertDuplicateStreams

        # Verify DuplicateStreams is registered
        assert "DuplicateStreams" in INFRASTRUCTURE_TRANSFORM_MAP
        assert INFRASTRUCTURE_TRANSFORM_MAP["DuplicateStreams"] == InsertDuplicateStreams

    def test_manual_registration(self):
        """Test manually registering a new infrastructure transform."""
        from qonnx.transformation.base import Transformation

        # Create a fake transform
        class FakeInsertTransform(Transformation):
            def apply(self, model):
                return (model, False)

        # Register it
        _register_infrastructure_transform("FakeKernel", FakeInsertTransform)

        # Verify registration
        assert "FakeKernel" in INFRASTRUCTURE_TRANSFORM_MAP
        assert INFRASTRUCTURE_TRANSFORM_MAP["FakeKernel"] == FakeInsertTransform

        # Clean up
        del INFRASTRUCTURE_TRANSFORM_MAP["FakeKernel"]
