############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
############################################################################

"""Tests for InsertDuplicateStreams transformation."""

import pytest
import numpy as np
from onnx import helper, TensorProto

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType

from brainsmith.transforms.insert_duplicate_streams import InsertDuplicateStreams


def make_fanout_model(fanout=2):
    """Create a simple ONNX model with tensor fanout.

    Args:
        fanout: Number of consumers for the tensor (default 2)

    Returns:
        ModelWrapper with a tensor that feeds multiple consumers
    """
    # Create input/output tensors
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 64, 64, 128])

    # Create outputs (one per fanout)
    outputs = []
    for i in range(fanout):
        outp = helper.make_tensor_value_info(f"outp{i}", TensorProto.FLOAT, [1, 64, 64, 128])
        outputs.append(outp)

    # Create intermediate tensor (fanout point)
    tensor_x = helper.make_tensor_value_info("tensor_x", TensorProto.FLOAT, [1, 64, 64, 128])

    # Create producer node
    conv = helper.make_node("Conv", ["inp", "weight"], ["tensor_x"], name="Conv_0")

    # Create consumer nodes (all consume tensor_x)
    consumer_nodes = []
    for i in range(fanout):
        if i % 2 == 0:
            node = helper.make_node("Add", ["tensor_x", f"bias{i}"], [f"outp{i}"], name=f"Add_{i}")
        else:
            node = helper.make_node("Mul", ["tensor_x", f"scale{i}"], [f"outp{i}"], name=f"Mul_{i}")
        consumer_nodes.append(node)

    # Create graph
    graph = helper.make_graph(
        [conv] + consumer_nodes,
        "test_fanout",
        [inp],
        outputs,
        value_info=[tensor_x]  # Add intermediate tensor to value_info
    )

    # Create model
    model = helper.make_model(graph, producer_name="test")
    model = ModelWrapper(model)

    # Set datatypes
    model.set_tensor_datatype("inp", DataType["INT8"])
    model.set_tensor_datatype("tensor_x", DataType["INT8"])
    for i in range(fanout):
        model.set_tensor_datatype(f"outp{i}", DataType["INT8"])

    # Set shapes explicitly
    model.set_tensor_shape("inp", [1, 64, 64, 128])
    model.set_tensor_shape("tensor_x", [1, 64, 64, 128])
    for i in range(fanout):
        model.set_tensor_shape(f"outp{i}", [1, 64, 64, 128])

    return model


def make_linear_model():
    """Create a linear ONNX model without fanout.

    Returns:
        ModelWrapper with linear graph (no multi-consumer tensors)
    """
    # Create input/output tensors
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 64, 64, 128])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 64, 64, 128])

    # Create linear chain: Conv → Add → Mul
    conv = helper.make_node("Conv", ["inp", "weight"], ["t1"], name="Conv_0")
    add = helper.make_node("Add", ["t1", "bias"], ["t2"], name="Add_0")
    mul = helper.make_node("Mul", ["t2", "scale"], ["outp"], name="Mul_0")

    # Create graph
    graph = helper.make_graph(
        [conv, add, mul],
        "test_linear",
        [inp],
        [outp]
    )

    # Create model
    model = helper.make_model(graph, producer_name="test")
    model = ModelWrapper(model)

    # Set datatypes
    model.set_tensor_datatype("inp", DataType["INT8"])
    model.set_tensor_datatype("outp", DataType["INT8"])

    return model


class TestInsertDuplicateStreams:
    """Test suite for InsertDuplicateStreams transformation."""

    def test_fanout_2_detection(self):
        """Test detection and insertion for fanout=2."""
        model = make_fanout_model(fanout=2)

        # Apply transform
        model_new, modified = InsertDuplicateStreams().apply(model)

        assert modified, "Should have inserted DuplicateStreams for fanout=2"

        # Check that DuplicateStreams was inserted
        dup_nodes = [n for n in model_new.graph.node if n.op_type == "DuplicateStreams"]
        assert len(dup_nodes) == 1, "Should have exactly 1 DuplicateStreams node"

        # Check node has correct I/O count
        dup_node = dup_nodes[0]
        assert len(dup_node.input) == 1, "DuplicateStreams should have 1 input"
        assert len(dup_node.output) == 2, "DuplicateStreams should have 2 outputs for fanout=2"

        # Check that original tensor is input to DuplicateStreams
        assert dup_node.input[0] == "tensor_x", "Should consume original tensor"

        # Check that consumers were rewired to use clones
        add_node = [n for n in model_new.graph.node if n.name == "Add_0"][0]
        mul_node = [n for n in model_new.graph.node if n.name == "Mul_1"][0]

        assert add_node.input[0] != "tensor_x", "Add should use clone, not original"
        assert mul_node.input[0] != "tensor_x", "Mul should use clone, not original"
        assert add_node.input[0] in dup_node.output, "Add should use DuplicateStreams output"
        assert mul_node.input[0] in dup_node.output, "Mul should use DuplicateStreams output"

    def test_fanout_3_variable_outputs(self):
        """Test fanout=3 creates node with 3 outputs."""
        model = make_fanout_model(fanout=3)

        # Apply transform
        model_new, modified = InsertDuplicateStreams().apply(model)

        assert modified, "Should have inserted DuplicateStreams for fanout=3"

        # Check DuplicateStreams has 3 outputs
        dup_nodes = [n for n in model_new.graph.node if n.op_type == "DuplicateStreams"]
        assert len(dup_nodes) == 1, "Should have exactly 1 DuplicateStreams node"
        assert len(dup_nodes[0].output) == 3, "Should have 3 outputs for fanout=3"

    def test_fanout_4_variable_outputs(self):
        """Test fanout=4 creates node with 4 outputs."""
        model = make_fanout_model(fanout=4)

        # Apply transform
        model_new, modified = InsertDuplicateStreams().apply(model)

        # Check DuplicateStreams has 4 outputs
        dup_nodes = [n for n in model_new.graph.node if n.op_type == "DuplicateStreams"]
        assert len(dup_nodes[0].output) == 4, "Should have 4 outputs for fanout=4"

    def test_no_fanout_no_insertion(self):
        """Test that linear graphs without fanout are unchanged."""
        model = make_linear_model()

        # Apply transform
        model_new, modified = InsertDuplicateStreams().apply(model)

        assert not modified, "Should not have modified linear graph"

        # Check no DuplicateStreams was inserted
        dup_nodes = [n for n in model_new.graph.node if n.op_type == "DuplicateStreams"]
        assert len(dup_nodes) == 0, "Should have 0 DuplicateStreams nodes in linear graph"

    def test_datatype_preservation(self):
        """Test that datatypes are preserved through duplication."""
        model = make_fanout_model(fanout=2)

        # Set specific datatype
        model.set_tensor_datatype("tensor_x", DataType["INT4"])

        # Apply transform
        model_new, modified = InsertDuplicateStreams().apply(model)

        # Check clone tensors have same datatype
        dup_node = [n for n in model_new.graph.node if n.op_type == "DuplicateStreams"][0]

        for clone_name in dup_node.output:
            clone_dt = model_new.get_tensor_datatype(clone_name)
            assert clone_dt == DataType["INT4"], f"Clone {clone_name} should have INT4 datatype"

    def test_shape_preservation(self):
        """Test that shapes are preserved through duplication."""
        model = make_fanout_model(fanout=2)

        original_shape = model.get_tensor_shape("tensor_x")

        # Apply transform
        model_new, modified = InsertDuplicateStreams().apply(model)

        # Check clone tensors have same shape
        dup_node = [n for n in model_new.graph.node if n.op_type == "DuplicateStreams"][0]

        for clone_name in dup_node.output:
            clone_shape = model_new.get_tensor_shape(clone_name)
            assert clone_shape == original_shape, f"Clone {clone_name} should have same shape"

    def test_node_domain(self):
        """Test that DuplicateStreams node has correct domain."""
        model = make_fanout_model(fanout=2)

        # Apply transform
        model_new, modified = InsertDuplicateStreams().apply(model)

        # Check domain
        dup_node = [n for n in model_new.graph.node if n.op_type == "DuplicateStreams"][0]
        assert dup_node.domain == "brainsmith.kernels", "Should have brainsmith.kernels domain"

    def test_multiple_fanout_tensors(self):
        """Test graph with multiple tensors having fanout."""
        # Create graph with 2 fanout tensors
        inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 64, 64, 128])
        out1 = helper.make_tensor_value_info("out1", TensorProto.FLOAT, [1, 64, 64, 128])
        out2 = helper.make_tensor_value_info("out2", TensorProto.FLOAT, [1, 64, 64, 128])
        out3 = helper.make_tensor_value_info("out3", TensorProto.FLOAT, [1, 64, 64, 128])
        out4 = helper.make_tensor_value_info("out4", TensorProto.FLOAT, [1, 64, 64, 128])

        # Intermediate tensors
        t1 = helper.make_tensor_value_info("t1", TensorProto.FLOAT, [1, 64, 64, 128])
        t2 = helper.make_tensor_value_info("t2", TensorProto.FLOAT, [1, 64, 64, 128])

        # First fanout: conv → [add1, mul1]
        conv = helper.make_node("Conv", ["inp", "w"], ["t1"], name="Conv_0")
        add1 = helper.make_node("Add", ["t1", "b1"], ["out1"], name="Add_1")
        mul1 = helper.make_node("Mul", ["t1", "s1"], ["t2"], name="Mul_1")

        # Second fanout: mul1 → [add2, mul2]
        add2 = helper.make_node("Add", ["t2", "b2"], ["out3"], name="Add_2")
        mul2 = helper.make_node("Mul", ["t2", "s2"], ["out4"], name="Mul_2")

        graph = helper.make_graph(
            [conv, add1, mul1, add2, mul2],
            "test_multi_fanout",
            [inp],
            [out1, out3, out4],
            value_info=[t1, t2]  # Add intermediates to value_info
        )

        model = helper.make_model(graph, producer_name="test")
        model = ModelWrapper(model)

        # Set shapes
        model.set_tensor_shape("inp", [1, 64, 64, 128])
        model.set_tensor_shape("t1", [1, 64, 64, 128])
        model.set_tensor_shape("t2", [1, 64, 64, 128])
        model.set_tensor_shape("out1", [1, 64, 64, 128])
        model.set_tensor_shape("out3", [1, 64, 64, 128])
        model.set_tensor_shape("out4", [1, 64, 64, 128])

        # Set datatypes
        model.set_tensor_datatype("inp", DataType["INT8"])
        model.set_tensor_datatype("t1", DataType["INT8"])
        model.set_tensor_datatype("t2", DataType["INT8"])

        # Apply transform
        model_new, modified = InsertDuplicateStreams().apply(model)

        # Should insert 2 DuplicateStreams nodes
        dup_nodes = [n for n in model_new.graph.node if n.op_type == "DuplicateStreams"]
        assert len(dup_nodes) == 2, "Should have 2 DuplicateStreams nodes"


class TestInsertDuplicateStreamsIntegration:
    """Test suite for InsertDuplicateStreams integration with other transforms."""

    def test_insertion_order_preserves_topology(self):
        """Test that DuplicateStreams is inserted in correct topological order."""
        model = make_fanout_model(fanout=2)

        # Apply transform
        model_new, modified = InsertDuplicateStreams().apply(model)

        # Get node indices
        node_names = [n.name for n in model_new.graph.node]
        conv_idx = node_names.index("Conv_0")
        dup_idx = [i for i, n in enumerate(model_new.graph.node) if n.op_type == "DuplicateStreams"][0]
        add_idx = node_names.index("Add_0")
        mul_idx = node_names.index("Mul_1")

        # DuplicateStreams should be after producer
        assert dup_idx > conv_idx, "DuplicateStreams should be after producer"

        # DuplicateStreams should be before consumers
        assert dup_idx < add_idx, "DuplicateStreams should be before consumers"
        assert dup_idx < mul_idx, "DuplicateStreams should be before consumers"

    def test_repeated_application_is_idempotent(self):
        """Test that applying transform multiple times doesn't insert duplicates."""
        model = make_fanout_model(fanout=2)

        # Apply transform twice
        model_new, modified1 = InsertDuplicateStreams().apply(model)
        model_final, modified2 = InsertDuplicateStreams().apply(model_new)

        # Second application should not modify (already has DuplicateStreams)
        assert modified1, "First application should modify"
        assert not modified2, "Second application should not modify"

        # Should still have exactly 1 DuplicateStreams node
        dup_nodes = [n for n in model_final.graph.node if n.op_type == "DuplicateStreams"]
        assert len(dup_nodes) == 1, "Should still have exactly 1 DuplicateStreams node"

    def test_with_existing_duplicate_streams(self):
        """Test behavior when model already has DuplicateStreams nodes."""
        # Create model with manual DuplicateStreams
        inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 64, 64, 128])
        out1 = helper.make_tensor_value_info("out1", TensorProto.FLOAT, [1, 64, 64, 128])
        out2 = helper.make_tensor_value_info("out2", TensorProto.FLOAT, [1, 64, 64, 128])

        conv = helper.make_node("Conv", ["inp", "w"], ["t1"], name="Conv_0")
        dup = helper.make_node(
            "DuplicateStreams",
            inputs=["t1"],
            outputs=["t1_clone0", "t1_clone1"],
            domain="brainsmith.kernels",
            name="DuplicateStreams_0"
        )
        add = helper.make_node("Add", ["t1_clone0", "b"], ["out1"], name="Add_0")
        mul = helper.make_node("Mul", ["t1_clone1", "s"], ["out2"], name="Mul_0")

        graph = helper.make_graph(
            [conv, dup, add, mul],
            "test_existing_dup",
            [inp],
            [out1, out2]
        )

        model = helper.make_model(graph, producer_name="test")
        model = ModelWrapper(model)

        # Apply transform
        model_new, modified = InsertDuplicateStreams().apply(model)

        # Should not modify (no tensor fanout after DuplicateStreams)
        assert not modified, "Should not modify graph with existing DuplicateStreams"

        # Should still have exactly 1 DuplicateStreams node
        dup_nodes = [n for n in model_new.graph.node if n.op_type == "DuplicateStreams"]
        assert len(dup_nodes) == 1, "Should have exactly 1 DuplicateStreams node"


class TestInsertDuplicateStreamsEdgeCases:
    """Test suite for edge cases and robustness."""

    def test_fanout_1_no_insertion(self):
        """Test that fanout=1 (single consumer) does not insert DuplicateStreams."""
        # Create graph where each tensor has exactly 1 consumer
        inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 64, 64, 128])
        outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 64, 64, 128])

        conv = helper.make_node("Conv", ["inp", "w"], ["t1"], name="Conv_0")
        add = helper.make_node("Add", ["t1", "b"], ["outp"], name="Add_0")

        graph = helper.make_graph([conv, add], "test_fanout_1", [inp], [outp])
        model = helper.make_model(graph, producer_name="test")
        model = ModelWrapper(model)

        # Apply transform
        model_new, modified = InsertDuplicateStreams().apply(model)

        # Should not insert for fanout=1
        assert not modified, "Should not insert DuplicateStreams for fanout=1"
        dup_nodes = [n for n in model_new.graph.node if n.op_type == "DuplicateStreams"]
        assert len(dup_nodes) == 0, "Should have 0 DuplicateStreams nodes"

    def test_empty_graph(self):
        """Test transform on empty graph."""
        inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 64, 64, 128])
        outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 64, 64, 128])

        graph = helper.make_graph([], "test_empty", [inp], [outp])
        model = helper.make_model(graph, producer_name="test")
        model = ModelWrapper(model)

        # Apply transform
        model_new, modified = InsertDuplicateStreams().apply(model)

        # Should not modify empty graph
        assert not modified, "Should not modify empty graph"

    def test_large_fanout_insertion(self):
        """Test insertion for large fanout (10 consumers)."""
        model = make_fanout_model(fanout=10)

        # Apply transform
        model_new, modified = InsertDuplicateStreams().apply(model)

        assert modified, "Should insert DuplicateStreams for fanout=10"

        # Check DuplicateStreams has 10 outputs
        dup_nodes = [n for n in model_new.graph.node if n.op_type == "DuplicateStreams"]
        assert len(dup_nodes) == 1, "Should have exactly 1 DuplicateStreams node"
        assert len(dup_nodes[0].output) == 10, "Should have 10 outputs for fanout=10"

    def test_graph_with_multiple_outputs(self):
        """Test transform handles graphs with multiple outputs gracefully."""
        # Create graph with multiple outputs (no actual fanout in nodes)
        inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 64, 64, 128])
        out1 = helper.make_tensor_value_info("out1", TensorProto.FLOAT, [1, 64, 64, 128])
        out2 = helper.make_tensor_value_info("out2", TensorProto.FLOAT, [1, 64, 64, 128])

        conv1 = helper.make_node("Conv", ["inp", "w1"], ["out1"], name="Conv_0")
        conv2 = helper.make_node("Conv", ["inp", "w2"], ["out2"], name="Conv_1")

        # Both outputs are graph outputs, inp has fanout=2
        graph = helper.make_graph(
            [conv1, conv2],
            "test_multi_output",
            [inp],
            [out1, out2],
        )

        model = helper.make_model(graph, producer_name="test")
        model = ModelWrapper(model)

        # Set shapes and datatypes
        model.set_tensor_shape("inp", [1, 64, 64, 128])
        model.set_tensor_shape("out1", [1, 64, 64, 128])
        model.set_tensor_shape("out2", [1, 64, 64, 128])
        model.set_tensor_datatype("inp", DataType["INT8"])

        # Apply transform
        model_new, modified = InsertDuplicateStreams().apply(model)

        # inp has fanout=2 (consumed by Conv_0 and Conv_1)
        assert modified, "Should insert DuplicateStreams for inp fanout"
        dup_nodes = [n for n in model_new.graph.node if n.op_type == "DuplicateStreams"]
        assert len(dup_nodes) == 1, "Should have exactly 1 DuplicateStreams node"

        # Verify DuplicateStreams inserted for inp
        dup_node = dup_nodes[0]
        assert dup_node.input[0] == "inp", "Should duplicate inp tensor"
        assert len(dup_node.output) == 2, "Should have 2 outputs for fanout=2"
