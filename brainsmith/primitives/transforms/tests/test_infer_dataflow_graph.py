############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
############################################################################

"""Tests for InferDataflowGraph meta-transform."""

import pytest
import numpy as np
from onnx import helper, TensorProto

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType

from brainsmith.primitives.transforms.infer_dataflow_graph import InferDataflowGraph
from brainsmith.primitives.transforms.insert_duplicate_streams import InsertDuplicateStreams
from brainsmith.primitives.transforms.infer_kernel_list import InferKernelList


def make_fanout_add_model():
    """Create model with both fanout AND ONNX Add nodes.

    This model tests both topology analysis (fanout) and pattern matching
    (Add → AddStreams conversion).

    Structure:
        inp → Conv → tensor_x (fanout=2) → Add_0 → outp0
                                         ↘ Add_1 → outp1

    Returns:
        ModelWrapper with tensor fanout and Add operations
    """
    # Inputs and outputs
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 64])
    outp0 = helper.make_tensor_value_info("outp0", TensorProto.FLOAT, [1, 64])
    outp1 = helper.make_tensor_value_info("outp1", TensorProto.FLOAT, [1, 64])

    # Intermediate tensors
    tensor_x = helper.make_tensor_value_info("tensor_x", TensorProto.FLOAT, [1, 64])
    bias0 = helper.make_tensor_value_info("bias0", TensorProto.FLOAT, [1, 64])
    bias1 = helper.make_tensor_value_info("bias1", TensorProto.FLOAT, [1, 64])

    # Create nodes
    conv = helper.make_node("Conv", ["inp", "weight"], ["tensor_x"], name="Conv_0")
    add0 = helper.make_node("Add", ["tensor_x", "bias0"], ["outp0"], name="Add_0")
    add1 = helper.make_node("Add", ["tensor_x", "bias1"], ["outp1"], name="Add_1")

    # Create graph
    graph = helper.make_graph(
        [conv, add0, add1],
        "test_fanout_add",
        [inp, bias0, bias1],
        [outp0, outp1],
        value_info=[tensor_x]  # Include intermediate tensor
    )

    model = helper.make_model(graph, producer_name="test")
    model = ModelWrapper(model)

    # Set datatypes (required for AddStreams inference)
    for tensor in ["inp", "tensor_x", "outp0", "outp1", "bias0", "bias1"]:
        model.set_tensor_datatype(tensor, DataType["INT8"])

    # Set shapes explicitly
    for tensor in ["inp", "tensor_x", "outp0", "outp1", "bias0", "bias1"]:
        model.set_tensor_shape(tensor, [1, 64])

    # Set layouts (required for AddStreams inference)
    import qonnx.core.data_layout as DataLayout
    for tensor in ["inp", "tensor_x", "outp0", "outp1", "bias0", "bias1"]:
        model.set_tensor_layout(tensor, DataLayout.NHWC)

    # Initialize weight
    weight_data = np.random.randn(64, 64).astype(np.float32)
    model.set_initializer("weight", weight_data)

    return model


def make_add_only_model():
    """Create model with only Add nodes (no fanout).

    Tests pattern matching without topology transforms.

    Returns:
        ModelWrapper with single Add operation
    """
    in0 = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [1, 64])
    in1 = helper.make_tensor_value_info("in1", TensorProto.FLOAT, [1, 64])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 64])

    add_node = helper.make_node("Add", ["in0", "in1"], ["out"])
    graph = helper.make_graph([add_node], "test_add", [in0, in1], [out])
    model = helper.make_model(graph, producer_name="test")
    model = ModelWrapper(model)

    # Set datatypes and layouts
    for tensor in ["in0", "in1", "out"]:
        model.set_tensor_datatype(tensor, DataType["INT8"])
        model.set_tensor_shape(tensor, [1, 64])
        import qonnx.core.data_layout as DataLayout
        model.set_tensor_layout(tensor, DataLayout.NHWC)

    return model


def make_fanout_only_model():
    """Create model with fanout but no ONNX operations to infer.

    Tests topology transforms without pattern matching.

    Returns:
        ModelWrapper with tensor fanout only
    """
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 64])
    outp0 = helper.make_tensor_value_info("outp0", TensorProto.FLOAT, [1, 64])
    outp1 = helper.make_tensor_value_info("outp1", TensorProto.FLOAT, [1, 64])

    # Intermediate tensor (fanout point)
    tensor_x = helper.make_tensor_value_info("tensor_x", TensorProto.FLOAT, [1, 64])

    # Conv produces tensor_x, two consumers read it
    conv = helper.make_node("Conv", ["inp", "weight"], ["tensor_x"], name="Conv_0")
    relu0 = helper.make_node("Relu", ["tensor_x"], ["outp0"], name="Relu_0")
    relu1 = helper.make_node("Relu", ["tensor_x"], ["outp1"], name="Relu_1")

    graph = helper.make_graph(
        [conv, relu0, relu1],
        "test_fanout",
        [inp],
        [outp0, outp1],
        value_info=[tensor_x]  # Include intermediate tensor
    )

    model = helper.make_model(graph, producer_name="test")
    model = ModelWrapper(model)

    # Set shapes explicitly
    for tensor in ["inp", "tensor_x", "outp0", "outp1"]:
        model.set_tensor_shape(tensor, [1, 64])

    # Initialize weight
    weight_data = np.random.randn(64, 64).astype(np.float32)
    model.set_initializer("weight", weight_data)

    return model


class TestInferDataflowGraph:
    """Test suite for InferDataflowGraph meta-transform."""

    def test_two_phase_workflow(self):
        """Verify topology → pattern execution order."""
        model = make_fanout_add_model()

        # Apply meta-transform
        transform = InferDataflowGraph()
        model, modified = transform.apply(model)

        assert modified, "Should have modified graph"

        # Phase 1: Topology phase should insert DuplicateStreams
        duplicate_nodes = [n for n in model.graph.node if n.op_type == "DuplicateStreams"]
        assert len(duplicate_nodes) == 1, "Should insert one DuplicateStreams node"

        # Phase 2: Pattern phase should convert Add → AddStreams
        addstreams_nodes = [n for n in model.graph.node if n.op_type == "AddStreams"]
        assert len(addstreams_nodes) == 2, "Should convert both Add nodes to AddStreams"

        # Original Add nodes should be gone
        add_nodes = [n for n in model.graph.node if n.op_type == "Add"]
        assert len(add_nodes) == 0, "Original Add nodes should be removed"

    def test_skip_topology(self):
        """Test skip_topology parameter."""
        model = make_fanout_add_model()

        # Apply with skip_topology=True
        transform = InferDataflowGraph(skip_topology=True)
        model, modified = transform.apply(model)

        # Should still convert Add → AddStreams (pattern matching)
        addstreams_nodes = [n for n in model.graph.node if n.op_type == "AddStreams"]
        assert len(addstreams_nodes) == 2, "Should still convert Add nodes"

        # Should NOT insert DuplicateStreams (topology skipped)
        duplicate_nodes = [n for n in model.graph.node if n.op_type == "DuplicateStreams"]
        assert len(duplicate_nodes) == 0, "Should not insert DuplicateStreams when skipped"

    def test_skip_pattern_matching(self):
        """Test skip_pattern_matching parameter."""
        model = make_fanout_add_model()

        # Apply with skip_pattern_matching=True
        transform = InferDataflowGraph(skip_pattern_matching=True)
        model, modified = transform.apply(model)

        # Should insert DuplicateStreams (topology)
        duplicate_nodes = [n for n in model.graph.node if n.op_type == "DuplicateStreams"]
        assert len(duplicate_nodes) == 1, "Should insert DuplicateStreams"

        # Should NOT convert Add → AddStreams (pattern matching skipped)
        add_nodes = [n for n in model.graph.node if n.op_type == "Add"]
        assert len(add_nodes) == 2, "Add nodes should remain unconverted"

        addstreams_nodes = [n for n in model.graph.node if n.op_type == "AddStreams"]
        assert len(addstreams_nodes) == 0, "Should not convert Add when pattern matching skipped"

    def test_both_phases_skipped(self):
        """Test that skipping both phases does nothing."""
        model = make_fanout_add_model()
        original_node_count = len(model.graph.node)

        # Apply with both phases skipped
        transform = InferDataflowGraph(skip_topology=True, skip_pattern_matching=True)
        model, modified = transform.apply(model)

        # Should not modify graph
        assert not modified, "Should not modify graph when both phases skipped"
        assert len(model.graph.node) == original_node_count, "Node count should be unchanged"

    def test_kernel_classes_passthrough(self):
        """Test that kernel_classes parameter is passed to InferKernelList."""
        from brainsmith.kernels.addstreams import AddStreams

        model = make_fanout_add_model()

        # Apply with specific kernel class
        transform = InferDataflowGraph(kernel_classes=[AddStreams])
        model, modified = transform.apply(model)

        assert modified, "Should have modified graph"

        # Should still work with explicit kernel list
        addstreams_nodes = [n for n in model.graph.node if n.op_type == "AddStreams"]
        assert len(addstreams_nodes) == 2, "Should convert Add nodes with explicit kernel list"

    def test_no_fanout_no_topology_changes(self):
        """Test model without fanout doesn't trigger topology transforms."""
        model = make_add_only_model()

        # Apply full transform
        transform = InferDataflowGraph()
        model, modified = transform.apply(model)

        # Should convert Add → AddStreams (pattern matching)
        addstreams_nodes = [n for n in model.graph.node if n.op_type == "AddStreams"]
        assert len(addstreams_nodes) == 1, "Should convert Add node"

        # Should NOT insert DuplicateStreams (no fanout)
        duplicate_nodes = [n for n in model.graph.node if n.op_type == "DuplicateStreams"]
        assert len(duplicate_nodes) == 0, "Should not insert DuplicateStreams without fanout"

    def test_no_onnx_ops_no_pattern_changes(self):
        """Test model without inferrable ops doesn't trigger pattern matching."""
        model = make_fanout_only_model()

        # Apply full transform
        transform = InferDataflowGraph()
        model, modified = transform.apply(model)

        # Should insert DuplicateStreams (fanout)
        duplicate_nodes = [n for n in model.graph.node if n.op_type == "DuplicateStreams"]
        assert len(duplicate_nodes) == 1, "Should insert DuplicateStreams for fanout"

        # Should NOT convert any nodes (no ONNX ops to infer)
        # Relu nodes should remain as-is
        relu_nodes = [n for n in model.graph.node if n.op_type == "Relu"]
        assert len(relu_nodes) == 2, "Relu nodes should remain unchanged"

    def test_idempotent_application(self):
        """Test that applying transform multiple times is safe."""
        model = make_fanout_add_model()

        # First application
        transform1 = InferDataflowGraph()
        model, modified1 = transform1.apply(model)
        assert modified1, "First application should modify graph"

        # Count nodes after first application
        node_types_1 = [n.op_type for n in model.graph.node]
        node_count_1 = len(model.graph.node)

        # Second application
        transform2 = InferDataflowGraph()
        model, modified2 = transform2.apply(model)

        # Should not modify graph again (already transformed)
        assert not modified2, "Second application should not modify graph"

        # Node structure should be identical
        node_types_2 = [n.op_type for n in model.graph.node]
        node_count_2 = len(model.graph.node)

        assert node_count_1 == node_count_2, "Node count should be unchanged"
        assert node_types_1 == node_types_2, "Node types should be unchanged"

    def test_graph_modified_flag_aggregation(self):
        """Test that graph_modified flag is correctly aggregated from both phases."""
        # Case 1: Only topology modifies
        model1 = make_fanout_only_model()
        transform1 = InferDataflowGraph()
        _, modified1 = transform1.apply(model1)
        assert modified1, "Should report modified when topology phase changes graph"

        # Case 2: Only pattern matching modifies
        model2 = make_add_only_model()
        transform2 = InferDataflowGraph()
        _, modified2 = transform2.apply(model2)
        assert modified2, "Should report modified when pattern phase changes graph"

        # Case 3: Both phases modify
        model3 = make_fanout_add_model()
        transform3 = InferDataflowGraph()
        _, modified3 = transform3.apply(model3)
        assert modified3, "Should report modified when both phases change graph"

        # Case 4: Neither phase modifies
        model4 = make_fanout_add_model()
        # First transform it
        model4 = model4.transform(InferDataflowGraph())
        # Second transform should do nothing
        transform4 = InferDataflowGraph()
        _, modified4 = transform4.apply(model4)
        assert not modified4, "Should report not modified when already transformed"


class TestInferDataflowGraphIntegration:
    """Integration tests for InferDataflowGraph."""

    def test_equivalent_to_manual_orchestration(self):
        """Verify InferDataflowGraph produces same result as manual transform chain."""
        model1 = make_fanout_add_model()
        model2 = make_fanout_add_model()

        # Method 1: Use InferDataflowGraph
        transform_meta = InferDataflowGraph()
        model1, modified1 = transform_meta.apply(model1)

        # Method 2: Manual orchestration
        model2, modified2a = InsertDuplicateStreams().apply(model2)
        model2, modified2b = InferKernelList().apply(model2)
        modified2 = modified2a or modified2b

        # Both should modify graph
        assert modified1 == modified2, "Both methods should report same modification status"

        # Both should have same number of nodes
        assert len(model1.graph.node) == len(model2.graph.node), "Node count should match"

        # Both should have same node types (order may differ)
        types1 = sorted([n.op_type for n in model1.graph.node])
        types2 = sorted([n.op_type for n in model2.graph.node])
        assert types1 == types2, "Node types should match between methods"

        # Check specific transformations
        duplicate_count1 = len([n for n in model1.graph.node if n.op_type == "DuplicateStreams"])
        duplicate_count2 = len([n for n in model2.graph.node if n.op_type == "DuplicateStreams"])
        assert duplicate_count1 == duplicate_count2 == 1, "Both should have 1 DuplicateStreams node"

        addstreams_count1 = len([n for n in model1.graph.node if n.op_type == "AddStreams"])
        addstreams_count2 = len([n for n in model2.graph.node if n.op_type == "AddStreams"])
        assert addstreams_count1 == addstreams_count2 == 2, "Both should have 2 AddStreams nodes"

    def test_complex_model_integration(self):
        """Test with more complex model structure."""
        # Create model with multiple fanouts and multiple Add nodes
        inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 64])
        outp0 = helper.make_tensor_value_info("outp0", TensorProto.FLOAT, [1, 64])
        outp1 = helper.make_tensor_value_info("outp1", TensorProto.FLOAT, [1, 64])
        outp2 = helper.make_tensor_value_info("outp2", TensorProto.FLOAT, [1, 64])

        # Intermediate tensors
        x1 = helper.make_tensor_value_info("x1", TensorProto.FLOAT, [1, 64])
        x2 = helper.make_tensor_value_info("x2", TensorProto.FLOAT, [1, 64])
        bias = helper.make_tensor_value_info("bias", TensorProto.FLOAT, [1, 64])

        # Create complex structure:
        # inp → Conv → x1 (fanout=2) → Add_0 → x2 (fanout=2) → Relu → outp0
        #                            ↘ Relu → outp1            ↘ Sigmoid → outp2
        conv = helper.make_node("Conv", ["inp", "weight"], ["x1"], name="Conv_0")
        add = helper.make_node("Add", ["x1", "bias"], ["x2"], name="Add_0")
        relu0 = helper.make_node("Relu", ["x1"], ["outp1"], name="Relu_0")
        relu1 = helper.make_node("Relu", ["x2"], ["outp0"], name="Relu_1")
        sigmoid = helper.make_node("Sigmoid", ["x2"], ["outp2"], name="Sigmoid_0")

        graph = helper.make_graph(
            [conv, add, relu0, relu1, sigmoid],
            "test_complex",
            [inp, bias],
            [outp0, outp1, outp2],
            value_info=[x1, x2]  # Include intermediate tensors
        )

        model = helper.make_model(graph, producer_name="test")
        model = ModelWrapper(model)

        # Set datatypes and layouts
        for tensor in ["inp", "x1", "x2", "outp0", "outp1", "outp2", "bias"]:
            model.set_tensor_datatype(tensor, DataType["INT8"])
            model.set_tensor_shape(tensor, [1, 64])
            import qonnx.core.data_layout as DataLayout
            model.set_tensor_layout(tensor, DataLayout.NHWC)

        # Initialize weight
        weight_data = np.random.randn(64, 64).astype(np.float32)
        model.set_initializer("weight", weight_data)

        # Apply transform
        transform = InferDataflowGraph()
        model, modified = transform.apply(model)

        assert modified, "Should modify graph"

        # Should insert 2 DuplicateStreams (x1 and x2 both have fanout=2)
        duplicate_nodes = [n for n in model.graph.node if n.op_type == "DuplicateStreams"]
        assert len(duplicate_nodes) == 2, "Should insert 2 DuplicateStreams nodes"

        # Should convert Add → AddStreams
        addstreams_nodes = [n for n in model.graph.node if n.op_type == "AddStreams"]
        assert len(addstreams_nodes) == 1, "Should convert Add node to AddStreams"

        # Original Add should be gone
        add_nodes = [n for n in model.graph.node if n.op_type == "Add"]
        assert len(add_nodes) == 0, "Original Add node should be removed"
