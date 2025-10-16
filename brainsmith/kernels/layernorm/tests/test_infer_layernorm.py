#!/usr/bin/env python3
############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################
"""
Tests for InferLayerNorm transformation.

This test suite validates that the InferLayerNorm transform correctly
converts FuncLayerNorm nodes to LayerNorm nodes with proper attributes.

Key validation points:
1. Transformation occurs (FuncLayerNorm → LayerNorm)
2. Node attributes are correct (SIMD, epsilon, datatypes)
3. ifm_dim and NumChannels are NOT present (inferred from kernel_model)
4. Shape and datatype inference works
5. Multiple scenarios (different shapes, datatypes, layouts)
"""

import pytest
import numpy as np
from onnx import helper, TensorProto

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes

from brainsmith.kernels.layernorm.infer_layernorm import InferLayerNorm
from brainsmith.kernels.layernorm.layernorm import LayerNorm


# ============================================================================
# Helper Functions
# ============================================================================

def create_funclayernorm_model(shape, datatype="FLOAT32", epsilon=1e-5, axis=-1):
    """Create a model with a FuncLayerNorm node.

    Args:
        shape: Input tensor shape (e.g., [1, 128, 768])
        datatype: FINN DataType name
        epsilon: LayerNorm epsilon value
        axis: Normalization axis

    Returns:
        ModelWrapper with FuncLayerNorm node
    """
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape)
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, shape)

    node = helper.make_node(
        "FuncLayerNorm",
        ["inp"],
        ["out"],
        axis=axis,
        epsilon=epsilon
    )

    graph = helper.make_graph([node], "test", [inp], [out])
    model = helper.make_model(graph)
    model = ModelWrapper(model)

    # Set datatypes
    dt = DataType[datatype]
    model.set_tensor_datatype("inp", dt)
    model.set_tensor_datatype("out", dt)

    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    return model


# ============================================================================
# Basic Transformation Tests
# ============================================================================

def test_infer_auto_layernorm_basic():
    """Test that InferLayerNorm converts FuncLayerNorm to LayerNorm."""
    print("\n=== Test 1: Basic Transformation ===")

    # Create model with FuncLayerNorm
    shape = [1, 128, 768]
    model = create_funclayernorm_model(shape, datatype="INT8", epsilon=1e-5)

    # Verify FuncLayerNorm exists
    func_nodes = [n for n in model.graph.node if n.op_type == "FuncLayerNorm"]
    assert len(func_nodes) == 1
    print(f"  ✓ Created model with FuncLayerNorm")

    # Apply transform
    model_transformed = model.transform(InferLayerNorm())
    print(f"  ✓ Transform applied")

    # Verify LayerNorm exists
    auto_nodes = [n for n in model_transformed.graph.node if n.op_type == "LayerNorm"]
    assert len(auto_nodes) == 1, f"Expected 1 LayerNorm node, got {len(auto_nodes)}"
    print(f"  ✓ LayerNorm node created")

    # Verify FuncLayerNorm removed
    func_nodes_after = [n for n in model_transformed.graph.node if n.op_type == "FuncLayerNorm"]
    assert len(func_nodes_after) == 0, "FuncLayerNorm should be removed"
    print(f"  ✓ FuncLayerNorm node removed")


def test_auto_layernorm_node_attributes():
    """Test that LayerNorm node has correct attributes."""
    print("\n=== Test 2: Node Attributes ===")

    shape = [1, 128, 768]
    epsilon = 1e-5
    datatype = "INT8"

    # Create and transform model
    model = create_funclayernorm_model(shape, datatype=datatype, epsilon=epsilon)
    model_transformed = model.transform(InferLayerNorm())

    # Get LayerNorm node
    node = [n for n in model_transformed.graph.node if n.op_type == "LayerNorm"][0]

    # Check required attributes
    attr_names = [a.name for a in node.attribute]

    # Should have these attributes
    assert "SIMD" in attr_names, "SIMD attribute missing"
    assert "epsilon" in attr_names, "epsilon attribute missing"
    assert "_input0Datatype" in attr_names, "_input0Datatype attribute missing"
    assert "_output0Datatype" in attr_names, "_output0Datatype attribute missing"
    print(f"  ✓ Required attributes present: SIMD, epsilon, datatypes")

    # Verify attribute values
    simd_val = helper.get_node_attr_value(node, "SIMD")
    epsilon_val = helper.get_node_attr_value(node, "epsilon")
    idt_val = helper.get_node_attr_value(node, "_input0Datatype")
    odt_val = helper.get_node_attr_value(node, "_output0Datatype")

    # Decode bytes to string if necessary
    if isinstance(idt_val, bytes):
        idt_val = idt_val.decode()
    if isinstance(odt_val, bytes):
        odt_val = odt_val.decode()

    assert simd_val == 1, f"SIMD should default to 1, got {simd_val}"
    assert abs(epsilon_val - epsilon) < 1e-10, f"Epsilon should be {epsilon}, got {epsilon_val}"
    assert idt_val == datatype, f"_input0Datatype should be {datatype}, got {idt_val}"
    assert odt_val == datatype, f"_output0Datatype should be {datatype}, got {odt_val}"
    print(f"  ✓ Attribute values correct:")
    print(f"    - SIMD: {helper.get_node_attr_value(node, 'SIMD')}")
    print(f"    - epsilon: {helper.get_node_attr_value(node, 'epsilon')}")
    print(f"    - _input0Datatype: {helper.get_node_attr_value(node, '_input0Datatype')}")
    print(f"    - _output0Datatype: {helper.get_node_attr_value(node, '_output0Datatype')}")


def test_no_redundant_attributes():
    """Test that ifm_dim and NumChannels are NOT present (key difference)."""
    print("\n=== Test 3: No Redundant Attributes ===")

    shape = [1, 128, 768]
    model = create_funclayernorm_model(shape)
    model_transformed = model.transform(InferLayerNorm())

    # Get LayerNorm node
    node = [n for n in model_transformed.graph.node if n.op_type == "LayerNorm"][0]
    attr_names = [a.name for a in node.attribute]

    # Should NOT have these attributes (inferred from tensor context)
    assert "ifm_dim" not in attr_names, "ifm_dim should not be present (inferred automatically)"
    assert "NumChannels" not in attr_names, "NumChannels should not be present (inferred automatically)"

    print(f"  ✓ ifm_dim not present (inferred from tensor context)")
    print(f"  ✓ NumChannels not present (inferred from tensor context)")
    print(f"  → Shape information automatically inferred via kernel_model")


# ============================================================================
# Shape and Datatype Tests
# ============================================================================

@pytest.mark.parametrize("shape", [
    [1, 128, 768],    # BERT base
    [1, 256, 1024],   # BERT large
    [2, 64, 512],     # Batch size 2
    [1, 1, 768],      # Single token
])
def test_different_shapes(shape):
    """Test transformation with different input shapes."""
    print(f"\n=== Test 4: Shape {shape} ===")

    model = create_funclayernorm_model(shape)
    model_transformed = model.transform(InferLayerNorm())

    auto_nodes = [n for n in model_transformed.graph.node if n.op_type == "LayerNorm"]
    assert len(auto_nodes) == 1

    print(f"  ✓ Transformation successful for shape {shape}")


@pytest.mark.parametrize("datatype", ["INT8", "INT16", "INT32", "FLOAT32"])
def test_different_datatypes(datatype):
    """Test transformation preserves different datatypes."""
    print(f"\n=== Test 5: Datatype {datatype} ===")

    shape = [1, 128, 768]
    model = create_funclayernorm_model(shape, datatype=datatype)
    model_transformed = model.transform(InferLayerNorm())

    # Get LayerNorm node
    node = [n for n in model_transformed.graph.node if n.op_type == "LayerNorm"][0]

    # Verify datatypes preserved (decode bytes if necessary)
    idt_val = helper.get_node_attr_value(node, "_input0Datatype")
    odt_val = helper.get_node_attr_value(node, "_output0Datatype")
    if isinstance(idt_val, bytes):
        idt_val = idt_val.decode()
    if isinstance(odt_val, bytes):
        odt_val = odt_val.decode()

    assert idt_val == datatype
    assert odt_val == datatype

    print(f"  ✓ Datatype {datatype} preserved correctly")


# ============================================================================
# Integration Tests
# ============================================================================

def test_auto_layernorm_instantiation():
    """Test that created LayerNorm node can be instantiated."""
    print("\n=== Test 6: LayerNorm Instantiation ===")

    shape = [1, 128, 768]
    model = create_funclayernorm_model(shape, datatype="INT8")
    model_transformed = model.transform(InferLayerNorm())

    # Get LayerNorm node
    node = [n for n in model_transformed.graph.node if n.op_type == "LayerNorm"][0]

    # Instantiate LayerNorm operator
    op_inst = LayerNorm(node)
    print(f"  ✓ LayerNorm instantiated successfully")

    # Refresh from model (required for kernel_model)
    op_inst.refresh_df_model(model_transformed)
    print(f"  ✓ Tensor context refreshed")

    # Access kernel_model (triggers validation and shape inference)
    kernel_model = op_inst.kernel_model
    print(f"  ✓ Kernel model built successfully")

    # Verify shape inference works
    normal_input_shape = op_inst.get_normal_input_shape()
    assert normal_input_shape == list(shape)
    print(f"  ✓ Shape inference working: {normal_input_shape}")

    # Verify datatypes work
    input_datatype = op_inst.get_input_datatype()
    assert input_datatype == DataType["INT8"]
    print(f"  ✓ Datatype inference working: {input_datatype}")


def test_auto_layernorm_execution():
    """Test that LayerNorm can execute in Python mode."""
    print("\n=== Test 7: LayerNorm Execution ===")

    shape = [2, 64, 256]
    model = create_funclayernorm_model(shape, datatype="FLOAT32")
    model_transformed = model.transform(InferLayerNorm())

    # Get LayerNorm node
    node = [n for n in model_transformed.graph.node if n.op_type == "LayerNorm"][0]

    # Instantiate and setup
    op_inst = LayerNorm(node)
    op_inst.refresh_df_model(model_transformed)

    # Create input data
    np.random.seed(42)
    input_data = np.random.randn(*shape).astype(np.float32)

    # Execute
    context = {"inp": input_data}
    op_inst.execute_node(context, model_transformed.graph)

    # Verify output exists and has correct shape
    output_data = context["out"]
    assert output_data.shape == tuple(shape)
    print(f"  ✓ Execution successful: output shape {output_data.shape}")

    # Verify normalization (mean ≈ 0, std ≈ 1)
    mean = np.mean(output_data, axis=-1)
    std = np.std(output_data, axis=-1)
    assert np.allclose(mean, 0, atol=1e-5)
    assert np.allclose(std, 1, atol=1e-5)
    print(f"  ✓ Normalization correct: mean≈0, std≈1")


# ============================================================================
# Edge Cases
# ============================================================================

def test_multiple_funclayernorm_nodes():
    """Test transformation with multiple FuncLayerNorm nodes."""
    print("\n=== Test 8: Multiple FuncLayerNorm Nodes ===")

    shape = [1, 128, 768]

    # Create model with multiple FuncLayerNorm nodes
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape)
    mid = helper.make_tensor_value_info("mid", TensorProto.FLOAT, shape)
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, shape)

    node1 = helper.make_node("FuncLayerNorm", ["inp"], ["mid"], axis=-1, epsilon=1e-5)
    node2 = helper.make_node("FuncLayerNorm", ["mid"], ["out"], axis=-1, epsilon=1e-5)

    graph = helper.make_graph([node1, node2], "test", [inp], [out])
    model = helper.make_model(graph)
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", DataType["INT8"])
    model.set_tensor_datatype("mid", DataType["INT8"])
    model.set_tensor_datatype("out", DataType["INT8"])

    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    # Apply transform
    model_transformed = model.transform(InferLayerNorm())

    auto_nodes = [n for n in model_transformed.graph.node if n.op_type == "LayerNorm"]
    assert len(auto_nodes) == 2, f"Expected 2 LayerNorm nodes, got {len(auto_nodes)}"

    func_nodes = [n for n in model_transformed.graph.node if n.op_type == "FuncLayerNorm"]
    assert len(func_nodes) == 0, "All FuncLayerNorm nodes should be removed"

    print(f"  ✓ Transformed 2 FuncLayerNorm nodes to 2 LayerNorm nodes")


def test_non_channel_axis_ignored():
    """Test that normalization on non-channel axis is ignored."""
    print("\n=== Test 9: Non-Channel Axis Ignored ===")

    shape = [1, 128, 768]

    # Create model with FuncLayerNorm on axis 1 (not last dimension)
    model = create_funclayernorm_model(shape, axis=1)

    # Apply transform
    model_transformed = model.transform(InferLayerNorm())

    # Should NOT transform (normalization not on last axis)
    func_nodes = [n for n in model_transformed.graph.node if n.op_type == "FuncLayerNorm"]
    assert len(func_nodes) == 1, "FuncLayerNorm should remain (not channel axis)"

    auto_nodes = [n for n in model_transformed.graph.node if n.op_type == "LayerNorm"]
    assert len(auto_nodes) == 0, "No LayerNorm should be created"

    print(f"  ✓ FuncLayerNorm on non-channel axis correctly ignored")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run all InferLayerNorm tests."""
    print("=" * 60)
    print("InferLayerNorm Transformation Tests")
    print("=" * 60)

    try:
        # Basic transformation tests
        test_infer_auto_layernorm_basic()
        test_auto_layernorm_node_attributes()
        test_no_redundant_attributes()

        # Shape and datatype tests
        for shape in [[1, 128, 768], [1, 256, 1024], [2, 64, 512]]:
            test_different_shapes(shape)

        for datatype in ["INT8", "INT16", "FLOAT32"]:
            test_different_datatypes(datatype)

        # Integration tests
        test_auto_layernorm_instantiation()
        test_auto_layernorm_execution()

        # Edge cases
        test_multiple_funclayernorm_nodes()
        test_non_channel_axis_ignored()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print("\nInferLayerNorm transform validated:")
        print("  ✓ Correctly transforms FuncLayerNorm to LayerNorm")
        print("  ✓ Sets proper attributes (SIMD, epsilon, datatypes)")
        print("  ✓ Does NOT set ifm_dim or NumChannels (inferred)")
        print("  ✓ Works with multiple shapes and datatypes")
        print("  ✓ LayerNorm instances work correctly")
        print("  ✓ Handles edge cases properly")

        return 0

    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ TEST FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
