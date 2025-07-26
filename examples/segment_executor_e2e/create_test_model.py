#!/usr/bin/env python3
"""Create a simple ONNX model for testing."""

import onnx
from onnx import helper, TensorProto
import numpy as np

# Create a simple model: input -> MatMul -> output
input_tensor = helper.make_tensor_value_info(
    "input", TensorProto.FLOAT, [1, 10]
)
output_tensor = helper.make_tensor_value_info(
    "output", TensorProto.FLOAT, [1, 5]
)

# Create weight tensor
weight_initializer = helper.make_tensor(
    "weight",
    TensorProto.FLOAT,
    [10, 5],
    np.random.randn(10, 5).astype(np.float32).flatten()
)

# Create MatMul node
matmul_node = helper.make_node(
    "MatMul",
    inputs=["input", "weight"],
    outputs=["output"],
    name="simple_matmul"
)

# Create graph
graph = helper.make_graph(
    [matmul_node],
    "simple_model",
    [input_tensor],
    [output_tensor],
    [weight_initializer]
)

# Create model
model = helper.make_model(graph)
onnx.save(model, "test_model.onnx")
print("Created test_model.onnx")