"""Utilities for creating test ONNX models."""

import onnx
from onnx import helper, TensorProto, ValueInfoProto
import numpy as np
from typing import List, Optional, Dict, Any
import pytest


def create_simple_model(
    input_shape: List[int] = [1, 3, 32, 32],
    output_shape: List[int] = [1, 10],
    op_type: str = "MatMul",
    model_name: str = "simple_test_model"
) -> onnx.ModelProto:
    """Create a simple single-operation ONNX model.
    
    Args:
        input_shape: Shape of input tensor
        output_shape: Shape of output tensor
        op_type: Type of operation to use
        model_name: Name of the model
        
    Returns:
        ONNX model proto
    """
    # Create input/output tensors
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, input_shape
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, output_shape
    )
    
    # Create weight initializer for MatMul
    if op_type == "MatMul":
        weight_shape = [np.prod(input_shape[1:]), np.prod(output_shape[1:])]
        weight_tensor = helper.make_tensor(
            "weight",
            TensorProto.FLOAT,
            weight_shape,
            np.random.randn(*weight_shape).astype(np.float32).flatten()
        )
        
        # Flatten input first
        flatten_node = helper.make_node(
            "Flatten",
            inputs=["input"],
            outputs=["input_flat"],
            axis=1
        )
        
        # MatMul operation
        matmul_node = helper.make_node(
            "MatMul",
            inputs=["input_flat", "weight"],
            outputs=["output"]
        )
        
        nodes = [flatten_node, matmul_node]
        initializers = [weight_tensor]
        
    elif op_type == "Add":
        # Simple Add with constant
        const_tensor = helper.make_tensor(
            "constant",
            TensorProto.FLOAT,
            input_shape,
            np.ones(input_shape).astype(np.float32).flatten()
        )
        
        add_node = helper.make_node(
            "Add",
            inputs=["input", "constant"],
            outputs=["output"]
        )
        
        nodes = [add_node]
        initializers = [const_tensor]
        
    else:
        # Generic single operation
        node = helper.make_node(
            op_type,
            inputs=["input"],
            outputs=["output"]
        )
        nodes = [node]
        initializers = []
    
    # Create graph
    graph_proto = helper.make_graph(
        nodes,
        model_name,
        [input_tensor],
        [output_tensor],
        initializer=initializers
    )
    
    # Create model
    model_proto = helper.make_model(
        graph_proto,
        producer_name="brainsmith_test"
    )
    
    # Set opset version
    model_proto.opset_import[0].version = 11
    
    return model_proto


def _save_model(model: onnx.ModelProto, path: str) -> None:
    """Save ONNX model to file (internal use only).
    
    Args:
        model: ONNX model proto
        path: Path to save the model
    """
    onnx.save(model, path)


@pytest.fixture
def simple_onnx_model(tmp_path):
    """Create a simple ONNX model for testing."""
    model = create_simple_model()
    model_path = tmp_path / "simple_model.onnx"
    _save_model(model, str(model_path))
    
    return model_path