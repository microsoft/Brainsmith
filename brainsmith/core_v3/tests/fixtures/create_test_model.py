"""
Create a simple ONNX model for testing.
"""

import onnx
from onnx import helper, TensorProto
import numpy as np


def create_simple_model():
    """Create a simple ONNX model with MatMul and Softmax operations."""
    
    # Input tensors
    input_tensor = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [1, 768]
    )
    
    # Weight tensor for MatMul
    weight_tensor = helper.make_tensor_value_info(
        'weight', TensorProto.FLOAT, [768, 256]
    )
    
    # Output tensor
    output_tensor = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, [1, 256]
    )
    
    # Create MatMul node
    matmul_node = helper.make_node(
        'MatMul',
        inputs=['input', 'weight'],
        outputs=['matmul_output'],
        name='matmul1'
    )
    
    # Create Softmax node
    softmax_node = helper.make_node(
        'Softmax',
        inputs=['matmul_output'],
        outputs=['output'],
        axis=-1,
        name='softmax1'
    )
    
    # Create the graph
    graph = helper.make_graph(
        [matmul_node, softmax_node],
        'simple_model',
        [input_tensor, weight_tensor],
        [output_tensor],
        []
    )
    
    # Create the model
    model = helper.make_model(
        graph,
        producer_name='brainsmith_test',
        opset_imports=[helper.make_opsetid("", 13)]
    )
    
    # Add metadata
    model.metadata_props.append(
        onnx.StringStringEntryProto(key='test_property', value='test_value')
    )
    
    return model


if __name__ == "__main__":
    # Create and save the model
    model = create_simple_model()
    onnx.save(model, "simple_model.onnx")
    print("Created simple_model.onnx")