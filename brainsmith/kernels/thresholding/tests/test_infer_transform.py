#!/usr/bin/env python3
"""
Test script for InferThresholdingAxi transform.

Creates a simple model with MultiThreshold node and verifies it gets
converted to ThresholdingAxi AutoHWCustomOp.
"""

import numpy as np
from onnx import helper, TensorProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.util.basic import qonnx_make_model

# Import our new transform
from brainsmith.transforms.core.infer_thresholding_axi import InferThresholdingAxi


def create_test_model():
    """Create a simple test model with MultiThreshold node."""
    
    # Define tensor shapes
    batch_size = 1
    channels = 4
    num_thresholds = 3  # Number of threshold levels
    
    # Create input tensor
    inp = helper.make_tensor_value_info(
        "inp", TensorProto.FLOAT, [batch_size, channels]
    )
    
    # Create threshold tensor 
    thresh = helper.make_tensor_value_info(
        "thresh", TensorProto.FLOAT, [channels, num_thresholds]
    )
    
    # Create output tensor
    outp = helper.make_tensor_value_info(
        "outp", TensorProto.FLOAT, [batch_size, channels]
    )
    
    # Create MultiThreshold node
    mt_node = helper.make_node(
        "MultiThreshold",
        inputs=["inp", "thresh"],
        outputs=["outp"],
        domain="qonnx.custom_op.general",
        out_scale=1.0,
        out_bias=0,
        name="MultiThreshold_0"
    )
    
    # Create graph
    graph = helper.make_graph(
        nodes=[mt_node],
        name="test_graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[thresh]
    )
    
    # Create model
    model = qonnx_make_model(graph)
    model = ModelWrapper(model)
    
    # Set datatypes
    model.set_tensor_datatype("inp", DataType["INT8"])
    model.set_tensor_datatype("thresh", DataType["INT8"])
    model.set_tensor_datatype("outp", DataType["UINT8"])
    
    # Add threshold initializer
    thresh_vals = np.array([
        [-10, 0, 10],
        [-10, 0, 10],
        [-10, 0, 10],
        [-10, 0, 10]
    ], dtype=np.float32)
    model.set_initializer("thresh", thresh_vals)
    
    return model


def test_transform():
    """Test the InferThresholdingAxi transform."""
    
    print("Creating test model...")
    model = create_test_model()
    
    # Print original graph
    print("\nOriginal model:")
    for node in model.graph.node:
        print(f"  Node: {node.op_type} ({node.name})")
        print(f"    Domain: {node.domain}")
        print(f"    Inputs: {list(node.input)}")
        print(f"    Outputs: {list(node.output)}")
    
    # Apply transform
    print("\nApplying InferThresholdingAxi transform...")
    transform = InferThresholdingAxi()
    model_transformed, modified = transform.apply(model)
    
    print(f"Graph modified: {modified}")
    
    # Print transformed graph
    print("\nTransformed model:")
    for node in model_transformed.graph.node:
        print(f"  Node: {node.op_type} ({node.name})")
        print(f"    Domain: {node.domain}")
        print(f"    Inputs: {list(node.input)}")
        print(f"    Outputs: {list(node.output)}")
        
        # Print node attributes
        print("    Attributes:")
        for attr in node.attribute:
            if attr.HasField('i'):
                print(f"      {attr.name}: {attr.i}")
            elif attr.HasField('f'):
                print(f"      {attr.name}: {attr.f}")
            elif attr.HasField('s'):
                print(f"      {attr.name}: {attr.s.decode()}")
    
    # Verify conversion
    if modified:
        # Check that we have a ThresholdingAxi node
        found_thresholding_axi = False
        for node in model_transformed.graph.node:
            if node.op_type == "ThresholdingAxi":
                found_thresholding_axi = True
                # Verify domain
                expected_domain = "brainsmith.kernels.thresholding.rtl"
                assert node.domain == expected_domain, f"Expected domain {expected_domain}, got {node.domain}"
                
                # Verify key attributes
                attrs = {attr.name: attr for attr in node.attribute}
                assert "CHANNELS" in attrs
                assert "PE" in attrs
                assert "BIAS" in attrs
                assert "inputDataType" in attrs
                assert "outputDataType" in attrs
                assert "thresholdDataType" in attrs
                
                print("\n✓ Conversion successful!")
                print(f"  ThresholdingAxi node created with:")
                print(f"    CHANNELS: {attrs['CHANNELS'].i}")
                print(f"    PE: {attrs['PE'].i}")
                print(f"    BIAS: {attrs['BIAS'].i}")
                print(f"    inputDataType: {attrs['inputDataType'].s.decode()}")
                print(f"    outputDataType: {attrs['outputDataType'].s.decode()}")
                print(f"    thresholdDataType: {attrs['thresholdDataType'].s.decode()}")
                
        assert found_thresholding_axi, "No ThresholdingAxi node found after transform!"
    else:
        print("\n✗ Transform did not modify the graph")


if __name__ == "__main__":
    test_transform()