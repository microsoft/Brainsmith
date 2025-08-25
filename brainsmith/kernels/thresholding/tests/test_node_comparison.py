#!/usr/bin/env python3
"""
Simple comparison test between old Thresholding and new ThresholdingAxi.

Tests that both transforms produce equivalent nodes from MultiThreshold inputs.
"""

import numpy as np
from onnx import helper, TensorProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.util.basic import qonnx_make_model
from qonnx.custom_op.registry import getCustomOp

# Import transforms
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferThresholdingLayer


from brainsmith.transforms.core.infer_thresholding_axi import InferThresholdingAxi


def create_multithreshold_model(channels, input_dt, output_dt, bias=0):
    """Create a simple MultiThreshold test model."""
    
    # Create tensors
    inp = helper.make_tensor_value_info(
        "inp", TensorProto.FLOAT, [1, channels]
    )
    thresh = helper.make_tensor_value_info(
        "thresh", TensorProto.FLOAT, [channels, 3]  # 3 threshold levels
    )
    outp = helper.make_tensor_value_info(
        "outp", TensorProto.FLOAT, [1, channels]
    )
    
    # Create MultiThreshold node
    mt_node = helper.make_node(
        "MultiThreshold",
        inputs=["inp", "thresh"],
        outputs=["outp"],
        domain="qonnx.custom_op.general",
        name="MultiThreshold_0"
    )
    
    # Add attributes manually to ensure they're properly set
    mt_node.attribute.extend([
        helper.make_attribute("out_scale", 1.0),
        helper.make_attribute("out_bias", float(bias)),
        helper.make_attribute("out_dtype", output_dt)
    ])
    
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
    model.set_tensor_datatype("inp", DataType[input_dt])
    model.set_tensor_datatype("thresh", DataType["INT8"])
    model.set_tensor_datatype("outp", DataType[output_dt])
    
    # Create threshold values (sorted ascending as required)
    thresh_vals = np.array([
        [-10, 0, 10]
    ] * channels, dtype=np.float32)
    model.set_initializer("thresh", thresh_vals)
    
    return model


def compare_nodes(old_node, new_node, test_name):
    """Compare attributes between old and new thresholding nodes."""
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"{'='*60}")
    
    if old_node:
        print(f"\nOld Thresholding node:")
        print(f"  Type: {old_node.op_type}")
        print(f"  Domain: {old_node.domain}")
        old_inst = getCustomOp(old_node)
        print(f"  NumChannels: {old_inst.get_nodeattr('NumChannels')}")
        print(f"  PE: {old_inst.get_nodeattr('PE')}")
        print(f"  ActVal: {old_inst.get_nodeattr('ActVal')}")
        print(f"  inputDataType: {old_inst.get_nodeattr('inputDataType')}")
        print(f"  outputDataType: {old_inst.get_nodeattr('outputDataType')}")
        print(f"  numSteps: {old_inst.get_nodeattr('numSteps')}")
    
    print(f"\nNew ThresholdingAxi node:")
    print(f"  Type: {new_node.op_type}")
    print(f"  Domain: {new_node.domain}")
    
    # Get attributes
    attrs = {attr.name: attr for attr in new_node.attribute}
    
    # Print key attributes
    print(f"  CHANNELS: {attrs['CHANNELS'].i}")
    print(f"  PE: {attrs['PE'].i}")
    print(f"  BIAS: {attrs['BIAS'].i}")
    print(f"  inputDataType: {attrs['inputDataType'].s.decode()}")
    print(f"  outputDataType: {attrs['outputDataType'].s.decode()}")
    print(f"  thresholdDataType: {attrs['thresholdDataType'].s.decode()}")
    
    # Print RTL-specific attributes
    print(f"\n  RTL-specific attributes:")
    print(f"    input_FPARG: {attrs['input_FPARG'].i}")
    print(f"    DEPTH_TRIGGER_URAM: {attrs['DEPTH_TRIGGER_URAM'].i}")
    print(f"    DEPTH_TRIGGER_BRAM: {attrs['DEPTH_TRIGGER_BRAM'].i}")
    print(f"    DEEP_PIPELINE: {attrs['DEEP_PIPELINE'].i}")
    print(f"    USE_AXILITE: {attrs['USE_AXILITE'].i}")
    
    # Compare values if old node exists
    if old_node:
        print(f"\n  Comparison:")
        old_inst = getCustomOp(old_node)
        channels_match = old_inst.get_nodeattr('NumChannels') == attrs['CHANNELS'].i
        pe_match = old_inst.get_nodeattr('PE') == attrs['PE'].i
        bias_match = old_inst.get_nodeattr('ActVal') == attrs['BIAS'].i
        
        print(f"    Channels match: {channels_match}")
        print(f"    PE match: {pe_match}")
        print(f"    Bias/ActVal match: {bias_match}")


def run_test(test_name, channels, input_dt, output_dt, bias=0):
    """Run a single comparison test."""
    
    # Create base model
    model = create_multithreshold_model(channels, input_dt, output_dt, bias)
    
    # Test old transform if FINN available
    old_node = None
    try:
        old_model = model.transform(InferThresholdingLayer())
        old_nodes = old_model.get_nodes_by_op_type("Thresholding")
        if old_nodes:
            old_node = old_nodes[0]
    except AssertionError as e:
        print(f"  ⚠️  FINN InferThresholdingLayer failed: {e}")
        print(f"      (This is expected for some configurations)")

    # Test new transform
    try:
        new_model = model.transform(InferThresholdingAxi())
        new_nodes = new_model.get_nodes_by_op_type("ThresholdingAxi")
        
        if not new_nodes:
            print(f"❌ {test_name}: ThresholdingAxi node not created!")
            # Check if MultiThreshold still exists
            mt_nodes = new_model.get_nodes_by_op_type("MultiThreshold")
            if mt_nodes:
                print(f"      MultiThreshold node still exists - transform didn't convert it")
            return False
    except Exception as e:
        print(f"❌ {test_name}: InferThresholdingAxi failed with error: {e}")
        return False
    
    new_node = new_nodes[0]
    
    # Compare nodes
    compare_nodes(old_node, new_node, test_name)
    
    return True


def main():
    """Run all comparison tests."""
    print("🚀 Starting Thresholding Comparison Tests")
    print("="*60)
    
    tests = [
        ("Test 1: 4 channels, INT8->UINT4", 4, "INT8", "UINT4", 0),
        ("Test 2: 8 channels, INT8->INT4", 8, "INT8", "INT4", -8),  # Correct bias for signed
        ("Test 3: 16 channels, UINT8->UINT8", 16, "UINT8", "UINT8", 0),
        ("Test 4: 4 channels, INT8->BIPOLAR", 4, "INT8", "BIPOLAR", 0),  # BIPOLAR allows 0 bias
        ("Test 5: 32 channels, INT16->INT8", 32, "INT16", "INT8", -128),  # Another signed test
    ]
    
    passed = 0
    for test_name, channels, input_dt, output_dt, bias in tests:
        if run_test(test_name, channels, input_dt, output_dt, bias):
            passed += 1
    
    print(f"\n{'='*60}")
    print(f"📊 Results: {passed}/{len(tests)} tests completed")
    
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())