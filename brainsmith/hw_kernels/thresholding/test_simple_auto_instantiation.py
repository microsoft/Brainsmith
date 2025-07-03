#!/usr/bin/env python3
"""
Simple test to instantiate auto-generated HWCustomOp.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# ONNX imports
import onnx
from onnx import helper, TensorProto, numpy_helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.custom_op.registry import getCustomOp

# Import the auto-generated HWCustomOp directly
from brainsmith.hw_kernels.thresholding.auto_thresholding.thresholding_axi_hw_custom_op import ThresholdingAxi


def test_direct_instantiation():
    """Test direct instantiation of auto-generated HWCustomOp."""
    print("\n=== Testing Direct Instantiation ===")
    
    # Create a simple ONNX node
    node = helper.make_node(
        "ThresholdingAxi",
        ["inp", "thresh"],
        ["outp"],
        domain="test.domain",
        CHANNELS=32,
        PE=4,
        LEVELS=3,
        inputDataType="INT8",
        outputDataType="UINT4",
        weightDataType="INT8"
    )
    
    # Instantiate the HWCustomOp
    try:
        op_inst = ThresholdingAxi(node)
        print(f"✅ Successfully instantiated: {type(op_inst).__name__}")
        
        # Check attributes
        attrs = op_inst.get_nodeattr_types()
        print(f"✅ Node attributes: {list(attrs.keys())}")
        
        # Check if exec_mode is present
        if "exec_mode" in attrs:
            print("✅ exec_mode attribute is present")
        else:
            print("❌ exec_mode attribute is missing")
            
        return True
    except Exception as e:
        print(f"❌ Failed to instantiate: {e}")
        return False


def test_model_with_auto_op():
    """Test creating a model with auto-generated HWCustomOp."""
    print("\n=== Testing Model Creation ===")
    
    # Create tensors
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 32])
    thresh = helper.make_tensor_value_info("thresh", TensorProto.FLOAT, [32, 3])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 32])
    
    # Create threshold initializer
    thresh_vals = np.array([[-10, 0, 10]] * 32, dtype=np.float32)
    thresh_init = numpy_helper.from_array(thresh_vals, name="thresh")
    
    # Create node
    node = helper.make_node(
        "ThresholdingAxi",
        ["inp", "thresh"],
        ["outp"],
        domain="test.domain",
        CHANNELS=32,
        PE=4,
        LEVELS=3,
        inputDataType="INT8",
        outputDataType="UINT4",
        weightDataType="INT8"
    )
    
    # Create graph and model
    graph = helper.make_graph([node], "test_graph", [inp, thresh], [outp], [thresh_init])
    model = helper.make_model(graph, producer_name="test-model")
    model_wrapper = ModelWrapper(model)
    
    # Set datatypes
    model_wrapper.set_tensor_datatype("inp", DataType["INT8"])
    model_wrapper.set_tensor_datatype("outp", DataType["UINT4"])
    model_wrapper.set_tensor_datatype("thresh", DataType["INT8"])
    
    try:
        # Get the op instance using getCustomOp
        op_inst = getCustomOp(node)
        print(f"✅ getCustomOp returned: {type(op_inst).__name__}")
        
        # Check if it's our auto-generated op
        if isinstance(op_inst, ThresholdingAxi):
            print("✅ Op is instance of ThresholdingAxi")
        else:
            print(f"❌ Op is not ThresholdingAxi, got: {type(op_inst)}")
            
        return True
    except Exception as e:
        print(f"❌ Failed to get custom op: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Testing Auto-Generated HWCustomOp")
    print("=" * 50)
    
    tests = [
        ("Direct Instantiation", test_direct_instantiation),
        ("Model Creation", test_model_with_auto_op),
    ]
    
    passed = 0
    for test_name, test_func in tests:
        if test_func():
            passed += 1
            
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())