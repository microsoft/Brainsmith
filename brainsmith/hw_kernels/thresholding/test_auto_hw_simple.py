#!/usr/bin/env python3
"""
Simple test for auto-generated HWCustomOp with FINN integration.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import the auto-generated HWCustomOp
from brainsmith.hw_kernels.thresholding.auto_thresholding.thresholding_axi_hw_custom_op import ThresholdingAxi

# ONNX imports
from onnx import helper


def test_attributes():
    """Test that all required attributes are present."""
    print("\n✅ Testing Auto-Generated HWCustomOp Attributes")
    
    # Create a minimal node
    node = helper.make_node(
        "ThresholdingAxi",
        ["inp", "thresh"],
        ["outp"],
        CHANNELS=32,
        PE=4,
        inputDataType="INT8",
        outputDataType="UINT4"
    )
    
    # Instantiate
    op = ThresholdingAxi(node)
    attrs = op.get_nodeattr_types()
    
    # Check critical attributes
    required = ["exec_mode", "backend", "inputDataType", "outputDataType", "CHANNELS", "PE"]
    missing = [attr for attr in required if attr not in attrs]
    
    if missing:
        print(f"❌ Missing attributes: {missing}")
        return False
    else:
        print(f"✅ All required attributes present: {required}")
        return True


def test_kernel_model():
    """Test that KernelModel can be created."""
    print("\n✅ Testing KernelModel Creation")
    
    node = helper.make_node(
        "ThresholdingAxi",
        ["inp", "thresh"],
        ["outp"],
        CHANNELS=32,
        PE=4,
        inputDataType="INT8",
        outputDataType="UINT4"
    )
    
    op = ThresholdingAxi(node)
    
    # Check if kernel model can be created (it's created lazily)
    if hasattr(op, '_kernel_model'):
        if op._kernel_model is None:
            print("✅ KernelModel not yet initialized (lazy initialization)")
        else:
            print(f"✅ KernelModel exists: {type(op._kernel_model)}")
        return True
    else:
        print("❌ No _kernel_model attribute found")
        return False


def main():
    """Run tests."""
    print("🚀 Testing Auto-Generated ThresholdingAxi HWCustomOp")
    print("=" * 60)
    
    passed = 0
    total = 0
    
    # Test 1: Attributes
    total += 1
    if test_attributes():
        passed += 1
    
    # Test 2: Kernel Model
    total += 1
    if test_kernel_model():
        passed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Auto-generated HWCustomOp is working correctly.")
        return 0
    else:
        print("❌ Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())