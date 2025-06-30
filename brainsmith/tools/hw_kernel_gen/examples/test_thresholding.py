############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Test script for ThresholdingOp implementation.

This script demonstrates how to use the manually implemented ThresholdingOp
and tests all FINN interface methods.
"""

import numpy as np
from typing import Dict, Any

# Import the manual implementation
from thresholding_manual import ThresholdingOp


class MockONNXNode:
    """Mock ONNX node for testing."""
    
    def __init__(self, name="Threshold_0"):
        self.name = name
        self.op_type = "Thresholding"
        self.input = ["input_tensor"]
        self.output = ["output_tensor"]
        self.attribute = []
        self._attr_dict = {}
    
    def add_attribute(self, name: str, value: Any):
        """Add attribute to node."""
        attr = type('obj', (object,), {
            'name': name,
            'value': value,
            'i': value if isinstance(value, int) else 0,
            'f': value if isinstance(value, float) else 0.0,
            's': value if isinstance(value, str) else ""
        })
        self.attribute.append(attr)
        self._attr_dict[name] = value
    
    def get_attr(self, name: str):
        """Get attribute value."""
        return self._attr_dict.get(name)


def test_basic_configuration():
    """Test basic ThresholdingOp configuration."""
    print("\n=== Test Basic Configuration ===")
    
    # Create node
    node = MockONNXNode()
    node.add_attribute("backend", "fpgadataflow")
    node.add_attribute("NumChannels", 64)
    node.add_attribute("numSteps", 1)
    node.add_attribute("input_dtype", "UINT8")
    node.add_attribute("output_dtype", "UINT1")
    
    # Create operation
    op = ThresholdingOp(node)
    
    # Test attribute access
    print(f"Backend: {op.get_nodeattr('backend')}")
    print(f"NumChannels: {op.get_nodeattr('NumChannels')}")
    print(f"Input dtype: {op.get_nodeattr('input_dtype')}")
    print(f"Output dtype: {op.get_nodeattr('output_dtype')}")
    
    return op


def test_finn_methods(op: ThresholdingOp):
    """Test all FINN abstract methods."""
    print("\n=== Test FINN Methods ===")
    
    # Test datatypes
    print("\nDatatypes:")
    try:
        input_dtype = op.get_input_datatype(0)
        print(f"  Input datatype: {input_dtype}")
        print(f"  Input bitwidth: {input_dtype.bitwidth()}")
    except Exception as e:
        print(f"  Error getting input datatype: {e}")
    
    try:
        output_dtype = op.get_output_datatype(0)
        print(f"  Output datatype: {output_dtype}")
        print(f"  Output bitwidth: {output_dtype.bitwidth()}")
    except Exception as e:
        print(f"  Error getting output datatype: {e}")
    
    # Test shapes
    print("\nShapes:")
    try:
        normal_in = op.get_normal_input_shape(0)
        print(f"  Normal input shape: {normal_in}")
    except Exception as e:
        print(f"  Error getting normal input shape: {e}")
    
    try:
        normal_out = op.get_normal_output_shape(0)
        print(f"  Normal output shape: {normal_out}")
    except Exception as e:
        print(f"  Error getting normal output shape: {e}")
    
    # Test stream widths
    print("\nStream Widths:")
    try:
        in_width = op.get_instream_width(0)
        print(f"  Input stream width: {in_width} bits")
    except Exception as e:
        print(f"  Error getting input stream width: {e}")
    
    try:
        out_width = op.get_outstream_width(0)
        print(f"  Output stream width: {out_width} bits")
    except Exception as e:
        print(f"  Error getting output stream width: {e}")
    
    # Test output values
    try:
        num_outputs = op.get_number_output_values()
        print(f"  Number of output values: {num_outputs}")
    except Exception as e:
        print(f"  Error getting output values: {e}")


def test_sdim_configuration():
    """Test SDIM configuration and folding."""
    print("\n=== Test SDIM Configuration ===")
    
    # Create node with SDIM
    node = MockONNXNode()
    node.add_attribute("backend", "fpgadataflow")
    node.add_attribute("NumChannels", 64)
    node.add_attribute("input_dtype", "UINT8")
    node.add_attribute("output_dtype", "UINT1")
    node.add_attribute("SIMD", 8)  # Process 8 channels in parallel
    
    op = ThresholdingOp(node)
    
    print(f"SIMD (legacy): {op.get_nodeattr('SIMD')}")
    
    # Test folded shapes
    try:
        folded_in = op.get_folded_input_shape(0)
        print(f"Folded input shape: {folded_in}")
    except Exception as e:
        print(f"Error getting folded input shape: {e}")
    
    try:
        folded_out = op.get_folded_output_shape(0)
        print(f"Folded output shape: {folded_out}")
    except Exception as e:
        print(f"Error getting folded output shape: {e}")
    
    # Test stream width with SDIM
    try:
        in_width = op.get_instream_width(0)
        print(f"Input stream width with SIMD=8: {in_width} bits")
        print(f"  = {op.get_nodeattr('SIMD')} channels * {8} bits/channel")
    except Exception as e:
        print(f"Error: {e}")


def test_resource_estimation():
    """Test resource estimation methods."""
    print("\n=== Test Resource Estimation ===")
    
    # Create node
    node = MockONNXNode()
    node.add_attribute("backend", "fpgadataflow")
    node.add_attribute("NumChannels", 256)
    node.add_attribute("numSteps", 3)  # Multi-thresholding
    node.add_attribute("input_dtype", "INT16")
    node.add_attribute("output_dtype", "UINT2")
    node.add_attribute("SIMD", 16)
    
    op = ThresholdingOp(node)
    
    # Test estimations
    print(f"BRAM estimation: {op.bram_estimation()} blocks")
    print(f"LUT estimation: {op.lut_estimation()}")
    print(f"DSP estimation: {op.dsp_estimation()}")
    print(f"URAM estimation: {op.uram_estimation()}")
    print(f"Expected cycles: {op.get_exp_cycles()}")


def test_verification():
    """Test node verification."""
    print("\n=== Test Verification ===")
    
    # Test valid configuration
    print("\nValid configuration:")
    node = MockONNXNode()
    node.add_attribute("backend", "fpgadataflow")
    node.add_attribute("NumChannels", 128)
    node.add_attribute("numSteps", 7)  # Needs 3 bits
    node.add_attribute("input_dtype", "UINT8")
    node.add_attribute("output_dtype", "UINT3")  # 3 bits, sufficient
    
    op = ThresholdingOp(node)
    for msg in op.verify_node():
        print(f"  {msg}")
    
    # Test invalid configuration
    print("\nInvalid configuration (insufficient output bits):")
    node2 = MockONNXNode("Threshold_1")
    node2.add_attribute("backend", "fpgadataflow")
    node2.add_attribute("NumChannels", 128)
    node2.add_attribute("numSteps", 7)  # Needs 3 bits
    node2.add_attribute("input_dtype", "UINT8")
    node2.add_attribute("output_dtype", "UINT2")  # Only 2 bits!
    
    op2 = ThresholdingOp(node2)
    for msg in op2.verify_node():
        print(f"  {msg}")


def test_execution():
    """Test node execution."""
    print("\n=== Test Execution ===")
    
    # Create node
    node = MockONNXNode()
    node.add_attribute("backend", "fpgadataflow")
    node.add_attribute("NumChannels", 16)
    node.add_attribute("input_dtype", "UINT8")
    node.add_attribute("output_dtype", "UINT1")
    
    op = ThresholdingOp(node)
    
    # Create execution context
    context = {
        "input_tensor": np.random.randint(0, 256, size=(1, 16), dtype=np.uint8)
    }
    
    print(f"Input tensor shape: {context['input_tensor'].shape}")
    print(f"Input tensor sample: {context['input_tensor'][0, :8]}")
    
    # Execute
    op.execute_node(context, None)
    
    if "output_tensor" in context:
        print(f"Output tensor shape: {context['output_tensor'].shape}")
        print(f"Output tensor sample: {context['output_tensor'][0, :8]}")
    else:
        print("No output tensor produced")


def main():
    """Run all tests."""
    print("ThresholdingOp Test Suite")
    print("=" * 60)
    
    # Basic configuration
    op = test_basic_configuration()
    
    # FINN methods
    test_finn_methods(op)
    
    # SDIM configuration
    test_sdim_configuration()
    
    # Resource estimation
    test_resource_estimation()
    
    # Verification
    test_verification()
    
    # Execution
    test_execution()
    
    print("\n" + "=" * 60)
    print("All tests completed!")


if __name__ == "__main__":
    main()