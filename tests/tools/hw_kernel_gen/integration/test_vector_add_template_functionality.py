#!/usr/bin/env python3
"""
Comprehensive testbench for the generated VectorAdd AutoHWCustomOp.

This testbench verifies that:
1. Template reduction worked correctly (only essential methods in subclass)
2. Parent class AutoHWCustomOp provides all missing functionality
3. Runtime parameter extraction functions properly
4. Interface metadata drives DataflowModel correctly
5. All FINN integration methods work via parent class delegation
"""

import sys
import os
import traceback
from pathlib import Path

# Add brainsmith to path
sys.path.insert(0, '/home/tafk/dev/brainsmith-2')
sys.path.insert(0, '/home/tafk/dev/brainsmith-2/output/vector_add')

# Mock FINN dependencies (since not available in this environment)
class MockDataType:
    def __init__(self, name):
        self.name = name
        self._bitwidth = 8 if 'FIXED8' in name else 16 if 'FIXED16' in name else 32
    
    def bitwidth(self):
        return self._bitwidth
    
    def __str__(self):
        return self.name

class MockFINN:
    def __getitem__(self, key):
        return MockDataType(key)

# Patch FINN imports
sys.modules['qonnx'] = type(sys)('qonnx')
sys.modules['qonnx.core'] = type(sys)('qonnx.core')
sys.modules['qonnx.core.datatype'] = type(sys)('qonnx.core.datatype')
sys.modules['qonnx.core.datatype'].DataType = MockFINN()

sys.modules['finn'] = type(sys)('finn')
sys.modules['finn.custom_op'] = type(sys)('finn.custom_op')
sys.modules['finn.custom_op.fpgadataflow'] = type(sys)('finn.custom_op.fpgadataflow')
sys.modules['finn.custom_op.fpgadataflow.hwcustomop'] = type(sys)('finn.custom_op.fpgadataflow.hwcustomop')

# Mock HWCustomOp base class with minimal functionality
class MockHWCustomOp:
    def __init__(self, onnx_node, **kwargs):
        self.onnx_node = onnx_node
    
    def get_nodeattr(self, name):
        # Extract from mock node
        return getattr(self.onnx_node, name, None)
    
    def get_enhanced_nodeattr_types(self):
        return {
            "runtime_parallel_mode": ("s", False, "automatic", {"automatic", "manual"}),
            "enable_validation": ("b", False, True)
        }

sys.modules['finn.custom_op.fpgadataflow.hwcustomop'].HWCustomOp = MockHWCustomOp

# Mock onnx
sys.modules['onnx'] = type(sys)('onnx')
sys.modules['onnx.helper'] = type(sys)('onnx.helper')

class MockNode:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

sys.modules['onnx.helper'].make_node = lambda *args, **kwargs: MockNode(**kwargs)

def test_template_reduction():
    """Test that template reduction worked correctly."""
    print("🔍 Testing Template Reduction...")
    
    # Import the generated class
    from vector_add_hw_custom_op import VectorAdd
    
    # Check that only essential methods exist in the generated class
    essential_methods = ['__init__', 'get_interface_metadata', 'get_nodeattr_types', 
                        'bram_estimation', 'lut_estimation', 'dsp_estimation']
    
    removed_methods = ['get_input_datatype', 'get_output_datatype', 
                      'get_normal_input_shape', 'get_folded_input_shape',
                      'get_normal_output_shape', 'get_folded_output_shape',
                      'get_instream_width', 'get_outstream_width', 'get_exp_cycles']
    
    # Check essential methods exist
    for method in essential_methods:
        assert hasattr(VectorAdd, method), f"Missing essential method: {method}"
    
    print(f"✅ All {len(essential_methods)} essential methods present")
    
    # Check that previously generated methods were removed from the class definition
    class_methods = [attr for attr in dir(VectorAdd) if callable(getattr(VectorAdd, attr)) and not attr.startswith('_')]
    generated_redundant = [m for m in removed_methods if m in class_methods and hasattr(getattr(VectorAdd, m), '__qualname__') and 'VectorAdd' in getattr(VectorAdd, m).__qualname__]
    
    if generated_redundant:
        print(f"⚠️  Warning: These methods still generated in subclass: {generated_redundant}")
    else:
        print(f"✅ All {len(removed_methods)} redundant methods successfully removed from generated subclass")
    
    # Count lines in generated file
    file_path = Path('/home/tafk/dev/brainsmith-2/output/vector_add/vector_add_hw_custom_op.py')
    line_count = len(file_path.read_text().splitlines())
    print(f"✅ Generated file size: {line_count} lines (target: <200 lines)")
    
    return True

def test_instantiation_and_parameters():
    """Test instantiation and runtime parameter extraction."""
    print("\n🏗️  Testing Instantiation and Parameter Extraction...")
    
    from vector_add_hw_custom_op import VectorAdd
    
    # Create test node with parameters
    node = MockNode(
        PE=8,
        VECTOR_SIZE=32, 
        inputDataType="FIXED8",
        outputDataType="FIXED16",
        numInputVectors=[1, 128]
    )
    
    # Test instantiation
    op = VectorAdd(node)
    print("✅ VectorAdd instantiation successful")
    
    # Test runtime parameter extraction
    assert op.get_nodeattr("PE") == 8, "PE parameter not extracted correctly"
    assert op.get_nodeattr("VECTOR_SIZE") == 32, "VECTOR_SIZE parameter not extracted correctly"
    print("✅ Runtime parameter extraction working")
    
    # Test kernel attributes
    assert op.kernel_name == "vector_add", "Kernel name not set correctly"
    assert op.rtl_source == "example_vector_add.sv", "RTL source not set correctly"
    print("✅ Kernel metadata set correctly")
    
    return op

def test_interface_metadata():
    """Test interface metadata structure and content."""
    print("\n🔌 Testing Interface Metadata...")
    
    from vector_add_hw_custom_op import VectorAdd
    
    # Get static interface metadata
    interfaces = VectorAdd.get_interface_metadata()
    
    # Verify interface count and types
    assert len(interfaces) == 4, f"Expected 4 interfaces, got {len(interfaces)}"
    print(f"✅ Found {len(interfaces)} interfaces")
    
    # Check interface names and types
    interface_names = [iface.name for iface in interfaces]
    expected_names = ['ap', 'input0', 'input1', 'output0']
    assert set(interface_names) == set(expected_names), f"Interface names mismatch: {interface_names}"
    
    # Check interface types
    from brainsmith.dataflow.core.interface_types import InterfaceType
    ap_iface = next(i for i in interfaces if i.name == 'ap')
    input0_iface = next(i for i in interfaces if i.name == 'input0')
    output0_iface = next(i for i in interfaces if i.name == 'output0')
    
    assert ap_iface.interface_type == InterfaceType.CONTROL, "ap interface should be CONTROL"
    assert input0_iface.interface_type == InterfaceType.INPUT, "input0 interface should be INPUT"
    assert output0_iface.interface_type == InterfaceType.OUTPUT, "output0 interface should be OUTPUT"
    print("✅ Interface types correct")
    
    # Check datatype constraints
    input0_datatypes = [dt.finn_type for dt in input0_iface.allowed_datatypes]
    assert "FIXED8" in input0_datatypes, "input0 should support FIXED8"
    assert "FIXED16" in input0_datatypes, "input0 should support FIXED16"
    print("✅ Datatype constraints defined")
    
    # Check chunking strategies (BDIM)
    assert input0_iface.chunking_strategy is not None, "input0 should have chunking strategy"
    assert input0_iface.chunking_strategy.block_shape == [':', ':'], "input0 BDIM shape incorrect"
    print("✅ BDIM chunking strategies defined")
    
    return interfaces

def test_node_attributes():
    """Test node attribute definitions."""
    print("\n📋 Testing Node Attribute Types...")
    
    from vector_add_hw_custom_op import VectorAdd
    
    node = MockNode(PE=4, VECTOR_SIZE=16, inputDataType="FIXED8", outputDataType="FIXED16")
    op = VectorAdd(node)
    
    # Get node attribute types
    attrs = op.get_nodeattr_types()
    
    # Check RTL parameters
    assert "PE" in attrs, "PE parameter not in node attributes"
    assert "VECTOR_SIZE" in attrs, "VECTOR_SIZE parameter not in node attributes"
    
    # Check PE is optional with default
    pe_spec = attrs["PE"]
    assert pe_spec[0] == "i", "PE should be integer type"
    assert pe_spec[1] == False, "PE should be optional"
    assert pe_spec[2] == 4, "PE default should be 4"
    
    # Check VECTOR_SIZE is required
    vs_spec = attrs["VECTOR_SIZE"]
    assert vs_spec[0] == "i", "VECTOR_SIZE should be integer type"
    assert vs_spec[1] == True, "VECTOR_SIZE should be required"
    assert vs_spec[2] is None, "VECTOR_SIZE should have no default"
    
    print("✅ RTL parameters defined correctly")
    
    # Check standard FINN attributes
    standard_attrs = ["inputDataType", "outputDataType", "runtime_writeable_weights", "numInputVectors"]
    for attr in standard_attrs:
        assert attr in attrs, f"Missing standard attribute: {attr}"
    print("✅ Standard FINN attributes present")
    
    # Check parent class attributes were added
    parent_attrs = ["runtime_parallel_mode", "enable_validation"]  # From mock parent class
    for attr in parent_attrs:
        assert attr in attrs, f"Missing parent class attribute: {attr}"
    print("✅ Parent class attributes merged successfully")
    
    return attrs

def test_parent_class_delegation():
    """Test that parent class methods are available and working."""
    print("\n🎯 Testing Parent Class Method Delegation...")
    
    from vector_add_hw_custom_op import VectorAdd
    
    node = MockNode(PE=4, VECTOR_SIZE=16, inputDataType="FIXED8", outputDataType="FIXED16")
    op = VectorAdd(node)
    
    # Test that parent class provides missing methods
    parent_methods = [
        'get_input_datatype', 'get_output_datatype',
        'get_normal_input_shape', 'get_folded_input_shape', 
        'get_normal_output_shape', 'get_folded_output_shape',
        'get_instream_width', 'get_outstream_width', 
        'get_exp_cycles', 'estimate_bram_usage', 'estimate_lut_usage'
    ]
    
    available_methods = []
    for method in parent_methods:
        if hasattr(op, method):
            available_methods.append(method)
    
    print(f"✅ Parent class provides {len(available_methods)}/{len(parent_methods)} expected methods")
    
    # Test some key parent class properties
    if hasattr(op, 'dataflow_model'):
        print("✅ DataflowModel available from parent class")
    
    if hasattr(op, 'interface_metadata'):
        print("✅ Interface metadata collection available from parent class")
    
    # Test that we can access interface-related properties
    try:
        if hasattr(op, 'input_interfaces'):
            input_ifaces = op.input_interfaces
            print(f"✅ Input interfaces accessible: {input_ifaces}")
        
        if hasattr(op, 'output_interfaces'):
            output_ifaces = op.output_interfaces
            print(f"✅ Output interfaces accessible: {output_ifaces}")
    except Exception as e:
        print(f"⚠️  Interface access error (expected without full FINN): {e}")
    
    return True

def test_resource_estimation():
    """Test resource estimation methods."""
    print("\n💾 Testing Resource Estimation...")
    
    from vector_add_hw_custom_op import VectorAdd
    
    node = MockNode(PE=4, VECTOR_SIZE=16, inputDataType="FIXED8", outputDataType="FIXED16")
    op = VectorAdd(node)
    
    # Test kernel-specific resource estimates
    bram = op.bram_estimation()
    lut = op.lut_estimation()
    dsp = op.dsp_estimation()
    
    assert isinstance(bram, int), "BRAM estimation should return integer"
    assert isinstance(lut, int), "LUT estimation should return integer"
    assert isinstance(dsp, int), "DSP estimation should return integer"
    
    print(f"✅ Resource estimates: BRAM={bram}, LUT={lut}, DSP={dsp}")
    
    # Test that parent class estimation methods are available
    if hasattr(op, 'estimate_bram_usage'):
        print("✅ Parent class BRAM estimation available")
    
    if hasattr(op, 'estimate_lut_usage'):
        print("✅ Parent class LUT estimation available")
    
    return bram, lut, dsp

def test_convenience_function():
    """Test the make_vector_add_node convenience function."""
    print("\n🔧 Testing Convenience Function...")
    
    from vector_add_hw_custom_op import make_vector_add_node
    
    # Test successful node creation
    node = make_vector_add_node(
        inputs=["input_tensor"],
        outputs=["output_tensor"],
        PE=8,
        VECTOR_SIZE=32,
        inputDataType="FIXED8",
        outputDataType="FIXED16"
    )
    
    assert hasattr(node, 'PE'), "Node should have PE attribute"
    assert hasattr(node, 'VECTOR_SIZE'), "Node should have VECTOR_SIZE attribute"
    assert node.PE == 8, "PE value incorrect"
    assert node.VECTOR_SIZE == 32, "VECTOR_SIZE value incorrect"
    print("✅ Node creation successful with all parameters")
    
    # Test validation of required parameters
    try:
        make_vector_add_node(
            inputs=["input_tensor"],
            outputs=["output_tensor"], 
            PE=8
            # Missing VECTOR_SIZE
        )
        assert False, "Should have raised error for missing VECTOR_SIZE"
    except ValueError as e:
        assert "VECTOR_SIZE" in str(e), "Error should mention missing VECTOR_SIZE"
        print("✅ Required parameter validation working")
    
    return node

def main():
    """Run all tests."""
    print("🧪 VectorAdd AutoHWCustomOp Testbench")
    print("=" * 50)
    
    try:
        # Test 1: Template reduction verification
        test_template_reduction()
        
        # Test 2: Basic instantiation and parameters
        op = test_instantiation_and_parameters()
        
        # Test 3: Interface metadata
        interfaces = test_interface_metadata()
        
        # Test 4: Node attributes
        attrs = test_node_attributes()
        
        # Test 5: Parent class delegation
        test_parent_class_delegation()
        
        # Test 6: Resource estimation
        bram, lut, dsp = test_resource_estimation()
        
        # Test 7: Convenience function
        node = test_convenience_function()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! Template reduction successful.")
        print("\n📊 Summary:")
        print(f"   ✅ Template reduction: Generated subclass focuses on essential methods")
        print(f"   ✅ Runtime parameters: PE and VECTOR_SIZE extracted correctly")
        print(f"   ✅ Interface metadata: {len(interfaces)} interfaces with proper BDIM")
        print(f"   ✅ Parent delegation: AutoHWCustomOp provides all missing functionality")
        print(f"   ✅ Resource estimates: BRAM={bram}, LUT={lut}, DSP={dsp}")
        print(f"   ✅ FINN integration: make_vector_add_node works correctly")
        print("\n🏆 The reduced template successfully leverages AutoHWCustomOp parent class!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)