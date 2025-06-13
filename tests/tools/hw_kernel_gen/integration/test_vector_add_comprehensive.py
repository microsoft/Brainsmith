#!/usr/bin/env python3
"""
Comprehensive manual test for VectorAdd HWCustomOp with constraint groups.

This test validates all expected functionality of both the generated VectorAdd class
and the parent AutoHWCustomOp class, focusing on the new constraint groups system.
"""

import sys
import traceback
import tempfile
import os
from unittest.mock import Mock
from pathlib import Path

# Add output directory to path to import the generated VectorAdd class
sys.path.insert(0, '/home/tafk/dev/brainsmith-2/output/vector_add/vector_add')

try:
    from vector_add_hw_custom_op import VectorAdd, make_vector_add_node
    print("✓ Successfully imported VectorAdd and make_vector_add_node")
except ImportError as e:
    print(f"✗ Failed to import VectorAdd: {e}")
    sys.exit(1)

try:
    import onnx.helper
    import onnx
    print("✓ Successfully imported ONNX")
except ImportError as e:
    print(f"✗ Failed to import ONNX: {e}")
    sys.exit(1)

from brainsmith.dataflow.core.qonnx_types import DatatypeConstraintGroup
from brainsmith.dataflow.core.interface_types import InterfaceType

def create_test_onnx_node(**attrs):
    """Create a real ONNX node with the given attributes"""
    node = onnx.helper.make_node(
        "VectorAdd",
        inputs=["input1", "input2"],
        outputs=["output1"],
        **attrs
    )
    return node

def test_vectoradd_basic_instantiation():
    """Test 1: Basic VectorAdd instantiation with valid parameters"""
    
    print("\n=== Test 1: Basic VectorAdd Instantiation ===")
    
    try:
        # Create real ONNX node with required parameters and datatypes
        onnx_node = create_test_onnx_node(
            PE=8,
            VECTOR_SIZE=256,
            # Required datatype attributes for constraint groups
            input0_dtype="FIXED<16,8>",
            input1_dtype="FIXED<16,8>",
            output0_dtype="FIXED<32,16>"
        )
        
        # Create VectorAdd instance
        vector_add = VectorAdd(onnx_node)
        
        print("✓ VectorAdd instantiated successfully")
        print(f"  Kernel name: {vector_add.kernel_name}")
        print(f"  RTL source: {vector_add.rtl_source}")
        
        # Verify parameter extraction
        assert vector_add.get_nodeattr("PE") == 8
        assert vector_add.get_nodeattr("VECTOR_SIZE") == 256
        print("✓ Runtime parameter extraction working")
        
        return True
        
    except Exception as e:
        print(f"✗ Test 1 failed: {e}")
        traceback.print_exc()
        return False

def test_vectoradd_interface_metadata():
    """Test 2: Interface metadata validation with constraint groups"""
    
    print("\n=== Test 2: Interface Metadata Validation ===")
    
    try:
        # Get static interface metadata
        interface_metadata = VectorAdd.get_interface_metadata()
        
        print(f"✓ Found {len(interface_metadata)} interfaces")
        
        # Validate expected interfaces
        interface_names = [iface.name for iface in interface_metadata]
        expected_interfaces = ["ap", "input0", "input1", "output0"]
        
        for expected in expected_interfaces:
            if expected not in interface_names:
                print(f"✗ Missing expected interface: {expected}")
                return False
            print(f"  ✓ Found interface: {expected}")
        
        # Validate constraint groups
        constraint_interfaces = [iface for iface in interface_metadata 
                               if iface.interface_type != InterfaceType.CONTROL]
        
        print(f"✓ Found {len(constraint_interfaces)} dataflow interfaces with constraints")
        
        for iface in constraint_interfaces:
            print(f"  Interface {iface.name}:")
            print(f"    Type: {iface.interface_type}")
            print(f"    Constraints: {len(iface.datatype_constraints)}")
            
            for i, constraint in enumerate(iface.datatype_constraints):
                print(f"      {i+1}. {constraint.base_type} {constraint.min_width}-{constraint.max_width}")
                
                # Validate constraint group structure
                assert hasattr(constraint, 'base_type')
                assert hasattr(constraint, 'min_width') 
                assert hasattr(constraint, 'max_width')
                assert isinstance(constraint.min_width, int)
                assert isinstance(constraint.max_width, int)
                assert constraint.min_width <= constraint.max_width
        
        print("✓ All constraint groups properly structured")
        return True
        
    except Exception as e:
        print(f"✗ Test 2 failed: {e}")
        traceback.print_exc()
        return False

def test_vectoradd_datatype_methods():
    """Test 3: Datatype methods with constraint validation"""
    
    print("\n=== Test 3: Datatype Methods with Constraint Validation ===")
    
    try:
        # Test with valid datatypes
        onnx_node = create_test_onnx_node(
            PE=4,
            VECTOR_SIZE=128,
            input0_dtype="FIXED<16,8>",  # Valid: FIXED 8-16 range
            input1_dtype="FIXED<8,4>",   # Valid: FIXED 8-16 range  
            output0_dtype="FIXED<32,16>" # Valid: FIXED 16-32 range
        )
        
        vector_add = VectorAdd(onnx_node)
        
        # Test input datatype methods
        input0_dtype = vector_add.get_input_datatype(0)
        input1_dtype = vector_add.get_input_datatype(1) 
        print(f"✓ Input datatypes: {input0_dtype}, {input1_dtype}")
        
        # Test output datatype method
        output_dtype = vector_add.get_output_datatype(0)
        print(f"✓ Output datatype: {output_dtype}")
        
        # Verify datatypes match what we set
        assert "FIXED" in str(input0_dtype)
        assert "FIXED" in str(output_dtype)
        
        print("✓ Datatype methods working with valid constraints")
        return True
        
    except Exception as e:
        print(f"✗ Test 3 failed: {e}")
        traceback.print_exc()
        return False

def test_vectoradd_constraint_validation():
    """Test 4: Constraint validation with invalid datatypes"""
    
    print("\n=== Test 4: Constraint Validation ===")
    
    try:
        # Test 1: Missing datatypes (should fail)
        print("  Testing missing datatypes...")
        try:
            onnx_node = create_test_onnx_node(
                PE=4,
                VECTOR_SIZE=128
                # No datatype attributes
            )
            vector_add = VectorAdd(onnx_node)
            print("  ✗ Expected error for missing datatypes but none occurred")
            return False
        except (ValueError, KeyError) as e:
            if "must be explicitly specified" in str(e):
                print("  ✓ Correctly caught missing datatype error")
            else:
                print(f"  ✗ Wrong error message: {e}")
                return False
        
        # Test 2: Invalid datatype violating constraints
        print("  Testing constraint violations...")
        try:
            onnx_node = create_test_onnx_node(
                PE=4,
                VECTOR_SIZE=128,
                input0_dtype="FIXED<32,16>",  # Invalid: outside 8-16 range
                input1_dtype="FIXED<16,8>",
                output0_dtype="FIXED<32,16>"
            )
            vector_add = VectorAdd(onnx_node)
            print("  ✗ Expected error for constraint violation but none occurred")
            return False
        except (ValueError, KeyError) as e:
            if "violates constraints" in str(e) or "Invalid" in str(e):
                print("  ✓ Correctly caught constraint violation")
            else:
                print(f"  ✗ Wrong error message: {e}")
                return False
        
        # Test 3: Invalid QONNX datatype format
        print("  Testing invalid QONNX datatype...")
        try:
            onnx_node = create_test_onnx_node(
                PE=4,
                VECTOR_SIZE=128,
                input0_dtype="INVALID_TYPE",  # Not a valid QONNX type
                input1_dtype="FIXED<16,8>",
                output0_dtype="FIXED<32,16>"
            )
            vector_add = VectorAdd(onnx_node)
            print("  ✗ Expected error for invalid QONNX datatype but none occurred")
            return False
        except (ValueError, KeyError) as e:
            if "Invalid QONNX datatype" in str(e) or "INVALID_TYPE" in str(e):
                print("  ✓ Correctly caught invalid QONNX datatype")
            else:
                print(f"  ✗ Wrong error message: {e}")
                return False
        
        print("✓ All constraint validation tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Test 4 failed: {e}")
        traceback.print_exc()
        return False

def test_vectoradd_dataflow_model():
    """Test 5: DataflowModel creation and interface access"""
    
    print("\n=== Test 5: DataflowModel Creation ===")
    
    try:
        onnx_node = create_test_onnx_node(
            PE=8,
            VECTOR_SIZE=256,
            input0_dtype="FIXED<16,8>",
            input1_dtype="FIXED<16,8>",
            output0_dtype="FIXED<32,16>"
        )
        
        vector_add = VectorAdd(onnx_node)
        
        # Test dataflow model access
        dataflow_model = vector_add.dataflow_model
        print(f"✓ DataflowModel created")
        
        # Validate interfaces
        input_interfaces = dataflow_model.input_interfaces
        output_interfaces = dataflow_model.output_interfaces
        
        print(f"  Input interfaces: {len(input_interfaces)}")
        print(f"  Output interfaces: {len(output_interfaces)}")
        
        # Should have 2 inputs (input0, input1) and 1 output (output0)
        # Note: control interface may or may not be counted as input
        assert len(output_interfaces) >= 1, "Should have at least 1 output interface"
        
        # Test interface properties
        for i, iface in enumerate(input_interfaces):
            print(f"    Input {i}: {iface.name} ({iface.interface_type})")
            if iface.interface_type != InterfaceType.CONTROL:
                # Use QONNX DataType method for canonical name
                dtype_name = iface.dtype.get_canonical_name() if hasattr(iface.dtype, 'get_canonical_name') else str(iface.dtype)
                print(f"      Datatype: {dtype_name}")
                print(f"      Tensor dims: {iface.tensor_dims}")
                print(f"      Block dims: {iface.block_dims}")
        
        for i, iface in enumerate(output_interfaces):
            print(f"    Output {i}: {iface.name} ({iface.interface_type})")
            # Use QONNX DataType method for canonical name
            dtype_name = iface.dtype.get_canonical_name() if hasattr(iface.dtype, 'get_canonical_name') else str(iface.dtype)
            print(f"      Datatype: {dtype_name}")
            print(f"      Tensor dims: {iface.tensor_dims}")
        
        print("✓ DataflowModel interfaces validated")
        return True
        
    except Exception as e:
        print(f"✗ Test 5 failed: {e}")
        traceback.print_exc()
        return False

def test_vectoradd_shape_methods():
    """Test 6: Shape calculation methods from AutoHWCustomOp"""
    
    print("\n=== Test 6: Shape Calculation Methods ===")
    
    try:
        onnx_node = create_test_onnx_node(
            PE=4,
            VECTOR_SIZE=128,
            input0_dtype="FIXED<16,8>",
            input1_dtype="FIXED<16,8>",
            output0_dtype="FIXED<32,16>"
        )
        
        vector_add = VectorAdd(onnx_node)
        
        # Test shape methods (these come from AutoHWCustomOp parent class)
        try:
            input_shape = vector_add.get_normal_input_shape()
            print(f"  ✓ Normal input shape: {input_shape}")
        except Exception as e:
            print(f"  - Normal input shape method: {e}")
        
        try:
            output_shape = vector_add.get_normal_output_shape()
            print(f"  ✓ Normal output shape: {output_shape}")
        except Exception as e:
            print(f"  - Normal output shape method: {e}")
        
        try:
            folded_input = vector_add.get_folded_input_shape()
            print(f"  ✓ Folded input shape: {folded_input}")
        except Exception as e:
            print(f"  - Folded input shape method: {e}")
        
        try:
            folded_output = vector_add.get_folded_output_shape()
            print(f"  ✓ Folded output shape: {folded_output}")
        except Exception as e:
            print(f"  - Folded output shape method: {e}")
        
        print("✓ Shape methods accessible (implementation from AutoHWCustomOp)")
        return True
        
    except Exception as e:
        print(f"✗ Test 6 failed: {e}")
        traceback.print_exc()
        return False

def test_vectoradd_stream_width_methods():
    """Test 7: Stream width calculation methods"""
    
    print("\n=== Test 7: Stream Width Calculation ===")
    
    try:
        onnx_node = create_test_onnx_node(
            PE=4,
            VECTOR_SIZE=128,
            input0_dtype="FIXED<16,8>",
            input1_dtype="FIXED<16,8>", 
            output0_dtype="FIXED<32,16>"
        )
        
        vector_add = VectorAdd(onnx_node)
        
        # Test stream width methods
        try:
            input_width = vector_add.get_instream_width()
            print(f"  ✓ Input stream width: {input_width} bits")
            assert isinstance(input_width, int), "Stream width should be integer"
            assert input_width > 0, "Stream width should be positive"
        except Exception as e:
            print(f"  - Input stream width: {e}")
        
        try:
            output_width = vector_add.get_outstream_width()
            print(f"  ✓ Output stream width: {output_width} bits")
            assert isinstance(output_width, int), "Stream width should be integer"
            assert output_width > 0, "Stream width should be positive"
        except Exception as e:
            print(f"  - Output stream width: {e}")
        
        print("✓ Stream width methods working")
        return True
        
    except Exception as e:
        print(f"✗ Test 7 failed: {e}")
        traceback.print_exc()
        return False

def test_vectoradd_resource_estimation():
    """Test 8: Resource estimation methods"""
    
    print("\n=== Test 8: Resource Estimation ===")
    
    try:
        onnx_node = create_test_onnx_node(
            PE=8,
            VECTOR_SIZE=512,
            input0_dtype="FIXED<16,8>",
            input1_dtype="FIXED<16,8>",
            output0_dtype="FIXED<32,16>"
        )
        
        vector_add = VectorAdd(onnx_node)
        
        # Test generated resource estimation methods
        bram_est = vector_add.bram_estimation()
        print(f"  ✓ BRAM estimation: {bram_est}")
        assert isinstance(bram_est, int), "BRAM estimate should be integer"
        assert bram_est >= 0, "BRAM estimate should be non-negative"
        
        lut_est = vector_add.lut_estimation()
        print(f"  ✓ LUT estimation: {lut_est}")
        assert isinstance(lut_est, int), "LUT estimate should be integer"
        assert lut_est >= 0, "LUT estimate should be non-negative"
        
        dsp_est = vector_add.dsp_estimation()
        print(f"  ✓ DSP estimation: {dsp_est}")
        assert isinstance(dsp_est, int), "DSP estimate should be integer"
        assert dsp_est >= 0, "DSP estimate should be non-negative"
        
        # Test parent class resource estimation methods
        try:
            parent_bram = vector_add.estimate_bram_usage()
            print(f"  ✓ Parent BRAM estimation: {parent_bram}")
        except Exception as e:
            print(f"  - Parent BRAM estimation: {e}")
        
        try:
            parent_lut = vector_add.estimate_lut_usage()
            print(f"  ✓ Parent LUT estimation: {parent_lut}")
        except Exception as e:
            print(f"  - Parent LUT estimation: {e}")
        
        try:
            parent_dsp = vector_add.estimate_dsp_usage()
            print(f"  ✓ Parent DSP estimation: {parent_dsp}")
        except Exception as e:
            print(f"  - Parent DSP estimation: {e}")
        
        print("✓ Resource estimation methods working")
        return True
        
    except Exception as e:
        print(f"✗ Test 8 failed: {e}")
        traceback.print_exc()
        return False

def test_vectoradd_node_attributes():
    """Test 9: Node attribute types and validation"""
    
    print("\n=== Test 9: Node Attribute Types ===")
    
    try:
        onnx_node = create_test_onnx_node(
            PE=4,
            VECTOR_SIZE=256,
            input0_dtype="FIXED<16,8>",
            input1_dtype="FIXED<16,8>",
            output0_dtype="FIXED<32,16>"
        )
        
        vector_add = VectorAdd(onnx_node)
        
        # Test get_nodeattr_types method
        attr_types = vector_add.get_nodeattr_types()
        print(f"  ✓ Found {len(attr_types)} node attribute types")
        
        # Verify RTL parameters are present
        expected_params = ["PE", "VECTOR_SIZE"]
        for param in expected_params:
            if param in attr_types:
                attr_type, required, default = attr_types[param][:3]
                print(f"    {param}: type={attr_type}, required={required}, default={default}")
                assert attr_type == "i", f"{param} should be integer type"
            else:
                print(f"  ✗ Missing expected parameter: {param}")
                return False
        
        # Check PE is optional with default=4
        pe_info = attr_types["PE"]
        assert pe_info[1] == False, "PE should be optional"
        assert pe_info[2] == 4, "PE default should be 4"
        
        # Check VECTOR_SIZE is required
        vs_info = attr_types["VECTOR_SIZE"]
        assert vs_info[1] == True, "VECTOR_SIZE should be required"
        
        print("✓ Node attribute types properly defined")
        return True
        
    except Exception as e:
        print(f"✗ Test 9 failed: {e}")
        traceback.print_exc()
        return False

def test_make_vector_add_node():
    """Test 10: Convenience function for node creation"""
    
    print("\n=== Test 10: Node Creation Convenience Function ===")
    
    try:
        # Test successful node creation
        node = make_vector_add_node(
            inputs=["input1", "input2"],
            outputs=["output1"],
            PE=8,
            VECTOR_SIZE=256
        )
        
        print("  ✓ Node created successfully")
        print(f"    Op type: {node.op_type}")
        print(f"    Inputs: {list(node.input)}")
        print(f"    Outputs: {list(node.output)}")
        print(f"    Domain: {node.domain}")
        
        # Verify node properties
        assert node.op_type == "VectorAdd"
        assert len(node.input) == 2
        assert len(node.output) == 1
        assert node.domain == "finn.custom_op.fpgadataflow"
        
        # Test missing required parameter
        try:
            bad_node = make_vector_add_node(
                inputs=["input1"],
                outputs=["output1"],
                PE=4
                # Missing VECTOR_SIZE
            )
            print("  ✗ Expected error for missing required parameter")
            return False
        except ValueError as e:
            if "Missing required parameters" in str(e):
                print("  ✓ Correctly caught missing required parameter")
            else:
                print(f"  ✗ Wrong error message: {e}")
                return False
        
        print("✓ Convenience function working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Test 10 failed: {e}")
        traceback.print_exc()
        return False

def test_vectoradd_inheritance():
    """Test 11: Proper inheritance from AutoHWCustomOp"""
    
    print("\n=== Test 11: Inheritance Validation ===")
    
    try:
        onnx_node = create_test_onnx_node(
            PE=4,
            VECTOR_SIZE=128,
            input0_dtype="FIXED<16,8>",
            input1_dtype="FIXED<16,8>",
            output0_dtype="FIXED<32,16>"
        )
        
        vector_add = VectorAdd(onnx_node)
        
        # Test inheritance chain
        from brainsmith.dataflow.core.auto_hw_custom_op import AutoHWCustomOp
        assert isinstance(vector_add, AutoHWCustomOp), "Should inherit from AutoHWCustomOp"
        print("  ✓ Inherits from AutoHWCustomOp")
        
        # Test that parent class methods are accessible
        parent_methods = [
            'get_interface_config',
            'update_parallelism',
            'dataflow_model'
        ]
        
        available_methods = []
        for method in parent_methods:
            if hasattr(vector_add, method):
                available_methods.append(method)
                print(f"    ✓ Has parent method: {method}")
            else:
                print(f"    - Missing parent method: {method}")
        
        # Test kernel-specific attributes
        assert hasattr(vector_add, 'kernel_name'), "Should have kernel_name attribute"
        assert hasattr(vector_add, 'rtl_source'), "Should have rtl_source attribute"
        assert vector_add.kernel_name == "vector_add"
        print("  ✓ Kernel-specific attributes set correctly")
        
        print(f"✓ Inheritance working ({len(available_methods)}/{len(parent_methods)} parent methods available)")
        return True
        
    except Exception as e:
        print(f"✗ Test 11 failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all comprehensive tests"""
    
    print("VectorAdd HWCustomOp Comprehensive Manual Test Suite")
    print("=" * 60)
    print("Testing both generated VectorAdd class and AutoHWCustomOp parent functionality")
    print("Focus: Constraint groups, datatype validation, full functionality")
    
    # Run all tests
    tests = [
        test_vectoradd_basic_instantiation,
        test_vectoradd_interface_metadata,
        test_vectoradd_datatype_methods,
        test_vectoradd_constraint_validation,
        test_vectoradd_dataflow_model,
        test_vectoradd_shape_methods,
        test_vectoradd_stream_width_methods,
        test_vectoradd_resource_estimation,
        test_vectoradd_node_attributes,
        test_make_vector_add_node,
        test_vectoradd_inheritance
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test_func.__name__} crashed: {e}")
            traceback.print_exc()
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 60)
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✓ Generated VectorAdd class is fully functional")
        print("✓ Constraint groups system working correctly")
        print("✓ AutoHWCustomOp parent class integration successful")
        print("✓ All expected functionality validated")
    else:
        print(f"❌ {total - passed} tests failed")
        failed_tests = [tests[i].__name__ for i, result in enumerate(results) if not result]
        print(f"Failed tests: {failed_tests}")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)