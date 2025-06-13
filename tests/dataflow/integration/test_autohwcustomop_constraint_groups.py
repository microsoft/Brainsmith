#!/usr/bin/env python3
"""Test AutoHWCustomOp integration with constraint groups"""

import sys
import traceback
from unittest.mock import Mock
from brainsmith.dataflow.core.auto_hw_custom_op import AutoHWCustomOp
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata
from brainsmith.dataflow.core.qonnx_types import DatatypeConstraintGroup
from brainsmith.dataflow.core.interface_types import InterfaceType

class MockAutoHWCustomOp(AutoHWCustomOp):
    """Mock AutoHWCustomOp that overrides get_nodeattr for testing"""
    
    def __init__(self, onnx_node, interface_metadata, test_node_attrs=None):
        self._test_node_attrs = test_node_attrs or {}
        super().__init__(onnx_node, interface_metadata)
    
    def get_nodeattr(self, name, default=None):
        """Override to use test attributes"""
        if name in self._test_node_attrs:
            return self._test_node_attrs[name]
        return default

def create_mock_onnx_node():
    """Create a minimal mock ONNX node for testing"""
    mock_node = Mock()
    mock_node.attribute = []
    return mock_node

def test_autohwcustomop_with_valid_datatypes():
    """Test AutoHWCustomOp with valid user-specified datatypes"""
    
    print("=== Test: AutoHWCustomOp with Valid Datatypes ===")
    
    # Create interface metadata with constraint groups
    input_metadata = InterfaceMetadata(
        name="in0_V_data_V",
        interface_type=InterfaceType.INPUT,
        datatype_constraints=[
            DatatypeConstraintGroup(base_type="UINT", min_width=8, max_width=16)
        ]
    )
    
    output_metadata = InterfaceMetadata(
        name="out0_V_data_V", 
        interface_type=InterfaceType.OUTPUT,
        datatype_constraints=[
            DatatypeConstraintGroup(base_type="UINT", min_width=8, max_width=16)
        ]
    )
    
    # Create mock ONNX node and test attributes
    mock_node = create_mock_onnx_node()
    test_attrs = {
        "in0_V_data_V_dtype": "UINT8",
        "out0_V_data_V_dtype": "UINT8"
    }
    
    try:
        # Create MockAutoHWCustomOp with test attributes
        hwop = MockAutoHWCustomOp(
            onnx_node=mock_node,
            interface_metadata=[input_metadata, output_metadata],
            test_node_attrs=test_attrs
        )
        
        # Test datatype methods
        input_dtype = hwop.get_input_datatype(0)
        output_dtype = hwop.get_output_datatype(0)
        
        print(f"✓ Input datatype: {input_dtype}")
        print(f"✓ Output datatype: {output_dtype}")
        print("✓ AutoHWCustomOp created successfully with valid datatypes")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        traceback.print_exc()
        return False

def test_autohwcustomop_missing_datatypes():
    """Test AutoHWCustomOp error handling when datatypes not specified"""
    
    print("\n=== Test: AutoHWCustomOp Missing Datatypes ===")
    
    # Create interface metadata
    input_metadata = InterfaceMetadata(
        name="in0_V_data_V",
        interface_type=InterfaceType.INPUT,
        datatype_constraints=[
            DatatypeConstraintGroup(base_type="UINT", min_width=8, max_width=16)
        ]
    )
    
    # Create mock ONNX node without datatype attributes
    mock_node = create_mock_onnx_node()
    
    try:
        # This should fail during construction because dataflow model building requires datatypes
        hwop = MockAutoHWCustomOp(
            onnx_node=mock_node,
            interface_metadata=[input_metadata],
            test_node_attrs={}  # No datatypes specified
        )
        
        print("✗ Expected ValueError for missing datatypes but none was raised")
        return False
        
    except ValueError as e:
        expected_msgs = ["must be explicitly specified", "_dtype"]
        if any(msg in str(e) for msg in expected_msgs):
            print(f"✓ Correctly caught missing datatype error: {e}")
            return True
        else:
            print(f"✗ Wrong error message: {e}")
            return False
    except Exception as e:
        print(f"✗ Unexpected error type: {e}")
        traceback.print_exc()
        return False

def test_autohwcustomop_invalid_datatypes():
    """Test AutoHWCustomOp constraint validation"""
    
    print("\n=== Test: AutoHWCustomOp Invalid Datatypes ===")
    
    # Create interface metadata with strict constraints
    input_metadata = InterfaceMetadata(
        name="in0_V_data_V",
        interface_type=InterfaceType.INPUT,
        datatype_constraints=[
            DatatypeConstraintGroup(base_type="UINT", min_width=8, max_width=8)  # Only UINT8 allowed
        ]
    )
    
    # Create mock ONNX node with invalid datatype (UINT16 when only UINT8 allowed)
    mock_node = create_mock_onnx_node()
    test_attrs = {
        "in0_V_data_V_dtype": "UINT16"  # Violates constraint
    }
    
    try:
        # This should fail during construction due to constraint violation
        hwop = MockAutoHWCustomOp(
            onnx_node=mock_node,
            interface_metadata=[input_metadata],
            test_node_attrs=test_attrs
        )
        
        print("✗ Expected ValueError for constraint violation but none was raised")
        return False
        
    except ValueError as e:
        expected_msgs = ["violates constraints", "UINT16"]
        if any(msg in str(e) for msg in expected_msgs):
            print(f"✓ Correctly caught constraint violation: {e}")
            return True
        else:
            print(f"✗ Wrong error message: {e}")
            return False
    except Exception as e:
        print(f"✗ Unexpected error type: {e}")
        traceback.print_exc()
        return False

def test_get_datatype_methods_missing_attr():
    """Test get_input_datatype/get_output_datatype when called directly with missing attributes"""
    
    print("\n=== Test: Direct Datatype Method Calls with Missing Attributes ===")
    
    # Create interface metadata
    input_metadata = InterfaceMetadata(
        name="in0_V_data_V",
        interface_type=InterfaceType.INPUT,
        datatype_constraints=[
            DatatypeConstraintGroup(base_type="UINT", min_width=8, max_width=16)
        ]
    )
    
    # Create mock ONNX node with valid datatype for construction
    mock_node = create_mock_onnx_node()
    test_attrs = {
        "in0_V_data_V_dtype": "UINT8"
    }
    
    try:
        # Create MockAutoHWCustomOp successfully
        hwop = MockAutoHWCustomOp(
            onnx_node=mock_node,
            interface_metadata=[input_metadata],
            test_node_attrs=test_attrs
        )
        
        # Now modify the test attributes to simulate missing datatype
        hwop._test_node_attrs = {}  # Remove all test attributes
        
        # This should fail
        input_dtype = hwop.get_input_datatype(0)
        print("✗ Expected ValueError for missing datatype attribute but none was raised")
        return False
        
    except ValueError as e:
        expected_msgs = ["must be explicitly specified", "_dtype"]
        if any(msg in str(e) for msg in expected_msgs):
            print(f"✓ Correctly caught missing datatype in get_input_datatype: {e}")
            return True
        else:
            print(f"✗ Wrong error message: {e}")
            return False
    except Exception as e:
        print(f"✗ Unexpected error type: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    
    print("AutoHWCustomOp Constraint Groups Integration Test")
    print("=" * 60)
    
    test1_passed = test_autohwcustomop_with_valid_datatypes()
    test2_passed = test_autohwcustomop_missing_datatypes()
    test3_passed = test_autohwcustomop_invalid_datatypes()
    test4_passed = test_get_datatype_methods_missing_attr()
    
    all_passed = test1_passed and test2_passed and test3_passed and test4_passed
    
    print("\n" + "=" * 60)
    print(f"Overall result: {'PASSED' if all_passed else 'FAILED'}")
    
    if all_passed:
        print("✓ All AutoHWCustomOp constraint group integration tests passed")
    else:
        print("✗ Some tests failed")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)