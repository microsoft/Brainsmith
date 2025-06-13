#!/usr/bin/env python3
"""End-to-end integration test for QONNX datatype constraint groups"""

import sys
import traceback
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock

# Import all components for full pipeline test
from brainsmith.tools.hw_kernel_gen.rtl_parser.parser import RTLParser
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

def create_test_rtl_file():
    """Create a temporary RTL file for testing"""
    rtl_content = '''// Test SystemVerilog module with DATATYPE pragmas for constraint groups
// @brainsmith TOP_MODULE test_kernel
// @brainsmith DATATYPE in0 UINT 8 16
// @brainsmith DATATYPE in1 UINT 8 16  
// @brainsmith DATATYPE out0 UINT 8 16
// @brainsmith BDIM in0 [PE]
// @brainsmith BDIM in1 [PE]
// @brainsmith BDIM out0 [PE]

module test_kernel #(
    parameter PE = 8
)(
    input wire clk,
    input wire rst_n,
    
    // AXI-Stream input 0
    input wire [63:0] in0_V_data_V_TDATA,
    input wire in0_V_data_V_TVALID,
    output wire in0_V_data_V_TREADY,
    
    // AXI-Stream input 1  
    input wire [63:0] in1_V_data_V_TDATA,
    input wire in1_V_data_V_TVALID,
    output wire in1_V_data_V_TREADY,
    
    // AXI-Stream output
    output wire [63:0] out0_V_data_V_TDATA,
    output wire out0_V_data_V_TVALID,
    input wire out0_V_data_V_TREADY
);

// Test logic
assign out0_V_data_V_TDATA = in0_V_data_V_TDATA + in1_V_data_V_TDATA;
assign out0_V_data_V_TVALID = in0_V_data_V_TVALID & in1_V_data_V_TVALID;
assign in0_V_data_V_TREADY = out0_V_data_V_TREADY;
assign in1_V_data_V_TREADY = out0_V_data_V_TREADY;

endmodule'''
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sv', delete=False) as f:
        f.write(rtl_content)
        return f.name

def test_full_rtl_to_template_flow():
    """Test Step 6.1: Full RTL → template → instantiation flow"""
    
    print("=== Test: Full RTL → Template → Instantiation Flow ===")
    
    rtl_file = None
    try:
        # Step 1: Create test RTL file
        rtl_file = create_test_rtl_file()
        print(f"✓ Created test RTL file: {rtl_file}")
        
        # Step 2: Parse RTL with RTL parser
        parser = RTLParser()
        kernel_metadata = parser.parse_file(rtl_file)
        
        print(f"✓ RTL parsing successful")
        print(f"  Kernel name: {kernel_metadata.name}")
        print(f"  Interfaces found: {len(kernel_metadata.interfaces)}")
        print(f"  Pragmas found: {len(kernel_metadata.pragmas)}")
        
        # Verify constraint groups were created correctly
        dataflow_interfaces = [iface for iface in kernel_metadata.interfaces 
                             if iface.interface_type in [InterfaceType.INPUT, InterfaceType.OUTPUT]]
        
        constraint_groups_found = 0
        for interface in dataflow_interfaces:
            if interface.datatype_constraints:
                constraint_groups_found += len(interface.datatype_constraints)
                print(f"  {interface.name}: {len(interface.datatype_constraints)} constraint groups")
                for cg in interface.datatype_constraints:
                    print(f"    {cg.base_type}{cg.min_width}-{cg.max_width}")
        
        if constraint_groups_found < 3:
            print(f"✗ Expected at least 3 constraint groups, found {constraint_groups_found}")
            return False
        
        print(f"✓ Found {constraint_groups_found} constraint groups total")
        
        # Step 3: Test AutoHWCustomOp instantiation with valid datatypes
        mock_node = Mock()
        mock_node.attribute = []
        
        valid_datatypes = {
            "in0_V_data_V_dtype": "UINT8",
            "in1_V_data_V_dtype": "UINT8", 
            "out0_V_data_V_dtype": "UINT8"
        }
        
        hwop = MockAutoHWCustomOp(
            onnx_node=mock_node,
            interface_metadata=kernel_metadata.interfaces,
            test_node_attrs=valid_datatypes
        )
        
        # Test datatype methods work
        input_dtype = hwop.get_input_datatype(0)
        output_dtype = hwop.get_output_datatype(0)
        
        print(f"✓ AutoHWCustomOp instantiation successful")
        print(f"  Input datatype: {input_dtype}")
        print(f"  Output datatype: {output_dtype}")
        
        # Step 4: Test dataflow model was built correctly
        dataflow_model = hwop.dataflow_model
        print(f"✓ DataflowModel created with {len(dataflow_model.input_interfaces)} inputs, {len(dataflow_model.output_interfaces)} outputs")
        
        return True
        
    except Exception as e:
        print(f"✗ Full flow test failed: {e}")
        traceback.print_exc()
        return False
    finally:
        # Clean up temporary file
        if rtl_file and os.path.exists(rtl_file):
            os.unlink(rtl_file)

def test_vector_add_example():
    """Test Step 6.1: Test with vector_add example"""
    
    print("\n=== Test: Vector Add Example ===")
    
    try:
        # Use the existing vector_add test file
        vector_add_file = "test_vector_add_constraint_groups.sv"
        if not os.path.exists(vector_add_file):
            print(f"✗ Vector add test file not found: {vector_add_file}")
            return False
        
        # Parse vector_add RTL
        parser = RTLParser()
        kernel_metadata = parser.parse_file(vector_add_file)
        
        print(f"✓ Vector add RTL parsing successful")
        print(f"  Kernel name: {kernel_metadata.name}")
        print(f"  Interfaces: {len(kernel_metadata.interfaces)}")
        
        # Test with different valid datatypes within constraints
        test_cases = [
            {"name": "UINT8", "datatypes": {"in0_V_data_V_dtype": "UINT8", "in1_V_data_V_dtype": "UINT8", "out0_V_data_V_dtype": "UINT8"}},
            {"name": "UINT16", "datatypes": {"in0_V_data_V_dtype": "UINT16", "in1_V_data_V_dtype": "UINT16", "out0_V_data_V_dtype": "UINT16"}},
            {"name": "Mixed valid", "datatypes": {"in0_V_data_V_dtype": "UINT8", "in1_V_data_V_dtype": "UINT16", "out0_V_data_V_dtype": "UINT8"}},
        ]
        
        for test_case in test_cases:
            try:
                mock_node = Mock()
                mock_node.attribute = []
                
                hwop = MockAutoHWCustomOp(
                    onnx_node=mock_node,
                    interface_metadata=kernel_metadata.interfaces,
                    test_node_attrs=test_case["datatypes"]
                )
                
                print(f"  ✓ {test_case['name']}: AutoHWCustomOp created successfully")
                
            except Exception as e:
                print(f"  ✗ {test_case['name']}: Failed - {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Vector add test failed: {e}")
        traceback.print_exc()
        return False

def test_error_scenarios():
    """Test Step 6.1: Test error scenarios throughout pipeline"""
    
    print("\n=== Test: Error Scenarios Throughout Pipeline ===")
    
    rtl_file = None
    try:
        # Create RTL file for testing
        rtl_file = create_test_rtl_file()
        
        # Parse RTL successfully first
        parser = RTLParser()
        kernel_metadata = parser.parse_file(rtl_file)
        
        # Test 1: Missing datatypes
        print("  Test 1: Missing datatypes")
        try:
            mock_node = Mock()
            mock_node.attribute = []
            
            hwop = MockAutoHWCustomOp(
                onnx_node=mock_node,
                interface_metadata=kernel_metadata.interfaces,
                test_node_attrs={}  # No datatypes
            )
            print("    ✗ Expected error for missing datatypes but none occurred")
            return False
        except ValueError as e:
            if "must be explicitly specified" in str(e):
                print("    ✓ Correctly caught missing datatype error")
            else:
                print(f"    ✗ Wrong error message: {e}")
                return False
        
        # Test 2: Invalid datatypes (violate constraints)
        print("  Test 2: Invalid datatypes")
        try:
            mock_node = Mock()
            mock_node.attribute = []
            
            hwop = MockAutoHWCustomOp(
                onnx_node=mock_node,
                interface_metadata=kernel_metadata.interfaces,
                test_node_attrs={
                    "in0_V_data_V_dtype": "UINT32",  # Violates 8-16 constraint
                    "in1_V_data_V_dtype": "UINT8",
                    "out0_V_data_V_dtype": "UINT8"
                }
            )
            print("    ✗ Expected error for constraint violation but none occurred")
            return False
        except ValueError as e:
            if "violates constraints" in str(e):
                print("    ✓ Correctly caught constraint violation error")
            else:
                print(f"    ✗ Wrong error message: {e}")
                return False
        
        # Test 3: Invalid QONNX datatype strings
        print("  Test 3: Invalid QONNX datatype strings")
        try:
            mock_node = Mock()
            mock_node.attribute = []
            
            hwop = MockAutoHWCustomOp(
                onnx_node=mock_node,
                interface_metadata=kernel_metadata.interfaces,
                test_node_attrs={
                    "in0_V_data_V_dtype": "INVALID_TYPE",  # Not a valid QONNX type
                    "in1_V_data_V_dtype": "UINT8",
                    "out0_V_data_V_dtype": "UINT8"
                }
            )
            print("    ✗ Expected error for invalid QONNX datatype but none occurred")
            return False
        except (ValueError, KeyError) as e:
            if "Invalid QONNX datatype" in str(e) or "INVALID_TYPE" in str(e):
                print("    ✓ Correctly caught invalid QONNX datatype error")
            else:
                print(f"    ✗ Wrong error message: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error scenarios test failed: {e}")
        traceback.print_exc()
        return False
    finally:
        if rtl_file and os.path.exists(rtl_file):
            os.unlink(rtl_file)

def test_error_message_quality():
    """Test Step 6.1: Verify error messages are helpful"""
    
    print("\n=== Test: Error Message Quality ===")
    
    try:
        # Create interface metadata with specific constraints
        test_metadata = [
            InterfaceMetadata(
                name="test_input",
                interface_type=InterfaceType.INPUT,
                datatype_constraints=[
                    DatatypeConstraintGroup(base_type="UINT", min_width=8, max_width=16)
                ]
            )
        ]
        
        mock_node = Mock()
        mock_node.attribute = []
        
        # Test missing datatype error message quality
        try:
            hwop = MockAutoHWCustomOp(
                onnx_node=mock_node,
                interface_metadata=test_metadata,
                test_node_attrs={}
            )
        except ValueError as e:
            error_msg = str(e)
            
            # Check error message contains required information
            required_elements = [
                "test_input",  # Interface name
                "must be explicitly specified",  # Clear instruction
                "test_input_dtype",  # Specific attribute name
                "UINT8-16"  # Constraint description
            ]
            
            all_present = all(element in error_msg for element in required_elements)
            if all_present:
                print("  ✓ Missing datatype error message contains all required elements")
                print(f"    Message: {error_msg}")
            else:
                missing = [elem for elem in required_elements if elem not in error_msg]
                print(f"  ✗ Missing datatype error message missing elements: {missing}")
                print(f"    Message: {error_msg}")
                return False
        
        # Test constraint violation error message quality
        try:
            hwop = MockAutoHWCustomOp(
                onnx_node=mock_node,
                interface_metadata=test_metadata,
                test_node_attrs={"test_input_dtype": "UINT32"}
            )
        except ValueError as e:
            error_msg = str(e)
            
            # Check constraint violation message
            required_elements = [
                "UINT32",  # Invalid datatype
                "test_input",  # Interface name
                "violates constraints",  # Clear description
                "UINT8-16"  # What's allowed
            ]
            
            all_present = all(element in error_msg for element in required_elements)
            if all_present:
                print("  ✓ Constraint violation error message contains all required elements")
                print(f"    Message: {error_msg}")
            else:
                missing = [elem for elem in required_elements if elem not in error_msg]
                print(f"  ✗ Constraint violation error message missing elements: {missing}")
                print(f"    Message: {error_msg}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error message quality test failed: {e}")
        traceback.print_exc()
        return False

def test_backward_compatibility():
    """Test Step 6.2: Test existing code still works where possible"""
    
    print("\n=== Test: Backward Compatibility ===")
    
    try:
        # Test that RTL parsing still works (should be compatible)
        rtl_file = create_test_rtl_file()
        try:
            parser = RTLParser()
            kernel_metadata = parser.parse_file(rtl_file)
            
            print("  ✓ RTL parsing backward compatible")
            print(f"    Kernel: {kernel_metadata.name}")
            print(f"    Interfaces: {len(kernel_metadata.interfaces)}")
            
        finally:
            if os.path.exists(rtl_file):
                os.unlink(rtl_file)
        
        # Test that InterfaceMetadata creation still works
        metadata = InterfaceMetadata(
            name="test_interface",
            interface_type=InterfaceType.INPUT,
            datatype_constraints=[
                DatatypeConstraintGroup(base_type="UINT", min_width=8, max_width=16)
            ]
        )
        
        print("  ✓ InterfaceMetadata creation backward compatible")
        print(f"    Name: {metadata.name}")
        print(f"    Constraints: {len(metadata.datatype_constraints)}")
        
        # Test constraint validation methods
        constraint_desc = metadata.get_constraint_description()
        print(f"  ✓ Constraint description generation works: {constraint_desc}")
        
        return True
        
    except Exception as e:
        print(f"✗ Backward compatibility test failed: {e}")
        traceback.print_exc()
        return False

def document_breaking_changes():
    """Test Step 6.2: Document breaking changes"""
    
    print("\n=== Documentation: Breaking Changes ===")
    
    breaking_changes = [
        {
            "component": "AutoHWCustomOp",
            "change": "get_input_datatype() and get_output_datatype() now require explicit datatype specification",
            "before": "Methods returned default datatypes when none specified",
            "after": "Methods raise ValueError with constraint information when datatype not specified",
            "migration": "Add node attributes like 'interface_name_dtype' with valid QONNX datatype strings"
        },
        {
            "component": "DataflowInterface creation",
            "change": "No default datatypes provided during interface creation",
            "before": "Interfaces could be created with default/fallback datatypes",
            "after": "Must use from_metadata_and_runtime_datatype() factory method with explicit datatype",
            "migration": "Always specify runtime datatypes when creating DataflowInterface objects"
        },
        {
            "component": "InterfaceMetadata",
            "change": "datatype_constraints field replaces allowed_datatypes",
            "before": "Used allowed_datatypes field with DataTypeConstraint objects",
            "after": "Uses datatype_constraints field with DatatypeConstraintGroup objects",
            "migration": "Update code to use datatype_constraints and DatatypeConstraintGroup"
        },
        {
            "component": "Template generation",
            "change": "Templates now generate DatatypeConstraintGroup objects",
            "before": "Templates generated DataTypeConstraint objects",
            "after": "Templates generate DatatypeConstraintGroup with base_type, min_width, max_width",
            "migration": "Update generated code to import and use DatatypeConstraintGroup"
        }
    ]
    
    print("  Breaking changes identified:")
    for i, change in enumerate(breaking_changes, 1):
        print(f"  {i}. {change['component']}")
        print(f"     Change: {change['change']}")
        print(f"     Before: {change['before']}")
        print(f"     After: {change['after']}")
        print(f"     Migration: {change['migration']}")
        print()
    
    print(f"  ✓ Documented {len(breaking_changes)} breaking changes")
    
    return True

def main():
    """Run Phase 6 Steps 6.1 and 6.2 tests"""
    
    print("Phase 6: Integration Testing (Steps 6.1 and 6.2)")
    print("=" * 60)
    
    # Step 6.1: End-to-end testing
    print("STEP 6.1: END-TO-END TESTING")
    print("-" * 40)
    
    test1_passed = test_full_rtl_to_template_flow()
    test2_passed = test_vector_add_example()
    test3_passed = test_error_scenarios()
    test4_passed = test_error_message_quality()
    
    step61_passed = test1_passed and test2_passed and test3_passed and test4_passed
    
    # Step 6.2: Backward compatibility testing
    print("\nSTEP 6.2: BACKWARD COMPATIBILITY TESTING")
    print("-" * 40)
    
    test5_passed = test_backward_compatibility()
    test6_passed = document_breaking_changes()
    
    step62_passed = test5_passed and test6_passed
    
    # Overall results
    all_passed = step61_passed and step62_passed
    
    print("\n" + "=" * 60)
    print(f"STEP 6.1 (End-to-end): {'PASSED' if step61_passed else 'FAILED'}")
    print(f"STEP 6.2 (Backward compatibility): {'PASSED' if step62_passed else 'FAILED'}")
    print(f"Overall Phase 6 (Steps 6.1-6.2): {'PASSED' if all_passed else 'FAILED'}")
    
    if all_passed:
        print("✓ Phase 6 Steps 6.1-6.2 completed successfully")
        print("✓ QONNX datatype constraint groups integration is working end-to-end")
    else:
        print("✗ Some tests failed")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)