#!/usr/bin/env python3
"""Test RTL parsing with constraint groups end-to-end"""

from brainsmith.tools.hw_kernel_gen.rtl_parser.parser import RTLParser
from brainsmith.dataflow.core.qonnx_types import DatatypeConstraintGroup

def test_rtl_parsing_with_constraint_groups():
    """Test complete RTL parsing with constraint groups"""
    
    print("=== Testing RTL Parsing with Constraint Groups ===")
    
    # Parse the test RTL file
    parser = RTLParser()
    
    try:
        kernel_metadata = parser.parse_file("test_vector_add_constraint_groups.sv")
        
        print(f"✓ RTL parsing successful")
        print(f"  Kernel name: {kernel_metadata.name}")
        print(f"  Pragmas found: {len(kernel_metadata.pragmas)}")
        print(f"  Interfaces found: {len(kernel_metadata.interfaces)}")
        
        # Check interfaces were created correctly
        print(f"\n--- Interfaces ---")
        for interface_metadata in kernel_metadata.interfaces:
            print(f"  {interface_metadata.name}: {interface_metadata.interface_type}")
            print(f"    Constraints: {len(interface_metadata.datatype_constraints)}")
            
            for constraint in interface_metadata.datatype_constraints:
                print(f"      {constraint.base_type}{constraint.min_width}-{constraint.max_width}")
        
        # Verify specific constraint groups
        in0_metadata = next((im for im in kernel_metadata.interfaces if im.name == "in0_V_data_V"), None)
        if in0_metadata:
            if len(in0_metadata.datatype_constraints) == 1:
                constraint = in0_metadata.datatype_constraints[0]
                if (constraint.base_type == "UINT" and 
                    constraint.min_width == 8 and 
                    constraint.max_width == 16):
                    print("✓ in0_V_data_V constraint group correctly applied")
                else:
                    print(f"✗ in0 constraint group incorrect: {constraint}")
                    return False
            else:
                print(f"✗ in0 should have 1 constraint, got {len(in0_metadata.datatype_constraints)}")
                return False
        else:
            print("✗ in0 interface not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ RTL parsing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_interface_metadata_creation():
    """Test that interfaces are created with correct constraint groups"""
    
    print("\n=== Testing Interface Metadata Creation ===")
    
    # Parse the RTL and extract interface metadata
    parser = RTLParser()
    
    try:
        kernel_metadata = parser.parse_file("test_vector_add_constraint_groups.sv")
        
        expected_interfaces = ["in0_V_data_V", "in1_V_data_V", "out0_V_data_V"]
        all_passed = True
        
        for expected_name in expected_interfaces:
            interface_metadata = next((im for im in kernel_metadata.interfaces if im.name == expected_name), None)
            
            if not interface_metadata:
                print(f"✗ {expected_name}: Interface not found")
                all_passed = False
                continue
            
            # Check constraint groups
            if len(interface_metadata.datatype_constraints) != 1:
                print(f"✗ {expected_name}: Expected 1 constraint group, got {len(interface_metadata.datatype_constraints)}")
                all_passed = False
                continue
            
            constraint = interface_metadata.datatype_constraints[0]
            if (constraint.base_type == "UINT" and 
                constraint.min_width == 8 and 
                constraint.max_width == 16):
                print(f"✓ {expected_name}: Constraint group correct (UINT8-16)")
            else:
                print(f"✗ {expected_name}: Constraint group incorrect - {constraint}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"✗ Interface metadata creation failed: {e}")
        return False

def test_pragma_application():
    """Test that DATATYPE pragmas are correctly applied"""
    
    print("\n=== Testing Pragma Application ===")
    
    parser = RTLParser()
    
    try:
        parsing_result = parser.parse_file("test_vector_add_constraint_groups.sv")
        
        # Check that we found the expected pragmas
        datatype_pragmas = [p for p in parsing_result.pragmas if p.type.value == "datatype"]
        
        if len(datatype_pragmas) != 3:
            print(f"✗ Expected 3 DATATYPE pragmas, found {len(datatype_pragmas)}")
            return False
        
        print(f"✓ Found {len(datatype_pragmas)} DATATYPE pragmas")
        
        # Check pragma parsing
        for pragma in datatype_pragmas:
            parsed_data = pragma.parsed_data
            interface_name = parsed_data.get("interface_name")
            base_type = parsed_data.get("base_type")
            min_width = parsed_data.get("min_width")
            max_width = parsed_data.get("max_width")
            
            if (base_type == "UINT" and min_width == 8 and max_width == 16):
                print(f"✓ {interface_name}: Pragma parsed correctly")
            else:
                print(f"✗ {interface_name}: Pragma parsing error - {parsed_data}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Pragma application failed: {e}")
        return False

def main():
    """Run all tests"""
    
    print("RTL to Constraint Groups Integration Test")
    print("=" * 50)
    
    test1_passed = test_rtl_parsing_with_constraint_groups()
    test2_passed = test_interface_metadata_creation()
    test3_passed = test_pragma_application()
    
    all_passed = test1_passed and test2_passed and test3_passed
    
    print("\n" + "=" * 50)
    print(f"Overall result: {'PASSED' if all_passed else 'FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    main()