#!/usr/bin/env python3
"""Test RTL parser pragma parsing with new constraint groups"""

from brainsmith.tools.hw_kernel_gen.rtl_parser.data import DatatypePragma, PragmaType
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata
from brainsmith.dataflow.core.interface_types import InterfaceType
from brainsmith.dataflow.core.qonnx_types import DatatypeConstraintGroup

def test_datatype_pragma_parsing():
    """Test DATATYPE pragma parsing with constraint groups"""
    
    print("=== Testing DATATYPE Pragma Parsing ===")
    
    # Test valid pragma parsing
    test_cases = [
        {
            "inputs": ["in0", "UINT", "8", "16"],
            "expected": {
                "interface_name": "in0",
                "base_type": "UINT", 
                "min_width": 8,
                "max_width": 16
            }
        },
        {
            "inputs": ["weights", "FIXED", "8", "8"],
            "expected": {
                "interface_name": "weights",
                "base_type": "FIXED",
                "min_width": 8,
                "max_width": 8
            }
        },
        {
            "inputs": ["out0", "INT", "4", "12"],
            "expected": {
                "interface_name": "out0",
                "base_type": "INT",
                "min_width": 4,
                "max_width": 12
            }
        }
    ]
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases):
        try:
            pragma = DatatypePragma(
                type=PragmaType.DATATYPE,
                inputs=test_case["inputs"],
                line_number=10 + i
            )
            
            # Check parsed data
            parsed = pragma.parsed_data
            expected = test_case["expected"]
            
            success = True
            for key, expected_value in expected.items():
                if parsed.get(key) != expected_value:
                    print(f"✗ Test {i+1}: {key} mismatch - expected {expected_value}, got {parsed.get(key)}")
                    success = False
                    all_passed = False
            
            if success:
                print(f"✓ Test {i+1}: Pragma parsing successful - {test_case['inputs']}")
            
        except Exception as e:
            print(f"✗ Test {i+1}: Failed with error - {e}")
            all_passed = False
    
    return all_passed

def test_datatype_pragma_constraint_generation():
    """Test DATATYPE pragma constraint group generation"""
    
    print("\n=== Testing Constraint Group Generation ===")
    
    # Create test pragma
    pragma = DatatypePragma(
        type=PragmaType.DATATYPE,
        inputs=["in0", "UINT", "8", "16"],
        line_number=1
    )
    
    all_passed = True
    
    try:
        # Test constraint group creation
        constraint_group = pragma._create_constraint_group()
        
        # Verify constraint group properties
        if constraint_group.base_type != "UINT":
            print(f"✗ Base type mismatch: expected UINT, got {constraint_group.base_type}")
            all_passed = False
        elif constraint_group.min_width != 8:
            print(f"✗ Min width mismatch: expected 8, got {constraint_group.min_width}")
            all_passed = False
        elif constraint_group.max_width != 16:
            print(f"✗ Max width mismatch: expected 16, got {constraint_group.max_width}")
            all_passed = False
        else:
            print(f"✓ Constraint group created: {constraint_group.base_type}{constraint_group.min_width}-{constraint_group.max_width}")
        
    except Exception as e:
        print(f"✗ Constraint group creation failed: {e}")
        all_passed = False
    
    return all_passed

def test_datatype_pragma_application():
    """Test DATATYPE pragma application to InterfaceMetadata"""
    
    print("\n=== Testing Pragma Application to Metadata ===")
    
    # Create test metadata
    metadata = InterfaceMetadata(
        name="in0",
        interface_type=InterfaceType.INPUT,
        datatype_constraints=[]
    )
    
    # Create test pragma
    pragma = DatatypePragma(
        type=PragmaType.DATATYPE,
        inputs=["in0", "UINT", "8", "16"],
        line_number=1
    )
    
    all_passed = True
    
    try:
        # Test pragma application
        if not pragma.applies_to_interface_metadata(metadata):
            print("✗ Pragma should apply to metadata but doesn't")
            all_passed = False
        else:
            print("✓ Pragma correctly applies to metadata")
        
        # Apply pragma
        updated_metadata = pragma.apply_to_metadata(metadata)
        
        # Check constraint was added
        if len(updated_metadata.datatype_constraints) != 1:
            print(f"✗ Expected 1 constraint, got {len(updated_metadata.datatype_constraints)}")
            all_passed = False
        else:
            constraint = updated_metadata.datatype_constraints[0]
            if constraint.base_type == "UINT" and constraint.min_width == 8 and constraint.max_width == 16:
                print(f"✓ Constraint correctly applied: {constraint.base_type}{constraint.min_width}-{constraint.max_width}")
            else:
                print(f"✗ Constraint values incorrect: {constraint}")
                all_passed = False
                
    except Exception as e:
        print(f"✗ Pragma application failed: {e}")
        all_passed = False
    
    return all_passed

def test_invalid_pragma_inputs():
    """Test invalid pragma inputs are rejected"""
    
    print("\n=== Testing Invalid Pragma Inputs ===")
    
    invalid_cases = [
        (["in0"], "Too few arguments"),
        (["in0", "INVALID_TYPE", "8", "16"], "Invalid base type"),
        (["in0", "UINT", "abc", "16"], "Non-numeric min_width"),
        (["in0", "UINT", "16", "8"], "min_width > max_width"),
        (["in0", "UINT", "0", "16"], "min_width must be positive")
    ]
    
    all_passed = True
    
    for i, (inputs, description) in enumerate(invalid_cases):
        try:
            pragma = DatatypePragma(
                type=PragmaType.DATATYPE,
                inputs=inputs,
                line_number=20 + i
            )
            print(f"✗ {description}: Should have failed but didn't")
            all_passed = False
            
        except Exception as e:
            print(f"✓ {description}: Correctly rejected - {str(e)[:60]}...")
    
    return all_passed

def main():
    """Run all tests"""
    
    print("RTL Parser Pragma Constraint Groups Test")
    print("=" * 50)
    
    test1_passed = test_datatype_pragma_parsing()
    test2_passed = test_datatype_pragma_constraint_generation()
    test3_passed = test_datatype_pragma_application()
    test4_passed = test_invalid_pragma_inputs()
    
    all_passed = test1_passed and test2_passed and test3_passed and test4_passed
    
    print("\n" + "=" * 50)
    print(f"Overall result: {'PASSED' if all_passed else 'FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    main()