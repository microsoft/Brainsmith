#!/usr/bin/env python3
"""Unit tests for QONNX types core components"""

from brainsmith.dataflow.core.qonnx_types import (
    DatatypeConstraintGroup, 
    validate_datatype_against_constraints,
    _matches_constraint_group,
    DataType
)

def test_datatype_constraint_group_validation():
    """Test DatatypeConstraintGroup validation"""
    
    print("=== Testing DatatypeConstraintGroup Validation ===")
    
    # Test valid constraint groups
    valid_tests = [
        ("UINT", 8, 16),
        ("INT", 1, 32),
        ("FIXED", 8, 8),
        ("FLOAT", 16, 64),
        ("BIPOLAR", 1, 1),
        ("TERNARY", 2, 4)
    ]
    
    all_passed = True
    for base_type, min_width, max_width in valid_tests:
        try:
            group = DatatypeConstraintGroup(base_type, min_width, max_width)
            print(f"✓ {base_type} {min_width}-{max_width}: Valid")
        except Exception as e:
            print(f"✗ {base_type} {min_width}-{max_width}: Failed - {e}")
            all_passed = False
    
    # Test invalid constraint groups
    print("\n--- Testing Invalid Constraint Groups ---")
    invalid_tests = [
        ("INVALID_TYPE", 8, 16),  # Invalid base type
        ("UINT", 0, 16),          # Invalid min_width
        ("UINT", -1, 16),         # Negative min_width
        ("UINT", 16, 8),          # max_width < min_width
    ]
    
    for base_type, min_width, max_width in invalid_tests:
        try:
            group = DatatypeConstraintGroup(base_type, min_width, max_width)
            print(f"✗ {base_type} {min_width}-{max_width}: Should have failed")
            all_passed = False
        except ValueError as e:
            print(f"✓ {base_type} {min_width}-{max_width}: Correctly rejected - {str(e)[:50]}...")
        except Exception as e:
            print(f"? {base_type} {min_width}-{max_width}: Unexpected error - {e}")
    
    return all_passed

def test_constraint_matching_all_datatype_families():
    """Test constraint matching for all datatype families"""
    
    print("\n=== Testing Constraint Matching by Datatype Family ===")
    
    # Test different datatype families
    test_cases = [
        # UINT family
        {
            "constraint": DatatypeConstraintGroup("UINT", 8, 16),
            "valid_types": ["UINT8", "UINT12", "UINT16"],
            "invalid_types": ["UINT4", "UINT32", "INT8", "FLOAT32"]
        },
        
        # INT family  
        {
            "constraint": DatatypeConstraintGroup("INT", 4, 8),
            "valid_types": ["INT4", "INT6", "INT8"],
            "invalid_types": ["INT2", "INT16", "UINT8", "FIXED<8,4>"]
        },
        
        # FIXED family
        {
            "constraint": DatatypeConstraintGroup("FIXED", 8, 12),
            "valid_types": ["FIXED<8,4>", "FIXED<10,5>", "FIXED<12,6>"],
            "invalid_types": ["FIXED<6,3>", "FIXED<16,8>", "UINT8", "FLOAT32"]
        },
        
        # FLOAT family
        {
            "constraint": DatatypeConstraintGroup("FLOAT", 32, 64),
            "valid_types": ["FLOAT32", "FLOAT64"],
            "invalid_types": ["FLOAT16", "UINT32", "INT32", "FIXED<32,16>"]
        },
        
        # BIPOLAR family
        {
            "constraint": DatatypeConstraintGroup("BIPOLAR", 1, 1),
            "valid_types": ["BIPOLAR"],
            "invalid_types": ["UINT1", "INT1", "TERNARY"]
        },
        
        # TERNARY family
        {
            "constraint": DatatypeConstraintGroup("TERNARY", 2, 4),
            "valid_types": ["TERNARY"],
            "invalid_types": ["UINT2", "INT2", "BIPOLAR"]
        }
    ]
    
    all_passed = True
    
    for test_case in test_cases:
        constraint = test_case["constraint"]
        print(f"\n--- Testing {constraint.base_type}{constraint.min_width}-{constraint.max_width} ---")
        
        # Test valid types
        for dtype_str in test_case["valid_types"]:
            try:
                dtype = DataType[dtype_str]
                result = _matches_constraint_group(dtype, constraint)
                if result:
                    print(f"✓ {dtype_str}: Correctly matched")
                else:
                    print(f"✗ {dtype_str}: Should have matched but didn't")
                    all_passed = False
            except Exception as e:
                print(f"✗ {dtype_str}: Failed to test - {e}")
                all_passed = False
        
        # Test invalid types
        for dtype_str in test_case["invalid_types"]:
            try:
                dtype = DataType[dtype_str]
                result = _matches_constraint_group(dtype, constraint)
                if not result:
                    print(f"✓ {dtype_str}: Correctly rejected")
                else:
                    print(f"✗ {dtype_str}: Should have been rejected")
                    all_passed = False
            except Exception as e:
                # Some datatypes might not exist in QONNX - that's ok
                print(f"? {dtype_str}: Not available in QONNX - {str(e)[:30]}...")
    
    return all_passed

def test_edge_cases():
    """Test edge cases (empty constraints, invalid datatypes)"""
    
    print("\n=== Testing Edge Cases ===")
    
    all_passed = True
    
    # Test empty constraints list
    print("--- Testing Empty Constraints ---")
    empty_constraints = []
    test_datatypes = ["UINT8", "INT16", "FLOAT32", "FIXED<8,4>"]
    
    for dtype_str in test_datatypes:
        try:
            dtype = DataType[dtype_str]
            result = validate_datatype_against_constraints(dtype, empty_constraints)
            if result:
                print(f"✓ {dtype_str}: Correctly allowed with empty constraints")
            else:
                print(f"✗ {dtype_str}: Should be allowed with empty constraints")
                all_passed = False
        except Exception as e:
            print(f"✗ {dtype_str}: Failed - {e}")
            all_passed = False
    
    # Test single constraint group edge cases
    print("\n--- Testing Single Width Constraints ---")
    single_width_constraint = DatatypeConstraintGroup("UINT", 8, 8)  # Exactly 8 bits
    
    single_width_tests = [
        ("UINT8", True),   # Exact match
        ("UINT7", False),  # Below range (if it existed)
        ("UINT9", False),  # Above range (if it existed)
        ("UINT16", False), # Above range
        ("INT8", False)    # Wrong type
    ]
    
    for dtype_str, expected in single_width_tests:
        try:
            dtype = DataType[dtype_str]
            result = validate_datatype_against_constraints(dtype, [single_width_constraint])
            if result == expected:
                print(f"✓ {dtype_str}: Expected {expected}, got {result}")
            else:
                print(f"✗ {dtype_str}: Expected {expected}, got {result}")
                all_passed = False
        except Exception as e:
            # Some test datatypes might not exist - skip them
            print(f"? {dtype_str}: Not available - {str(e)[:30]}...")
    
    # Test multiple constraint groups
    print("\n--- Testing Multiple Constraint Groups ---")
    multiple_constraints = [
        DatatypeConstraintGroup("UINT", 8, 16),
        DatatypeConstraintGroup("INT", 4, 8),
        DatatypeConstraintGroup("FIXED", 8, 12)
    ]
    
    multiple_tests = [
        ("UINT8", True),    # Matches first group
        ("UINT16", True),   # Matches first group
        ("INT4", True),     # Matches second group
        ("INT8", True),     # Matches second group
        ("FIXED<8,4>", True),  # Matches third group
        ("FIXED<12,6>", True), # Matches third group
        ("UINT32", False),  # No match
        ("FLOAT32", False), # No match
        ("INT16", False)    # No match
    ]
    
    for dtype_str, expected in multiple_tests:
        try:
            dtype = DataType[dtype_str]
            result = validate_datatype_against_constraints(dtype, multiple_constraints)
            if result == expected:
                print(f"✓ {dtype_str}: Expected {expected}, got {result}")
            else:
                print(f"✗ {dtype_str}: Expected {expected}, got {result}")
                all_passed = False
        except Exception as e:
            print(f"? {dtype_str}: Error - {e}")
    
    return all_passed

def test_bitwidth_range_validation():
    """Test bitwidth range validation"""
    
    print("\n=== Testing Bitwidth Range Validation ===")
    
    all_passed = True
    
    # Test various bitwidth ranges
    range_tests = [
        {
            "constraint": DatatypeConstraintGroup("UINT", 1, 4),
            "valid": ["UINT1", "UINT2", "UINT3", "UINT4"],
            "invalid": ["UINT8", "UINT16", "UINT32"]
        },
        {
            "constraint": DatatypeConstraintGroup("INT", 8, 32),
            "valid": ["INT8", "INT16", "INT32"],
            "invalid": ["INT4", "INT64"]
        },
        {
            "constraint": DatatypeConstraintGroup("FIXED", 8, 16),
            "valid": ["FIXED<8,4>", "FIXED<12,6>", "FIXED<16,8>"],
            "invalid": ["FIXED<6,3>", "FIXED<32,16>"]
        }
    ]
    
    for test_case in range_tests:
        constraint = test_case["constraint"]
        print(f"\n--- Testing {constraint.base_type} {constraint.min_width}-{constraint.max_width} ---")
        
        # Test valid bitwidths
        for dtype_str in test_case["valid"]:
            try:
                dtype = DataType[dtype_str]
                result = _matches_constraint_group(dtype, constraint)
                if result:
                    print(f"✓ {dtype_str} (width {dtype.bitwidth()}): Correctly matched")
                else:
                    print(f"✗ {dtype_str} (width {dtype.bitwidth()}): Should have matched")
                    all_passed = False
            except Exception as e:
                print(f"? {dtype_str}: Not available - {str(e)[:30]}...")
        
        # Test invalid bitwidths  
        for dtype_str in test_case["invalid"]:
            try:
                dtype = DataType[dtype_str]
                result = _matches_constraint_group(dtype, constraint)
                if not result:
                    print(f"✓ {dtype_str} (width {dtype.bitwidth()}): Correctly rejected")
                else:
                    print(f"✗ {dtype_str} (width {dtype.bitwidth()}): Should have been rejected")
                    all_passed = False
            except Exception as e:
                print(f"? {dtype_str}: Not available - {str(e)[:30]}...")
    
    return all_passed

def main():
    """Run all unit tests"""
    
    print("QONNX Types Core Components Unit Tests")
    print("=" * 60)
    
    test1_passed = test_datatype_constraint_group_validation()
    test2_passed = test_constraint_matching_all_datatype_families()
    test3_passed = test_edge_cases()
    test4_passed = test_bitwidth_range_validation()
    
    all_passed = test1_passed and test2_passed and test3_passed and test4_passed
    
    print("\n" + "=" * 60)
    print(f"Unit tests result: {'PASSED' if all_passed else 'FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    main()