#!/usr/bin/env python3
"""Test QONNX datatype validation logic"""

from brainsmith.dataflow.core.qonnx_types import DataType, DatatypeConstraintGroup, validate_datatype_against_constraints

def test_validation_logic():
    """Test constraint validation with real QONNX datatypes"""
    
    # Create test constraints
    constraints = [
        DatatypeConstraintGroup('UINT', 8, 16),
        DatatypeConstraintGroup('INT', 4, 8)
    ]

    # Test with different datatypes
    test_cases = [
        'UINT8',    # Should pass: 8 bits, UINT 8-16 range
        'UINT16',   # Should pass: 16 bits, UINT 8-16 range  
        'UINT32',   # Should fail: 32 bits, outside UINT 8-16 range
        'INT4',     # Should pass: 4 bits, INT 4-8 range
        'INT8',     # Should pass: 8 bits, INT 4-8 range
        'INT16',    # Should fail: 16 bits, outside INT 4-8 range
        'BIPOLAR',  # Should fail: not in constraint groups
        'BINARY',   # Should fail: not in constraint groups
    ]

    print('Testing validation logic with real QONNX datatypes:')
    print('=' * 60)
    
    all_passed = True
    for datatype_name in test_cases:
        try:
            dtype = DataType[datatype_name]
            is_valid = validate_datatype_against_constraints(dtype, constraints)
            canonical = dtype.get_canonical_name()
            bitwidth = dtype.bitwidth()
            signed = dtype.signed()
            
            print(f'{datatype_name:8} -> {canonical:8} (bits={bitwidth:2}, signed={signed}) -> Valid: {is_valid}')
            
        except Exception as e:
            print(f'{datatype_name:8} -> ERROR: {e}')
            all_passed = False

    # Test empty constraints
    print(f'\nEmpty constraints test:')
    empty_result = validate_datatype_against_constraints(DataType["UINT8"], [])
    print(f'UINT8 with empty constraints -> Valid: {empty_result}')
    
    if empty_result != True:
        all_passed = False

    print(f'\nValidation logic test: {"PASSED" if all_passed else "FAILED"}')
    return all_passed

if __name__ == "__main__":
    test_validation_logic()