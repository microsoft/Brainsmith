#!/usr/bin/env python3
"""Test updated InterfaceMetadata with constraint groups"""

from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata
from brainsmith.dataflow.core.interface_types import InterfaceType
from brainsmith.dataflow.core.qonnx_types import DatatypeConstraintGroup, DataType

def test_interface_metadata():
    """Test InterfaceMetadata with constraint groups"""
    
    # Create interface metadata with constraint groups
    metadata = InterfaceMetadata(
        name="input0",
        interface_type=InterfaceType.INPUT,
        datatype_constraints=[
            DatatypeConstraintGroup("UINT", 8, 16),
            DatatypeConstraintGroup("INT", 4, 8)
        ]
    )
    
    print(f"Interface: {metadata.name}")
    print(f"Type: {metadata.interface_type}")
    print(f"Constraints: {metadata.get_constraint_description()}")
    
    # Test datatype validation
    test_cases = [
        ("UINT8", True),   # Should pass: 8 bits, UINT 8-16 range
        ("UINT16", True),  # Should pass: 16 bits, UINT 8-16 range
        ("UINT32", False), # Should fail: 32 bits, outside range
        ("INT4", True),    # Should pass: 4 bits, INT 4-8 range
        ("INT16", False),  # Should fail: 16 bits, outside range
        ("BIPOLAR", False) # Should fail: not in constraint groups
    ]
    
    print("\nDatatype validation tests:")
    all_passed = True
    for datatype_name, expected in test_cases:
        try:
            dtype = DataType[datatype_name]
            result = metadata.validates_datatype(dtype)
            status = "PASS" if result == expected else "FAIL"
            if result != expected:
                all_passed = False
            print(f"  {datatype_name:8} -> {result:5} (expected {expected:5}) [{status}]")
        except Exception as e:
            print(f"  {datatype_name:8} -> ERROR: {e}")
            all_passed = False
    
    # Test empty constraints
    empty_metadata = InterfaceMetadata(
        name="unconstrained",
        interface_type=InterfaceType.OUTPUT
    )
    print(f"\nEmpty constraints: {empty_metadata.get_constraint_description()}")
    empty_result = empty_metadata.validates_datatype(DataType["UINT8"])
    print(f"UINT8 with empty constraints -> Valid: {empty_result}")
    
    if empty_result != True:
        all_passed = False
    
    print(f"\nInterface metadata test: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed

if __name__ == "__main__":
    test_interface_metadata()