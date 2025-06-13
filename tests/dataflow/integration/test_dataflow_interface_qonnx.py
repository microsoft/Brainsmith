#!/usr/bin/env python3
"""Comprehensive test for DataflowInterface QONNX integration"""

from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata
from brainsmith.dataflow.core.interface_types import InterfaceType
from brainsmith.dataflow.core.qonnx_types import DatatypeConstraintGroup
from brainsmith.dataflow.core.dataflow_interface import DataflowInterface

def test_dataflow_interface_creation():
    """Test DataflowInterface creation with valid datatypes"""
    
    # Create interface metadata with constraints
    metadata = InterfaceMetadata(
        name="test_input",
        interface_type=InterfaceType.INPUT,
        datatype_constraints=[
            DatatypeConstraintGroup("UINT", 8, 16),
            DatatypeConstraintGroup("FIXED", 8, 16)
        ]
    )
    
    print("=== Testing DataflowInterface Creation ===")
    print(f"Metadata: {metadata.name} - {metadata.get_constraint_description()}")
    
    # Test valid datatypes
    valid_tests = [
        ("UINT8", [1, 128, 768], [1, 8, 96], [1, 1, 8]),
        ("UINT16", [1, 64, 384], [1, 4, 48], [1, 1, 4]),
        ("FIXED<8,4>", [1, 32], [1, 8], [1, 2]),
        ("FIXED<12,6>", [10, 20], [2, 4], [1, 2])
    ]
    
    all_passed = True
    for dtype_str, tensor_dims, block_dims, stream_dims in valid_tests:
        try:
            interface = DataflowInterface.from_metadata_and_runtime_datatype(
                metadata=metadata,
                runtime_datatype=dtype_str,
                tensor_dims=tensor_dims,
                block_dims=block_dims,
                stream_dims=stream_dims
            )
            
            # Test interface properties
            print(f"✓ {dtype_str}: {interface}")
            print(f"  Bitwidth: {interface.dtype.bitwidth()}")
            print(f"  Canonical name: {interface.dtype.get_canonical_name()}")
            print(f"  Stream width: {interface.calculate_stream_width()} bits")
            print(f"  Memory footprint: {interface.get_memory_footprint()} bits")
            
        except Exception as e:
            print(f"✗ {dtype_str}: Failed - {e}")
            all_passed = False
    
    # Test invalid datatypes
    print("\n=== Testing Invalid Datatypes ===")
    invalid_tests = [
        "UINT32",    # Outside range
        "INT8",      # Not in constraint groups
        "FLOAT32",   # Not in constraint groups
        "INVALID"    # Invalid type
    ]
    
    for dtype_str in invalid_tests:
        try:
            interface = DataflowInterface.from_metadata_and_runtime_datatype(
                metadata=metadata,
                runtime_datatype=dtype_str,
                tensor_dims=[1, 128],
                block_dims=[1, 8],
                stream_dims=[1, 1]
            )
            print(f"✗ {dtype_str}: Should have failed but succeeded")
            all_passed = False
        except ValueError as e:
            print(f"✓ {dtype_str}: Correctly rejected - {str(e)[:80]}...")
        except Exception as e:
            print(f"? {dtype_str}: Unexpected error - {e}")
    
    return all_passed

def test_dataflow_interface_methods():
    """Test DataflowInterface methods with QONNX types"""
    
    print("\n=== Testing DataflowInterface Methods ===")
    
    # Create a test interface
    metadata = InterfaceMetadata(
        name="method_test",
        interface_type=InterfaceType.INPUT,
        datatype_constraints=[DatatypeConstraintGroup("UINT", 8, 8)]
    )
    
    interface = DataflowInterface.from_metadata_and_runtime_datatype(
        metadata=metadata,
        runtime_datatype="UINT8",
        tensor_dims=[1, 128, 768],
        block_dims=[1, 16, 96],
        stream_dims=[1, 2, 8]
    )
    
    all_passed = True
    
    # Test various methods
    test_methods = [
        ("calculate_stream_width", lambda: interface.calculate_stream_width()),
        ("get_memory_footprint", lambda: interface.get_memory_footprint()),
        ("calculate_total_elements", lambda: interface.calculate_total_elements()),
        ("calculate_elements_per_block", lambda: interface.calculate_elements_per_block()),
        ("calculate_total_blocks", lambda: interface.calculate_total_blocks()),
        ("get_num_blocks", lambda: interface.get_num_blocks()),
        ("calculate_cII", lambda: interface.calculate_cII()),
        ("get_transfer_cycles", lambda: interface.get_transfer_cycles())
    ]
    
    for method_name, method_call in test_methods:
        try:
            result = method_call()
            print(f"✓ {method_name}: {result}")
        except Exception as e:
            print(f"✗ {method_name}: Failed - {e}")
            all_passed = False
    
    # Test validation
    try:
        validation_result = interface.validate()
        print(f"✓ validate: Valid={validation_result.is_valid}")
        if not validation_result.is_valid:
            for error in validation_result.errors:
                print(f"    Error: {error.message}")
    except Exception as e:
        print(f"✗ validate: Failed - {e}")
        all_passed = False
    
    return all_passed

def test_empty_constraints():
    """Test interfaces with empty constraints"""
    
    print("\n=== Testing Empty Constraints ===")
    
    metadata = InterfaceMetadata(
        name="unconstrained",
        interface_type=InterfaceType.OUTPUT
        # No datatype_constraints - should allow anything
    )
    
    # Test various datatypes with empty constraints
    test_types = ["UINT8", "INT16", "FLOAT32", "FIXED<8,4>", "BIPOLAR"]
    all_passed = True
    
    for dtype_str in test_types:
        try:
            interface = DataflowInterface.from_metadata_and_runtime_datatype(
                metadata=metadata,
                runtime_datatype=dtype_str,
                tensor_dims=[10, 20],
                block_dims=[2, 4],
                stream_dims=[1, 2]
            )
            print(f"✓ {dtype_str}: {interface.dtype.get_canonical_name()}")
        except Exception as e:
            print(f"✗ {dtype_str}: Failed - {e}")
            all_passed = False
    
    return all_passed

def main():
    """Run all tests"""
    print("DataflowInterface QONNX Integration Tests")
    print("=" * 50)
    
    test1_passed = test_dataflow_interface_creation()
    test2_passed = test_dataflow_interface_methods()
    test3_passed = test_empty_constraints()
    
    all_passed = test1_passed and test2_passed and test3_passed
    
    print("\n" + "=" * 50)
    print(f"Overall result: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed

if __name__ == "__main__":
    main()