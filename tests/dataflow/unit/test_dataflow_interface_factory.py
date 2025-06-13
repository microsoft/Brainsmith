#!/usr/bin/env python3
"""Test DataflowInterface factory method with runtime validation"""

from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata
from brainsmith.dataflow.core.interface_types import InterfaceType
from brainsmith.dataflow.core.qonnx_types import DatatypeConstraintGroup
from brainsmith.dataflow.core.dataflow_interface import DataflowInterface

def test_factory_method():
    """Test DataflowInterface factory method with constraint validation"""
    
    # Create interface metadata with constraint groups
    metadata = InterfaceMetadata(
        name="input0",
        interface_type=InterfaceType.INPUT,
        datatype_constraints=[
            DatatypeConstraintGroup("UINT", 8, 16),
            DatatypeConstraintGroup("INT", 4, 8)
        ]
    )
    
    print(f"Testing factory method for interface: {metadata.name}")
    print(f"Constraints: {metadata.get_constraint_description()}")
    
    # Test valid datatype creation
    try:
        interface = DataflowInterface.from_metadata_and_runtime_datatype(
            metadata=metadata,
            runtime_datatype="UINT8",
            tensor_dims=[1, 128, 768],
            block_dims=[1, 8, 96],
            stream_dims=[1, 1, 8]
        )
        print(f"✓ Valid UINT8: Created interface with dtype {interface.dtype.get_canonical_name()}")
    except Exception as e:
        print(f"✗ Valid UINT8 failed: {e}")
        return False
    
    # Test invalid datatype (violates constraints)
    try:
        interface = DataflowInterface.from_metadata_and_runtime_datatype(
            metadata=metadata,
            runtime_datatype="UINT32",  # Outside UINT 8-16 range
            tensor_dims=[1, 128, 768],
            block_dims=[1, 8, 96],
            stream_dims=[1, 1, 8]
        )
        print(f"✗ Invalid UINT32: Should have failed but created interface")
        return False
    except ValueError as e:
        print(f"✓ Invalid UINT32: Correctly rejected - {e}")
    except Exception as e:
        print(f"✗ Invalid UINT32: Unexpected error - {e}")
        return False
    
    # Test invalid datatype string
    try:
        interface = DataflowInterface.from_metadata_and_runtime_datatype(
            metadata=metadata,
            runtime_datatype="INVALID_TYPE",
            tensor_dims=[1, 128, 768],
            block_dims=[1, 8, 96],
            stream_dims=[1, 1, 8]
        )
        print(f"✗ Invalid datatype string: Should have failed but created interface")
        return False
    except ValueError as e:
        print(f"✓ Invalid datatype string: Correctly rejected - {e}")
    except Exception as e:
        print(f"✗ Invalid datatype string: Unexpected error - {e}")
        return False
    
    # Test with empty constraints (should allow anything)
    empty_metadata = InterfaceMetadata(
        name="unconstrained",
        interface_type=InterfaceType.OUTPUT
    )
    
    try:
        interface = DataflowInterface.from_metadata_and_runtime_datatype(
            metadata=empty_metadata,
            runtime_datatype="FLOAT32",
            tensor_dims=[1, 128],
            block_dims=[1, 8],
            stream_dims=[1, 1]
        )
        print(f"✓ Empty constraints: Created interface with dtype {interface.dtype.get_canonical_name()}")
    except Exception as e:
        print(f"✗ Empty constraints failed: {e}")
        return False
    
    print(f"\nFactory method test: PASSED")
    return True

if __name__ == "__main__":
    test_factory_method()