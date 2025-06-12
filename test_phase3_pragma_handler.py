#!/usr/bin/env python3
"""
Test script for Phase 3 PragmaHandler refactor.

Tests the refactored PragmaHandler using the new chain-of-responsibility pattern.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from brainsmith.tools.hw_kernel_gen.rtl_parser.pragma import PragmaHandler
from brainsmith.tools.hw_kernel_gen.rtl_parser.data import (
    DatatypePragma, BDimPragma, WeightPragma, PragmaType, Interface, Direction, Port, ValidationResult
)
from brainsmith.dataflow.core.interface_types import InterfaceType


def create_test_interface(name: str, interface_type: InterfaceType = InterfaceType.INPUT) -> Interface:
    """Create a test interface for testing."""
    test_port = Port(name=f"{name}_data", direction=Direction.INPUT, width="32")
    validation_result = ValidationResult(valid=True, message="Test interface")
    
    return Interface(
        name=name,
        type=interface_type,
        ports={"data": test_port},
        validation_result=validation_result,
        metadata={}
    )


def test_chain_of_responsibility():
    """Test that pragmas are applied in chain-of-responsibility pattern."""
    print("=== Testing Chain-of-Responsibility Pattern ===")
    
    # Create pragma handler
    handler = PragmaHandler()
    
    # Create test interface
    interface = create_test_interface("in0")
    
    # Create multiple pragmas affecting the same interface
    pragmas = [
        DatatypePragma(
            type=PragmaType.DATATYPE,
            inputs=["in0", "UINT,INT", "8", "16"],
            line_number=10
        ),
        BDimPragma(
            type=PragmaType.BDIM,
            inputs=["in0", "-1", "[16]"],
            line_number=20
        ),
        WeightPragma(
            type=PragmaType.WEIGHT,
            inputs=["in0"],
            line_number=30
        )
    ]
    
    # Apply all pragmas using chain-of-responsibility
    metadata = handler.create_interface_metadata(interface, pragmas)
    
    # Verify all effects were applied
    assert metadata.name == "in0", "Metadata name should be preserved"
    assert metadata.interface_type == InterfaceType.WEIGHT, "WeightPragma should have changed interface type to WEIGHT"
    assert len(metadata.allowed_datatypes) > 0, "DatatypePragma should have added datatype constraints"
    assert metadata.chunking_strategy is not None, "BDimPragma should have set chunking strategy"
    
    print("✅ Chain-of-responsibility pattern working correctly")


def test_pragma_filtering():
    """Test that only relevant pragmas are applied to each interface."""
    print("=== Testing Pragma Filtering ===")
    
    # Create pragma handler
    handler = PragmaHandler()
    
    # Create multiple interfaces
    in_interface = create_test_interface("in0")
    out_interface = create_test_interface("out0", InterfaceType.OUTPUT)
    
    # Create pragmas for different interfaces
    pragmas = [
        DatatypePragma(
            type=PragmaType.DATATYPE,
            inputs=["in0", "UINT", "8", "8"],
            line_number=10
        ),
        DatatypePragma(
            type=PragmaType.DATATYPE,
            inputs=["out0", "INT", "16", "16"],
            line_number=11
        ),
        WeightPragma(
            type=PragmaType.WEIGHT,
            inputs=["in0"],
            line_number=20
        )
    ]
    
    # Test in0 interface - should get DATATYPE and WEIGHT pragmas
    in_metadata = handler.create_interface_metadata(in_interface, pragmas)
    assert in_metadata.interface_type == InterfaceType.WEIGHT, "in0 should be marked as WEIGHT"
    
    # Test out0 interface - should only get DATATYPE pragma
    out_metadata = handler.create_interface_metadata(out_interface, pragmas)
    assert out_metadata.interface_type == InterfaceType.OUTPUT, "out0 should remain OUTPUT type"
    
    print("✅ Pragma filtering working correctly")


def test_base_metadata_creation():
    """Test that base metadata is created correctly from interface structure."""
    print("=== Testing Base Metadata Creation ===")
    
    # Create pragma handler
    handler = PragmaHandler()
    
    # Create interface with specific port width
    interface = create_test_interface("test_interface")
    
    # Create metadata with no pragmas (should use base constraints only)
    metadata = handler.create_interface_metadata(interface, [])
    
    # Verify base metadata
    assert metadata.name == "test_interface", "Should preserve interface name"
    assert metadata.interface_type == InterfaceType.INPUT, "Should preserve interface type"
    assert len(metadata.allowed_datatypes) > 0, "Should have base datatype constraints"
    assert metadata.chunking_strategy is not None, "Should have default chunking strategy"
    
    print("✅ Base metadata creation working correctly")


def test_error_handling():
    """Test that errors in pragma application don't break the chain."""
    print("=== Testing Error Handling ===")
    
    # Create pragma handler
    handler = PragmaHandler()
    
    # Create test interface
    interface = create_test_interface("test")
    
    # Create a pragma with invalid data (will cause error during application)
    class BadPragma(DatatypePragma):
        def apply_to_interface_metadata(self, interface, metadata):
            raise Exception("Simulated pragma error")
    
    bad_pragma = BadPragma(
        type=PragmaType.DATATYPE,
        inputs=["test", "UINT", "8", "8"],
        line_number=100
    )
    
    good_pragma = WeightPragma(
        type=PragmaType.WEIGHT,
        inputs=["test"],
        line_number=101
    )
    
    # Apply pragmas - bad pragma should fail but good pragma should still work
    metadata = handler.create_interface_metadata(interface, [bad_pragma, good_pragma])
    
    # Verify that good pragma was still applied despite bad pragma error
    assert metadata.interface_type == InterfaceType.WEIGHT, "Good pragma should still be applied after bad pragma fails"
    
    print("✅ Error handling working correctly")


def main():
    """Run all tests."""
    print("Testing Phase 3 PragmaHandler Refactor")
    print("======================================")
    
    try:
        test_chain_of_responsibility()
        test_pragma_filtering()
        test_base_metadata_creation()
        test_error_handling()
        
        print("\n🎉 All Phase 3 tests passed successfully!")
        print("✅ Chain-of-responsibility pattern is working correctly")
        print("✅ Pragma filtering ensures only relevant pragmas are applied")
        print("✅ Base metadata creation works correctly")
        print("✅ Error handling prevents bad pragmas from breaking the chain")
        print("✅ PragmaHandler refactor is complete and functional")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())