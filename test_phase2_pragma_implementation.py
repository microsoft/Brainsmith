#!/usr/bin/env python3
"""
Test script for Phase 2 pragma implementation.

Tests the enhanced pragma system with all three pragma types:
- DatatypePragma
- BDimPragma  
- WeightPragma
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from brainsmith.tools.hw_kernel_gen.rtl_parser.data import (
    DatatypePragma, BDimPragma, WeightPragma, PragmaType, Interface, Direction, Port
)
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata, DataTypeConstraint
from brainsmith.dataflow.core.interface_types import InterfaceType
from brainsmith.dataflow.core.block_chunking import DefaultChunkingStrategy


def create_test_interface(name: str, interface_type: InterfaceType = InterfaceType.INPUT) -> Interface:
    """Create a test interface for testing."""
    test_port = Port(name=f"{name}_data", direction=Direction.INPUT, width="32")
    
    from brainsmith.tools.hw_kernel_gen.rtl_parser.data import ValidationResult
    validation_result = ValidationResult(valid=True, message="Test interface")
    
    return Interface(
        name=name,
        type=interface_type,
        ports={"data": test_port},
        validation_result=validation_result,
        metadata={}
    )


def create_test_metadata(interface_name: str) -> InterfaceMetadata:
    """Create base test metadata."""
    return InterfaceMetadata(
        name=interface_name,
        interface_type=InterfaceType.INPUT,
        allowed_datatypes=[
            DataTypeConstraint(finn_type="UINT8", bit_width=8, signed=False),
            DataTypeConstraint(finn_type="INT8", bit_width=8, signed=True)
        ],
        chunking_strategy=DefaultChunkingStrategy()
    )


def test_datatype_pragma():
    """Test DatatypePragma implementation."""
    print("=== Testing DatatypePragma ===")
    
    # Create test pragma
    pragma = DatatypePragma(
        type=PragmaType.DATATYPE,
        inputs=["in0", "UINT,INT", "1", "16"],
        line_number=10
    )
    
    # Create test interface
    interface = create_test_interface("in0")
    
    # Test applies_to_interface
    assert pragma.applies_to_interface(interface), "DatatypePragma should apply to matching interface"
    
    # Test with non-matching interface
    other_interface = create_test_interface("out0")
    assert not pragma.applies_to_interface(other_interface), "DatatypePragma should not apply to non-matching interface"
    
    # Test apply_to_interface_metadata
    base_metadata = create_test_metadata("in0")
    updated_metadata = pragma.apply_to_interface_metadata(interface, base_metadata)
    
    assert updated_metadata.name == "in0", "Metadata name should be preserved"
    assert len(updated_metadata.allowed_datatypes) > 0, "Should have datatype constraints"
    
    print("✅ DatatypePragma tests passed")


def test_bdim_pragma():
    """Test BDimPragma implementation.""" 
    print("=== Testing BDimPragma ===")
    
    # Test enhanced format
    pragma_enhanced = BDimPragma(
        type=PragmaType.BDIM,
        inputs=["in0_V_data_V", "-1", "[16]"],
        line_number=20
    )
    
    # Create test interface
    interface = create_test_interface("in0_V_data_V")
    
    # Test applies_to_interface
    assert pragma_enhanced.applies_to_interface(interface), "BDimPragma should apply to matching interface"
    
    # Test apply_to_interface_metadata
    base_metadata = create_test_metadata("in0_V_data_V")
    updated_metadata = pragma_enhanced.apply_to_interface_metadata(interface, base_metadata)
    
    assert updated_metadata.name == "in0_V_data_V", "Metadata name should be preserved"
    assert updated_metadata.chunking_strategy is not None, "Should have chunking strategy"
    
    # Test legacy format
    pragma_legacy = BDimPragma(
        type=PragmaType.BDIM,
        inputs=["out0", "PE*CHANNELS", "1"],
        line_number=21
    )
    
    legacy_interface = create_test_interface("out0")
    assert pragma_legacy.applies_to_interface(legacy_interface), "Legacy BDimPragma should apply to matching interface"
    
    print("✅ BDimPragma tests passed")


def test_weight_pragma():
    """Test WeightPragma implementation."""
    print("=== Testing WeightPragma ===")
    
    # Create test pragma
    pragma = WeightPragma(
        type=PragmaType.WEIGHT,
        inputs=["weights", "bias"],
        line_number=30
    )
    
    # Create test interfaces
    weights_interface = create_test_interface("weights")
    bias_interface = create_test_interface("bias")
    data_interface = create_test_interface("data")
    
    # Test applies_to_interface
    assert pragma.applies_to_interface(weights_interface), "WeightPragma should apply to weights interface"
    assert pragma.applies_to_interface(bias_interface), "WeightPragma should apply to bias interface"
    assert not pragma.applies_to_interface(data_interface), "WeightPragma should not apply to data interface"
    
    # Test apply_to_interface_metadata
    base_metadata = create_test_metadata("weights")
    updated_metadata = pragma.apply_to_interface_metadata(weights_interface, base_metadata)
    
    assert updated_metadata.name == "weights", "Metadata name should be preserved"
    assert updated_metadata.interface_type == InterfaceType.WEIGHT, "Interface type should be changed to WEIGHT"
    
    print("✅ WeightPragma tests passed")


def test_interface_name_matching():
    """Test interface name matching logic."""
    print("=== Testing Interface Name Matching ===")
    
    # Test exact match
    pragma = DatatypePragma(
        type=PragmaType.DATATYPE,
        inputs=["in0", "UINT", "8", "8"],
        line_number=40
    )
    
    exact_interface = create_test_interface("in0")
    assert pragma.applies_to_interface(exact_interface), "Should match exact interface name"
    
    # Test prefix match
    prefix_interface = create_test_interface("in0_V_data_V")
    assert pragma.applies_to_interface(prefix_interface), "Should match with prefix"
    
    # Test AXI suffix handling
    axi_pragma = DatatypePragma(
        type=PragmaType.DATATYPE,
        inputs=["in0_V_data_V", "UINT", "8", "8"],
        line_number=41
    )
    
    base_interface = create_test_interface("in0")
    assert axi_pragma.applies_to_interface(base_interface), "Should match base name from AXI suffix"
    
    print("✅ Interface name matching tests passed")


def main():
    """Run all tests."""
    print("Testing Phase 2 Pragma Implementation")
    print("=====================================")
    
    try:
        test_datatype_pragma()
        test_bdim_pragma()
        test_weight_pragma()
        test_interface_name_matching()
        
        print("\n🎉 All Phase 2 tests passed successfully!")
        print("✅ DatatypePragma, BDimPragma, and WeightPragma are working correctly")
        print("✅ Interface name matching is working correctly")
        print("✅ All pragma types can apply their effects to InterfaceMetadata")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())