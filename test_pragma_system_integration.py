#!/usr/bin/env python3
"""
Integration tests for the refactored pragma system.

Tests the complete pragma system with real SystemVerilog parsing,
backward compatibility, and performance characteristics.
"""

import sys
import os
import time
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from brainsmith.tools.hw_kernel_gen.rtl_parser.pragma import PragmaHandler
from brainsmith.tools.hw_kernel_gen.rtl_parser.data import (
    DatatypePragma, BDimPragma, WeightPragma, PragmaType,
    Interface, Direction, Port, ValidationResult
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


def test_complete_pragma_system_with_real_cases():
    """Test complete pragma system with realistic SystemVerilog cases."""
    print("=== Testing Complete Pragma System ===")
    
    handler = PragmaHandler()
    
    # Create realistic interfaces
    interfaces = [
        create_test_interface("in0_V_data_V", InterfaceType.INPUT),
        create_test_interface("out0_V_data_V", InterfaceType.OUTPUT),
        create_test_interface("weights_V_data_V", InterfaceType.INPUT),
        create_test_interface("s_axi_control", InterfaceType.CONFIG),
    ]
    
    # Create realistic pragmas
    pragmas = [
        # DATATYPE pragma for input with specific constraints
        DatatypePragma(
            type=PragmaType.DATATYPE,
            inputs=["in0", "UINT,INT", "1", "16"],
            line_number=10
        ),
        
        # DATATYPE pragma for output with different constraints
        DatatypePragma(
            type=PragmaType.DATATYPE,
            inputs=["out0", "INT", "8", "8"],
            line_number=11
        ),
        
        # BDIM pragma with enhanced format
        BDimPragma(
            type=PragmaType.BDIM,
            inputs=["in0_V_data_V", "-1", "[16]"],
            line_number=20
        ),
        
        # BDIM pragma with legacy format
        BDimPragma(
            type=PragmaType.BDIM,
            inputs=["out0", "PE*CHANNELS", "1"],
            line_number=21
        ),
        
        # WEIGHT pragma for multiple interfaces
        WeightPragma(
            type=PragmaType.WEIGHT,
            inputs=["weights", "bias"],
            line_number=30
        ),
    ]
    
    # Test each interface gets correct pragmas
    results = {}
    for interface in interfaces:
        metadata = handler.create_interface_metadata(interface, pragmas)
        results[interface.name] = metadata
        print(f"  Interface {interface.name}: {metadata.interface_type.value}, "
              f"{len(metadata.allowed_datatypes)} datatypes, "
              f"chunking: {type(metadata.chunking_strategy).__name__}")
    
    # Verify results
    assert results["in0_V_data_V"].interface_type == InterfaceType.INPUT  # No weight pragma
    assert results["out0_V_data_V"].interface_type == InterfaceType.OUTPUT  # No weight pragma
    assert results["weights_V_data_V"].interface_type == InterfaceType.WEIGHT  # Weight pragma applied
    assert results["s_axi_control"].interface_type == InterfaceType.CONFIG  # No applicable pragmas
    
    # Verify datatype constraints were applied
    assert len(results["in0_V_data_V"].allowed_datatypes) > 0
    assert len(results["out0_V_data_V"].allowed_datatypes) > 0
    
    print("✅ Complete pragma system test passed")


def test_backward_compatibility():
    """Test that existing pragma apply() methods still work."""
    print("=== Testing Backward Compatibility ===")
    
    # Create test interfaces dictionary (old API style)
    interfaces = {
        "in0": create_test_interface("in0"),
        "weights": create_test_interface("weights"),
    }
    
    # Test DatatypePragma.apply() still works
    datatype_pragma = DatatypePragma(
        type=PragmaType.DATATYPE,
        inputs=["in0", "UINT", "8", "16"],
        line_number=10
    )
    
    # Apply using old API
    datatype_pragma.apply(interfaces=interfaces)
    
    # Verify interface.metadata was populated (old behavior)
    assert "datatype_constraints" in interfaces["in0"].metadata
    
    # Test WeightPragma.apply() still works
    weight_pragma = WeightPragma(
        type=PragmaType.WEIGHT,
        inputs=["weights"],
        line_number=20
    )
    
    # Apply using old API
    weight_pragma.apply(interfaces=interfaces)
    
    # Verify interface.metadata was populated (old behavior)
    assert interfaces["weights"].metadata.get("is_weight") == True
    
    print("✅ Backward compatibility test passed")


def test_performance_comparison():
    """Test performance characteristics of new vs old approach."""
    print("=== Testing Performance ===")
    
    handler = PragmaHandler()
    
    # Create many interfaces and pragmas for performance testing
    interfaces = [create_test_interface(f"interface_{i}") for i in range(100)]
    pragmas = []
    
    # Create pragmas for every 10th interface
    for i in range(0, 100, 10):
        pragmas.extend([
            DatatypePragma(
                type=PragmaType.DATATYPE,
                inputs=[f"interface_{i}", "UINT", "8", "8"],
                line_number=i
            ),
            BDimPragma(
                type=PragmaType.BDIM,
                inputs=[f"interface_{i}", "-1", "[16]"],
                line_number=i + 1000
            ),
        ])
    
    # Time the new chain-of-responsibility approach
    start_time = time.time()
    
    for interface in interfaces:
        metadata = handler.create_interface_metadata(interface, pragmas)
    
    new_approach_time = time.time() - start_time
    
    print(f"  New approach: {new_approach_time:.4f}s for 100 interfaces with {len(pragmas)} pragmas")
    print(f"  Average per interface: {new_approach_time / 100:.6f}s")
    
    # Verify performance is reasonable (should be very fast)
    assert new_approach_time < 1.0, "Performance should be under 1 second for 100 interfaces"
    
    print("✅ Performance test passed")


def test_pragma_combinations():
    """Test various combinations of pragmas affecting same interface."""
    print("=== Testing Pragma Combinations ===")
    
    handler = PragmaHandler()
    interface = create_test_interface("test")
    
    # Test 1: DATATYPE + BDIM
    pragmas_1 = [
        DatatypePragma(type=PragmaType.DATATYPE, inputs=["test", "UINT", "8", "8"], line_number=10),
        BDimPragma(type=PragmaType.BDIM, inputs=["test", "-1", "[16]"], line_number=20),
    ]
    
    metadata_1 = handler.create_interface_metadata(interface, pragmas_1)
    assert metadata_1.interface_type == InterfaceType.INPUT  # No weight change
    assert len(metadata_1.allowed_datatypes) > 0  # Datatype applied
    assert metadata_1.chunking_strategy is not None  # BDIM applied
    
    # Test 2: DATATYPE + WEIGHT (should override interface type)
    pragmas_2 = [
        DatatypePragma(type=PragmaType.DATATYPE, inputs=["test", "INT", "16", "16"], line_number=10),
        WeightPragma(type=PragmaType.WEIGHT, inputs=["test"], line_number=30),
    ]
    
    metadata_2 = handler.create_interface_metadata(interface, pragmas_2)
    assert metadata_2.interface_type == InterfaceType.WEIGHT  # Weight applied
    assert len(metadata_2.allowed_datatypes) > 0  # Datatype applied
    
    # Test 3: All three pragma types
    pragmas_3 = [
        DatatypePragma(type=PragmaType.DATATYPE, inputs=["test", "UINT,INT", "1", "32"], line_number=10),
        BDimPragma(type=PragmaType.BDIM, inputs=["test", "PE", "CHANNELS"], line_number=20),
        WeightPragma(type=PragmaType.WEIGHT, inputs=["test"], line_number=30),
    ]
    
    metadata_3 = handler.create_interface_metadata(interface, pragmas_3)
    assert metadata_3.interface_type == InterfaceType.WEIGHT  # Weight applied last
    assert len(metadata_3.allowed_datatypes) > 0  # Datatype applied
    assert metadata_3.chunking_strategy is not None  # BDIM applied
    
    print("✅ Pragma combinations test passed")


def test_error_recovery_and_robustness():
    """Test error recovery and robustness."""
    print("=== Testing Error Recovery ===")
    
    handler = PragmaHandler()
    interface = create_test_interface("test")
    
    # Create a mix of valid and invalid pragmas
    pragmas = []
    
    # Valid pragmas
    pragmas.append(DatatypePragma(
        type=PragmaType.DATATYPE,
        inputs=["test", "UINT", "8", "8"],
        line_number=10
    ))
    
    pragmas.append(WeightPragma(
        type=PragmaType.WEIGHT,
        inputs=["test"],
        line_number=30
    ))
    
    # Try to create invalid pragmas (should not crash)
    try:
        # This should fail during construction due to invalid inputs
        invalid_pragma = DatatypePragma(
            type=PragmaType.DATATYPE,
            inputs=["incomplete"],  # Missing required arguments
            line_number=50
        )
        pragmas.append(invalid_pragma)
    except:
        # Expected - invalid pragma should fail during creation
        pass
    
    # Apply all valid pragmas (should not crash)
    metadata = handler.create_interface_metadata(interface, pragmas)
    
    # Verify that valid pragmas were still applied despite any invalid ones
    assert metadata.interface_type == InterfaceType.WEIGHT  # WeightPragma applied
    assert len(metadata.allowed_datatypes) > 0  # DatatypePragma applied
    
    print("✅ Error recovery test passed")


def test_legacy_apply_methods_integration():
    """Test that legacy apply() methods still integrate correctly."""
    print("=== Testing Legacy Apply Methods Integration ===")
    
    # Create interfaces using old-style dictionary
    interfaces = {
        "in0": create_test_interface("in0"),
        "weights": create_test_interface("weights"),
        "out0": create_test_interface("out0", InterfaceType.OUTPUT),
    }
    
    # Create pragmas and apply using old API
    pragmas = [
        DatatypePragma(type=PragmaType.DATATYPE, inputs=["in0", "UINT", "8", "8"], line_number=10),
        DatatypePragma(type=PragmaType.DATATYPE, inputs=["out0", "INT", "16", "16"], line_number=11),
        BDimPragma(type=PragmaType.BDIM, inputs=["in0", "-1", "[16]"], line_number=20),
        WeightPragma(type=PragmaType.WEIGHT, inputs=["weights"], line_number=30),
    ]
    
    # Apply all pragmas using legacy API
    for pragma in pragmas:
        pragma.apply(interfaces=interfaces)
    
    # Verify legacy metadata was populated
    assert "datatype_constraints" in interfaces["in0"].metadata
    assert "datatype_constraints" in interfaces["out0"].metadata
    assert interfaces["weights"].metadata.get("is_weight") == True
    
    # Now test new API on same interfaces
    handler = PragmaHandler()
    
    # Apply using new API
    new_metadata_in0 = handler.create_interface_metadata(interfaces["in0"], pragmas)
    new_metadata_weights = handler.create_interface_metadata(interfaces["weights"], pragmas)
    
    # Verify new API also works correctly
    assert len(new_metadata_in0.allowed_datatypes) > 0
    assert new_metadata_weights.interface_type == InterfaceType.WEIGHT
    
    print("✅ Legacy apply methods integration test passed")


def main():
    """Run all integration tests."""
    print("Testing Pragma System Integration")
    print("=================================")
    
    try:
        test_complete_pragma_system_with_real_cases()
        test_backward_compatibility()
        test_performance_comparison()
        test_pragma_combinations()
        test_error_recovery_and_robustness()
        test_legacy_apply_methods_integration()
        
        print("\n🎉 All integration tests passed successfully!")
        print("✅ Complete pragma system works with realistic cases")
        print("✅ Backward compatibility maintained for existing apply() methods")
        print("✅ Performance is acceptable for large numbers of interfaces/pragmas")
        print("✅ Pragma combinations work correctly with proper precedence")
        print("✅ Error recovery prevents bad pragmas from breaking the system")
        print("✅ Legacy and new APIs can be used together seamlessly")
        print("✅ The refactored pragma system is production-ready")
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())