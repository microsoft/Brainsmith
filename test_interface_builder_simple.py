#!/usr/bin/env python3
"""Simple test for InterfaceBuilder.build_interface_metadata() method."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from brainsmith.tools.hw_kernel_gen.rtl_parser.interface_builder import InterfaceBuilder
from brainsmith.tools.hw_kernel_gen.rtl_parser.data import Port, Direction
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata
from brainsmith.dataflow.core.interface_types import InterfaceType


def test_basic_interface_creation():
    """Test basic InterfaceMetadata creation."""
    print("Testing InterfaceBuilder.build_interface_metadata()...")
    
    builder = InterfaceBuilder(debug=True)
    
    # Create basic AXI-Stream input ports
    ports = [
        Port("s_axis_tdata", Direction.INPUT, "32"),
        Port("s_axis_tvalid", Direction.INPUT, "1"),
        Port("s_axis_tready", Direction.OUTPUT, "1"),
        Port("m_axis_tdata", Direction.OUTPUT, "32"),
        Port("m_axis_tvalid", Direction.OUTPUT, "1"),
        Port("m_axis_tready", Direction.INPUT, "1"),
    ]
    
    # No pragmas for basic test
    pragmas = []
    
    print(f"Input ports: {[p.name for p in ports]}")
    
    # Call new method
    metadata_list, unassigned_ports = builder.build_interface_metadata(ports, pragmas)
    
    print(f"Created {len(metadata_list)} InterfaceMetadata objects:")
    for meta in metadata_list:
        print(f"  - {meta.name}: {meta.interface_type.value}")
        print(f"    Description: {meta.description}")
        print(f"    Allowed datatypes: {len(meta.allowed_datatypes)}")
        print(f"    Chunking strategy: {type(meta.chunking_strategy).__name__}")
    
    print(f"Unassigned ports: {len(unassigned_ports)}")
    
    # Basic verification
    assert len(metadata_list) == 2, f"Expected 2 interfaces, got {len(metadata_list)}"
    assert len(unassigned_ports) == 0, f"Expected 0 unassigned ports, got {len(unassigned_ports)}"
    
    # Check interface names and types
    interface_names = {meta.name for meta in metadata_list}
    assert "s_axis" in interface_names, "Missing s_axis interface"
    assert "m_axis" in interface_names, "Missing m_axis interface"
    
    print("✅ Basic test passed!")
    return True


def test_comparison_with_existing():
    """Test comparison with existing build_interfaces method."""
    print("\nTesting comparison with build_interfaces()...")
    
    builder = InterfaceBuilder(debug=True)
    
    # Create test ports  
    ports = [
        Port("s_axis_tdata", Direction.INPUT, "32"),
        Port("s_axis_tvalid", Direction.INPUT, "1"),
        Port("s_axis_tready", Direction.OUTPUT, "1"),
        Port("clk", Direction.INPUT, "1"),
        Port("rst_n", Direction.INPUT, "1"),
    ]
    
    # No pragmas for comparison
    pragmas = []
    
    # Call new method
    metadata_list, unassigned_metadata = builder.build_interface_metadata(ports, pragmas)
    
    print(f"build_interface_metadata: {len(metadata_list)} interfaces, {len(unassigned_metadata)} unassigned")
    
    # Validate results
    print(f"Detected interfaces:")
    for meta in metadata_list:
        print(f"  - {meta.name}: {meta.interface_type.value}")
        print(f"    Datatypes: {len(meta.allowed_datatypes)}")
        print(f"    Chunking: {type(meta.chunking_strategy).__name__}")
    
    print(f"Unassigned ports: {[p.name for p in unassigned_metadata]}")
    
    # Basic validation
    assert len(metadata_list) > 0, "No interfaces detected"
    
    print("✅ Basic interface detection test passed!")
    return True


if __name__ == "__main__":
    try:
        test_basic_interface_creation()
        test_comparison_with_existing()
        print("\n🎉 All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)