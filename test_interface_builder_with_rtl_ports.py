#!/usr/bin/env python3
"""Test InterfaceBuilder with ports extracted from real SystemVerilog file."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from brainsmith.tools.hw_kernel_gen.rtl_parser.interface_builder import InterfaceBuilder
from brainsmith.tools.hw_kernel_gen.rtl_parser.data import Port, Direction
from brainsmith.dataflow.core.interface_types import InterfaceType


def create_thresholding_axi_ports():
    """Create ports that represent thresholding_axi.sv interface."""
    # Based on typical thresholding_axi.sv structure
    ports = [
        # Global signals
        Port("ap_clk", Direction.INPUT, "1"),
        Port("ap_rst_n", Direction.INPUT, "1"),
        
        # AXI-Stream input interface
        Port("s_axis_V_data_V_TDATA", Direction.INPUT, "32"),
        Port("s_axis_V_data_V_TVALID", Direction.INPUT, "1"),
        Port("s_axis_V_data_V_TREADY", Direction.OUTPUT, "1"),
        
        # AXI-Stream output interface  
        Port("m_axis_V_data_V_TDATA", Direction.OUTPUT, "32"),
        Port("m_axis_V_data_V_TVALID", Direction.OUTPUT, "1"),
        Port("m_axis_V_data_V_TREADY", Direction.INPUT, "1"),
        
        # AXI-Lite configuration interface (if present)
        Port("s_axi_control_AWADDR", Direction.INPUT, "12"),
        Port("s_axi_control_AWVALID", Direction.INPUT, "1"),
        Port("s_axi_control_AWREADY", Direction.OUTPUT, "1"),
        Port("s_axi_control_WDATA", Direction.INPUT, "32"),
        Port("s_axi_control_WSTRB", Direction.INPUT, "4"),
        Port("s_axi_control_WVALID", Direction.INPUT, "1"),
        Port("s_axi_control_WREADY", Direction.OUTPUT, "1"),
        Port("s_axi_control_BRESP", Direction.OUTPUT, "2"),
        Port("s_axi_control_BVALID", Direction.OUTPUT, "1"),
        Port("s_axi_control_BREADY", Direction.INPUT, "1"),
        Port("s_axi_control_ARADDR", Direction.INPUT, "12"),
        Port("s_axi_control_ARVALID", Direction.INPUT, "1"),
        Port("s_axi_control_ARREADY", Direction.OUTPUT, "1"),
        Port("s_axi_control_RDATA", Direction.OUTPUT, "32"),
        Port("s_axi_control_RRESP", Direction.OUTPUT, "2"),
        Port("s_axi_control_RVALID", Direction.OUTPUT, "1"),
        Port("s_axi_control_RREADY", Direction.INPUT, "1"),
    ]
    
    return ports


def test_thresholding_style_interfaces():
    """Test InterfaceBuilder with thresholding-style interfaces."""
    print("Testing InterfaceBuilder with thresholding-style port structure...")
    
    builder = InterfaceBuilder(debug=True)
    ports = create_thresholding_axi_ports()
    pragmas = []  # No pragmas for basic test
    
    print(f"Input ports ({len(ports)}):")
    for port in ports:
        print(f"  - {port.name}: {port.direction.value} {port.width}")
    
    # Test the new build_interface_metadata method
    metadata_list, unassigned_ports = builder.build_interface_metadata(ports, pragmas)
    
    print(f"\n--- InterfaceBuilder Results ---")
    print(f"Created {len(metadata_list)} InterfaceMetadata objects:")
    for meta in metadata_list:
        print(f"  - {meta.name}: {meta.interface_type.value}")
        print(f"    Description: {meta.description}")
        print(f"    Allowed datatypes: {len(meta.allowed_datatypes)}")
        for dt in meta.allowed_datatypes:
            print(f"      * {dt.finn_type} ({dt.bit_width} bits, signed={dt.signed})")
        print(f"    Chunking strategy: {type(meta.chunking_strategy).__name__}")
    
    print(f"\nUnassigned ports: {len(unassigned_ports)}")
    for port in unassigned_ports:
        print(f"  - {port.name}: {port.direction.value} {port.width}")
    
    # Verify expected interfaces
    interface_names = {meta.name for meta in metadata_list}
    interface_types = {meta.interface_type for meta in metadata_list}
    
    print(f"\n--- Verification ---")
    print(f"Interface names detected: {interface_names}")
    print(f"Interface types detected: {interface_types}")
    
    # Expected interfaces for thresholding_axi
    expected_interfaces = {
        # Global control should be detected
        InterfaceType.CONTROL,
        # AXI-Stream interfaces
        InterfaceType.INPUT,  # s_axis
        InterfaceType.OUTPUT, # m_axis  
        # AXI-Lite config interface
        InterfaceType.CONFIG  # s_axi_control
    }
    
    found_types = {meta.interface_type for meta in metadata_list}
    
    if expected_interfaces.issubset(found_types):
        print("✅ All expected interface types detected")
    else:
        missing = expected_interfaces - found_types
        print(f"❌ Missing interface types: {missing}")
        return False
    
    # Verify metadata quality
    print(f"\n--- Interface Metadata Quality Check ---")
    
    for meta in metadata_list:
        print(f"Interface '{meta.name}':")
        print(f"  Type: {meta.interface_type.value}")
        print(f"  Datatypes: {len(meta.allowed_datatypes)}")
        print(f"  Chunking: {type(meta.chunking_strategy).__name__}")
        
        # Verify basic metadata properties
        if not meta.name:
            print(f"❌ Interface missing name")
            return False
        if not meta.interface_type:
            print(f"❌ Interface missing type")
            return False
        if not meta.allowed_datatypes:
            print(f"❌ Interface missing allowed datatypes")
            return False
    
    print("✅ All interface metadata has required properties")
    return True


if __name__ == "__main__":
    try:
        success = test_thresholding_style_interfaces()
        if success:
            print("\n🎉 Thresholding-style interface test passed!")
        else:
            print("\n❌ Thresholding-style interface test failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)