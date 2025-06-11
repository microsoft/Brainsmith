#!/usr/bin/env python3
"""
Test interface builder with unified interface types
"""

import sys
sys.path.insert(0, '/home/tafk/dev/brainsmith-2')

from brainsmith.dataflow.core.interface_types import InterfaceType
from brainsmith.tools.hw_kernel_gen.rtl_parser.data import Port, Direction
from brainsmith.tools.hw_kernel_gen.rtl_parser.interface_builder import InterfaceBuilder

def test_complete_interface_building():
    """Test complete interface building pipeline with unified types"""
    
    # Create ports for a typical RTL module
    ports = [
        # Global control signals
        Port("clk", Direction.INPUT, "1"),
        Port("rst_n", Direction.INPUT, "1"),
        
        # AXI-Stream input (proper suffix naming)
        Port("in0_TDATA", Direction.INPUT, "[7:0]"),
        Port("in0_TVALID", Direction.INPUT, "1"),
        Port("in0_TREADY", Direction.OUTPUT, "1"),
        
        # AXI-Stream output (proper suffix naming)
        Port("out0_TDATA", Direction.OUTPUT, "[7:0]"),
        Port("out0_TVALID", Direction.OUTPUT, "1"),
        Port("out0_TREADY", Direction.INPUT, "1"),
        
        # AXI-Stream weights (proper suffix naming)
        Port("weights_TDATA", Direction.INPUT, "[7:0]"),
        Port("weights_TVALID", Direction.INPUT, "1"),
        Port("weights_TREADY", Direction.OUTPUT, "1"),
        
        # AXI-Lite config
        Port("s_axi_control_AWADDR", Direction.INPUT, "[31:0]"),
        Port("s_axi_control_AWVALID", Direction.INPUT, "1"),
        Port("s_axi_control_AWREADY", Direction.OUTPUT, "1"),
        Port("s_axi_control_WDATA", Direction.INPUT, "[31:0]"),
        Port("s_axi_control_WSTRB", Direction.INPUT, "[3:0]"),
        Port("s_axi_control_WVALID", Direction.INPUT, "1"),
        Port("s_axi_control_WREADY", Direction.OUTPUT, "1"),
        Port("s_axi_control_BRESP", Direction.OUTPUT, "[1:0]"),
        Port("s_axi_control_BVALID", Direction.OUTPUT, "1"),
        Port("s_axi_control_BREADY", Direction.INPUT, "1"),
        
        # Some unassigned ports
        Port("debug_signal", Direction.OUTPUT, "1"),
        Port("status_reg", Direction.OUTPUT, "[31:0]"),
    ]
    
    # Build interfaces
    builder = InterfaceBuilder(debug=True)
    interfaces, unassigned_ports = builder.build_interfaces(ports)
    
    print(f"Found {len(interfaces)} interfaces:")
    for name, interface in interfaces.items():
        print(f"  - {name}: {interface.type} (protocol: {interface.type.protocol})")
    
    print(f"\\nUnassigned ports: {[p.name for p in unassigned_ports]}")
    
    # Verify expected interfaces were created
    assert len(interfaces) >= 4  # Should have at least control, input, output, weight, config
    
    # Find interfaces by type
    control_interfaces = [iface for iface in interfaces.values() if iface.type == InterfaceType.CONTROL]
    input_interfaces = [iface for iface in interfaces.values() if iface.type == InterfaceType.INPUT]
    output_interfaces = [iface for iface in interfaces.values() if iface.type == InterfaceType.OUTPUT]
    weight_interfaces = [iface for iface in interfaces.values() if iface.type == InterfaceType.WEIGHT]
    config_interfaces = [iface for iface in interfaces.values() if iface.type == InterfaceType.CONFIG]
    
    # Verify we have the expected interface types
    assert len(control_interfaces) >= 1, f"Expected control interface, got: {[i.name for i in control_interfaces]}"
    assert len(input_interfaces) >= 1, f"Expected input interface, got: {[i.name for i in input_interfaces]}"
    assert len(output_interfaces) >= 1, f"Expected output interface, got: {[i.name for i in output_interfaces]}"
    assert len(weight_interfaces) >= 1, f"Expected weight interface, got: {[i.name for i in weight_interfaces]}"
    assert len(config_interfaces) >= 1, f"Expected config interface, got: {[i.name for i in config_interfaces]}"
    
    # Verify interface properties
    for interface in interfaces.values():
        # All interfaces should have valid protocol
        assert interface.type.protocol in ["axi_stream", "axi_lite", "global_control"], f"Invalid protocol for {interface.name}: {interface.type.protocol}"
        
        # Dataflow interfaces should be AXI-Stream
        if interface.type.is_dataflow:
            assert interface.type.protocol == "axi_stream", f"Dataflow interface {interface.name} should use AXI-Stream protocol"
        
        # Config interfaces should be AXI-Lite
        if interface.type == InterfaceType.CONFIG:
            assert interface.type.protocol == "axi_lite", f"Config interface {interface.name} should use AXI-Lite protocol"
        
        # Control interfaces should be global control
        if interface.type == InterfaceType.CONTROL:
            assert interface.type.protocol == "global_control", f"Control interface {interface.name} should use global control protocol"
    
    print("✅ Complete interface building pipeline works correctly!")
    
    return interfaces, unassigned_ports

def test_interface_metadata():
    """Test that interface metadata is preserved correctly"""
    
    # Create simple AXI-Stream input
    ports = [
        Port("input_data_TDATA", Direction.INPUT, "[15:0]"),
        Port("input_data_TVALID", Direction.INPUT, "1"),
        Port("input_data_TREADY", Direction.OUTPUT, "1"),
    ]
    
    builder = InterfaceBuilder()
    interfaces, _ = builder.build_interfaces(ports)
    
    # Should have one input interface
    assert len(interfaces) == 1
    interface = list(interfaces.values())[0]
    
    # Check metadata is preserved
    assert interface.type == InterfaceType.INPUT
    assert interface.metadata['direction'] == Direction.INPUT
    assert interface.metadata['data_width_expr'] == "[15:0]"
    
    print("✅ Interface metadata preservation works correctly!")

if __name__ == "__main__":
    test_complete_interface_building()
    test_interface_metadata()
    print("🎉 All interface builder unified type tests passed!")