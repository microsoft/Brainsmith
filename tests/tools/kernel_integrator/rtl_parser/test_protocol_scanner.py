############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Unit tests for the Protocol Scanner's scanning functionality.

Tests the interface detection and extraction including:
- AXI-Stream pattern recognition
- AXI-Lite pattern recognition  
- Global control signal detection
- Port grouping by naming conventions
- Mixed protocol handling
"""

import pytest
from pathlib import Path

from brainsmith.tools.kernel_integrator.rtl_parser.protocol_validator import ProtocolScanner
from brainsmith.tools.kernel_integrator.types.rtl import Port, PortGroup
from brainsmith.core.dataflow.types import InterfaceType, ProtocolType
from brainsmith.tools.kernel_integrator.types.rtl import PortDirection

from .utils.rtl_builder import RTLBuilder


class TestProtocolScannerScanning:
    """Test cases for Protocol Scanner's scanning functionality."""
    
    def test_scan_axi_stream_patterns(self, protocol_validator):
        """Test detecting AXI-Stream interface patterns."""
        ports = [
            Port("s_axis_input_tdata", PortDirection.INPUT, "31:0"),
            Port("s_axis_input_tvalid", PortDirection.INPUT, "1"),
            Port("s_axis_input_tready", PortDirection.OUTPUT, "1"),
            Port("s_axis_input_tlast", PortDirection.INPUT, "1"),
            Port("m_axis_output_tdata", PortDirection.OUTPUT, "31:0"),
            Port("m_axis_output_tvalid", PortDirection.OUTPUT, "1"),
            Port("m_axis_output_tready", PortDirection.INPUT, "1"),
            Port("clk", PortDirection.INPUT, "1"),
            Port("rst", PortDirection.INPUT, "1")
        ]
        
        interfaces_by_protocol, unassigned = protocol_validator.scan(ports)
        
        # Should find 2 AXI-Stream interfaces + global control
        assert ProtocolType.AXI_STREAM in interfaces_by_protocol
        assert ProtocolType.CONTROL in interfaces_by_protocol
        
        # Check AXI-Stream interfaces
        axi_stream_interfaces = interfaces_by_protocol[ProtocolType.AXI_STREAM]
        assert "s_axis_input" in axi_stream_interfaces
        assert "m_axis_output" in axi_stream_interfaces
        
        # Check slave interface
        slave_interface = axi_stream_interfaces["s_axis_input"]
        assert len(slave_interface.ports) == 4
        assert "TDATA" in slave_interface.ports
        assert "TVALID" in slave_interface.ports
        assert "TREADY" in slave_interface.ports
        assert "TLAST" in slave_interface.ports
        
        # Check master interface
        master_interface = axi_stream_interfaces["m_axis_output"]
        assert len(master_interface.ports) == 3  # No tlast
    
    def test_scan_axi_lite_patterns(self, protocol_validator):
        """Test detecting AXI-Lite interface patterns."""
        ports = [
            # AXI-Lite slave interface
            Port("s_axi_awaddr", PortDirection.INPUT, "31:0"),
            Port("s_axi_awvalid", PortDirection.INPUT, "1"),
            Port("s_axi_awready", PortDirection.OUTPUT, "1"),
            Port("s_axi_wdata", PortDirection.INPUT, "31:0"),
            Port("s_axi_wstrb", PortDirection.INPUT, "3:0"),
            Port("s_axi_wvalid", PortDirection.INPUT, "1"),
            Port("s_axi_wready", PortDirection.OUTPUT, "1"),
            Port("s_axi_bresp", PortDirection.OUTPUT, "1:0"),
            Port("s_axi_bvalid", PortDirection.OUTPUT, "1"),
            Port("s_axi_bready", PortDirection.INPUT, "1"),
            Port("s_axi_araddr", PortDirection.INPUT, "31:0"),
            Port("s_axi_arvalid", PortDirection.INPUT, "1"),
            Port("s_axi_arready", PortDirection.OUTPUT, "1"),
            Port("s_axi_rdata", PortDirection.OUTPUT, "31:0"),
            Port("s_axi_rresp", PortDirection.OUTPUT, "1:0"),
            Port("s_axi_rvalid", PortDirection.OUTPUT, "1"),
            Port("s_axi_rready", PortDirection.INPUT, "1"),
            Port("clk", PortDirection.INPUT, "1"),
            Port("rst_n", PortDirection.INPUT, "1")
        ]
        
        interfaces_by_protocol, unassigned = protocol_validator.scan(ports)
        
        # Find AXI-Lite interface
        assert ProtocolType.AXI_LITE in interfaces_by_protocol
        axi_lite_interfaces = interfaces_by_protocol[ProtocolType.AXI_LITE]
        assert "s_axi" in axi_lite_interfaces
        
        axi_lite = axi_lite_interfaces["s_axi"]
        assert len(axi_lite.ports) == 17
        
        # Check for all required AXI-Lite signals
        assert "AWADDR" in axi_lite.ports
        assert "WDATA" in axi_lite.ports
        assert "ARADDR" in axi_lite.ports
        assert "RDATA" in axi_lite.ports
    
    def test_scan_global_control_signals(self, protocol_validator):
        """Test detecting global control signals."""
        ports = [
            Port("clk", PortDirection.INPUT, "1"),
            Port("rst", PortDirection.INPUT, "1"),
            Port("ap_clk", PortDirection.INPUT, "1"),
            Port("ap_rst_n", PortDirection.INPUT, "1"),
            Port("aclk", PortDirection.INPUT, "1"),
            Port("aresetn", PortDirection.INPUT, "1"),
            Port("enable", PortDirection.INPUT, "1"),
            Port("data_in", PortDirection.INPUT, "31:0"),
            Port("data_out", PortDirection.OUTPUT, "31:0")
        ]
        
        # Control signals should fail scan() since not all ports can be classified
        with pytest.raises(ValueError, match="Unassigned ports detected"):
            interfaces_by_protocol, unassigned = protocol_validator.scan(ports)
    
    def test_scan_multiple_interfaces(self, protocol_validator):
        """Test scanning multiple interfaces of same type."""
        ports = [
            # First input stream
            Port("s_axis_input0_tdata", PortDirection.INPUT, "31:0"),
            Port("s_axis_input0_tvalid", PortDirection.INPUT, "1"),
            Port("s_axis_input0_tready", PortDirection.OUTPUT, "1"),
            
            # Second input stream
            Port("s_axis_input1_tdata", PortDirection.INPUT, "15:0"),
            Port("s_axis_input1_tvalid", PortDirection.INPUT, "1"),
            Port("s_axis_input1_tready", PortDirection.OUTPUT, "1"),
            
            # Weight stream
            Port("s_axis_weights_tdata", PortDirection.INPUT, "7:0"),
            Port("s_axis_weights_tvalid", PortDirection.INPUT, "1"),
            Port("s_axis_weights_tready", PortDirection.OUTPUT, "1"),
            
            # Output stream
            Port("m_axis_output_tdata", PortDirection.OUTPUT, "31:0"),
            Port("m_axis_output_tvalid", PortDirection.OUTPUT, "1"),
            Port("m_axis_output_tready", PortDirection.INPUT, "1"),
            
            Port("clk", PortDirection.INPUT, "1"),
            Port("rst", PortDirection.INPUT, "1")
        ]
        
        interfaces_by_protocol, unassigned = protocol_validator.scan(ports)
        
        # Find all AXI-Stream groups
        assert ProtocolType.AXI_STREAM in interfaces_by_protocol
        axi_stream_interfaces = interfaces_by_protocol[ProtocolType.AXI_STREAM]
        assert len(axi_stream_interfaces) == 4
        
        # Check names
        assert "s_axis_input0" in axi_stream_interfaces
        assert "s_axis_input1" in axi_stream_interfaces
        assert "s_axis_weights" in axi_stream_interfaces
        assert "m_axis_output" in axi_stream_interfaces
    
    def test_scan_unmatched_ports(self, protocol_validator):
        """Test handling of ports that don't match any pattern."""
        ports = [
            Port("clk", PortDirection.INPUT, "1"),
            Port("rst", PortDirection.INPUT, "1"),
            Port("custom_data_in", PortDirection.INPUT, "31:0"),
            Port("custom_valid", PortDirection.INPUT, "1"),
            Port("custom_ready", PortDirection.OUTPUT, "1"),
            Port("status_flags", PortDirection.OUTPUT, "7:0"),
            Port("config_reg", PortDirection.INPUT, "15:0"),
            Port("interrupt", PortDirection.OUTPUT, "1")
        ]
        
        # Should raise error for unassigned ports
        with pytest.raises(ValueError, match="Unassigned ports detected"):
            interfaces_by_protocol, unassigned = protocol_validator.scan(ports)
    
    def test_scan_name_variations(self, protocol_validator):
        """Test handling various naming conventions."""
        ports = [
            # Uppercase variation
            Port("S_AXIS_TDATA", PortDirection.INPUT, "31:0"),
            Port("S_AXIS_TVALID", PortDirection.INPUT, "1"),
            Port("S_AXIS_TREADY", PortDirection.OUTPUT, "1"),
            
            # Mixed case
            Port("m_Axis_Output_TData", PortDirection.OUTPUT, "31:0"),
            Port("m_Axis_Output_TValid", PortDirection.OUTPUT, "1"),
            Port("m_Axis_Output_TReady", PortDirection.INPUT, "1"),
            
            # Underscores in different places
            Port("s_axis__input__tdata", PortDirection.INPUT, "7:0"),
            Port("s_axis__input__tvalid", PortDirection.INPUT, "1"),
            Port("s_axis__input__tready", PortDirection.OUTPUT, "1"),
            
            Port("clk", PortDirection.INPUT, "1"),
            Port("rst_n", PortDirection.INPUT, "1")
        ]
        
        interfaces_by_protocol, unassigned = protocol_validator.scan(ports)
        
        # Scanner should handle case variations
        assert ProtocolType.AXI_STREAM in interfaces_by_protocol
        axi_stream_interfaces = interfaces_by_protocol[ProtocolType.AXI_STREAM]
        assert len(axi_stream_interfaces) >= 2
    
    def test_partial_interface_detection(self, protocol_validator):
        """Test detecting incomplete interfaces."""
        ports = [
            # Partial AXI-Stream (missing tready)
            Port("s_axis_partial_tdata", PortDirection.INPUT, "31:0"),
            Port("s_axis_partial_tvalid", PortDirection.INPUT, "1"),
            
            # Another partial (only tdata)
            Port("m_axis_minimal_tdata", PortDirection.OUTPUT, "15:0"),
            
            # Complete interface for comparison
            Port("s_axis_complete_tdata", PortDirection.INPUT, "7:0"),
            Port("s_axis_complete_tvalid", PortDirection.INPUT, "1"),
            Port("s_axis_complete_tready", PortDirection.OUTPUT, "1"),
            
            Port("clk", PortDirection.INPUT, "1")
        ]
        
        # Scan should succeed, grouping all ports
        interfaces_by_protocol, unassigned = protocol_validator.scan(ports)
        
        # Validation would fail later when building interfaces
        assert ProtocolType.AXI_STREAM in interfaces_by_protocol
        axi_stream_interfaces = interfaces_by_protocol[ProtocolType.AXI_STREAM]
        
        # Check partial interface was detected
        assert "s_axis_partial" in axi_stream_interfaces
        partial_interface = axi_stream_interfaces["s_axis_partial"]
        assert len(partial_interface.ports) == 2
    
    def test_empty_port_list(self, protocol_validator):
        """Test scanning empty port list."""
        ports = []
        
        interfaces_by_protocol, unassigned = protocol_validator.scan(ports)
        
        assert all(len(interfaces) == 0 for interfaces in interfaces_by_protocol.values())
        assert unassigned == []
    
    def test_single_port_interface(self, protocol_validator):
        """Test handling single-port interfaces."""
        ports = [
            Port("clk", PortDirection.INPUT, "1"),
            Port("rst", PortDirection.INPUT, "1"),
            Port("rst_n", PortDirection.INPUT, "1"),
        ]
        
        interfaces_by_protocol, unassigned = protocol_validator.scan(ports)
        
        # Control signals should be grouped
        assert ProtocolType.CONTROL in interfaces_by_protocol
        control_interfaces = interfaces_by_protocol[ProtocolType.CONTROL]
        
        # All ports should be accounted for
        total_ports_in_interfaces = sum(
            len(interface.ports) 
            for interfaces in interfaces_by_protocol.values() 
            for interface in interfaces.values()
        )
        assert total_ports_in_interfaces + len(unassigned) == len(ports)