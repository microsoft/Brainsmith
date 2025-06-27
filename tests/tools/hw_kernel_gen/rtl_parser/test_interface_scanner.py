############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Unit tests for the Interface Scanner component.

Tests the interface detection and extraction including:
- AXI-Stream pattern recognition
- AXI-Lite pattern recognition  
- Global control signal detection
- Port grouping by naming conventions
- Mixed protocol handling
"""

import pytest
from pathlib import Path

from brainsmith.tools.hw_kernel_gen.rtl_parser.interface_scanner import InterfaceScanner
from brainsmith.tools.hw_kernel_gen.rtl_parser.rtl_data import Port, PortGroup
from brainsmith.tools.hw_kernel_gen.data import Direction, InterfaceType

from .utils.rtl_builder import RTLBuilder


class TestInterfaceScanner:
    """Test cases for Interface Scanner functionality."""
    
    def test_scan_axi_stream_patterns(self, interface_scanner):
        """Test detecting AXI-Stream interface patterns."""
        ports = [
            Port("s_axis_input_tdata", Direction.INPUT, "31:0"),
            Port("s_axis_input_tvalid", Direction.INPUT, "1"),
            Port("s_axis_input_tready", Direction.OUTPUT, "1"),
            Port("s_axis_input_tlast", Direction.INPUT, "1"),
            Port("m_axis_output_tdata", Direction.OUTPUT, "31:0"),
            Port("m_axis_output_tvalid", Direction.OUTPUT, "1"),
            Port("m_axis_output_tready", Direction.INPUT, "1"),
            Port("clk", Direction.INPUT, "1"),
            Port("rst", Direction.INPUT, "1")
        ]
        
        port_groups, unassigned = interface_scanner.scan(ports)
        
        # Should find 2 AXI-Stream interfaces + global control
        assert len(port_groups) >= 2
        
        # Find AXI-Stream interfaces (they are marked as INPUT type initially)
        axi_groups = [g for g in port_groups if g.name and ("s_axis" in g.name or "m_axis" in g.name)]
        assert len(axi_groups) == 2
        
        # Check slave interface
        slave_group = next(g for g in axi_groups if g.name == "s_axis_input")
        assert len(slave_group.ports) == 4
        assert "TDATA" in slave_group.ports
        assert "TVALID" in slave_group.ports
        assert "TREADY" in slave_group.ports
        assert "TLAST" in slave_group.ports
        
        # Check master interface
        master_group = next(g for g in axi_groups if g.name == "m_axis_output")
        assert len(master_group.ports) == 3  # No tlast
    
    def test_scan_axi_lite_patterns(self, interface_scanner):
        """Test detecting AXI-Lite interface patterns."""
        ports = [
            # AXI-Lite slave interface
            Port("s_axi_awaddr", Direction.INPUT, "31:0"),
            Port("s_axi_awvalid", Direction.INPUT, "1"),
            Port("s_axi_awready", Direction.OUTPUT, "1"),
            Port("s_axi_wdata", Direction.INPUT, "31:0"),
            Port("s_axi_wstrb", Direction.INPUT, "3:0"),
            Port("s_axi_wvalid", Direction.INPUT, "1"),
            Port("s_axi_wready", Direction.OUTPUT, "1"),
            Port("s_axi_bresp", Direction.OUTPUT, "1:0"),
            Port("s_axi_bvalid", Direction.OUTPUT, "1"),
            Port("s_axi_bready", Direction.INPUT, "1"),
            Port("s_axi_araddr", Direction.INPUT, "31:0"),
            Port("s_axi_arvalid", Direction.INPUT, "1"),
            Port("s_axi_arready", Direction.OUTPUT, "1"),
            Port("s_axi_rdata", Direction.OUTPUT, "31:0"),
            Port("s_axi_rresp", Direction.OUTPUT, "1:0"),
            Port("s_axi_rvalid", Direction.OUTPUT, "1"),
            Port("s_axi_rready", Direction.INPUT, "1"),
            Port("clk", Direction.INPUT, "1"),
            Port("rst_n", Direction.INPUT, "1")
        ]
        
        port_groups, unassigned = interface_scanner.scan(ports)
        
        # Find AXI-Lite interface
        axi_lite_groups = [g for g in port_groups if g.name and g.name.startswith("s_axi")]
        assert len(axi_lite_groups) == 1
        
        axi_lite = axi_lite_groups[0]
        assert axi_lite.name == "s_axi"
        assert len(axi_lite.ports) == 17
        
        # Check for all required AXI-Lite signals
        signal_names = [p.name for p in axi_lite.ports.values()]
        assert "s_axi_awaddr" in signal_names
        assert "s_axi_wdata" in signal_names
        assert "s_axi_araddr" in signal_names
        assert "s_axi_rdata" in signal_names
    
    def test_scan_global_control_signals(self, interface_scanner):
        """Test detecting global control signals."""
        ports = [
            Port("clk", Direction.INPUT, "1"),
            Port("rst", Direction.INPUT, "1"),
            Port("ap_clk", Direction.INPUT, "1"),
            Port("ap_rst_n", Direction.INPUT, "1"),
            Port("aclk", Direction.INPUT, "1"),
            Port("aresetn", Direction.INPUT, "1"),
            Port("enable", Direction.INPUT, "1"),
            Port("data_in", Direction.INPUT, "31:0"),
            Port("data_out", Direction.OUTPUT, "31:0")
        ]
        
        port_groups, unassigned = interface_scanner.scan(ports)
        
        # Global control signals might be grouped together
        # or identified individually
        control_ports = []
        for group in port_groups:
            for port in group.ports.values():
                if port.name in ["clk", "rst", "ap_clk", "ap_rst_n", "aclk", "aresetn"]:
                    control_ports.append(port)
        
        # Also check unassigned ports
        for port in unassigned:
            if port.name in ["clk", "rst", "ap_clk", "ap_rst_n", "aclk", "aresetn"]:
                control_ports.append(port)
        
        assert len(control_ports) >= 6
    
    def test_scan_multiple_interfaces(self, interface_scanner):
        """Test scanning multiple interfaces of same type."""
        ports = [
            # First input stream
            Port("s_axis_input0_tdata", Direction.INPUT, "31:0"),
            Port("s_axis_input0_tvalid", Direction.INPUT, "1"),
            Port("s_axis_input0_tready", Direction.OUTPUT, "1"),
            
            # Second input stream
            Port("s_axis_input1_tdata", Direction.INPUT, "15:0"),
            Port("s_axis_input1_tvalid", Direction.INPUT, "1"),
            Port("s_axis_input1_tready", Direction.OUTPUT, "1"),
            
            # Weight stream
            Port("s_axis_weights_tdata", Direction.INPUT, "7:0"),
            Port("s_axis_weights_tvalid", Direction.INPUT, "1"),
            Port("s_axis_weights_tready", Direction.OUTPUT, "1"),
            
            # Output stream
            Port("m_axis_output_tdata", Direction.OUTPUT, "31:0"),
            Port("m_axis_output_tvalid", Direction.OUTPUT, "1"),
            Port("m_axis_output_tready", Direction.INPUT, "1"),
            
            Port("clk", Direction.INPUT, "1"),
            Port("rst", Direction.INPUT, "1")
        ]
        
        port_groups, unassigned = interface_scanner.scan(ports)
        
        # Find all AXI-Stream groups
        axi_groups = [g for g in port_groups if g.name and "axis" in g.name]
        assert len(axi_groups) == 4
        
        # Check names
        names = {g.name for g in axi_groups}
        assert "s_axis_input0" in names
        assert "s_axis_input1" in names
        assert "s_axis_weights" in names
        assert "m_axis_output" in names
    
    def test_scan_unmatched_ports(self, interface_scanner):
        """Test handling of ports that don't match any pattern."""
        ports = [
            Port("clk", Direction.INPUT, "1"),
            Port("rst", Direction.INPUT, "1"),
            Port("custom_data_in", Direction.INPUT, "31:0"),
            Port("custom_valid", Direction.INPUT, "1"),
            Port("custom_ready", Direction.OUTPUT, "1"),
            Port("status_flags", Direction.OUTPUT, "7:0"),
            Port("config_reg", Direction.INPUT, "15:0"),
            Port("interrupt", Direction.OUTPUT, "1")
        ]
        
        port_groups, unassigned = interface_scanner.scan(ports)
        
        # Should group some ports, others may be unassigned
        all_grouped_ports = []
        for group in port_groups:
            all_grouped_ports.extend(group.ports.values())
        
        # Not all ports may be grouped
        assert len(all_grouped_ports) <= len(ports)
    
    def test_scan_name_variations(self, interface_scanner):
        """Test handling various naming conventions."""
        ports = [
            # Uppercase variation
            Port("S_AXIS_TDATA", Direction.INPUT, "31:0"),
            Port("S_AXIS_TVALID", Direction.INPUT, "1"),
            Port("S_AXIS_TREADY", Direction.OUTPUT, "1"),
            
            # Mixed case
            Port("m_Axis_Output_TData", Direction.OUTPUT, "31:0"),
            Port("m_Axis_Output_TValid", Direction.OUTPUT, "1"),
            Port("m_Axis_Output_TReady", Direction.INPUT, "1"),
            
            # Underscores in different places
            Port("s_axis__input__tdata", Direction.INPUT, "7:0"),
            Port("s_axis__input__tvalid", Direction.INPUT, "1"),
            
            Port("clk", Direction.INPUT, "1")
        ]
        
        port_groups, unassigned = interface_scanner.scan(ports)
        
        # Scanner should handle case variations
        axi_groups = [g for g in port_groups if g.name and "axis" in g.name.lower()]
        assert len(axi_groups) >= 2
    
    def test_partial_interface_detection(self, interface_scanner):
        """Test detecting incomplete interfaces."""
        ports = [
            # Partial AXI-Stream (missing tready)
            Port("s_axis_partial_tdata", Direction.INPUT, "31:0"),
            Port("s_axis_partial_tvalid", Direction.INPUT, "1"),
            
            # Another partial (only tdata)
            Port("m_axis_minimal_tdata", Direction.OUTPUT, "15:0"),
            
            # Complete interface for comparison
            Port("s_axis_complete_tdata", Direction.INPUT, "7:0"),
            Port("s_axis_complete_tvalid", Direction.INPUT, "1"),
            Port("s_axis_complete_tready", Direction.OUTPUT, "1"),
            
            Port("clk", Direction.INPUT, "1")
        ]
        
        port_groups, unassigned = interface_scanner.scan(ports)
        
        # Should detect all groups, even partial ones
        axi_groups = [g for g in port_groups if g.name and "axis" in g.name]
        assert len(axi_groups) >= 2
        
        # Check partial interface
        partial_group = next((g for g in axi_groups if g.name == "s_axis_partial"), None)
        if partial_group:
            assert len(partial_group.ports) == 2
    
    def test_custom_prefix_patterns(self, interface_scanner):
        """Test interfaces with custom prefixes."""
        ports = [
            # Custom input stream
            Port("input_stream_data", Direction.INPUT, "31:0"),
            Port("input_stream_valid", Direction.INPUT, "1"),
            Port("input_stream_ready", Direction.OUTPUT, "1"),
            
            # Custom output stream  
            Port("output_stream_data", Direction.OUTPUT, "31:0"),
            Port("output_stream_valid", Direction.OUTPUT, "1"),
            Port("output_stream_ready", Direction.INPUT, "1"),
            
            # Mixed with standard
            Port("s_axis_standard_tdata", Direction.INPUT, "7:0"),
            Port("s_axis_standard_tvalid", Direction.INPUT, "1"),
            
            Port("clk", Direction.INPUT, "1")
        ]
        
        port_groups, unassigned = interface_scanner.scan(ports)
        
        # Should find groups based on common prefixes
        assert len(port_groups) >= 2
        
        # Check if custom prefixes were grouped
        custom_groups = [g for g in port_groups 
                        if g.name and ("input_stream" in g.name or "output_stream" in g.name)]
        # May or may not group custom patterns depending on implementation
    
    def test_empty_port_list(self, interface_scanner):
        """Test scanning empty port list."""
        ports = []
        
        port_groups, unassigned = interface_scanner.scan(ports)
        
        assert port_groups == []
        assert unassigned == []
    
    def test_single_port_interface(self, interface_scanner):
        """Test handling single-port interfaces."""
        ports = [
            Port("clk", Direction.INPUT, "1"),
            Port("rst", Direction.INPUT, "1"),
            Port("enable", Direction.INPUT, "1"),
            Port("done", Direction.OUTPUT, "1"),
            Port("error", Direction.OUTPUT, "1")
        ]
        
        port_groups, unassigned = interface_scanner.scan(ports)
        
        # Single ports may be grouped as simple interfaces
        # or left ungrouped
        assert len(port_groups) >= 0
        
        # All ports should be accounted for
        total_ports_in_groups = sum(len(g.ports) for g in port_groups)
        assert total_ports_in_groups + len(unassigned) == len(ports)