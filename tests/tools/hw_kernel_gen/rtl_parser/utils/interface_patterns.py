############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""RTL patterns for interface testing.

This module provides pre-defined RTL patterns focused on interface detection,
protocol validation, and edge cases in interface parsing.
"""

from typing import List, Dict, Optional, Set
from .rtl_builder import RTLBuilder


class InterfacePatterns:
    """RTL patterns for interface testing."""
    
    @staticmethod
    def partial_axi_stream(missing_signals: List[str], 
                          prefix: str = "s_axis_data") -> str:
        """Create AXI-Stream interface with missing signals.
        
        Args:
            missing_signals: List of signals to omit (e.g., ["tready", "tlast"])
            prefix: Interface prefix
            
        Returns:
            RTL with incomplete AXI-Stream interface
        """
        builder = RTLBuilder().module("partial_axi").add_global_control()
        
        # Full signal set
        all_signals = {
            "tdata": ("input", "31:0"),
            "tvalid": ("input", ""),
            "tready": ("output", ""),
            "tlast": ("input", ""),
            "tkeep": ("input", "3:0"),
            "tstrb": ("input", "3:0"),
            "tid": ("input", "7:0"),
            "tdest": ("input", "3:0"),
            "tuser": ("input", "0:0")
        }
        
        # Add signals except missing ones
        for signal, (direction, width) in all_signals.items():
            if signal not in missing_signals:
                port_name = f"{prefix}_{signal}"
                builder.port(port_name, direction, width if width else "1")
        
        # Add output to make valid module
        if prefix.startswith("s_"):
            builder.axi_stream_master("m_axis_out", "32")
        else:
            builder.axi_stream_slave("s_axis_in", "32")
        
        return builder.build()
    
    @staticmethod
    def mixed_protocols() -> str:
        """Module with multiple protocol types mixed."""
        return (RTLBuilder()
                .module("mixed_protocols")
                .add_global_control()
                # AXI-Stream input
                .axi_stream_slave("s_axis_data", "32")
                # AXI-Lite control
                .axi_lite_slave("s_axi_control", "16", "32")
                # Custom interface that looks like AXI but isn't
                .port("custom_data", "input", "31:0")
                .port("custom_valid", "input")
                .port("custom_accept", "output")  # Not 'ready'
                # Another AXI-Stream
                .axi_stream_master("m_axis_result", "64")
                # Partial interface
                .port("partial_tdata", "input", "15:0")
                .port("partial_tvalid", "input")
                # No tready - incomplete
                .build())
    
    @staticmethod
    def custom_interface(signal_pattern: Dict[str, Dict]) -> str:
        """Create module with custom (non-standard) interface.
        
        Args:
            signal_pattern: Dict of signal_name -> {direction, width, protocol_hint}
        
        Example:
            {
                "data_bus": {"direction": "input", "width": "64", "hint": "data"},
                "data_en": {"direction": "input", "width": "1", "hint": "control"},
                "data_ack": {"direction": "output", "width": "1", "hint": "control"}
            }
        """
        builder = RTLBuilder().module("custom_interface").add_global_control()
        
        for signal_name, properties in signal_pattern.items():
            direction = properties.get("direction", "input")
            width = properties.get("width", "1")
            builder.port(signal_name, direction, width)
        
        return builder.build()
    
    @staticmethod
    def axi_stream_variants() -> str:
        """Different valid AXI-Stream configurations."""
        return (RTLBuilder()
                .module("axi_variants")
                .add_global_control()
                .parameter("DATA_WIDTH", "32")
                .parameter("USER_WIDTH", "4")
                .parameter("ID_WIDTH", "8")
                # Minimal AXI-Stream
                .port("s_axis_minimal_tdata", "input", "DATA_WIDTH-1:0")
                .port("s_axis_minimal_tvalid", "input")
                .port("s_axis_minimal_tready", "output")
                # With TLAST
                .port("s_axis_last_tdata", "input", "DATA_WIDTH-1:0")
                .port("s_axis_last_tvalid", "input")
                .port("s_axis_last_tready", "output")
                .port("s_axis_last_tlast", "input")
                # With sideband signals
                .port("s_axis_full_tdata", "input", "DATA_WIDTH-1:0")
                .port("s_axis_full_tvalid", "input")
                .port("s_axis_full_tready", "output")
                .port("s_axis_full_tlast", "input")
                .port("s_axis_full_tkeep", "input", "(DATA_WIDTH/8)-1:0")
                .port("s_axis_full_tstrb", "input", "(DATA_WIDTH/8)-1:0")
                .port("s_axis_full_tid", "input", "ID_WIDTH-1:0")
                .port("s_axis_full_tdest", "input", "3:0")
                .port("s_axis_full_tuser", "input", "USER_WIDTH-1:0")
                # Output interface
                .axi_stream_master("m_axis_out", "DATA_WIDTH")
                .build())
    
    @staticmethod
    def mixed_directions() -> str:
        """Interfaces with incorrect or mixed signal directions."""
        return (RTLBuilder()
                .module("mixed_directions")
                .add_global_control()
                # Input interface with wrong ready direction
                .port("s_axis_bad1_tdata", "input", "31:0")
                .port("s_axis_bad1_tvalid", "input")
                .port("s_axis_bad1_tready", "input")  # Should be output
                # Output interface with wrong valid direction
                .port("m_axis_bad2_tdata", "output", "31:0")
                .port("m_axis_bad2_tvalid", "input")  # Should be output
                .port("m_axis_bad2_tready", "input")
                # Bidirectional (inout) - unusual for AXI
                .port("weird_axis_tdata", "inout", "31:0")
                .port("weird_axis_tvalid", "inout")
                .port("weird_axis_tready", "inout")
                .build())
    
    @staticmethod
    def signal_name_variations() -> str:
        """Different naming conventions that might be interfaces."""
        return (RTLBuilder()
                .module("naming_variations")
                .add_global_control()
                # Standard naming
                .axi_stream_slave("s_axis_standard", "32")
                # Uppercase
                .port("S_AXIS_UPPER_TDATA", "input", "31:0")
                .port("S_AXIS_UPPER_TVALID", "input")
                .port("S_AXIS_UPPER_TREADY", "output")
                # Underscore variations
                .port("s_axis_under_score_tdata", "input", "31:0")
                .port("s_axis_under_score_tvalid", "input") 
                .port("s_axis_under_score_tready", "output")
                # No underscore before signal type
                .port("saxiscompacttdata", "input", "31:0")
                .port("saxiscompacttvalid", "input")
                .port("saxiscompacttready", "output")
                # Different separator
                .port("s-axis-dash-tdata", "input", "31:0")
                .port("s-axis-dash-tvalid", "input")
                .port("s-axis-dash-tready", "output")
                # Prefix variations
                .port("slave_axis_prefix_tdata", "input", "31:0")
                .port("slave_axis_prefix_tvalid", "input")
                .port("slave_axis_prefix_tready", "output")
                # Output
                .axi_stream_master("m_axis_output", "32")
                .build())
    
    @staticmethod
    def incomplete_axi_lite() -> str:
        """AXI-Lite interface with missing channels."""
        return (RTLBuilder()
                .module("incomplete_axilite")
                .add_global_control()
                .parameter("ADDR_WIDTH", "32")
                .parameter("DATA_WIDTH", "32")
                # Complete write address channel
                .port("s_axi_awaddr", "input", "ADDR_WIDTH-1:0")
                .port("s_axi_awvalid", "input")
                .port("s_axi_awready", "output")
                # Missing write data channel
                # Only has partial write data
                .port("s_axi_wdata", "input", "DATA_WIDTH-1:0")
                # Missing wvalid, wready, wstrb
                # Complete write response
                .port("s_axi_bresp", "output", "1:0")
                .port("s_axi_bvalid", "output")
                .port("s_axi_bready", "input")
                # Missing entire read channel
                # Add some AXI-Stream to ensure module has interfaces
                .axi_stream_slave("s_axis_data", "32")
                .axi_stream_master("m_axis_result", "32")
                .build())
    
    @staticmethod
    def protocol_detection_threshold() -> str:
        """Test protocol detection with various completeness levels."""
        return (RTLBuilder()
                .module("detection_threshold")
                .add_global_control()
                # 100% complete AXI-Stream
                .axi_stream_slave("s_axis_complete", "32")
                # 66% complete (missing tready)
                .port("s_axis_partial1_tdata", "input", "31:0")
                .port("s_axis_partial1_tvalid", "input")
                # 33% complete (only tdata)
                .port("s_axis_partial2_tdata", "input", "31:0")
                # Has all signals but wrong directions
                .port("s_axis_wrongdir_tdata", "output", "31:0")  # Wrong
                .port("s_axis_wrongdir_tvalid", "output")  # Wrong
                .port("s_axis_wrongdir_tready", "input")   # Wrong
                # Has extra signals
                .port("s_axis_extra_tdata", "input", "31:0")
                .port("s_axis_extra_tvalid", "input")
                .port("s_axis_extra_tready", "output")
                .port("s_axis_extra_tbusy", "output")   # Non-standard
                .port("s_axis_extra_terror", "output")  # Non-standard
                # Output
                .axi_stream_master("m_axis_output", "32")
                .build())
    
    @staticmethod
    def interface_grouping_edge_cases() -> str:
        """Edge cases for interface grouping logic."""
        return (RTLBuilder()
                .module("grouping_edges")
                .add_global_control()
                # Scattered signals that should group
                .port("my_if_valid", "input")
                .port("unrelated_signal", "input", "7:0")
                .port("my_if_data", "input", "31:0")
                .port("another_unrelated", "output")
                .port("my_if_ready", "output")
                # Similar names but different interfaces
                .port("s_axis_a_tdata", "input", "31:0")
                .port("s_axis_a_tvalid", "input")
                .port("s_axis_b_tdata", "input", "15:0")  # Different width
                .port("s_axis_b_tvalid", "input")
                # Missing middle signals
                .port("iface_0_signal", "input")
                .port("iface_2_signal", "input")  # No iface_1
                .port("iface_3_signal", "input")
                # Numeric suffixes
                .port("data_bus_0", "input", "31:0")
                .port("data_bus_1", "input", "31:0")
                .port("data_bus_2", "input", "31:0")
                .port("data_valid", "input", "2:0")  # Bit per bus?
                # Output
                .axi_stream_master("m_axis_out", "32")
                .build())
    
    @staticmethod
    def width_mismatches() -> str:
        """Interfaces with incompatible signal widths."""
        return (RTLBuilder()
                .module("width_mismatches")
                .add_global_control()
                .parameter("WIDTH1", "32")
                .parameter("WIDTH2", "64")
                # Consistent widths
                .port("s_axis_good_tdata", "input", "WIDTH1-1:0")
                .port("s_axis_good_tkeep", "input", "(WIDTH1/8)-1:0")
                .port("s_axis_good_tvalid", "input")
                .port("s_axis_good_tready", "output")
                # Inconsistent widths
                .port("s_axis_bad_tdata", "input", "WIDTH1-1:0")
                .port("s_axis_bad_tkeep", "input", "(WIDTH2/8)-1:0")  # Wrong
                .port("s_axis_bad_tvalid", "input")
                .port("s_axis_bad_tready", "output")
                # Keep width doesn't match data
                .port("s_axis_mismatch_tdata", "input", "31:0")
                .port("s_axis_mismatch_tkeep", "input", "7:0")  # Should be 3:0
                .port("s_axis_mismatch_tvalid", "input")
                .port("s_axis_mismatch_tready", "output")
                # Output
                .axi_stream_master("m_axis_out", "WIDTH1")
                .build())
    
    @staticmethod
    def global_control_variations() -> str:
        """Different global control interface patterns."""
        return (RTLBuilder()
                .module("control_variations")
                # Standard ap_clk/ap_rst_n
                .port("ap_clk", "input")
                .port("ap_rst_n", "input")
                # Alternative names
                .port("aclk", "input")
                .port("aresetn", "input")
                # Another variant
                .port("clk", "input")
                .port("rst", "input")
                .port("rstn", "input")
                # With enable
                .port("clock", "input")
                .port("reset", "input")
                .port("enable", "input")
                # Data interface
                .axi_stream_slave("s_axis_data", "32")
                .axi_stream_master("m_axis_data", "32")
                .build())
    
    @staticmethod
    def interface_without_ready() -> str:
        """Streaming interfaces missing backpressure."""
        return (RTLBuilder()
                .module("no_backpressure")
                .add_global_control()
                # Valid/data only (no ready)
                .port("stream_in_data", "input", "31:0")
                .port("stream_in_valid", "input")
                # Another without ready
                .port("fifo_write_data", "input", "63:0") 
                .port("fifo_write_enable", "input")
                # Output without ready
                .port("stream_out_data", "output", "31:0")
                .port("stream_out_valid", "output")
                # Standard interface for comparison
                .axi_stream_slave("s_axis_ref", "32")
                .axi_stream_master("m_axis_ref", "32")
                .build())
    
    @staticmethod
    def multiple_similar_interfaces() -> str:
        """Multiple interfaces with similar patterns."""
        builder = RTLBuilder().module("multi_similar").add_global_control()
        
        # Generate 5 similar input interfaces
        for i in range(5):
            prefix = f"s_axis_in{i}"
            builder.axi_stream_slave(prefix, "32")
        
        # Generate 3 similar output interfaces  
        for i in range(3):
            prefix = f"m_axis_out{i}"
            builder.axi_stream_master(prefix, "32")
        
        # Add some that break the pattern
        builder.port("s_axis_special_tdata", "input", "63:0")  # Different width
        builder.port("s_axis_special_tvalid", "input")
        builder.port("s_axis_special_tready", "output")
        
        return builder.build()
    
    @staticmethod
    def ambiguous_protocol_detection() -> str:
        """Interfaces that could match multiple protocols."""
        return (RTLBuilder()
                .module("ambiguous_protocols")
                .add_global_control()
                # Could be AXI-Stream or custom valid/ready
                .port("if1_data", "input", "31:0")
                .port("if1_valid", "input") 
                .port("if1_ready", "output")
                # Could be FIFO or custom
                .port("if2_din", "input", "31:0")
                .port("if2_wr_en", "input")
                .port("if2_full", "output")
                # Could be RAM or custom
                .port("if3_addr", "input", "9:0")
                .port("if3_data", "inout", "31:0")
                .port("if3_en", "input")
                .port("if3_we", "input")
                # Clear AXI-Stream for reference
                .axi_stream_slave("s_axis_clear", "32")
                .axi_stream_master("m_axis_clear", "32")
                .build())