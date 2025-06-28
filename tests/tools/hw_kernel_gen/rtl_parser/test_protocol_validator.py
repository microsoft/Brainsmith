############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Unit tests for the Protocol Validator component.

Tests the protocol validation functionality including:
- AXI-Stream protocol validation
- AXI-Lite protocol validation
- Global control signal validation
- Direction checking
- Missing/extra signal detection
"""

import pytest
from pathlib import Path

from brainsmith.tools.hw_kernel_gen.rtl_parser.protocol_validator import ProtocolValidator
from brainsmith.tools.hw_kernel_gen.rtl_parser.rtl_data import Port, PortGroup
from brainsmith.tools.hw_kernel_gen.data import InterfaceType
from brainsmith.tools.hw_kernel_gen.rtl_parser.rtl_data import PortDirection, InterfaceDirection

from .utils.rtl_builder import RTLBuilder


class TestProtocolValidator:
    """Test cases for Protocol Validator functionality."""
    
    def test_validate_axi_stream_complete(self, protocol_validator):
        """Test validating complete AXI-Stream interfaces."""
        # Create valid AXI-Stream slave interface
        slave_group = PortGroup(
            interface_type=InterfaceType.INPUT,
            name="s_axis_input",
            ports={
                "TDATA": Port("s_axis_input_tdata", PortDirection.INPUT, "31:0"),
                "TVALID": Port("s_axis_input_tvalid", PortDirection.INPUT, "1"),
                "TREADY": Port("s_axis_input_tready", PortDirection.OUTPUT, "1"),
                "TLAST": Port("s_axis_input_tlast", PortDirection.INPUT, "1")
            }
        )
        
        result = protocol_validator.validate(slave_group)
        assert result.valid
        assert slave_group.metadata.get('direction') == InterfaceDirection.INPUT
        assert slave_group.metadata.get('data_width_expr') == "31:0"
        
        # Create valid AXI-Stream master interface
        master_group = PortGroup(
            interface_type=InterfaceType.OUTPUT,
            name="m_axis_output",
            ports={
                "TDATA": Port("m_axis_output_tdata", PortDirection.OUTPUT, "15:0"),
                "TVALID": Port("m_axis_output_tvalid", PortDirection.OUTPUT, "1"),
                "TREADY": Port("m_axis_output_tready", PortDirection.INPUT, "1")
            }
        )
        
        result = protocol_validator.validate(master_group)
        assert result.valid
        assert master_group.metadata.get('direction') == InterfaceDirection.OUTPUT
        assert master_group.metadata.get('data_width_expr') == "15:0"
    
    def test_validate_axi_stream_missing_signals(self, protocol_validator):
        """Test AXI-Stream validation with missing required signals."""
        # Missing TVALID
        group = PortGroup(
            interface_type=InterfaceType.INPUT,
            name="s_axis_partial",
            ports={
                "TDATA": Port("s_axis_partial_tdata", PortDirection.INPUT, "31:0"),
                "TREADY": Port("s_axis_partial_tready", PortDirection.OUTPUT, "1")
            }
        )
        
        result = protocol_validator.validate(group)
        assert not result.valid
        assert "Missing required signal" in result.message
        assert "TVALID" in result.message
    
    def test_validate_axi_stream_wrong_directions(self, protocol_validator):
        """Test AXI-Stream validation with incorrect directions."""
        # Mixed directions (not all forward or all backward)
        group = PortGroup(
            interface_type=InterfaceType.INPUT,
            name="s_axis_mixed",
            ports={
                "TDATA": Port("s_axis_mixed_tdata", PortDirection.INPUT, "31:0"),
                "TVALID": Port("s_axis_mixed_tvalid", PortDirection.OUTPUT, "1"),  # Wrong!
                "TREADY": Port("s_axis_mixed_tready", PortDirection.OUTPUT, "1")
            }
        )
        
        result = protocol_validator.validate(group)
        assert not result.valid
        assert "Invalid signal directions" in result.message
    
    def test_validate_axi_stream_extra_signals(self, protocol_validator):
        """Test AXI-Stream validation with unexpected signals."""
        group = PortGroup(
            interface_type=InterfaceType.INPUT,
            name="s_axis_extra",
            ports={
                "TDATA": Port("s_axis_extra_tdata", PortDirection.INPUT, "31:0"),
                "TVALID": Port("s_axis_extra_tvalid", PortDirection.INPUT, "1"),
                "TREADY": Port("s_axis_extra_tready", PortDirection.OUTPUT, "1"),
                "TUSER": Port("s_axis_extra_tuser", PortDirection.INPUT, "7:0")  # Unexpected
            }
        )
        
        result = protocol_validator.validate(group)
        assert not result.valid
        assert "Unexpected signal" in result.message
        assert "TUSER" in result.message
    
    def test_validate_axi_lite_complete(self, protocol_validator):
        """Test validating complete AXI-Lite interface."""
        group = PortGroup(
            interface_type=InterfaceType.CONFIG,
            name="s_axi",
            ports={
                # Write Address Channel
                "AWADDR": Port("s_axi_awaddr", PortDirection.INPUT, "31:0"),
                "AWVALID": Port("s_axi_awvalid", PortDirection.INPUT, "1"),
                "AWREADY": Port("s_axi_awready", PortDirection.OUTPUT, "1"),
                # Write Data Channel
                "WDATA": Port("s_axi_wdata", PortDirection.INPUT, "31:0"),
                "WSTRB": Port("s_axi_wstrb", PortDirection.INPUT, "3:0"),
                "WVALID": Port("s_axi_wvalid", PortDirection.INPUT, "1"),
                "WREADY": Port("s_axi_wready", PortDirection.OUTPUT, "1"),
                # Write Response Channel
                "BRESP": Port("s_axi_bresp", PortDirection.OUTPUT, "1:0"),
                "BVALID": Port("s_axi_bvalid", PortDirection.OUTPUT, "1"),
                "BREADY": Port("s_axi_bready", PortDirection.INPUT, "1"),
                # Read Address Channel
                "ARADDR": Port("s_axi_araddr", PortDirection.INPUT, "31:0"),
                "ARVALID": Port("s_axi_arvalid", PortDirection.INPUT, "1"),
                "ARREADY": Port("s_axi_arready", PortDirection.OUTPUT, "1"),
                # Read Data Channel
                "RDATA": Port("s_axi_rdata", PortDirection.OUTPUT, "31:0"),
                "RRESP": Port("s_axi_rresp", PortDirection.OUTPUT, "1:0"),
                "RVALID": Port("s_axi_rvalid", PortDirection.OUTPUT, "1"),
                "RREADY": Port("s_axi_rready", PortDirection.INPUT, "1")
            }
        )
        
        result = protocol_validator.validate(group)
        assert result.valid
        assert group.interface_type == InterfaceType.CONFIG
    
    def test_validate_axi_lite_write_only(self, protocol_validator):
        """Test AXI-Lite validation with write channels only."""
        group = PortGroup(
            interface_type=InterfaceType.CONFIG,
            name="s_axi_wo",
            ports={
                # Write Address Channel
                "AWADDR": Port("s_axi_wo_awaddr", PortDirection.INPUT, "31:0"),
                "AWVALID": Port("s_axi_wo_awvalid", PortDirection.INPUT, "1"),
                "AWREADY": Port("s_axi_wo_awready", PortDirection.OUTPUT, "1"),
                # Write Data Channel
                "WDATA": Port("s_axi_wo_wdata", PortDirection.INPUT, "31:0"),
                "WSTRB": Port("s_axi_wo_wstrb", PortDirection.INPUT, "3:0"),
                "WVALID": Port("s_axi_wo_wvalid", PortDirection.INPUT, "1"),
                "WREADY": Port("s_axi_wo_wready", PortDirection.OUTPUT, "1"),
                # Write Response Channel
                "BRESP": Port("s_axi_wo_bresp", PortDirection.OUTPUT, "1:0"),
                "BVALID": Port("s_axi_wo_bvalid", PortDirection.OUTPUT, "1"),
                "BREADY": Port("s_axi_wo_bready", PortDirection.INPUT, "1")
            }
        )
        
        result = protocol_validator.validate(group)
        assert result.valid  # Write-only is valid
    
    def test_validate_axi_lite_partial_channel(self, protocol_validator):
        """Test AXI-Lite validation with incomplete channel."""
        # Missing WSTRB from write channel
        group = PortGroup(
            interface_type=InterfaceType.CONFIG,
            name="s_axi_partial",
            ports={
                "AWADDR": Port("s_axi_partial_awaddr", PortDirection.INPUT, "31:0"),
                "AWVALID": Port("s_axi_partial_awvalid", PortDirection.INPUT, "1"),
                "AWREADY": Port("s_axi_partial_awready", PortDirection.OUTPUT, "1"),
                "WDATA": Port("s_axi_partial_wdata", PortDirection.INPUT, "31:0"),
                # Missing WSTRB
                "WVALID": Port("s_axi_partial_wvalid", PortDirection.INPUT, "1"),
                "WREADY": Port("s_axi_partial_wready", PortDirection.OUTPUT, "1")
            }
        )
        
        result = protocol_validator.validate(group)
        assert not result.valid
        assert "Partial write" in result.message
    
    def test_validate_global_control(self, protocol_validator):
        """Test global control signal validation."""
        # Valid control group
        group = PortGroup(
            interface_type=InterfaceType.CONTROL,
            name="<NO_PREFIX>",
            ports={
                "clk": Port("clk", PortDirection.INPUT, "1"),
                "rst_n": Port("rst_n", PortDirection.INPUT, "1")
            }
        )
        
        result = protocol_validator.validate(group)
        assert result.valid
        assert group.interface_type == InterfaceType.CONTROL
        
        # Control with optional signals
        group_with_optional = PortGroup(
            interface_type=InterfaceType.CONTROL,
            name="<NO_PREFIX>",
            ports={
                "clk": Port("clk", PortDirection.INPUT, "1"),
                "rst_n": Port("rst_n", PortDirection.INPUT, "1"),
                "clk2x": Port("clk2x", PortDirection.INPUT, "1")
            }
        )
        
        result = protocol_validator.validate(group_with_optional)
        assert result.valid
    
    def test_validate_global_control_missing(self, protocol_validator):
        """Test global control validation with missing signals."""
        group = PortGroup(
            interface_type=InterfaceType.CONTROL,
            name="<NO_PREFIX>",
            ports={
                "clk": Port("clk", PortDirection.INPUT, "1")
                # Missing rst_n
            }
        )
        
        result = protocol_validator.validate(group)
        assert not result.valid
        assert "Missing required signal" in result.message
        assert "rst_n" in result.message.lower()
    
    def test_validate_unknown_interface_type(self, protocol_validator):
        """Test validation of unknown interface type."""
        # Create a group with an invalid type (using a string instead of enum)
        group = PortGroup(
            interface_type=InterfaceType.INPUT,  # Will be changed to simulate unknown
            name="unknown",
            ports={}
        )
        # Manually set to None to simulate unknown type
        group.interface_type = None
        
        result = protocol_validator.validate(group)
        assert not result.valid
        assert "Unknown interface type" in result.message
    
    def test_validate_type_determination(self, protocol_validator):
        """Test interface type determination for AXI-Stream."""
        # Weight interface
        weight_group = PortGroup(
            interface_type=InterfaceType.INPUT,  # Initially INPUT
            name="s_axis_weights",
            ports={
                "TDATA": Port("s_axis_weights_tdata", PortDirection.INPUT, "7:0"),
                "TVALID": Port("s_axis_weights_tvalid", PortDirection.INPUT, "1"),
                "TREADY": Port("s_axis_weights_tready", PortDirection.OUTPUT, "1")
            }
        )
        
        result = protocol_validator.validate(weight_group)
        assert result.valid
        assert weight_group.interface_type == InterfaceType.WEIGHT
        
        # Output interface with numbered suffix
        output_group = PortGroup(
            interface_type=InterfaceType.OUTPUT,
            name="m_axis_output0",
            ports={
                "TDATA": Port("m_axis_output0_tdata", PortDirection.OUTPUT, "31:0"),
                "TVALID": Port("m_axis_output0_tvalid", PortDirection.OUTPUT, "1"),
                "TREADY": Port("m_axis_output0_tready", PortDirection.INPUT, "1")
            }
        )
        
        result = protocol_validator.validate(output_group)
        assert result.valid
        assert output_group.interface_type == InterfaceType.OUTPUT