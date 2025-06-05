############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

import pytest
import logging
from typing import List

from brainsmith.tools.hw_kernel_gen.rtl_parser.data import Port, Direction, InterfaceType
from brainsmith.tools.hw_kernel_gen.rtl_parser.interface_scanner import InterfaceScanner
from brainsmith.tools.hw_kernel_gen.rtl_parser.protocol_validator import AXI_LITE_SUFFIXES

logger = logging.getLogger(__name__)

# =============================================================================
# LOCAL FIXTURES (Not shared - specific to interface scanner tests)
# =============================================================================

@pytest.fixture
def unassigned_ports_list():
    """Returns a list of ports that don't belong to any standard interface."""
    return [
        Port(name="custom_signal", direction=Direction.INPUT, width="8"),
        Port(name="another_port", direction=Direction.OUTPUT, width="1"),
        Port(name="config_INVALID", direction=Direction.INPUT, width="1"), # Looks like AXI-Lite but isn't
        Port(name="in0_TKEEP", direction=Direction.INPUT, width="4"), # Looks like AXI-Stream but isn't supported suffix
    ]

# --- Tests ---

def test_scan_only_global(scanner, global_ports):
    groups, remaining = scanner.scan(global_ports)
    assert len(groups) == 1
    assert not remaining
    assert groups[0].interface_type == InterfaceType.GLOBAL_CONTROL
    assert groups[0].name == "ap"
    assert set(groups[0].ports.keys()) == {"clk", "rst_n", "clk2x"}  # Include optional signal

def test_scan_only_axis(scanner, axi_stream_in_ports, axi_stream_out_ports):
    all_axis_ports = axi_stream_in_ports + axi_stream_out_ports
    groups, remaining = scanner.scan(all_axis_ports)
    assert len(groups) == 2
    assert not remaining

    groups.sort(key=lambda g: g.name) # Sort by name for consistent checking

    assert groups[0].interface_type == InterfaceType.AXI_STREAM
    assert groups[0].name == "in0"
    # Assertions expect UPPERCASE keys - include optional TLAST
    assert set(groups[0].ports.keys()) == {"TDATA", "TVALID", "TREADY", "TLAST"}

    assert groups[1].interface_type == InterfaceType.AXI_STREAM
    assert groups[1].name == "out1"
    # Assertions expect UPPERCASE keys
    assert set(groups[1].ports.keys()) == {"TDATA", "TVALID", "TREADY"}

def test_scan_only_axilite(scanner, axilite_config_ports):
    groups, remaining = scanner.scan(axilite_config_ports)
    assert len(groups) == 1
    assert not remaining
    assert groups[0].interface_type == InterfaceType.AXI_LITE
    assert groups[0].name == "config"
    # Check that the fixture contains the expected signals (doesn't include optional PROT signals)
    expected_keys = {"AWADDR", "AWVALID", "AWREADY", "WDATA", "WSTRB", "WVALID", "WREADY", 
                     "BRESP", "BVALID", "BREADY", "ARADDR", "ARVALID", "ARREADY", 
                     "RDATA", "RRESP", "RVALID", "RREADY"}
    assert set(groups[0].ports.keys()) == expected_keys

def test_scan_only_unassigned(scanner, unassigned_ports_list):
    groups, remaining = scanner.scan(unassigned_ports_list)
    assert not groups # Expect no groups formed
    assert len(remaining) == len(unassigned_ports_list)
    assert {p.name for p in remaining} == {p.name for p in unassigned_ports_list}

def test_scan_mixed(scanner, global_ports, axi_stream_in_ports, axilite_config_ports, unassigned_ports_list):
    all_ports = global_ports + axi_stream_in_ports + axilite_config_ports + unassigned_ports_list
    groups, remaining = scanner.scan(all_ports)

    assert len(groups) == 3 # global, in0, config
    assert len(remaining) == len(unassigned_ports_list)
    assert {p.name for p in remaining} == {p.name for p in unassigned_ports_list}

    group_map = {g.name: g for g in groups}
    assert "ap" in group_map and group_map["ap"].interface_type == InterfaceType.GLOBAL_CONTROL
    assert "in0" in group_map and group_map["in0"].interface_type == InterfaceType.AXI_STREAM
    assert "config" in group_map and group_map["config"].interface_type == InterfaceType.AXI_LITE

    # Assertions expect UPPERCASE keys
    assert set(group_map["in0"].ports.keys()) == {"TDATA", "TVALID", "TREADY", "TLAST"}
    expected_axilite_keys = {"AWADDR", "AWVALID", "AWREADY", "WDATA", "WSTRB", "WVALID", "WREADY", 
                             "BRESP", "BVALID", "BREADY", "ARADDR", "ARVALID", "ARREADY", 
                             "RDATA", "RRESP", "RVALID", "RREADY"}
    assert set(group_map["config"].ports.keys()) == expected_axilite_keys

def test_scan_empty(scanner):
    groups, remaining = scanner.scan([])
    assert not groups
    assert not remaining

def test_scan_axis_partial(scanner):
    # Test with partial AXI-Stream interface (missing TREADY)
    partial_axis_ports = [
        Port(name="in1_TDATA", direction=Direction.INPUT, width="32"),
        Port(name="in1_TVALID", direction=Direction.INPUT, width="1"),
        # Missing TREADY
    ]
    groups, remaining = scanner.scan(partial_axis_ports)
    assert len(groups) == 1  # Scanner groups partial interfaces, validator checks completeness
    assert not remaining  # All matching ports should be assigned to the group
    assert groups[0].interface_type == InterfaceType.AXI_STREAM
    assert groups[0].name == "in1"
    assert set(groups[0].ports.keys()) == {"TDATA", "TVALID"}

def test_scan_axilite_partial(scanner):
    # Test with partial AXI-Lite interface (only write address channel)
    partial_axilite_ports = [
        Port(name="config_AWADDR", direction=Direction.INPUT, width="32"),
        Port(name="config_AWVALID", direction=Direction.INPUT, width="1"),
        Port(name="config_AWREADY", direction=Direction.OUTPUT, width="1"),
        # Missing other channels
    ]
    groups, remaining = scanner.scan(partial_axilite_ports)
    assert len(groups) == 1  # Scanner groups partial interfaces, validator checks completeness
    assert not remaining  # All matching ports should be assigned to the group
    assert groups[0].interface_type == InterfaceType.AXI_LITE
    assert groups[0].name == "config"
    assert set(groups[0].ports.keys()) == {"AWADDR", "AWVALID", "AWREADY"}

def test_scan_case_insensitivity(scanner):
    # Test that scanning works with lowercase and mixed case suffixes
    case_insensitive_ports = [
        Port(name="test_tdata", direction=Direction.INPUT, width="32"),
        Port(name="test_TValid", direction=Direction.INPUT, width="1"),
        Port(name="test_TREADY", direction=Direction.OUTPUT, width="1"),
    ]
    groups, remaining = scanner.scan(case_insensitive_ports)
    assert len(groups) == 1
    assert not remaining
    assert groups[0].interface_type == InterfaceType.AXI_STREAM
    assert groups[0].name == "test"

def test_scan_vivado_suffixes(scanner):
    # Test AXI-Stream with Vivado-style suffixes (with _V)
    vivado_axis_ports = [
        Port(name="output_V_TDATA", direction=Direction.OUTPUT, width="64"),
        Port(name="output_V_TVALID", direction=Direction.OUTPUT, width="1"),
        Port(name="output_V_TREADY", direction=Direction.INPUT, width="1"),
    ]
    groups, remaining = scanner.scan(vivado_axis_ports)
    assert len(groups) == 1
    assert not remaining
    assert groups[0].interface_type == InterfaceType.AXI_STREAM
    assert groups[0].name == "output_V"
    # Check that ports are correctly mapped
    expected_keys = {"TDATA", "TVALID", "TREADY"}
    assert set(groups[0].ports.keys()) == expected_keys

# =============================================================================
# IMPLEMENTATION DETAIL TESTS
# =============================================================================

def test_regex_generation(scanner):
    """Test that the scanner generates proper regex patterns."""
    # This is testing implementation details, but important for robustness
    patterns = scanner.regex_maps
    assert InterfaceType.GLOBAL_CONTROL in patterns
    assert InterfaceType.AXI_STREAM in patterns
    assert InterfaceType.AXI_LITE in patterns

def test_signal_normalization(scanner):
    """Test that signals are properly normalized to uppercase."""
    ports = [
        Port(name="test_tdata", direction=Direction.INPUT, width="32"),
        Port(name="test_TVALID", direction=Direction.INPUT, width="1"),
        Port(name="test_tready", direction=Direction.OUTPUT, width="1"),
    ]
    groups, remaining = scanner.scan(ports)
    assert len(groups) == 1
    group = groups[0]
    # All signal suffixes should be normalized to uppercase
    assert "TDATA" in group.ports
    assert "TVALID" in group.ports  
    assert "TREADY" in group.ports
    # Original case should not be present
    assert "tdata" not in group.ports
    assert "tready" not in group.ports

# =============================================================================
# EDGE CASE TESTS
# =============================================================================

def test_scan_duplicate_prefixes_different_types(scanner):
    """Test behavior when same prefix is used for different interface types."""
    # This should not happen in well-designed modules, but test robustness
    conflicting_ports = [
        # Global control with "test" prefix
        Port(name="test_clk", direction=Direction.INPUT, width="1"),
        Port(name="test_rst_n", direction=Direction.INPUT, width="1"),
        # AXI-Stream with same "test" prefix
        Port(name="test_TDATA", direction=Direction.INPUT, width="32"),
        Port(name="test_TVALID", direction=Direction.INPUT, width="1"),
        Port(name="test_TREADY", direction=Direction.OUTPUT, width="1"),
    ]
    groups, remaining = scanner.scan(conflicting_ports)
    # Should prefer one interface type over another (implementation dependent)
    # At minimum, should not crash and should handle gracefully
    assert isinstance(groups, list)
    assert isinstance(remaining, list)
    # Total ports should be preserved
    total_assigned = sum(len(g.ports) for g in groups)
    assert total_assigned + len(remaining) == len(conflicting_ports)

def test_scan_empty_prefix(scanner):
    """Test behavior with ports that have empty or invalid prefixes."""
    invalid_prefix_ports = [
        Port(name="TDATA", direction=Direction.INPUT, width="32"),   # No prefix at all
        Port(name="TVALID", direction=Direction.INPUT, width="1"),   # No prefix at all  
        Port(name="TREADY", direction=Direction.OUTPUT, width="1"),  # No prefix at all
    ]
    groups, remaining = scanner.scan(invalid_prefix_ports)
    # Scanner actually creates a group with special name '<NO_PREFIX>' for these
    assert len(groups) == 1
    assert groups[0].name == "<NO_PREFIX>"
    assert groups[0].interface_type == InterfaceType.AXI_STREAM
    assert not remaining
