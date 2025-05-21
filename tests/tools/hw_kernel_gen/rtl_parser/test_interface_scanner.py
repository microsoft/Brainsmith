############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

import pytest # Ensure pytest is imported
import re # Add import for regex
import logging # Ensure logging is imported

from brainsmith.tools.hw_kernel_gen.rtl_parser.data import Port, Direction, InterfaceType
from brainsmith.tools.hw_kernel_gen.rtl_parser.interface_scanner import InterfaceScanner
from brainsmith.tools.hw_kernel_gen.rtl_parser.protocol_validator import AXI_LITE_SUFFIXES

logger = logging.getLogger(__name__)

# --- Fixtures ---

@pytest.fixture
def scanner():
    return InterfaceScanner()

@pytest.fixture
def global_ports():
    return [
        Port(name="ap_clk", direction=Direction.INPUT, width="1"),
        Port(name="ap_rst_n", direction=Direction.INPUT, width="1"),
        Port(name="ap_clk2x", direction=Direction.INPUT, width="1"), # Optional
    ]

@pytest.fixture
def axis_in_ports():
    return [
        Port(name="in0_TDATA", direction=Direction.INPUT, width="32"),
        Port(name="in0_TVALID", direction=Direction.INPUT, width="1"),
        Port(name="in0_TREADY", direction=Direction.OUTPUT, width="1"),
        Port(name="in0_TLAST", direction=Direction.INPUT, width="1"), # Optional
    ]

@pytest.fixture
def axis_out_ports():
    return [
        Port(name="out1_TDATA", direction=Direction.OUTPUT, width="64"),
        Port(name="out1_TVALID", direction=Direction.OUTPUT, width="1"),
        Port(name="out1_TREADY", direction=Direction.INPUT, width="1"),
    ]

@pytest.fixture
def axilite_ports_full():
    return [
        # Write
        Port(name="config_AWADDR", direction=Direction.INPUT, width="6"),
        Port(name="config_AWPROT", direction=Direction.INPUT, width="3"),
        Port(name="config_AWVALID", direction=Direction.INPUT, width="1"),
        Port(name="config_AWREADY", direction=Direction.OUTPUT, width="1"),
        Port(name="config_WDATA", direction=Direction.INPUT, width="32"),
        Port(name="config_WSTRB", direction=Direction.INPUT, width="4"),
        Port(name="config_WVALID", direction=Direction.INPUT, width="1"),
        Port(name="config_WREADY", direction=Direction.OUTPUT, width="1"),
        Port(name="config_BRESP", direction=Direction.OUTPUT, width="2"),
        Port(name="config_BVALID", direction=Direction.OUTPUT, width="1"),
        Port(name="config_BREADY", direction=Direction.INPUT, width="1"),
        # Read
        Port(name="config_ARADDR", direction=Direction.INPUT, width="6"),
        Port(name="config_ARPROT", direction=Direction.INPUT, width="3"),
        Port(name="config_ARVALID", direction=Direction.INPUT, width="1"),
        Port(name="config_ARREADY", direction=Direction.OUTPUT, width="1"),
        Port(name="config_RDATA", direction=Direction.OUTPUT, width="32"),
        Port(name="config_RRESP", direction=Direction.OUTPUT, width="2"),
        Port(name="config_RVALID", direction=Direction.OUTPUT, width="1"),
        Port(name="config_RREADY", direction=Direction.INPUT, width="1"),
    ]

@pytest.fixture
def unassigned_ports_list():
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
    assert set(groups[0].ports.keys()) == {"clk", "rst_n", "clk2x"}

def test_scan_only_axis(scanner, axis_in_ports, axis_out_ports):
    all_axis_ports = axis_in_ports + axis_out_ports
    groups, remaining = scanner.scan(all_axis_ports)
    assert len(groups) == 2
    assert not remaining

    groups.sort(key=lambda g: g.name) # Sort by name for consistent checking

    assert groups[0].interface_type == InterfaceType.AXI_STREAM
    assert groups[0].name == "in0"
    # Assertions expect UPPERCASE keys
    assert set(groups[0].ports.keys()) == {"TDATA", "TVALID", "TREADY", "TLAST"}

    assert groups[1].interface_type == InterfaceType.AXI_STREAM
    assert groups[1].name == "out1"
    # Assertions expect UPPERCASE keys
    assert set(groups[1].ports.keys()) == {"TDATA", "TVALID", "TREADY"} # out1 only has these 3

def test_scan_only_axilite(scanner, axilite_ports_full):
    groups, remaining = scanner.scan(axilite_ports_full)
    assert len(groups) == 1
    assert not remaining
    assert groups[0].interface_type == InterfaceType.AXI_LITE
    assert groups[0].name == "config"
    # Assertion expects UPPERCASE keys
    expected_keys = set(AXI_LITE_SUFFIXES.keys())
    assert set(groups[0].ports.keys()) == expected_keys

def test_scan_only_unassigned(scanner, unassigned_ports_list):
    groups, remaining = scanner.scan(unassigned_ports_list)
    assert not groups # Expect no groups formed
    assert len(remaining) == len(unassigned_ports_list)
    assert {p.name for p in remaining} == {p.name for p in unassigned_ports_list}

def test_scan_mixed(scanner, global_ports, axis_in_ports, axilite_ports_full, unassigned_ports_list):
    all_ports = global_ports + axis_in_ports + axilite_ports_full + unassigned_ports_list
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
    expected_axilite_keys = set(AXI_LITE_SUFFIXES.keys())
    assert set(group_map["config"].ports.keys()) == expected_axilite_keys

def test_scan_empty(scanner):
    groups, remaining = scanner.scan([])
    assert not groups
    assert not remaining

def test_scan_axis_partial(scanner):
    ports = [
        Port(name="in0_TDATA", direction=Direction.INPUT, width="32"),
        Port(name="in0_TVALID", direction=Direction.INPUT, width="1"),
        # Missing TREADY
    ]
    groups, remaining = scanner.scan(ports)
    assert len(groups) == 1
    assert not remaining # Scanner groups potential interfaces, validator checks completeness
    assert groups[0].interface_type == InterfaceType.AXI_STREAM
    assert groups[0].name == "in0"
    assert set(groups[0].ports.keys()) == {"TDATA", "TVALID"}

def test_scan_axilite_partial(scanner):
    ports = [
        Port(name="config_AWADDR", direction=Direction.INPUT, width="6"),
        Port(name="config_AWVALID", direction=Direction.INPUT, width="1"),
        Port(name="config_AWREADY", direction=Direction.OUTPUT, width="1"),
        # Only partial write channel
    ]
    groups, remaining = scanner.scan(ports)
    assert len(groups) == 1
    assert not remaining # Scanner groups potential interfaces, validator checks completeness
    assert groups[0].interface_type == InterfaceType.AXI_LITE
    assert groups[0].name == "config"
    assert set(groups[0].ports.keys()) == {"AWADDR", "AWVALID", "AWREADY"}

def test_scan_case_insensitivity(scanner):
    ports = [
        Port(name="ap_clk", direction=Direction.INPUT, width="1"), # Lowercase global
        Port(name="ap_RST_N", direction=Direction.INPUT, width="1"), # Uppercase global
        Port(name="in0_tdata", direction=Direction.INPUT, width="32"), # Lowercase stream
        Port(name="in0_TVALID", direction=Direction.INPUT, width="1"), # Mixed case stream
        Port(name="in0_tready", direction=Direction.OUTPUT, width="1"), # Lowercase stream
        Port(name="config_awaddr", direction=Direction.INPUT, width="6"), # Mixed case lite
        Port(name="config_AWVALID", direction=Direction.INPUT, width="1"), # Mixed case lite
        Port(name="config_awready", direction=Direction.OUTPUT, width="1"), # Lowercase lite
        # Add enough required signals for AXI-Lite write channel
        Port(name="config_WDATA", direction=Direction.INPUT, width="32"),
        Port(name="config_WSTRB", direction=Direction.INPUT, width="4"),
        Port(name="config_WVALID", direction=Direction.INPUT, width="1"),
        Port(name="config_WREADY", direction=Direction.OUTPUT, width="1"),
        Port(name="config_BRESP", direction=Direction.OUTPUT, width="2"),
        Port(name="config_BVALID", direction=Direction.OUTPUT, width="1"),
        Port(name="config_BREADY", direction=Direction.INPUT, width="1"),
    ]
    groups, remaining = scanner.scan(ports)
    assert len(groups) == 3 # ap, in0, config
    assert not remaining

    group_map = {g.name: g for g in groups}
    assert "ap" in group_map
    assert "in0" in group_map
    assert "config" in group_map

    # Updated to expect canonical suffix keys (as defined in the suffix dictionaries)
    # Global signals should use lowercase (based on GLOBAL_SIGNAL_SUFFIXES keys)
    assert set(group_map["ap"].ports.keys()) == {"clk", "rst_n"} 
    # AXI Stream signals should use uppercase (based on AXI_STREAM_SUFFIXES keys)
    assert set(group_map["in0"].ports.keys()) == {"TDATA", "TVALID", "TREADY"}
    # AXI Lite signals should use uppercase (based on AXI_LITE_SUFFIXES keys)
    assert {"AWADDR", "AWVALID", "AWREADY", "WDATA", "WSTRB", "WVALID", "WREADY", "BRESP", "BVALID", "BREADY"}.issubset(group_map["config"].ports.keys())

def test_scan_vivado_suffixes(scanner):
    ports = [
        Port(name="in0_V_TDATA", direction=Direction.INPUT, width="32"),
        Port(name="in0_V_TVALID", direction=Direction.INPUT, width="1"),
        Port(name="in0_V_TREADY", direction=Direction.OUTPUT, width="1"),
        Port(name="out0_V_TDATA", direction=Direction.OUTPUT, width="32"),
        Port(name="out0_V_TVALID", direction=Direction.OUTPUT, width="1"),
        Port(name="out0_V_TREADY", direction=Direction.INPUT, width="1"),
    ]
    groups, remaining = scanner.scan(ports)
    assert len(groups) == 2
    assert not remaining
    groups.sort(key=lambda g: g.name)
    assert groups[0].name == "in0_V"
    assert groups[1].name == "out0_V"
    # Assertions expect UPPERCASE keys
    assert set(groups[0].ports.keys()) == {"TDATA", "TVALID", "TREADY"}
    assert set(groups[1].ports.keys()) == {"TDATA", "TVALID", "TREADY"}
