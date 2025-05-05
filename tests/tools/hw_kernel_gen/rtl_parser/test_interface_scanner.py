############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

import pytest
import re # Add import for regex
import logging # Add import for logging

from brainsmith.tools.hw_kernel_gen.rtl_parser.data import Port, Direction
from brainsmith.tools.hw_kernel_gen.rtl_parser.interface_types import InterfaceType
from brainsmith.tools.hw_kernel_gen.rtl_parser.interface_scanner import InterfaceScanner

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
def axis_out_ports_v():
    return [
        Port(name="out1_V_TDATA", direction=Direction.OUTPUT, width="64"),
        Port(name="out1_V_TVALID", direction=Direction.OUTPUT, width="1"),
        Port(name="out1_V_TREADY", direction=Direction.INPUT, width="1"),
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
    assert groups[0].name == "global"
    assert set(groups[0].ports.keys()) == {"ap_clk", "ap_rst_n", "ap_clk2x"}

def test_scan_only_axis(scanner, axis_in_ports, axis_out_ports_v):
    all_axis_ports = axis_in_ports + axis_out_ports_v
    groups, remaining = scanner.scan(all_axis_ports)
    assert len(groups) == 2
    assert not remaining

    groups.sort(key=lambda g: g.name) # Sort by name for consistent checking

    assert groups[0].interface_type == InterfaceType.AXI_STREAM
    assert groups[0].name == "in0"
    assert set(groups[0].ports.keys()) == {"TDATA", "TVALID", "TREADY", "TLAST"}

    assert groups[1].interface_type == InterfaceType.AXI_STREAM
    assert groups[1].name == "out1"
    assert set(groups[1].ports.keys()) == {"TDATA", "TVALID", "TREADY"}

def test_scan_only_axilite(scanner, axilite_ports_full):
    groups, remaining = scanner.scan(axilite_ports_full)
    assert len(groups) == 1
    assert not remaining
    assert groups[0].interface_type == InterfaceType.AXI_LITE
    assert groups[0].name == "config"
    assert len(groups[0].ports) == len(axilite_ports_full)
    # Extract expected base names by removing the 'config_' prefix
    expected_base_names = {p.name.replace("config_", "") for p in axilite_ports_full}
    assert set(groups[0].ports.keys()) == expected_base_names

def test_scan_only_unassigned(scanner, unassigned_ports_list):
    groups, remaining = scanner.scan(unassigned_ports_list)
    assert not groups
    assert len(remaining) == len(unassigned_ports_list)
    assert set(p.name for p in remaining) == set(p.name for p in unassigned_ports_list)

def test_scan_mixed(scanner, global_ports, axis_in_ports, axilite_ports_full, unassigned_ports_list):
    all_ports = global_ports + axis_in_ports + axilite_ports_full + unassigned_ports_list
    groups, remaining = scanner.scan(all_ports)

    assert len(groups) == 3 # Global, AXIS in0, AXILite config
    assert len(remaining) == len(unassigned_ports_list)
    assert set(p.name for p in remaining) == set(p.name for p in unassigned_ports_list)

    group_map = {g.name: g for g in groups}
    assert "global" in group_map and group_map["global"].interface_type == InterfaceType.GLOBAL_CONTROL
    assert "in0" in group_map and group_map["in0"].interface_type == InterfaceType.AXI_STREAM
    assert "config" in group_map and group_map["config"].interface_type == InterfaceType.AXI_LITE

    assert len(group_map["global"].ports) == len(global_ports)
    assert len(group_map["in0"].ports) == len(axis_in_ports)
    assert len(group_map["config"].ports) == len(axilite_ports_full)

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
