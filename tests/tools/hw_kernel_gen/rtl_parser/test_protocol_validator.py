############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

import pytest
import dataclasses
import logging
from typing import List

from brainsmith.tools.hw_kernel_gen.rtl_parser.data import Port, Direction, InterfaceType, PortGroup
from brainsmith.tools.hw_kernel_gen.rtl_parser.protocol_validator import ProtocolValidator
from brainsmith.tools.hw_kernel_gen.rtl_parser.interface_scanner import InterfaceScanner

logger = logging.getLogger(__name__)

# --- Fixtures ---

@pytest.fixture
def validator():
    return ProtocolValidator()

@pytest.fixture
def global_ports() -> List[Port]:
    """Fixture for standard global control ports."""
    return [
        Port(name="ap_clk", direction=Direction.INPUT, width="1"),
        Port(name="ap_rst_n", direction=Direction.INPUT, width="1"),
    ]

@pytest.fixture
def axis_in_ports() -> List[Port]:
    """Fixture for a standard AXI-Stream input interface."""
    return [
        Port(name="in0_TDATA", direction=Direction.INPUT, width="32"),
        Port(name="in0_TVALID", direction=Direction.INPUT, width="1"),
        Port(name="in0_TREADY", direction=Direction.OUTPUT, width="1"),
        Port(name="in0_TLAST", direction=Direction.INPUT, width="1"), # Optional but common
    ]

@pytest.fixture
def axilite_write_ports():
    return [
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
    ]

@pytest.fixture
def axilite_read_ports():
    return [
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
def axilite_ports_full(axilite_write_ports, axilite_read_ports) -> List[Port]:
    """Fixture combining full AXI-Lite write and read ports."""
    return axilite_write_ports + axilite_read_ports

@pytest.fixture
def axis_in_ports_with_widths() -> List[Port]:
    """AXI-Stream input ports with specific width expressions."""
    return [
        Port(name="in_data_TDATA", direction=Direction.INPUT, width="[AXIS_WIDTH-1:0]"),
        Port(name="in_data_TVALID", direction=Direction.INPUT, width="1"),
        Port(name="in_data_TREADY", direction=Direction.OUTPUT, width="1"),
        Port(name="in_data_TLAST", direction=Direction.INPUT, width="1"), # Optional
    ]

@pytest.fixture
def axilite_ports_with_widths() -> List[Port]:
    """AXI-Lite ports with specific width expressions."""
    return [
        # Write Address Channel
        Port(name="config_AWADDR", direction=Direction.INPUT, width="[AXIL_ADDR_WIDTH-1:0]"),
        Port(name="config_AWPROT", direction=Direction.INPUT, width="[2:0]"), # Example fixed width
        Port(name="config_AWVALID", direction=Direction.INPUT, width="1"),
        Port(name="config_AWREADY", direction=Direction.OUTPUT, width="1"),
        # Write Data Channel
        Port(name="config_WDATA", direction=Direction.INPUT, width="[AXIL_DATA_WIDTH-1:0]"),
        Port(name="config_WSTRB", direction=Direction.INPUT, width="[(AXIL_DATA_WIDTH/8)-1:0]"),
        Port(name="config_WVALID", direction=Direction.INPUT, width="1"),
        Port(name="config_WREADY", direction=Direction.OUTPUT, width="1"),
        # Write Response Channel
        Port(name="config_BRESP", direction=Direction.OUTPUT, width="[1:0]"),
        Port(name="config_BVALID", direction=Direction.OUTPUT, width="1"),
        Port(name="config_BREADY", direction=Direction.INPUT, width="1"),
        # Read Address Channel
        Port(name="config_ARADDR", direction=Direction.INPUT, width="[AXIL_ADDR_WIDTH-1:0]"),
        Port(name="config_ARPROT", direction=Direction.INPUT, width="[2:0]"),
        Port(name="config_ARVALID", direction=Direction.INPUT, width="1"),
        Port(name="config_ARREADY", direction=Direction.OUTPUT, width="1"),
        # Read Data Channel
        Port(name="config_RDATA", direction=Direction.OUTPUT, width="[AXIL_DATA_WIDTH-1:0]"),
        Port(name="config_RRESP", direction=Direction.OUTPUT, width="[1:0]"),
        Port(name="config_RVALID", direction=Direction.OUTPUT, width="1"),
        Port(name="config_RREADY", direction=Direction.INPUT, width="1"),
    ]

# --- Helper Functions ---

def create_port_group(interface_type: InterfaceType, prefix: str, ports: List[Port]) -> PortGroup:
    """Helper function to create a PortGroup for testing."""
    scanner = InterfaceScanner()
    groups, unassigned = scanner.scan(ports)
    for group in groups:
        print(f"Group: {group.name}, Type: {group.interface_type.value}, Ports: {list(group.ports.keys())}")
    for port in unassigned:
        print(f"Unassigned Port: {port.name}, Direction: {port.direction.value}, Width: {port.width}")
    assert len(groups) == 1, "Expected exactly one group from scanner"
    assert len(unassigned) == 0, "Expected no unassigned ports"
    assert groups[0].interface_type == interface_type, f"Expected interface type {interface_type}, got {groups[0].interface_type}"
    assert groups[0].name == prefix, f"Expected group name {prefix}, got {groups[0].name}"
    return groups[0]


# --- Global Signal Tests ---

def test_validate_global_valid(scanner, validator):
    ports = [
        Port(name="ap_clk", direction=Direction.INPUT, width="1"),
        Port(name="ap_rst_n", direction=Direction.INPUT, width="1"),
        Port(name="ap_clk2x", direction=Direction.INPUT, width="1"), # Optional
    ]
    group = create_port_group(InterfaceType.GLOBAL_CONTROL, "ap", ports)
    result = validator.validate_global_control(group)
    assert result.valid
    assert result.message is None

def test_validate_global_missing_required(validator, scanner):
    ports = [
        Port(name="ap_clk", direction=Direction.INPUT, width="1"),
        # Missing ap_rst_n
    ]
    group = create_port_group(InterfaceType.GLOBAL_CONTROL, "ap", ports)
    result = validator.validate_global_control(group)
    assert not result.valid
    assert "Global Control: Missing required signal(s) in 'ap': {'RST_N'}" in result.message

def test_validate_global_wrong_direction(validator, scanner):
    ports = [
        Port(name="ap_clk", direction=Direction.OUTPUT, width="1"), # Wrong direction
        Port(name="ap_rst_n", direction=Direction.INPUT, width="1"),
    ]
    group = create_port_group(InterfaceType.GLOBAL_CONTROL, "ap", ports)
    result = validator.validate_global_control(group)
    assert not result.valid
    assert "Invalid global signal 'ap_clk': Incorrect direction" in result.message

# --- AXI-Stream Tests ---

@pytest.mark.parametrize("prefix, ports_list, expected_valid", [
    ("in0", [
        Port(name="in0_TDATA", direction=Direction.INPUT, width="32"),
        Port(name="in0_TVALID", direction=Direction.INPUT, width="1"),
        Port(name="in0_TREADY", direction=Direction.OUTPUT, width="1"),
        Port(name="in0_TLAST", direction=Direction.INPUT, width="1"), # Optional
    ], True),
    ("out1_v", [
        Port(name="out1_v_TDATA", direction=Direction.OUTPUT, width="64"),
        Port(name="out1_v_TVALID", direction=Direction.OUTPUT, width="1"),
        Port(name="out1_v_TREADY", direction=Direction.INPUT, width="1"),
    ], True),
    ("m_axis", [
        Port(name="m_axis_TDATA", direction=Direction.OUTPUT, width="8"),
        Port(name="m_axis_TVALID", direction=Direction.OUTPUT, width="1"),
        Port(name="m_axis_TREADY", direction=Direction.INPUT, width="1"),
    ], True), # Input based on 'm'
    ("s_axis", [
        Port(name="s_axis_TDATA", direction=Direction.INPUT, width="16"),
        Port(name="s_axis_TVALID", direction=Direction.INPUT, width="1"),
        Port(name="s_axis_TREADY", direction=Direction.OUTPUT, width="1"),
    ], True), # Output based on 's'
    ("in0", [
        Port(name="in0_TDATA", direction=Direction.INPUT, width="32"),
        Port(name="in0_TVALID", direction=Direction.INPUT, width="1"),
        # Missing TREADY
    ], False), # Missing required
    ("in0", [
        Port(name="in0_TDATA", direction=Direction.OUTPUT, width="32"), # Wrong direction
        Port(name="in0_TVALID", direction=Direction.INPUT, width="1"),
        Port(name="in0_TREADY", direction=Direction.OUTPUT, width="1"),
    ], False), # Wrong direction
    # Width check removed, so width=7 is now valid from validator perspective
    ("in0", [
        Port(name="in0_TDATA", direction=Direction.INPUT, width="7"),
        Port(name="in0_TVALID", direction=Direction.INPUT, width="1"),
        Port(name="in0_TREADY", direction=Direction.OUTPUT, width="1"),
    ], True),
])
def test_validate_axi_stream(validator, prefix, ports_list, expected_valid):
    group = create_port_group(InterfaceType.AXI_STREAM, prefix, ports_list)
    result = validator.validate_axi_stream(group)
    assert result.valid == expected_valid
    if not expected_valid:
        assert result.message is not None

def test_validate_axis_metadata(validator, axis_in_ports_with_widths):
    """Test that AXI-Stream validation extracts width metadata."""
    group = create_port_group(InterfaceType.AXI_STREAM, "in_data", axis_in_ports_with_widths)
    result = validator.validate_axi_stream(group)
    assert result.valid
    assert "data_width_expr" in group.metadata
    assert group.metadata["data_width_expr"] == "[AXIS_WIDTH-1:0]"

# --- AXI-Lite Tests ---

def test_validate_axilite_full(validator, axilite_ports_full):
    # Use create_port_group instead
    group = create_port_group(InterfaceType.AXI_LITE, "config", axilite_ports_full)
    result = validator.validate_axi_lite(group)
    assert result.valid
    assert result.message is None

def test_validate_axilite_write_only(validator, axilite_write_ports):
    # Use create_port_group instead
    group = create_port_group(InterfaceType.AXI_LITE, "config", axilite_write_ports)
    result = validator.validate_axi_lite(group)
    assert result.valid
    assert result.message is None

def test_validate_axilite_read_only(validator, axilite_read_ports):
    # Use create_port_group instead
    group = create_port_group(InterfaceType.AXI_LITE, "config", axilite_read_ports)
    result = validator.validate_axi_lite(group)
    assert result.valid
    assert result.message is None

def test_validate_axilite_missing_write_required(validator, axilite_read_ports):
    # Ensure we have enough write ports to trigger the 'has_write_channel' check,
    # but are missing a required one (AWVALID is missing here).
    write_ports_missing = [
        Port(name="config_AWADDR", direction=Direction.INPUT, width="6"),
        # Missing AWVALID
        Port(name="config_AWREADY", direction=Direction.OUTPUT, width="1"),
        Port(name="config_WDATA", direction=Direction.INPUT, width="32"),
        Port(name="config_WSTRB", direction=Direction.INPUT, width="4"),
        Port(name="config_WVALID", direction=Direction.INPUT, width="1"),
        Port(name="config_WREADY", direction=Direction.OUTPUT, width="1"),
        Port(name="config_BRESP", direction=Direction.OUTPUT, width="2"),
        Port(name="config_BVALID", direction=Direction.OUTPUT, width="1"),
        Port(name="config_BREADY", direction=Direction.INPUT, width="1"),
    ]
    ports = write_ports_missing + axilite_read_ports

    # Use create_port_group instead
    group = create_port_group(InterfaceType.AXI_LITE, "config", ports)
    result = validator.validate_axi_lite(group)
    assert not result.valid
    assert "AXI-Lite: Partial read, missing required signal(s) in 'config': {'AWVALID'}" in result.message

def test_validate_axilite_missing_read_required(validator, axilite_write_ports):
    read_ports_missing = [
        Port(name="config_ARADDR", direction=Direction.INPUT, width="6"),
        # Missing ARVALID
        Port(name="config_ARREADY", direction=Direction.OUTPUT, width="1"),
        # ... other valid read ports ...
    ]
    ports = read_ports_missing + axilite_write_ports

    # Use create_port_group instead
    group = create_port_group(InterfaceType.AXI_LITE, "config", ports)

    result = validator.validate_axi_lite(group)
    assert not result.valid
    assert "Missing required AXI-Lite read signals" in result.message
    assert "ARVALID" in result.message # Assuming ARVALID is the missing one

def test_validate_axilite_wrong_direction(validator, axilite_ports_full):
    # Modify one port's direction
    modified_ports = []
    for p in axilite_ports_full:
        if p.name == "config_AWREADY":
             # Incorrect direction (should be OUTPUT)
            modified_ports.append(dataclasses.replace(p, direction=Direction.INPUT))
        else:
            modified_ports.append(p)

    # Use create_port_group instead
    group = create_port_group(InterfaceType.AXI_LITE, "config", modified_ports)

    result = validator.validate_axi_lite(group)
    assert not result.valid
    assert "Invalid AXI-Lite signal 'config_AWREADY'" in result.message
    assert "Incorrect direction" in result.message

def test_validate_axilite_metadata(validator, axilite_ports_with_widths):
    """Test that AXI-Lite validation extracts width metadata."""
    group = create_port_group(InterfaceType.AXI_LITE, "config", axilite_ports_with_widths)
    result = validator.validate_axi_lite(group)
    assert result.valid
    assert "addr_width_expr" in group.metadata
    assert group.metadata["addr_width_expr"] == "[AXIL_ADDR_WIDTH-1:0]"
    assert "data_width_expr" in group.metadata
    assert group.metadata["data_width_expr"] == "[AXIL_DATA_WIDTH-1:0]"
    assert "strb_width_expr" in group.metadata
    assert group.metadata["strb_width_expr"] == "[(AXIL_DATA_WIDTH/8)-1:0]"


# --- General Validate Dispatch Test ---

def test_validate_dispatch(validator, global_ports, axis_in_ports, axilite_ports_full):
    # Global group
    global_group = create_port_group(InterfaceType.GLOBAL_CONTROL, "ap", global_ports)
    result_global = validator.validate(global_group)
    assert result_global.valid

    # AXI-Stream group
    axis_group = create_port_group(InterfaceType.AXI_STREAM, "in0", axis_in_ports)
    result_axis = validator.validate(axis_group)
    assert result_axis.valid

    # AXI-Lite group
    # Use create_port_group instead
    axilite_group = create_port_group(InterfaceType.AXI_LITE, "config", axilite_ports_full)
    result_axilite = validator.validate(axilite_group)
    assert result_axilite.valid

    # Unknown group
    unknown_group = PortGroup(interface_type=InterfaceType.UNKNOWN, name="unknown", ports={"foo": Port("foo", Direction.INPUT)})
    result_unknown = validator.validate(unknown_group)
    assert result_unknown.valid # Should skip validation
