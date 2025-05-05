# filepath: /home/tafk/dev/brainsmith/tests/tools/hw_kernel_gen/rtl_parser/test_protocol_validator.py
import pytest

from brainsmith.tools.hw_kernel_gen.rtl_parser.data import Port, Direction
from brainsmith.tools.hw_kernel_gen.rtl_parser.interface_types import InterfaceType, PortGroup
from brainsmith.tools.hw_kernel_gen.rtl_parser.protocol_validator import ProtocolValidator

# --- Fixtures ---

@pytest.fixture
def validator():
    return ProtocolValidator()

# --- Helper Function ---

def create_port_group(interface_type: InterfaceType, name: str, ports_list: list) -> PortGroup:
    group = PortGroup(interface_type=interface_type, name=name)
    for port in ports_list:
        # Determine key based on type
        key = port.name
        if interface_type == InterfaceType.AXI_STREAM:
            # Simplified suffix extraction for test setup
            match = [s for s in ["_TDATA", "_TVALID", "_TREADY", "_TLAST"] if port.name.endswith(s)]
            if match:
                key = match[0]
        group.add_port(port, key=key)
    return group

# --- Global Signal Tests ---

def test_validate_global_valid(validator):
    ports = [
        Port(name="ap_clk", direction=Direction.INPUT, width="1"),
        Port(name="ap_rst_n", direction=Direction.INPUT, width="1"),
        Port(name="ap_clk2x", direction=Direction.INPUT, width="1"), # Optional
    ]
    group = create_port_group(InterfaceType.GLOBAL_CONTROL, "global", ports)
    result = validator.validate_global_signals(group)
    assert result.valid
    assert result.message is None

def test_validate_global_missing_required(validator):
    ports = [
        Port(name="ap_clk", direction=Direction.INPUT, width="1"),
        # Missing ap_rst_n
    ]
    group = create_port_group(InterfaceType.GLOBAL_CONTROL, "global", ports)
    result = validator.validate_global_signals(group)
    assert not result.valid
    assert "Missing required global signals: {'ap_rst_n'}" in result.message

def test_validate_global_wrong_direction(validator):
    ports = [
        Port(name="ap_clk", direction=Direction.OUTPUT, width="1"), # Wrong direction
        Port(name="ap_rst_n", direction=Direction.INPUT, width="1"),
    ]
    group = create_port_group(InterfaceType.GLOBAL_CONTROL, "global", ports)
    result = validator.validate_global_signals(group)
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

# --- AXI-Lite Tests ---

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

def test_validate_axilite_full(validator, axilite_write_ports, axilite_read_ports):
    ports = axilite_write_ports + axilite_read_ports
    group = create_port_group(InterfaceType.AXI_LITE, "config", ports)
    result = validator.validate_axi_lite(group)
    assert result.valid
    assert result.message is None

def test_validate_axilite_write_only(validator, axilite_write_ports):
    group = create_port_group(InterfaceType.AXI_LITE, "config", axilite_write_ports)
    result = validator.validate_axi_lite(group)
    assert result.valid
    assert result.message is None

def test_validate_axilite_read_only(validator, axilite_read_ports):
    group = create_port_group(InterfaceType.AXI_LITE, "config", axilite_read_ports)
    result = validator.validate_axi_lite(group)
    assert result.valid
    assert result.message is None

def test_validate_axilite_missing_write_required(validator, axilite_read_ports):
    write_ports_missing = [
        Port(name="config_AWADDR", direction=Direction.INPUT, width="6"),
        # Missing AWVALID
        Port(name="config_AWREADY", direction=Direction.OUTPUT, width="1"),
        # ... other valid write ports ...
    ]
    ports = write_ports_missing + axilite_read_ports
    group = create_port_group(InterfaceType.AXI_LITE, "config", ports)
    result = validator.validate_axi_lite(group)
    assert not result.valid
    assert "Missing required AXI-Lite write signals" in result.message
    assert "config_AWVALID" in result.message # Check specific missing signal if possible

def test_validate_axilite_missing_read_required(validator, axilite_write_ports):
    read_ports_missing = [
        Port(name="config_ARADDR", direction=Direction.INPUT, width="6"),
        # Missing ARVALID
        Port(name="config_ARREADY", direction=Direction.OUTPUT, width="1"),
        # ... other valid read ports ...
    ]
    ports = axilite_write_ports + read_ports_missing
    group = create_port_group(InterfaceType.AXI_LITE, "config", ports)
    result = validator.validate_axi_lite(group)
    assert not result.valid
    assert "Missing required AXI-Lite read signals" in result.message
    assert "config_ARVALID" in result.message

def test_validate_axilite_wrong_direction(validator, axilite_write_ports):
    ports = list(axilite_write_ports)
    # Find and modify a port's direction
    for i, p in enumerate(ports):
        if p.name == "config_AWREADY":
            ports[i] = Port(name="config_AWREADY", direction=Direction.INPUT, width="1") # Should be OUTPUT
            break
    group = create_port_group(InterfaceType.AXI_LITE, "config", ports)
    result = validator.validate_axi_lite(group)
    assert not result.valid
    assert "Invalid AXI-Lite signal 'config_AWREADY': Incorrect direction" in result.message

def test_validate_axilite_empty(validator):
    group = create_port_group(InterfaceType.AXI_LITE, "config", [])
    result = validator.validate_axi_lite(group)
    assert not result.valid # Empty group is not valid
    assert "has no recognized read or write signals" in result.message

# --- General Validate Dispatch Test ---

def test_validate_dispatch(validator):
    """Tests that the main validate method dispatches correctly."""

    # 1. Test Global Dispatch (using a valid case)
    global_ports = [
        Port(name="ap_clk", direction=Direction.INPUT, width="1"),
        Port(name="ap_rst_n", direction=Direction.INPUT, width="1"),
    ]
    global_group = create_port_group(InterfaceType.GLOBAL_CONTROL, "global", global_ports)
    result_global = validator.validate(global_group)
    # Check if it looks like a valid global result (indirectly checks dispatch)
    assert result_global.valid
    assert result_global.message is None

    # 2. Test AXI-Stream Dispatch (using a valid case)
    axis_ports = [
        Port(name="in0_TDATA", direction=Direction.INPUT, width="32"),
        Port(name="in0_TVALID", direction=Direction.INPUT, width="1"),
        Port(name="in0_TREADY", direction=Direction.OUTPUT, width="1"),
    ]
    axis_group = create_port_group(InterfaceType.AXI_STREAM, "in0", axis_ports)
    result_axis = validator.validate(axis_group)
    # Check if it looks like a valid AXI-Stream result
    assert result_axis.valid
    assert result_axis.message is None

    # 3. Test AXI-Lite Dispatch (using a valid read-only case)
    axilite_ports = [
        Port(name="config_ARADDR", direction=Direction.INPUT, width="6"),
        Port(name="config_ARVALID", direction=Direction.INPUT, width="1"),
        Port(name="config_ARREADY", direction=Direction.OUTPUT, width="1"),
        Port(name="config_RDATA", direction=Direction.OUTPUT, width="32"),
        Port(name="config_RVALID", direction=Direction.OUTPUT, width="1"),
        Port(name="config_RREADY", direction=Direction.INPUT, width="1"),
        Port(name="config_ARPROT", direction=Direction.INPUT, width="3"), # Added
        Port(name="config_RRESP", direction=Direction.OUTPUT, width="2"), # Added
    ]
    axilite_group = create_port_group(InterfaceType.AXI_LITE, "config", axilite_ports)
    result_axilite = validator.validate(axilite_group)
    # Check if it looks like a valid AXI-Lite result
    assert result_axilite.valid
    assert result_axilite.message is None

    # 4. Test UNKNOWN type handling
    # Assuming UNKNOWN type exists in InterfaceType enum
    try:
        unknown_type = InterfaceType.UNKNOWN
        unknown_group = PortGroup(unknown_type, name="other")
        result_unknown = validator.validate(unknown_group)
        assert result_unknown.valid
        assert "Skipping validation for UNKNOWN group" in result_unknown.message
    except AttributeError:
        # Handle case where InterfaceType.UNKNOWN might not exist
        print("Skipping UNKNOWN type test as InterfaceType.UNKNOWN not found.")

    # 5. Test an invalid case to ensure dispatch happens and returns invalid
    invalid_axis_ports = [
        Port(name="in1_TDATA", direction=Direction.INPUT, width="32"),
        # Missing TVALID, TREADY
    ]
    invalid_axis_group = create_port_group(InterfaceType.AXI_STREAM, "in1", invalid_axis_ports)
    result_invalid_axis = validator.validate(invalid_axis_group)
    assert not result_invalid_axis.valid
    assert "Missing required AXI-Stream signals" in result_invalid_axis.message
