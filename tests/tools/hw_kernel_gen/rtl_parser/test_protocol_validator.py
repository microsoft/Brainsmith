# filepath: /home/tafk/dev/brainsmith/tests/tools/hw_kernel_gen/rtl_parser/test_protocol_validator.py
import pytest
import dataclasses
from typing import List, Dict # Added Dict

from brainsmith.tools.hw_kernel_gen.rtl_parser.data import Port, Direction
from brainsmith.tools.hw_kernel_gen.rtl_parser.interface_types import InterfaceType, PortGroup
from brainsmith.tools.hw_kernel_gen.rtl_parser.protocol_validator import ProtocolValidator

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

# --- ADDED: Combined AXI-Lite Ports Fixture ---
@pytest.fixture
def axilite_ports_full(axilite_write_ports, axilite_read_ports) -> List[Port]:
    """Fixture combining full AXI-Lite write and read ports."""
    return axilite_write_ports + axilite_read_ports
# --- END ADDED ---

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

def create_generic_axilite_group(ports: list, name: str = "config") -> PortGroup:
    """Helper to create PortGroup with generic keys for AXI-Lite."""
    generic_ports_dict = {}
    prefix = name + "_" # Assuming prefix matches group name
    for port in ports:
        base_name = port.name.replace(prefix, "")
        generic_ports_dict[base_name] = port
    return PortGroup(interface_type=InterfaceType.AXI_LITE, name=name, ports=generic_ports_dict)

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

def test_validate_axilite_full(validator, axilite_ports_full):
    # MODIFIED: Use helper to create group with generic keys
    group = create_generic_axilite_group(axilite_ports_full, "config")
    result = validator.validate_axi_lite(group)
    # Original assertions remain
    assert result.valid
    assert result.message is None

def test_validate_axilite_write_only(validator, axilite_write_ports):
    # MODIFIED: Use helper to create group with generic keys
    group = create_generic_axilite_group(axilite_write_ports, "config")
    result = validator.validate_axi_lite(group)
    # Original assertions remain
    assert result.valid
    assert result.message is None

def test_validate_axilite_read_only(validator, axilite_read_ports):
    # MODIFIED: Use helper to create group with generic keys
    group = create_generic_axilite_group(axilite_read_ports, "config")
    result = validator.validate_axi_lite(group)
    # Original assertions remain
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

    # MODIFIED: Use helper to create group with generic keys
    group = create_generic_axilite_group(ports, "config")
    # --- END MODIFICATION ---

    result = validator.validate_axi_lite(group)
    # Original assertions remain
    assert not result.valid
    assert "Missing required AXI-Lite write signals" in result.message
    assert "AWVALID" in result.message

def test_validate_axilite_missing_read_required(validator, axilite_write_ports):
    read_ports_missing = [
        Port(name="config_ARADDR", direction=Direction.INPUT, width="6"),
        # Missing ARVALID
        Port(name="config_ARREADY", direction=Direction.OUTPUT, width="1"),
        # ... other valid read ports ...
    ]
    ports = read_ports_missing + axilite_write_ports

    # MODIFIED: Use helper to create group with generic keys
    group = create_generic_axilite_group(ports, "config")
    # --- END MODIFICATION ---

    result = validator.validate_axi_lite(group)
    # Original assertions remain
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

    # MODIFIED: Use helper to create group with generic keys
    group = create_generic_axilite_group(modified_ports, "config")
    # --- END MODIFICATION ---

    result = validator.validate_axi_lite(group)
    # Original assertions remain
    assert not result.valid
    assert "Invalid AXI-Lite signal 'config_AWREADY'" in result.message
    assert "Incorrect direction" in result.message

# --- General Validate Dispatch Test ---

def test_validate_dispatch(validator, global_ports, axis_in_ports, axilite_ports_full):
    # Global group (no change needed)
    global_group = PortGroup(interface_type=InterfaceType.GLOBAL_CONTROL, name="global", ports={p.name: p for p in global_ports})
    result_global = validator.validate(global_group)
    assert result_global.valid

    # AXI-Stream group (no change needed, assuming keys are suffixes like _TDATA)
    axis_group = PortGroup(interface_type=InterfaceType.AXI_STREAM, name="in0", ports={p.name.replace("in0", ""): p for p in axis_in_ports})
    result_axis = validator.validate(axis_group)
    assert result_axis.valid

    # AXI-Lite group (MODIFIED: Use helper)
    axilite_group = create_generic_axilite_group(axilite_ports_full, "config")
    result_axilite = validator.validate(axilite_group)
    # Original assertions remain
    assert result_axilite.valid # This should now pass

    # Unknown group (no change needed)
    unknown_group = PortGroup(interface_type=InterfaceType.UNKNOWN, name="unknown", ports={"foo": Port("foo", Direction.INPUT)})
    result_unknown = validator.validate(unknown_group)
    assert result_unknown.valid # Should skip validation
