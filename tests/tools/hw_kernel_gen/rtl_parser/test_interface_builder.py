# filepath: /home/tafk/dev/brainsmith/tests/tools/hw_kernel_gen/rtl_parser/test_interface_builder.py
import pytest
import logging

from brainsmith.tools.hw_kernel_gen.rtl_parser.data import Port, Direction
from brainsmith.tools.hw_kernel_gen.rtl_parser.interface_types import InterfaceType, Interface
from brainsmith.tools.hw_kernel_gen.rtl_parser.interface_builder import InterfaceBuilder

# --- Fixtures (can reuse or adapt from other test files if needed) ---

@pytest.fixture
def builder():
    return InterfaceBuilder()

@pytest.fixture
def builder_debug():
    return InterfaceBuilder(debug=True)

@pytest.fixture
def ports_all_valid_mixed():
    return [
        # Global
        Port(name="ap_clk", direction=Direction.INPUT, width="1"),
        Port(name="ap_rst_n", direction=Direction.INPUT, width="1"),
        # AXIS In
        Port(name="in0_TDATA", direction=Direction.INPUT, width="32"),
        Port(name="in0_TVALID", direction=Direction.INPUT, width="1"),
        Port(name="in0_TREADY", direction=Direction.OUTPUT, width="1"),
        # AXIS Out
        Port(name="out1_V_TDATA", direction=Direction.OUTPUT, width="64"),
        Port(name="out1_V_TVALID", direction=Direction.OUTPUT, width="1"),
        Port(name="out1_V_TREADY", direction=Direction.INPUT, width="1"),
        # AXI-Lite (Write only)
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
def ports_with_invalid_axis():
    return [
        Port(name="ap_clk", direction=Direction.INPUT, width="1"),
        Port(name="ap_rst_n", direction=Direction.INPUT, width="1"),
        # Invalid AXIS (missing TREADY)
        Port(name="in0_TDATA", direction=Direction.INPUT, width="32"),
        Port(name="in0_TVALID", direction=Direction.INPUT, width="1"),
        # Valid AXIS
        Port(name="out1_TDATA", direction=Direction.OUTPUT, width="16"),
        Port(name="out1_TVALID", direction=Direction.OUTPUT, width="1"),
        Port(name="out1_TREADY", direction=Direction.INPUT, width="1"),
    ]

@pytest.fixture
def ports_with_unassigned():
    return [
        Port(name="ap_clk", direction=Direction.INPUT, width="1"),
        Port(name="ap_rst_n", direction=Direction.INPUT, width="1"),
        Port(name="in0_TDATA", direction=Direction.INPUT, width="32"),
        Port(name="in0_TVALID", direction=Direction.INPUT, width="1"),
        Port(name="in0_TREADY", direction=Direction.OUTPUT, width="1"),
        Port(name="custom_signal", direction=Direction.INOUT, width="8"),
        Port(name="another_sig", direction=Direction.INPUT, width="1"),
    ]

# --- Tests ---

def test_build_all_valid(builder, ports_all_valid_mixed):
    interfaces, unassigned = builder.build_interfaces(ports_all_valid_mixed)

    assert not unassigned
    assert len(interfaces) == 4 # global, in0, out1, config

    assert "global" in interfaces
    assert interfaces["global"].type == InterfaceType.GLOBAL_CONTROL
    assert len(interfaces["global"].ports) == 2

    assert "in0" in interfaces
    assert interfaces["in0"].type == InterfaceType.AXI_STREAM
    assert len(interfaces["in0"].ports) == 3 # TDATA, TVALID, TREADY

    assert "out1" in interfaces # Note: scanner extracts prefix as 'out1_V', builder uses it
    assert interfaces["out1"].type == InterfaceType.AXI_STREAM
    assert len(interfaces["out1"].ports) == 3 # TDATA, TVALID, TREADY

    assert "config" in interfaces
    assert interfaces["config"].type == InterfaceType.AXI_LITE
    assert len(interfaces["config"].ports) == 11 # Write channel only

    # Check validation status
    for iface in interfaces.values():
        assert iface.validation_result.valid

def test_build_with_invalid_group(builder, ports_with_invalid_axis, caplog):
    caplog.set_level(logging.WARNING)
    interfaces, unassigned = builder.build_interfaces(ports_with_invalid_axis)

    assert len(interfaces) == 2 # global, out1
    assert "global" in interfaces
    assert "out1" in interfaces
    assert "in0" not in interfaces # Should fail validation

    assert len(unassigned) == 2 # The two ports from the failed in0 group
    unassigned_names = {p.name for p in unassigned}
    assert unassigned_names == {"in0_TDATA", "in0_TVALID"}

    # Check logs for warning
    assert "Validation failed for potential interface 'in0' (axis)" in caplog.text
    assert "Missing required AXI-Stream signals" in caplog.text

def test_build_with_unassigned(builder, ports_with_unassigned, caplog):
    caplog.set_level(logging.WARNING) # Ensure warnings are captured if any
    interfaces, unassigned = builder.build_interfaces(ports_with_unassigned)

    assert len(interfaces) == 2 # global, in0
    assert "global" in interfaces
    assert "in0" in interfaces

    assert len(unassigned) == 2
    unassigned_names = {p.name for p in unassigned}
    assert unassigned_names == {"custom_signal", "another_sig"}

    # Should be no validation warnings in this case
    assert "Validation failed" not in caplog.text

def test_build_empty(builder):
    interfaces, unassigned = builder.build_interfaces([])
    assert not interfaces
    assert not unassigned

def test_build_only_unassigned(builder):
    ports = [
        Port(name="custom1", direction=Direction.INPUT, width="1"),
        Port(name="custom2", direction=Direction.OUTPUT, width="1"),
    ]
    interfaces, unassigned = builder.build_interfaces(ports)
    assert not interfaces
    assert len(unassigned) == 2
    assert {p.name for p in unassigned} == {"custom1", "custom2"}

def test_build_debug_logging(builder_debug, ports_with_invalid_axis, caplog):
    caplog.set_level(logging.DEBUG)
    builder_debug.build_interfaces(ports_with_invalid_axis)

    # Check for specific debug messages
    assert "Successfully validated and built interface: global (global)" in caplog.text
    assert "Successfully validated and built interface: out1 (axis)" in caplog.text
    assert "Validation failed for potential interface 'in0' (axis)" in caplog.text
    assert "Ports from failed group 'in0': ['in0_TDATA', 'in0_TVALID']" in caplog.text
