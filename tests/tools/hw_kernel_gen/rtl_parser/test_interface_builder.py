############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

import pytest
import logging

from brainsmith.tools.hw_kernel_gen.rtl_parser.data import Port, Direction, InterfaceType
from brainsmith.tools.hw_kernel_gen.rtl_parser.interface_builder import InterfaceBuilder

# Local fixtures that are not shared across test files yet
# (All core fixtures like interface_builder, global_ports, etc. are now in conftest.py)

# --- Tests ---

def test_build_all_valid(interface_builder, ports_all_valid_mixed):
    interfaces, unassigned = interface_builder.build_interfaces(ports_all_valid_mixed)

    assert not unassigned
    assert len(interfaces) == 4 # ap, in0, out1_V, config

    assert "ap" in interfaces
    assert interfaces["ap"].type == InterfaceType.GLOBAL_CONTROL
    assert len(interfaces["ap"].ports) == 2

    assert "in0" in interfaces
    assert interfaces["in0"].type == InterfaceType.AXI_STREAM
    assert len(interfaces["in0"].ports) == 3 # TDATA, TVALID, TREADY

    assert "out1_V" in interfaces
    assert interfaces["out1_V"].type == InterfaceType.AXI_STREAM
    assert len(interfaces["out1_V"].ports) == 3 # TDATA, TVALID, TREADY

    assert "config" in interfaces
    assert interfaces["config"].type == InterfaceType.AXI_LITE
    assert len(interfaces["config"].ports) == 10 # Write channel only

    # Check validation status
    for iface in interfaces.values():
        assert iface.validation_result.valid

def test_build_with_invalid_group(interface_builder, ports_with_invalid_axis, caplog):
    caplog.set_level(logging.WARNING)
    interfaces, unassigned = interface_builder.build_interfaces(ports_with_invalid_axis)

    assert len(interfaces) == 2 # ap, out1
    assert "ap" in interfaces
    assert "out1" in interfaces
    assert "in0" not in interfaces # Should fail validation

    assert len(unassigned) == 2 # The two ports from the failed in0 group
    unassigned_names = {p.name for p in unassigned}
    assert unassigned_names == {"in0_TDATA", "in0_TVALID"}

    # Check logs for warning
    assert "Validation failed for potential interface 'in0' (axistream)" in caplog.text
    assert "Missing required signal(s) in 'in0': {'TREADY'}" in caplog.text

def test_build_with_unassigned(interface_builder, ports_with_unassigned, caplog):
    caplog.set_level(logging.WARNING) # Ensure warnings are captured if any
    interfaces, unassigned = interface_builder.build_interfaces(ports_with_unassigned)

    assert len(interfaces) == 2 # ap, in0
    assert "ap" in interfaces
    assert "in0" in interfaces

    assert len(unassigned) == 2
    unassigned_names = {p.name for p in unassigned}
    assert unassigned_names == {"custom_enable", "debug_counter"}

    # Should be no validation warnings in this case
    assert "Validation failed" not in caplog.text

def test_build_empty(interface_builder):
    interfaces, unassigned = interface_builder.build_interfaces([])
    assert not interfaces
    assert not unassigned

def test_build_only_unassigned(interface_builder):
    ports = [
        Port(name="custom1", direction=Direction.INPUT, width="1"),
        Port(name="custom2", direction=Direction.OUTPUT, width="1"),
    ]
    interfaces, unassigned = interface_builder.build_interfaces(ports)
    assert not interfaces
    assert len(unassigned) == 2
    assert {p.name for p in unassigned} == {"custom1", "custom2"}

def test_build_debug_logging(interface_builder_debug, ports_with_invalid_axis, caplog):
    caplog.set_level(logging.DEBUG)
    interface_builder_debug.build_interfaces(ports_with_invalid_axis)

    # Check for specific debug messages
    assert "Successfully validated and built interface: ap (global)" in caplog.text
    assert "Successfully validated and built interface: out1 (axistream)" in caplog.text
    assert "Validation failed for potential interface 'in0' (axistream)" in caplog.text
    assert "Ports from failed group 'in0': ['in0_TDATA', 'in0_TVALID']" in caplog.text
