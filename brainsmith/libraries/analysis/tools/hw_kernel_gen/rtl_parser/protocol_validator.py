############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Validates interface protocol requirements for identified PortGroups.

Checks if groups of ports identified by the InterfaceScanner adhere to the
rules defined for specific protocols (Global, AXI-Stream, AXI-Lite), such as
presence of required signals and correct port directions. Protocol definitions
(signal names, requirements) are defined as constants in this module.
"""

import logging
from typing import Dict, Set, List, Tuple

from .data import Port, Direction, PortGroup, ValidationResult, InterfaceType

# --- Protocol Definitions ---
# Define known signal patterns based on RTL_Parser-Data-Analysis.md
GLOBAL_SIGNAL_SUFFIXES = {
    "clk": {"direction": Direction.INPUT, "required": True},
    "rst_n": {"direction": Direction.INPUT, "required": True},
    "clk2x": {"direction": Direction.INPUT, "required": False},
}

# Suffixes for AXI-Stream signals (direction defaults to slave, but both supported) 
AXI_STREAM_SUFFIXES = {
    "TDATA": {"direction": Direction.INPUT, "required": True},
    "TVALID": {"direction": Direction.INPUT, "required": True},
    "TREADY": {"direction": Direction.OUTPUT, "required": True},
    "TLAST": {"direction": Direction.INPUT, "required": False}, # Optional
}

# Suffixes for AXI-Lite signals
AXI_LITE_SUFFIXES = {
    # Write Address Channel
    "AWADDR": {"direction": Direction.INPUT, "required": True},
    "AWPROT": {"direction": Direction.INPUT, "required": False}, # Optional
    "AWVALID": {"direction": Direction.INPUT, "required": True},
    "AWREADY": {"direction": Direction.OUTPUT, "required": True},
    # Write Data Channel
    "WDATA": {"direction": Direction.INPUT, "required": True},
    "WSTRB": {"direction": Direction.INPUT, "required": True},
    "WVALID": {"direction": Direction.INPUT, "required": True},
    "WREADY": {"direction": Direction.OUTPUT, "required": True},
    # Write Response Channel
    "BRESP": {"direction": Direction.OUTPUT, "required": True},
    "BVALID": {"direction": Direction.OUTPUT, "required": True},
    "BREADY": {"direction": Direction.INPUT, "required": True},
    # Read Address Channel
    "ARADDR": {"direction": Direction.INPUT, "required": True},
    "ARPROT": {"direction": Direction.INPUT, "required": False}, # Optional
    "ARVALID": {"direction": Direction.INPUT, "required": True},
    "ARREADY": {"direction": Direction.OUTPUT, "required": True},
    # Read Data Channel
    "RDATA": {"direction": Direction.OUTPUT, "required": True},
    "RRESP": {"direction": Direction.OUTPUT, "required": True},
    "RVALID": {"direction": Direction.OUTPUT, "required": True},
    "RREADY": {"direction": Direction.INPUT, "required": True},
}

# Helper sets for channel identification (using UPPERCASE keys now)
AXI_LITE_WRITE_SUFFIXES = {k: v for k, v in AXI_LITE_SUFFIXES.items() if k.startswith('AW') or k.startswith('W') or k.startswith('B')}
AXI_LITE_READ_SUFFIXES = {k: v for k, v in AXI_LITE_SUFFIXES.items() if k.startswith('AR') or k.startswith('R')}


logger = logging.getLogger(__name__)


class ProtocolValidator:
    """Validates PortGroups against defined interface protocol rules."""

    def __init__(self, debug: bool = False):
        """Initializes the ProtocolValidator."""
        self.debug = debug
        self.input_count = 0
        self.output_count = 0

    def validate(self, group: PortGroup) -> ValidationResult:
        """Dispatches validation to the appropriate method based on group type."""
        itype = group.interface_type
        logger.debug(f"Validating {itype} group '{group.name}'. Received ports: {list(group.ports.keys())}")
        
        if itype == InterfaceType.GLOBAL_CONTROL:
            return self.validate_global_control(group)
        elif itype == InterfaceType.AXI_STREAM:
            return self.validate_axi_stream(group)
        elif itype == InterfaceType.AXI_LITE:
            return self.validate_axi_lite(group)
        else:
            return ValidationResult(False, f"Unknown interface type '{itype}' for group '{group.name}'.")

    def _check_required_signals(self, group_ports: Dict[str, Port], required_spec: Dict[str, Dict]) -> Tuple[Set[str], Set[str]]:
        """Checks if all required signals (keys) are present in the group's ports, and filters for any unexpected signals.
        
        Returns:
            Tuple of (missing_signals, unexpected_signals)
        """
        present_keys = {key.upper() for key in group_ports.keys()}
        required_keys = {key.upper() for key, spec in required_spec.items() if spec["required"] is True}
        optional_keys = {key.upper() for key, spec in required_spec.items() if spec["required"] is False}
        missing = required_keys - present_keys
        unexpected = present_keys - required_keys - optional_keys
        return missing, unexpected

    def validate_global_control(self, group: PortGroup) -> ValidationResult:
        if group.interface_type != InterfaceType.GLOBAL_CONTROL:
            return ValidationResult(False, "Invalid group type for Global Control validation.")
        
        # Check against required & expected signals
        missing, unexpected = self._check_required_signals(group.ports, GLOBAL_SIGNAL_SUFFIXES)
        if missing:
            return ValidationResult(False, f"Global Control: Missing required signal(s) in '{group.name}': {missing}")
        if unexpected:
            return ValidationResult(False, f"Global Control: Unexpected signal in '{group.name}': {unexpected}")

        # Determine direction
        incorrect_ports = [
            f"{port_name} (expected: {GLOBAL_SIGNAL_SUFFIXES[port_name]['direction']}, got: {port.direction})"
            for port_name, port in group.ports.items()
            if port_name in GLOBAL_SIGNAL_SUFFIXES and port.direction != GLOBAL_SIGNAL_SUFFIXES[port_name]["direction"]
        ]
        
        direction = len(incorrect_ports) == 0
        if not direction:
            return ValidationResult(False, f"Global Control: Incorrect direction in '{group.name}': {incorrect_ports}")

        logger.debug(f"  Validation successful for Global Control group '{group.name}'")
        return ValidationResult(True)

    def validate_axi_stream(self, group: PortGroup) -> ValidationResult:
        if group.interface_type != InterfaceType.AXI_STREAM:
            return ValidationResult(False, "Invalid group type for AXI-Stream validation.")

        # Check against required & expected signals
        missing, unexpected = self._check_required_signals(group.ports, AXI_STREAM_SUFFIXES)
        if missing:
            return ValidationResult(False, f"AXI-Stream: Missing required signal(s) in '{group.name}': {missing}")
        if unexpected:
            return ValidationResult(False, f"AXI-Stream: Unexpected signal in '{group.name}': {unexpected}")

        # Determine direction consistency
        incorrect_ports = []
        direction_matches = []
        
        for port_name, port in group.ports.items():
            if port_name in AXI_STREAM_SUFFIXES:
                expected_dir = AXI_STREAM_SUFFIXES[port_name]["direction"]
                if port.direction != expected_dir:
                    incorrect_ports.append(f"{port_name} (expected: {expected_dir}, got: {port.direction})")
                direction_matches.append(port.direction == expected_dir)
        
        # Check if all directions match (forward) or all are inverted (backward)
        all_forward = all(direction_matches)
        all_backward = not any(direction_matches)
        
        if not (all_forward or all_backward):
            return ValidationResult(False, f"AXI-Stream: Invalid signal directions in '{group.name}': {incorrect_ports}")
        
        # Set interface direction metadata
        group.metadata['direction'] = Direction.INPUT if all_forward else Direction.OUTPUT

        # Extract data width metadata
        tdata_port = group.ports.get("TDATA")
        if tdata_port:
            group.metadata['data_width_expr'] = tdata_port.width

        logger.debug(f"  Validation successful for AXI-Stream group '{group.name}'")
        return ValidationResult(True)

    def validate_axi_lite(self, group: PortGroup) -> ValidationResult:
        if group.interface_type != InterfaceType.AXI_LITE:
            return ValidationResult(False, "Invalid group type for AXI-Lite validation.")
        
        # Check against required & expected signals
        missing, unexpected = self._check_required_signals(group.ports, AXI_LITE_SUFFIXES)
        has_write_channel = any(AXI_LITE_WRITE_SUFFIXES[sig]['required'] and sig not in missing for sig in AXI_LITE_WRITE_SUFFIXES)
        has_read_channel = any(AXI_LITE_READ_SUFFIXES[sig]['required'] and sig not in missing for sig in AXI_LITE_READ_SUFFIXES)
        if has_write_channel and any(sig in AXI_LITE_WRITE_SUFFIXES for sig in missing):
            return ValidationResult(False, f"AXI-Lite: Partial write, missing required signal(s) in '{group.name}': {missing}")
        if has_read_channel and any(sig in AXI_LITE_READ_SUFFIXES for sig in missing):
            return ValidationResult(False, f"AXI-Lite: Partial read, missing required signal(s) in '{group.name}': {missing}")
        if not has_write_channel and not has_read_channel:
            return ValidationResult(False, f"AXI-Lite: Not enough valid signals in '{group.name}' for read or write.")
        if unexpected:
            return ValidationResult(False, f"AXI-Lite: Unexpected signal in '{group.name}': {unexpected}")

        # Determine direction
        incorrect_ports = [
            f"{port_name} (expected: {AXI_LITE_SUFFIXES[port_name]['direction']}, got: {port.direction})"
            for port_name, port in group.ports.items()
            if port_name in AXI_LITE_SUFFIXES and port.direction != AXI_LITE_SUFFIXES[port_name]["direction"]
        ]
        
        directions_valid = len(incorrect_ports) == 0
        if not directions_valid:
            return ValidationResult(False, f"AXI-Lite: Incorrect direction in '{group.name}': {incorrect_ports}")

        # TODO: Add checks for response signal widths

        # Extract metadata
        if has_write_channel:
            # Validate static channel sizes
            awaddr_port = group.ports.get("AWADDR")
            wdata_port = group.ports.get("WDATA")
            wstrb_port = group.ports.get("WSTRB")
            group.metadata['write_width_expr'] = {
                "addr": awaddr_port.width,
                "data": wdata_port.width,
                "strobe": wstrb_port.width,
            }
            # TODO: Add robust WSTRB width support, allowing it to be better defined by a local param or standard
        if has_read_channel:
            # Validate static channel sizes
            araddr_port = group.ports.get("ARADDR")
            rdata_port = group.ports.get("RDATA")
            group.metadata['read_width_expr'] = {
                "addr": araddr_port.width,
                "data": rdata_port.width
            }

        logger.debug(f"  Validation successful for AXI-Lite group '{group.name}'")
        return ValidationResult(True)

    def _assign_wrapper_names(self, interfaces: List[PortGroup]) -> None:
        """Assign wrapper names to interfaces based on their type."""
        input_count, output_count = 0, 0

        for group in interfaces:
            if group.interface_type == InterfaceType.GLOBAL_CONTROL:
                for signal in group.ports:
                    group.metadata["wrapper_name"] = signal
            elif group.interface_type == InterfaceType.AXI_STREAM:
                if group.metadata['direction'] == Direction.INPUT:
                    group.metadata["wrapper_name"] = f"in{input_count}"
                    input_count += 1
                elif group.metadata['direction'] == Direction.OUTPUT:
                    group.metadata["wrapper_name"] = f"out{output_count}"
                    output_count += 1
            elif group.interface_type == InterfaceType.AXI_LITE:
                group.metadata["wrapper_name"] = "config"