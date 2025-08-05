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

from brainsmith.core.dataflow.types import InterfaceType
from brainsmith.tools.kernel_integrator.types.core import PortDirection
from brainsmith.tools.kernel_integrator.types.rtl import Port, PortGroup, ProtocolValidationResult

# --- Protocol Definitions ---
# Define known signal patterns based on RTL_Parser-Data-Analysis.md
GLOBAL_SIGNAL_SUFFIXES = {
    "clk": {"direction": PortDirection.INPUT, "required": True},
    "rst_n": {"direction": PortDirection.INPUT, "required": True},
    "clk2x": {"direction": PortDirection.INPUT, "required": False},
}

# Suffixes for AXI-Stream signals (direction defaults to slave, but both supported) 
AXI_STREAM_SUFFIXES = {
    "TDATA": {"direction": PortDirection.INPUT, "required": True},
    "TVALID": {"direction": PortDirection.INPUT, "required": True},
    "TREADY": {"direction": PortDirection.OUTPUT, "required": True},
    "TLAST": {"direction": PortDirection.INPUT, "required": False}, # Optional
}

# Suffixes for AXI-Lite signals
AXI_LITE_SUFFIXES = {
    # Write Address Channel
    "AWADDR": {"direction": PortDirection.INPUT, "required": True},
    "AWPROT": {"direction": PortDirection.INPUT, "required": False}, # Optional
    "AWVALID": {"direction": PortDirection.INPUT, "required": True},
    "AWREADY": {"direction": PortDirection.OUTPUT, "required": True},
    # Write Data Channel
    "WDATA": {"direction": PortDirection.INPUT, "required": True},
    "WSTRB": {"direction": PortDirection.INPUT, "required": True},
    "WVALID": {"direction": PortDirection.INPUT, "required": True},
    "WREADY": {"direction": PortDirection.OUTPUT, "required": True},
    # Write Response Channel
    "BRESP": {"direction": PortDirection.OUTPUT, "required": True},
    "BVALID": {"direction": PortDirection.OUTPUT, "required": True},
    "BREADY": {"direction": PortDirection.INPUT, "required": True},
    # Read Address Channel
    "ARADDR": {"direction": PortDirection.INPUT, "required": True},
    "ARPROT": {"direction": PortDirection.INPUT, "required": False}, # Optional
    "ARVALID": {"direction": PortDirection.INPUT, "required": True},
    "ARREADY": {"direction": PortDirection.OUTPUT, "required": True},
    # Read Data Channel
    "RDATA": {"direction": PortDirection.OUTPUT, "required": True},
    "RRESP": {"direction": PortDirection.OUTPUT, "required": True},
    "RVALID": {"direction": PortDirection.OUTPUT, "required": True},
    "RREADY": {"direction": PortDirection.INPUT, "required": True},
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

    def validate(self, group: PortGroup) -> ProtocolValidationResult:
        """Dispatches validation to the appropriate method based on group type."""
        itype = group.interface_type
        logger.debug(f"Validating {itype} group '{group.name}'. Received ports: {list(group.ports.keys())}")
        
        # Dispatch based on protocol (using interface type properties)
        if itype == InterfaceType.CONTROL:
            return self.validate_global_control(group)
        elif itype in [InterfaceType.INPUT, InterfaceType.OUTPUT, InterfaceType.WEIGHT]:
            return self.validate_axi_stream(group)
        elif itype == InterfaceType.CONFIG:
            return self.validate_axi_lite(group)
        else:
            return ProtocolValidationResult(False, f"Unknown interface type '{itype}' for group '{group.name}'.")

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

    def validate_global_control(self, group: PortGroup) -> ProtocolValidationResult:
        if group.interface_type != InterfaceType.CONTROL:
            return ProtocolValidationResult(False, "Invalid group type for Global Control validation.")
        
        # Set final interface type to CONTROL (global control signals)
        group.interface_type = InterfaceType.CONTROL
        
        # Check against required & expected signals
        missing, unexpected = self._check_required_signals(group.ports, GLOBAL_SIGNAL_SUFFIXES)
        if missing:
            return ProtocolValidationResult(False, f"Global Control: Missing required signal(s) in '{group.name}': {missing}")
        if unexpected:
            return ProtocolValidationResult(False, f"Global Control: Unexpected signal in '{group.name}': {unexpected}")

        # Determine direction
        incorrect_ports = [
            f"{port_name} (expected: {GLOBAL_SIGNAL_SUFFIXES[port_name]['direction']}, got: {port.direction})"
            for port_name, port in group.ports.items()
            if port_name in GLOBAL_SIGNAL_SUFFIXES and port.direction != GLOBAL_SIGNAL_SUFFIXES[port_name]["direction"]
        ]
        
        direction = len(incorrect_ports) == 0
        if not direction:
            return ProtocolValidationResult(False, f"Global Control: Incorrect direction in '{group.name}': {incorrect_ports}")

        logger.debug(f"  Validation successful for Global Control group '{group.name}'")
        return ProtocolValidationResult(True)

    def validate_axi_stream(self, group: PortGroup) -> ProtocolValidationResult:
        # Accept any preliminary AXI-Stream type (INPUT, OUTPUT, WEIGHT)
        if group.interface_type not in [InterfaceType.INPUT, InterfaceType.OUTPUT, InterfaceType.WEIGHT]:
            return ProtocolValidationResult(False, "Invalid group type for AXI-Stream validation.")

        # Check against required & expected signals
        missing, unexpected = self._check_required_signals(group.ports, AXI_STREAM_SUFFIXES)
        if missing:
            return ProtocolValidationResult(False, f"AXI-Stream: Missing required signal(s) in '{group.name}': {missing}")
        if unexpected:
            return ProtocolValidationResult(False, f"AXI-Stream: Unexpected signal in '{group.name}': {unexpected}")

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
            return ProtocolValidationResult(False, f"AXI-Stream: Invalid signal directions in '{group.name}': {incorrect_ports}")
        
        # Set interface direction metadata
        direction = "input" if all_forward else "output"
        group.metadata['direction'] = direction

        # Extract data width metadata
        tdata_port = group.ports.get("TDATA")
        if tdata_port:
            group.metadata['data_width_expr'] = tdata_port.width
        
        # Determine specific dataflow interface type based on direction and naming
        group.interface_type = self._determine_dataflow_type(group.name, direction)

        logger.debug(f"  Validation successful for AXI-Stream group '{group.name}' → {group.interface_type}")
        return ProtocolValidationResult(True)

    def validate_axi_lite(self, group: PortGroup) -> ProtocolValidationResult:
        if group.interface_type != InterfaceType.CONFIG:
            return ProtocolValidationResult(False, "Invalid group type for AXI-Lite validation.")
        
        # Set final interface type to CONFIG (AXI-Lite always for configuration)
        group.interface_type = InterfaceType.CONFIG
        
        # Check against required & expected signals
        missing, unexpected = self._check_required_signals(group.ports, AXI_LITE_SUFFIXES)
        has_write_channel = any(AXI_LITE_WRITE_SUFFIXES[sig]['required'] and sig not in missing for sig in AXI_LITE_WRITE_SUFFIXES)
        has_read_channel = any(AXI_LITE_READ_SUFFIXES[sig]['required'] and sig not in missing for sig in AXI_LITE_READ_SUFFIXES)
        if has_write_channel and any(sig in AXI_LITE_WRITE_SUFFIXES for sig in missing):
            return ProtocolValidationResult(False, f"AXI-Lite: Partial write, missing required signal(s) in '{group.name}': {missing}")
        if has_read_channel and any(sig in AXI_LITE_READ_SUFFIXES for sig in missing):
            return ProtocolValidationResult(False, f"AXI-Lite: Partial read, missing required signal(s) in '{group.name}': {missing}")
        if not has_write_channel and not has_read_channel:
            return ProtocolValidationResult(False, f"AXI-Lite: Not enough valid signals in '{group.name}' for read or write.")
        if unexpected:
            return ProtocolValidationResult(False, f"AXI-Lite: Unexpected signal in '{group.name}': {unexpected}")

        # Determine direction
        incorrect_ports = [
            f"{port_name} (expected: {AXI_LITE_SUFFIXES[port_name]['direction']}, got: {port.direction})"
            for port_name, port in group.ports.items()
            if port_name in AXI_LITE_SUFFIXES and port.direction != AXI_LITE_SUFFIXES[port_name]["direction"]
        ]
        
        directions_valid = len(incorrect_ports) == 0
        if not directions_valid:
            return ProtocolValidationResult(False, f"AXI-Lite: Incorrect direction in '{group.name}': {incorrect_ports}")

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
        return ProtocolValidationResult(True)

    def _determine_dataflow_type(self, interface_name: str, direction: str) -> InterfaceType:
        """Determine dataflow interface type from name patterns and direction."""
        name_lower = interface_name.lower()
        
        # Weight interface patterns
        if any(pattern in name_lower for pattern in ['weight', 'weights', 'param', 'coeff']):
            return InterfaceType.WEIGHT
        
        # Input/output based on direction
        if direction == "input":
            return InterfaceType.INPUT
        elif direction == "output":
            return InterfaceType.OUTPUT
        else:
            return InterfaceType.INPUT  # Default to input for unknown directions

    def _assign_wrapper_names(self, interfaces: List[PortGroup]) -> None:
        """Assign wrapper names to interfaces based on their type."""
        input_count, output_count = 0, 0

        for group in interfaces:
            if group.interface_type == InterfaceType.CONTROL:
                for signal in group.ports:
                    group.metadata["wrapper_name"] = signal
            elif group.interface_type in [InterfaceType.INPUT, InterfaceType.OUTPUT, InterfaceType.WEIGHT]:
                if group.interface_type == InterfaceType.INPUT:
                    group.metadata["wrapper_name"] = f"in{input_count}"
                    input_count += 1
                elif group.interface_type == InterfaceType.OUTPUT:
                    group.metadata["wrapper_name"] = f"out{output_count}"
                    output_count += 1
                elif group.interface_type == InterfaceType.WEIGHT:
                    group.metadata["wrapper_name"] = f"weight{input_count}"  # Weights count as inputs
                    input_count += 1
            elif group.interface_type == InterfaceType.CONFIG:
                group.metadata["wrapper_name"] = "config"