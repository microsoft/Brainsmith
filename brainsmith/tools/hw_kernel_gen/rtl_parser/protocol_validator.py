############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Validates interface protocol requirements for identified PortGroups."""

import logging
from typing import Dict, Set, Optional

from brainsmith.tools.hw_kernel_gen.rtl_parser.data import Port, Direction # Ensure Direction is imported
from brainsmith.tools.hw_kernel_gen.rtl_parser.interface_types import PortGroup, ValidationResult, InterfaceType

# --- MOVED Protocol Definitions ---
# Define known signal patterns based on RTL_Parser-Data-Analysis.md
GLOBAL_SIGNALS = {
    "ap_clk": {"direction": Direction.INPUT, "required": True},
    "ap_rst_n": {"direction": Direction.INPUT, "required": True},
    "ap_clk2x": {"direction": Direction.INPUT, "required": False},
}

# Suffixes for AXI-Stream. Direction depends on prefix (in/out) and will be validated later.
# The key used in PortGroup will be the suffix (e.g., TDATA)
AXI_STREAM_SUFFIXES = {
    "TDATA": {"required": True},
    "TVALID": {"required": True},
    "TREADY": {"required": True},
    "TLAST": {"required": False},
}

# AXI-Lite signals. The key used in PortGroup will be the full signal name.
AXI_LITE_WRITE_SUFFIXES = {
    "AWADDR": {"direction": Direction.INPUT, "required": True},
    "AWPROT": {"direction": Direction.INPUT, "required": True},
    "AWVALID": {"direction": Direction.INPUT, "required": True},
    "AWREADY": {"direction": Direction.OUTPUT, "required": True},
    "WDATA": {"direction": Direction.INPUT, "required": True},
    "WSTRB": {"direction": Direction.INPUT, "required": True},
    "WVALID": {"direction": Direction.INPUT, "required": True},
    "WREADY": {"direction": Direction.OUTPUT, "required": True},
    "BRESP": {"direction": Direction.OUTPUT, "required": True},
    "BVALID": {"direction": Direction.OUTPUT, "required": True},
    "BREADY": {"direction": Direction.INPUT, "required": True},
}

AXI_LITE_READ_SUFFIXES = {
    "ARADDR": {"direction": Direction.INPUT, "required": True},
    "ARPROT": {"direction": Direction.INPUT, "required": True},
    "ARVALID": {"direction": Direction.INPUT, "required": True},
    "ARREADY": {"direction": Direction.OUTPUT, "required": True},
    "RDATA": {"direction": Direction.OUTPUT, "required": True},
    "RRESP": {"direction": Direction.OUTPUT, "required": True},
    "RVALID": {"direction": Direction.OUTPUT, "required": True},
    "RREADY": {"direction": Direction.INPUT, "required": True},
}

# Combined AXI-Lite for easier checking
AXI_LITE_SUFFIXES = {**AXI_LITE_WRITE_SUFFIXES, **AXI_LITE_READ_SUFFIXES}
# --- END MOVED Protocol Definitions ---

logger = logging.getLogger(__name__)

class ProtocolValidator:
    """Validates interface protocol requirements for PortGroups."""

    def __init__(self, debug: bool = False):
        self.debug = debug # Placeholder for potential future debugging logs

    def _check_required_signals(self, group_ports: Dict[str, Port], required_spec: Dict[str, Dict]) -> Set[str]:
        """Checks if all required signals are present in the group."""
        present_keys = set(group_ports.keys())
        required_keys = {key for key, spec in required_spec.items() if spec.get("required", False)}
        missing = required_keys - present_keys
        return missing

    def _validate_port_properties(self, port: Port, expected_direction: Direction) -> Optional[str]:
        """Validates direction and potentially other properties of a port."""
        if port.direction != expected_direction:
            return f"Incorrect direction: expected {expected_direction.value}, got {port.direction.value}"
        # Add other property checks here if needed (e.g., width for specific signals)
        return None

    def validate_global_signals(self, group: PortGroup) -> ValidationResult:
        """Validate global control signals based on GLOBAL_SIGNALS definition."""
        if group.interface_type != InterfaceType.GLOBAL_CONTROL:
            return ValidationResult(False, "Invalid group type for global signal validation.")

        missing = self._check_required_signals(group.ports, GLOBAL_SIGNALS)
        if missing:
            return ValidationResult(False, f"Missing required global signals: {missing}")

        # Validate properties of present signals
        for signal_name, port in group.ports.items():
            if signal_name in GLOBAL_SIGNALS:
                expected_direction = GLOBAL_SIGNALS[signal_name]["direction"]
                error = self._validate_port_properties(port, expected_direction)
                if error:
                    return ValidationResult(False, f"Invalid global signal '{signal_name}': {error}")
            # else: # Optional signal, validation might differ or be skipped

        return ValidationResult(True)

    def validate_axi_stream(self, group: PortGroup) -> ValidationResult:
        """Validate AXI-Stream interface based on AXI_STREAM_SUFFIXES and naming conventions."""
        if group.interface_type != InterfaceType.AXI_STREAM:
            return ValidationResult(False, "Invalid group type for AXI-Stream validation.")
        if not group.name:
             return ValidationResult(False, "AXI-Stream group missing name (prefix).")

        # --- MODIFIED: Use keys without underscore ---
        missing = self._check_required_signals(group.ports, AXI_STREAM_SUFFIXES)
        if missing:
            return ValidationResult(False, f"Missing required AXI-Stream signals for '{group.name}': {missing}")
        # --- END MODIFICATION ---

        # Determine expected direction based on prefix/suffix
        lname = group.name.lower()
        is_input_stream = lname.startswith("in") or lname.startswith("s_") or lname.endswith("_s") # Slave/Input
        is_output_stream = lname.startswith("out") or lname.startswith("m_") or lname.endswith("_m") # Master/Output

        if not is_input_stream and not is_output_stream:
             logger.warning(f"Could not determine direction (input/output) for AXI-Stream '{group.name}' based on naming convention.")
             # Proceed, but direction checks will be skipped

        # --- MODIFIED: Use key without underscore ---
        tdata_port = group.ports.get("TDATA")
        if not tdata_port:
            # Should have been caught by missing check, but defensive coding
            return ValidationResult(False, f"AXI-Stream '{group.name}' missing TDATA.")
        # --- END MODIFICATION ---

        # Validate signal directions based on inferred stream direction
        for suffix, port in group.ports.items():
            expected_direction = None
            # --- MODIFIED: Use keys without underscore ---
            if suffix == "TDATA" or suffix == "TVALID" or suffix == "TLAST":
                # Corrected logic: Input stream means these signals are INPUT
                if is_input_stream: expected_direction = Direction.INPUT
                elif is_output_stream: expected_direction = Direction.OUTPUT
            elif suffix == "TREADY":
                 # Corrected logic: Input stream means TREADY is OUTPUT
                if is_input_stream: expected_direction = Direction.OUTPUT
                elif is_output_stream: expected_direction = Direction.INPUT
            # --- END MODIFICATION ---

            if expected_direction: # Only validate if direction is clear
                error = self._validate_port_properties(port, expected_direction)
                if error:
                    # Construct the full port name for the error message
                    full_port_name = f"{group.name}{suffix}"
                    # Handle the '_V_' case if necessary based on actual port name
                    if '_V' in port.name and '_V' not in full_port_name:
                         # Attempt to reconstruct name more accurately if needed, simple version for now
                         pass # Or adjust full_port_name based on port.name if prefix logic is complex
                    return ValidationResult(False, f"Invalid AXI-Stream signal '{port.name}': {error}") # Use actual port name
            elif not is_input_stream and not is_output_stream:
                 logger.debug(f"Skipping direction validation for '{port.name}' due to ambiguous stream direction.")
            # else: # Should not happen if suffix is known

        return ValidationResult(True)

    def validate_axi_lite(self, group: PortGroup) -> ValidationResult:
        """Validate AXI-Lite interface based on AXI_LITE signal definitions."""
        if group.interface_type != InterfaceType.AXI_LITE:
            return ValidationResult(False, "Invalid group type for AXI-Lite validation.")
        if not group.name:
             return ValidationResult(False, "AXI-Lite group missing name (prefix).") # Added check

        # group.ports now uses generic keys (e.g., "AWADDR", "WDATA")
        present_signals = set(group.ports.keys())

        # Check which channels are potentially present based on generic signal names
        has_write_channel = any(sig in present_signals for sig in AXI_LITE_WRITE_SUFFIXES)
        has_read_channel = any(sig in present_signals for sig in AXI_LITE_READ_SUFFIXES)

        if not has_write_channel and not has_read_channel:
            # Use group.name (prefix) in the error message
            return ValidationResult(False, f"AXI-Lite group '{group.name}' has no recognized read or write signals.")

        # Check required signals for present channels using generic keys
        if has_write_channel:
            # AXI_LITE_WRITE_SUFFIXES uses generic keys
            missing_write = self._check_required_signals(group.ports, AXI_LITE_WRITE_SUFFIXES)
            if missing_write:
                # Use group.name (prefix) in the error message
                return ValidationResult(False, f"Missing required AXI-Lite write signals for '{group.name}': {missing_write}")

        if has_read_channel:
            # AXI_LITE_READ_SUFFIXES uses generic keys
            missing_read = self._check_required_signals(group.ports, AXI_LITE_READ_SUFFIXES)
            if missing_read:
                # Use group.name (prefix) in the error message
                return ValidationResult(False, f"Missing required AXI-Lite read signals for '{group.name}': {missing_read}")

        # Validate properties of all present AXI-Lite signals
        # Iterate through the generic signal names found in the group
        for signal_name, port in group.ports.items():
            # AXI_LITE_SUFFIXES uses generic keys
            if signal_name in AXI_LITE_SUFFIXES:
                expected_direction = AXI_LITE_SUFFIXES[signal_name]["direction"]
                error = self._validate_port_properties(port, expected_direction)
                if error:
                    # Report the error using the original port name for clarity
                    return ValidationResult(False, f"Invalid AXI-Lite signal '{port.name}': {error}")
            # else: Signal not in our known list, might be an error or extension (ignore for now)

        # TODO: Add check for address width consistency if needed

        return ValidationResult(True)

    def validate(self, group: PortGroup) -> ValidationResult:
        """Validates a PortGroup based on its type."""
        if group.interface_type == InterfaceType.GLOBAL_CONTROL:
            return self.validate_global_signals(group)
        elif group.interface_type == InterfaceType.AXI_STREAM:
            return self.validate_axi_stream(group)
        elif group.interface_type == InterfaceType.AXI_LITE:
            return self.validate_axi_lite(group)
        elif group.interface_type == InterfaceType.UNKNOWN:
            logger.debug(f"Skipping validation for UNKNOWN group '{group.name}'")
            return ValidationResult(True, f"Skipping validation for UNKNOWN group '{group.name}'")
        else:
            logger.warning(f"Unknown interface type encountered during validation: {group.interface_type}")
            return ValidationResult(False, f"Unknown interface type: {group.interface_type}")

