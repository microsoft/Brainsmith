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
from typing import Dict, Set, Optional

from brainsmith.tools.hw_kernel_gen.rtl_parser.data import Port, Direction, PortGroup, ValidationResult, InterfaceType

# --- Protocol Definitions ---
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
# --- END Protocol Definitions ---

logger = logging.getLogger(__name__)

class ProtocolValidator:
    """Validates PortGroups against defined interface protocol rules."""

    def __init__(self, debug: bool = False):
        """Initializes the ProtocolValidator."""
        self.debug = debug

    def _check_required_signals(self, group_ports: Dict[str, Port], required_spec: Dict[str, Dict]) -> Set[str]:
        """Checks if all required signals (keys) are present in the group's ports."""
        present_keys = set(group_ports.keys())
        required_keys = {key for key, spec in required_spec.items() if spec.get("required", False)}
        missing = required_keys - present_keys
        return missing

    def _validate_port_properties(self, port: Port, expected_direction: Direction) -> Optional[str]:
        """Validates the direction of a port against the expected direction."""
        if port.direction != expected_direction:
            return f"Incorrect direction: expected {expected_direction.value}, got {port.direction.value}"
        # Add other property checks here if needed (e.g., width for specific signals)
        return None

    def validate_global_signals(self, group: PortGroup) -> ValidationResult:
        """Validates a PortGroup identified as potential Global Control signals.

        Checks for presence of required signals (e.g., ap_clk, ap_rst_n) and
        verifies the direction of present signals against GLOBAL_SIGNALS definition.

        Args:
            group: The PortGroup to validate (must have type GLOBAL_CONTROL).

        Returns:
            ValidationResult indicating success or failure with a message.
        """
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
        """Validates a PortGroup identified as a potential AXI-Stream interface.

        Checks for required signal suffixes (TDATA, TVALID, TREADY) and validates
        port directions based on inferred stream direction (input/output) from the
        group's name prefix (e.g., "in0", "out_data").

        Args:
            group: The PortGroup to validate (must have type AXI_STREAM and a name).

        Returns:
            ValidationResult indicating success or failure with a message.
        """
        if group.interface_type != InterfaceType.AXI_STREAM:
            return ValidationResult(False, "Invalid group type for AXI-Stream validation.")
        if not group.name:
             return ValidationResult(False, "AXI-Stream group missing name (prefix).")

        # Check for required signals using keys without underscore
        missing = self._check_required_signals(group.ports, AXI_STREAM_SUFFIXES)
        if missing:
            return ValidationResult(False, f"Missing required AXI-Stream signals for '{group.name}': {missing}")

        # Determine expected direction based on prefix/suffix
        lname = group.name.lower()
        is_input_stream = lname.startswith("in") or lname.startswith("s_") or lname.endswith("_s") # Slave/Input
        is_output_stream = lname.startswith("out") or lname.startswith("m_") or lname.endswith("_m") # Master/Output

        if not is_input_stream and not is_output_stream:
             logger.warning(f"Could not determine direction (input/output) for AXI-Stream '{group.name}' based on naming convention.")
             # Proceed, but direction checks will be skipped

        tdata_port = group.ports.get("TDATA")
        if not tdata_port:
            # Should have been caught by missing check, but defensive coding
            return ValidationResult(False, f"AXI-Stream '{group.name}' missing TDATA.")
        # Store data width expression
        group.metadata['data_width_expr'] = tdata_port.width

        # Store keep width expression if TKeep exists
        tkeep_port = group.ports.get("TKEEP") # Assuming TKEEP is the key used by scanner
        if tkeep_port:
            group.metadata['keep_width_expr'] = tkeep_port.width

        # Validate signal directions based on inferred stream direction
        for suffix, port in group.ports.items():
            expected_direction = None
            # Use keys without underscore for checks
            if suffix == "TDATA" or suffix == "TVALID" or suffix == "TLAST":
                # Corrected logic: Input stream means these signals are INPUT
                if is_input_stream: expected_direction = Direction.INPUT
                elif is_output_stream: expected_direction = Direction.OUTPUT
            elif suffix == "TREADY":
                 # Corrected logic: Input stream means TREADY is OUTPUT
                if is_input_stream: expected_direction = Direction.OUTPUT
                elif is_output_stream: expected_direction = Direction.INPUT

            if expected_direction: # Only validate if direction is clear
                error = self._validate_port_properties(port, expected_direction)
                if error:
                    # Report the error using the original port name for clarity
                    return ValidationResult(False, f"Invalid AXI-Stream signal '{port.name}': {error}")
            elif not is_input_stream and not is_output_stream:
                 logger.debug(f"Skipping direction validation for '{port.name}' due to ambiguous stream direction.")
            # else: # Should not happen if suffix is known

        return ValidationResult(True)

    def validate_axi_lite(self, group: PortGroup) -> ValidationResult:
        """Validates a PortGroup identified as a potential AXI-Lite interface.

        Checks for the presence of required signals for read and/or write channels
        based on which signals are present. Validates the direction of present
        signals against AXI_LITE_SUFFIXES definitions.

        Args:
            group: The PortGroup to validate (must have type AXI_LITE and a name).

        Returns:
            ValidationResult indicating success or failure with a message.
        """
        if group.interface_type != InterfaceType.AXI_LITE:
            return ValidationResult(False, "Invalid group type for AXI-Lite validation.")
        if not group.name:
             return ValidationResult(False, "AXI-Lite group missing name (prefix).")

        # group.ports now uses generic keys (e.g., "AWADDR", "WDATA")
        present_signals = set(group.ports.keys())

        # --- Extract and store width information ---
        addr_width_expr = None
        data_width_expr = None
        strb_width_expr = None

        # Address Width (Check AWADDR first, then ARADDR)
        awaddr_port = group.ports.get("AWADDR")
        if awaddr_port:
            addr_width_expr = awaddr_port.width
        else:
            araddr_port = group.ports.get("ARADDR")
            if araddr_port:
                addr_width_expr = araddr_port.width
        if addr_width_expr:
            group.metadata['addr_width_expr'] = addr_width_expr

        # Data Width (Check WDATA first, then RDATA)
        wdata_port = group.ports.get("WDATA")
        if wdata_port:
            data_width_expr = wdata_port.width
        else:
            rdata_port = group.ports.get("RDATA")
            if rdata_port:
                data_width_expr = rdata_port.width
        if data_width_expr:
            group.metadata['data_width_expr'] = data_width_expr

        # Strobe Width
        wstrb_port = group.ports.get("WSTRB")
        if wstrb_port:
            group.metadata['strb_width_expr'] = wstrb_port.width
        # --- End Width Extraction ---


        # Determine if read/write channels are present based on generic signal names
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
        """Validates a PortGroup by dispatching to the appropriate type-specific validator.

        Args:
            group: The PortGroup to validate.

        Returns:
            ValidationResult indicating success or failure with a message.
        """
        if group.interface_type == InterfaceType.GLOBAL_CONTROL:
            return self.validate_global_signals(group)
        elif group.interface_type == InterfaceType.AXI_STREAM:
            return self.validate_axi_stream(group)
        elif group.interface_type == InterfaceType.AXI_LITE:
            return self.validate_axi_lite(group)
        elif group.interface_type == InterfaceType.UNKNOWN:
            logger.debug(f"Skipping validation for UNKNOWN group '{group.name}'")
            # Return True as UNKNOWN groups are not invalid, just unclassified
            return ValidationResult(True, f"Skipping validation for UNKNOWN group '{group.name}'")
        else:
            logger.warning(f"Unknown interface type encountered during validation: {group.interface_type}")
            return ValidationResult(False, f"Unknown interface type: {group.interface_type}")

