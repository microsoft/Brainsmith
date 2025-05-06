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

# Suffixes for AXI-Stream signals
AXI_STREAM_SUFFIXES = {
    "tdata": {"required": True},
    "tvalid": {"required": True},
    "tready": {"required": True},
    "tlast": {"required": False},
}

# AXI-Lite signals. Use lowercase keys.
AXI_LITE_WRITE_SUFFIXES = {
    "awaddr": {"direction": Direction.INPUT, "required": True},
    "awprot": {"direction": Direction.INPUT, "required": False},
    "awvalid": {"direction": Direction.INPUT, "required": True},
    "awready": {"direction": Direction.OUTPUT, "required": True},
    "wdata": {"direction": Direction.INPUT, "required": True},
    "wstrb": {"direction": Direction.INPUT, "required": True},
    "wvalid": {"direction": Direction.INPUT, "required": True},
    "wready": {"direction": Direction.OUTPUT, "required": True},
    "bresp": {"direction": Direction.OUTPUT, "required": True},
    "bvalid": {"direction": Direction.OUTPUT, "required": True},
    "bready": {"direction": Direction.INPUT, "required": True},
}

AXI_LITE_READ_SUFFIXES = {
    "araddr": {"direction": Direction.INPUT, "required": True},
    "arprot": {"direction": Direction.INPUT, "required": False},
    "arvalid": {"direction": Direction.INPUT, "required": True},
    "arready": {"direction": Direction.OUTPUT, "required": True},
    "rdata": {"direction": Direction.OUTPUT, "required": True},
    "rresp": {"direction": Direction.OUTPUT, "required": True},
    "rvalid": {"direction": Direction.OUTPUT, "required": True},
    "rready": {"direction": Direction.INPUT, "required": True},
}

# Combined AXI-Lite for easier checking (now with lowercase keys)
AXI_LITE_SUFFIXES = {**AXI_LITE_WRITE_SUFFIXES, **AXI_LITE_READ_SUFFIXES}
# --- END Protocol Definitions ---

logger = logging.getLogger(__name__)

class ProtocolValidator:
    """Validates PortGroups against defined interface protocol rules."""

    def __init__(self, debug: bool = False):
        """Initializes the ProtocolValidator."""
        self.debug = debug

    def validate(self, group: PortGroup) -> ValidationResult:
        """Dispatches validation to the appropriate method based on group type."""

        if group.interface_type == InterfaceType.GLOBAL_CONTROL:
            return self.validate_global_signals(group)
        elif group.interface_type == InterfaceType.AXI_STREAM:
            return self.validate_axi_stream(group)
        elif group.interface_type == InterfaceType.AXI_LITE:
            return self.validate_axi_lite(group)
        else:
            # Should not happen if scanner only produces known types
            logger.error(f"Unknown interface type '{group.interface_type}' encountered during validation.")
            return ValidationResult(False, f"Unknown interface type: {group.interface_type}")

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

        # --- ADDED LOGGING ---
        logger.debug(f"Validating Global Control group. Received ports: {list(group.ports.keys())}")
        # --- END LOGGING ---

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

        # --- ADDED LOGGING ---
        if self.debug:
            logger.debug(f"--- Validating AXI-Stream Group: '{group.name}' ---")
            logger.debug(f"  Ports received: {list(group.ports.keys())}")
        # --- END LOGGING ---

        # Check for required signals using lowercase keys
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

        # Use lowercase "tdata"
        tdata_port = group.ports.get("tdata")
        if not tdata_port:
            return ValidationResult(False, f"AXI-Stream '{group.name}' missing tdata.")
        group.metadata['data_width_expr'] = tdata_port.width

        # Get tkeep port if present (using lowercase key)
        tkeep_port = group.ports.get("tkeep")
        if tkeep_port:
            group.metadata['keep_width_expr'] = tkeep_port.width
            # Note: We are not *requiring* tkeep, just extracting width if present

        # Validate signal directions based on inferred stream direction
        for suffix, port in group.ports.items(): # suffix is lowercase
            expected_direction = None
            # Use lowercase keys for checks - ONLY check core AXI-Stream signals
            if suffix == "tdata" or suffix == "tvalid" or suffix == "tlast": # Removed tkeep, tstrb, etc.
                if is_input_stream: expected_direction = Direction.INPUT
                elif is_output_stream: expected_direction = Direction.OUTPUT
            elif suffix == "tready":
                if is_input_stream: expected_direction = Direction.OUTPUT
                elif is_output_stream: expected_direction = Direction.INPUT
            # else: # Ignore other optional signals like tkeep for direction validation for now

            if expected_direction:
                # --- ADDED LOGGING ---
                if self.debug:
                    logger.debug(f"  Validating port '{port.name}': Actual Direction='{port.direction.value}', Expected Direction='{expected_direction.value}'")
                # --- END LOGGING ---
                error = self._validate_port_properties(port, expected_direction)
                if error:
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

        # --- ADDED LOGGING ---
        if self.debug:
            logger.debug(f"--- Validating AXI-Lite Group: '{group.name}' ---")
            logger.debug(f"  Ports received: {list(group.ports.keys())}")
        # --- END LOGGING ---

        present_signals = set(group.ports.keys()) # Lowercase keys from scanner

        # Determine which channels are present based on signals
        has_write_channel = any(sig in AXI_LITE_WRITE_SUFFIXES for sig in present_signals)
        has_read_channel = any(sig in AXI_LITE_READ_SUFFIXES for sig in present_signals)

        # Check required signals for present channels
        if has_write_channel:
            missing_write = self._check_required_signals(group.ports, AXI_LITE_WRITE_SUFFIXES)
            if missing_write:
                return ValidationResult(False, f"Missing required AXI-Lite write signals for '{group.name}': {missing_write}")
        if has_read_channel:
            missing_read = self._check_required_signals(group.ports, AXI_LITE_READ_SUFFIXES)
            if missing_read:
                 return ValidationResult(False, f"Missing required AXI-Lite read signals for '{group.name}': {missing_read}")

        # If neither channel seems present based on *any* signal, it's invalid
        if not has_write_channel and not has_read_channel:
             return ValidationResult(False, f"AXI-Lite group '{group.name}' has no recognizable read or write signals.")

        # Validate properties (direction) for all present signals
        for signal_name, port in group.ports.items():
            if signal_name in AXI_LITE_SUFFIXES:
                expected_direction = AXI_LITE_SUFFIXES[signal_name]["direction"]
                # --- ADDED LOGGING ---
                if self.debug:
                    logger.debug(f"  Validating port '{port.name}': Actual Direction='{port.direction.value}', Expected Direction='{expected_direction.value}'")
                # --- END LOGGING ---
                error = self._validate_port_properties(port, expected_direction)
                if error:
                    return ValidationResult(False, f"Invalid AXI-Lite signal '{port.name}': {error}")

        # --- Extract and store width information (using lowercase keys) ---
        addr_width_expr = None
        data_width_expr = None

        # Address Width (Check "awaddr" first, then "araddr")
        awaddr_port = group.ports.get("awaddr")
        if awaddr_port:
            addr_width_expr = awaddr_port.width # Use the actual parsed width string
            logger.debug(f"  Extracted ADDR_WIDTH from awaddr: {addr_width_expr}")
        else:
            araddr_port = group.ports.get("araddr")
            if araddr_port:
                addr_width_expr = araddr_port.width
                logger.debug(f"  Extracted ADDR_WIDTH from araddr: {addr_width_expr}")
        if addr_width_expr:
            group.metadata['addr_width_expr'] = addr_width_expr # Store in metadata

        # Data Width (Check "wdata" first, then "rdata")
        wdata_port = group.ports.get("wdata")
        if wdata_port:
            data_width_expr = wdata_port.width
            logger.debug(f"  Extracted DATA_WIDTH from wdata: {data_width_expr}")
        else:
            rdata_port = group.ports.get("rdata")
            if rdata_port:
                data_width_expr = rdata_port.width
                logger.debug(f"  Extracted DATA_WIDTH from rdata: {data_width_expr}")
        if data_width_expr:
             group.metadata['data_width_expr'] = data_width_expr # Store in metadata

        return ValidationResult(True)

