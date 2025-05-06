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
    "TDATA": {"direction": None, "required": True},
    "TVALID": {"direction": None, "required": True},
    "TREADY": {"direction": None, "required": True},
    "TLAST": {"direction": None, "required": False}, # Optional
}

# Suffixes for AXI-Lite signals
AXI_LITE_SUFFIXES = {
    # Write Address Channel
    "AWADDR": {"direction": Direction.INPUT, "required": True},
    "AWPROT": {"direction": Direction.INPUT, "required": False}, # Optional but usually present
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
    "ARPROT": {"direction": Direction.INPUT, "required": False}, # Optional but usually present
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

    def validate(self, group: PortGroup) -> ValidationResult:
        """Dispatches validation to the appropriate method based on group type."""
        if group.interface_type == InterfaceType.GLOBAL_CONTROL:
            return self.validate_global_signals(group)
        elif group.interface_type == InterfaceType.AXI_STREAM:
            return self.validate_axi_stream(group)
        elif group.interface_type == InterfaceType.AXI_LITE:
            return self.validate_axi_lite(group)
        else:
            # <<< FIX: Return True for UNKNOWN type to skip validation >>>
            logger.debug(f"Skipping validation for unknown interface type '{group.interface_type}' in group '{group.name}'.")
            return ValidationResult(True) # Treat unknown types as valid (or rather, skip validation)

    def _check_required_signals(self, group_ports: Dict[str, Port], required_spec: Dict[str, Dict]) -> Set[str]:
        """Checks if all required signals (keys) are present in the group's ports."""
        # group_ports keys are now UPPERCASE from scanner
        present_keys_upper = set(group_ports.keys())
        # required_spec keys are UPPERCASE from definitions
        required_keys_upper = {key for key, spec in required_spec.items() if spec.get("required", False)}
        missing_upper = required_keys_upper - present_keys_upper
        return missing_upper # Return missing keys in UPPERCASE

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

        # Check for required signals using UPPERCASE spec keys
        # group.ports keys are now UPPERCASE
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

        # Use UPPERCASE keys for metadata and checks
        tdata_port = group.ports.get("TDATA") # Use UPPERCASE key
        if not tdata_port:
             return ValidationResult(False, f"Internal Error: TDATA port not found in group '{group.name}' despite passing required check.")
        group.metadata['data_width_expr'] = tdata_port.width

        tkeep_port = group.ports.get("TKEEP") # Use UPPERCASE key
        if tkeep_port:
            group.metadata['keep_width_expr'] = tkeep_port.width

        # Validate signal directions
        for suffix_upper, port in group.ports.items(): # Iterate through UPPERCASE keys
            if suffix_upper in AXI_STREAM_SUFFIXES:
                # Determine expected direction for THIS suffix
                expected_direction = None
                spec = AXI_STREAM_SUFFIXES[suffix_upper] # Use UPPERCASE key
                # Default directions based on stream type
                if is_input_stream: # Slave/Input
                    expected_direction = Direction.INPUT if suffix_upper == "TVALID" or suffix_upper == "TDATA" or suffix_upper == "TLAST" or suffix_upper == "TKEEP" else Direction.OUTPUT # TREADY is output
                elif is_output_stream: # Master/Output
                    expected_direction = Direction.OUTPUT if suffix_upper == "TVALID" or suffix_upper == "TDATA" or suffix_upper == "TLAST" or suffix_upper == "TKEEP" else Direction.INPUT # TREADY is input

                # Override direction if explicitly defined in spec (though unlikely for AXI-Stream)
                if spec.get("direction") is not None:
                    expected_direction = spec["direction"]

                if expected_direction:
                    error = self._validate_port_properties(port, expected_direction)
                    if error:
                        return ValidationResult(False, f"Invalid AXI-Stream signal '{port.name}': {error}")

        logger.debug(f"  Validation successful for AXI-Stream group '{group.name}'")
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

        present_signals_upper = set(group.ports.keys()) # UPPERCASE keys from scanner

        # Check presence using UPPERCASE keys
        has_write_channel = any(sig in AXI_LITE_WRITE_SUFFIXES for sig in present_signals_upper)
        has_read_channel = any(sig in AXI_LITE_READ_SUFFIXES for sig in present_signals_upper)
        logger.debug(f"  Channel presence: Write={has_write_channel}, Read={has_read_channel}")

        # Check required signals using UPPERCASE spec keys
        if has_write_channel:
            missing_write = self._check_required_signals(group.ports, AXI_LITE_WRITE_SUFFIXES)
            if missing_write:
                logger.debug(f"  Validation failed: Missing required write signals: {missing_write}")
                # Missing keys are returned in UPPERCASE
                return ValidationResult(False, f"Missing required AXI-Lite write signals for '{group.name}': {missing_write}")
        if has_read_channel:
            missing_read = self._check_required_signals(group.ports, AXI_LITE_READ_SUFFIXES)
            if missing_read:
                 logger.debug(f"  Validation failed: Missing required read signals: {missing_read}")
                 # Missing keys are returned in UPPERCASE
                 return ValidationResult(False, f"Missing required AXI-Lite read signals for '{group.name}': {missing_read}")

        # If neither channel seems present based on *any* signal, it's invalid
        if not has_write_channel and not has_read_channel:
             return ValidationResult(False, f"AXI-Lite group '{group.name}' has no recognizable read or write signals.")

        # Validate properties (direction) for all present signals
        for signal_upper, port in group.ports.items(): # Iterate through UPPERCASE keys
             if signal_upper in AXI_LITE_SUFFIXES:
                expected_direction = AXI_LITE_SUFFIXES[signal_upper]["direction"] # Use UPPERCASE key
                # --- ADDED LOGGING ---
                if self.debug:
                    logger.debug(f"  Validating port '{port.name}': Actual Direction='{port.direction.value}', Expected Direction='{expected_direction.value}'")
                # --- END LOGGING ---
                error = self._validate_port_properties(port, expected_direction)
                if error:
                    logger.debug(f"  Validation failed: Port '{port.name}' property error: {error}")
                    return ValidationResult(False, f"Invalid AXI-Lite signal '{port.name}': {error}")

        # --- Extract and store width information ---
        addr_width = None
        data_width = None
        strb_width = None # <<< ADD strb_width >>>
        awaddr_port = group.ports.get("AWADDR")
        araddr_port = group.ports.get("ARADDR")
        wdata_port = group.ports.get("WDATA")
        rdata_port = group.ports.get("RDATA")
        wstrb_port = group.ports.get("WSTRB") # <<< GET WSTRB port >>>

        if awaddr_port: addr_width = awaddr_port.width
        elif araddr_port: addr_width = araddr_port.width

        if wdata_port: data_width = wdata_port.width
        elif rdata_port: data_width = rdata_port.width

        # <<< ADD: Extract strb_width >>>
        if wstrb_port: strb_width = wstrb_port.width

        if addr_width: group.metadata['addr_width_expr'] = addr_width
        if data_width: group.metadata['data_width_expr'] = data_width
        # <<< ADD: Store strb_width_expr >>>
        if strb_width: group.metadata['strb_width_expr'] = strb_width
        # --- End width extraction ---

        logger.debug(f"  Validation successful for AXI-Lite group '{group.name}'")
        return ValidationResult(True)

