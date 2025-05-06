############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Scans a list of SystemVerilog ports to identify potential interface groups.

Uses naming conventions (regex patterns based on protocol definitions) to group
ports belonging to Global Control, AXI-Stream, or AXI-Lite interfaces.
Ports that don't match known patterns are returned as unassigned.
"""

from collections import defaultdict
from typing import List, Dict, Optional, Tuple

from brainsmith.tools.hw_kernel_gen.rtl_parser.data import Port, InterfaceType, PortGroup
# Import protocol definitions for regex generation and global signal check
from brainsmith.tools.hw_kernel_gen.rtl_parser.protocol_validator import (
    GLOBAL_SIGNALS,
    AXI_STREAM_SUFFIXES, # UPPERCASE keys
    AXI_LITE_SUFFIXES    # UPPERCASE keys
)
import re
import logging

logger = logging.getLogger(__name__)

# --- Regex Generation (using required keys) ---
required_axi_stream_keys = {k for k, v in AXI_STREAM_SUFFIXES.items() if v.get("required", False)}
required_axi_lite_keys = {k for k, v in AXI_LITE_SUFFIXES.items() if v.get("required", False)}

AXI_STREAM_SPLIT_REGEX = re.compile(
    r"^(?P<prefix>.*?)(?:_V)?_(?P<suffix>(" + "|".join(re.escape(k) for k in required_axi_stream_keys) + r"))$",
    re.IGNORECASE
) if required_axi_stream_keys else re.compile(r"$.^")

AXI_LITE_SPLIT_REGEX = re.compile(
    r"^(?P<prefix>.*?)(?:_V)?_(?P<signal>(" + "|".join(re.escape(k) for k in required_axi_lite_keys) + r"))$",
    re.IGNORECASE
) if required_axi_lite_keys else re.compile(r"$.^")

# --- Regex for Optional Signals (used in second pass) ---
# Match prefix, optional _V_, then underscore, then ANY known suffix/signal
OPTIONAL_AXI_STREAM_SPLIT_REGEX = re.compile(
    r"^(?P<prefix>.*?)(?:_V)?_(?P<suffix>(" + "|".join(re.escape(k) for k in AXI_STREAM_SUFFIXES.keys()) + r"))$",
    re.IGNORECASE
)
OPTIONAL_AXI_LITE_SPLIT_REGEX = re.compile(
    r"^(?P<prefix>.*?)(?:_V)?_(?P<signal>(" + "|".join(re.escape(k) for k in AXI_LITE_SUFFIXES.keys()) + r"))$",
    re.IGNORECASE
)
# --- End Regex Generation ---


class InterfaceScanner:
    """Scans and groups ports into potential interfaces based on naming conventions."""

    def __init__(self, debug: bool = False):
        """Initializes the InterfaceScanner."""
        self.debug = debug

    def _is_global_signal(self, port_name: str) -> bool:
        """Checks if a port name matches a known global signal name (case-insensitive)."""
        # <<< FIX: Perform case-insensitive check >>>
        return port_name.lower() in (key.lower() for key in GLOBAL_SIGNALS.keys())

    def _get_axi_stream_parts(self, port_name: str) -> Optional[Tuple[str, str]]:
        """Extracts prefix and UPPERCASE suffix from a potential AXI-Stream port name (only matches required suffixes)."""
        match = AXI_STREAM_SPLIT_REGEX.match(port_name)
        if match:
            prefix = match.group("prefix")
            suffix = match.group("suffix")
            suffix_upper = suffix.upper()
            if suffix_upper in required_axi_stream_keys:
                if prefix:
                    return prefix, suffix_upper
                else:
                    logger.debug(f"Port '{port_name}' matched AXI stream suffix but has empty prefix. Ignoring.")
        return None

    def _get_axi_lite_parts(self, port_name: str) -> Optional[Tuple[str, str]]:
        """Extracts prefix and UPPERCASE signal name from a potential AXI-Lite port name (only matches required signals)."""
        match = AXI_LITE_SPLIT_REGEX.match(port_name)
        if match:
            prefix = match.group("prefix")
            signal = match.group("signal")
            signal_upper = signal.upper()
            if signal_upper in required_axi_lite_keys:
                if prefix:
                    return prefix, signal_upper
                else:
                    logger.debug(f"Port '{port_name}' matched AXI lite signal but has empty prefix. Ignoring.")
        return None

    def scan(self, ports: List[Port]) -> Tuple[List[PortGroup], List[Port]]:
        """
        Scans a list of ports and groups them into potential interfaces.

        Iterates through ports, attempting to classify them as Global, AXI-Stream,
        or AXI-Lite based on naming patterns defined by regexes and known signal names.

        Args:
            ports: A list of Port objects extracted from the RTL.

        Returns:
            A tuple containing:
                - A list of identified PortGroup objects, ready for validation.
                - A list of Port objects that did not match any known pattern.
        """
        identified_groups_dict: Dict[Tuple[InterfaceType, str], PortGroup] = {} # Use dict for easier lookup
        assigned_port_names: set[str] = set()

        # 1. Identify Global Signals
        global_ports: Dict[str, Port] = {}
        for port in ports:
            if self._is_global_signal(port.name):
                global_ports[port.name] = port
                assigned_port_names.add(port.name)
        if global_ports:
            logger.debug(f"Creating Global Control group with ports: {list(global_ports.keys())}")
            group_key = (InterfaceType.GLOBAL_CONTROL, "global")
            identified_groups_dict[group_key] = PortGroup(
                interface_type=InterfaceType.GLOBAL_CONTROL,
                name="global",
                ports=global_ports
            )

        # 2. First Pass: Identify Groups based on REQUIRED signals
        # Use lowercase prefix as key for case-insensitivity
        axi_stream_groups_temp: Dict[str, Dict[str, Port]] = defaultdict(dict)
        axi_lite_groups_temp: Dict[str, Dict[str, Port]] = defaultdict(dict)

        for port in ports:
            if port.name in assigned_port_names:
                continue

            # Check AXI-Stream (required signals only)
            stream_parts = self._get_axi_stream_parts(port.name)
            if stream_parts:
                prefix, suffix_upper = stream_parts
                prefix_lower = prefix.lower() # <<< Normalize prefix case >>>
                axi_stream_groups_temp[prefix_lower][suffix_upper] = port
                assigned_port_names.add(port.name)
                logger.debug(f"Pass 1: Assigned '{port.name}' to potential AXI-Stream group '{prefix_lower}' (suffix: {suffix_upper})")
                continue

            # Check AXI-Lite (required signals only)
            lite_parts = self._get_axi_lite_parts(port.name)
            if lite_parts:
                prefix, signal_upper = lite_parts
                prefix_lower = prefix.lower() # <<< Normalize prefix case >>>
                axi_lite_groups_temp[prefix_lower][signal_upper] = port
                assigned_port_names.add(port.name)
                logger.debug(f"Pass 1: Assigned '{port.name}' to potential AXI-Lite group '{prefix_lower}' (signal: {signal_upper})")
                continue

        # Create PortGroup objects from first pass
        for prefix_lower, ports_dict in axi_stream_groups_temp.items():
            group_key = (InterfaceType.AXI_STREAM, prefix_lower)
            identified_groups_dict[group_key] = PortGroup(
                interface_type=InterfaceType.AXI_STREAM,
                name=prefix_lower, # Use normalized name
                ports=ports_dict
            )
            logger.debug(f"Created AXI-Stream PortGroup '{prefix_lower}' with signals: {list(ports_dict.keys())}")

        for prefix_lower, ports_dict in axi_lite_groups_temp.items():
            group_key = (InterfaceType.AXI_LITE, prefix_lower)
            identified_groups_dict[group_key] = PortGroup(
                interface_type=InterfaceType.AXI_LITE,
                name=prefix_lower, # Use normalized name
                ports=ports_dict
            )
            logger.debug(f"Created AXI-Lite PortGroup '{prefix_lower}' with signals: {list(ports_dict.keys())}")

        # 3. Second Pass: Assign Optional Signals to Existing Groups
        remaining_ports = [p for p in ports if p.name not in assigned_port_names]
        still_unassigned_ports = []

        for port in remaining_ports:
            assigned_in_pass2 = False

            # Try matching optional AXI-Stream
            opt_stream_match = OPTIONAL_AXI_STREAM_SPLIT_REGEX.match(port.name)
            if opt_stream_match:
                prefix = opt_stream_match.group("prefix")
                suffix = opt_stream_match.group("suffix")
                prefix_lower = prefix.lower() # Normalize prefix
                suffix_upper = suffix.upper()

                # Check if it's an optional signal AND the group exists
                if not AXI_STREAM_SUFFIXES[suffix_upper].get("required", False):
                    group_key = (InterfaceType.AXI_STREAM, prefix_lower)
                    if group_key in identified_groups_dict:
                        identified_groups_dict[group_key].ports[suffix_upper] = port
                        assigned_port_names.add(port.name) # Mark as assigned now
                        assigned_in_pass2 = True
                        logger.debug(f"Pass 2: Assigned optional AXI-Stream '{port.name}' to group '{prefix_lower}'")

            # Try matching optional AXI-Lite if not assigned yet
            if not assigned_in_pass2:
                opt_lite_match = OPTIONAL_AXI_LITE_SPLIT_REGEX.match(port.name)
                if opt_lite_match:
                    prefix = opt_lite_match.group("prefix")
                    signal = opt_lite_match.group("signal")
                    prefix_lower = prefix.lower() # Normalize prefix
                    signal_upper = signal.upper()

                    # Check if it's an optional signal AND the group exists
                    if not AXI_LITE_SUFFIXES[signal_upper].get("required", False):
                        group_key = (InterfaceType.AXI_LITE, prefix_lower)
                        if group_key in identified_groups_dict:
                            identified_groups_dict[group_key].ports[signal_upper] = port
                            assigned_port_names.add(port.name) # Mark as assigned now
                            assigned_in_pass2 = True
                            logger.debug(f"Pass 2: Assigned optional AXI-Lite '{port.name}' to group '{prefix_lower}'")

            if not assigned_in_pass2:
                still_unassigned_ports.append(port)
                logger.debug(f"Port '{port.name}' remains unassigned after Pass 2.")


        # Final logging and return
        final_groups = list(identified_groups_dict.values())
        if self.debug:
            logger.debug(f"--- Interface Scan Complete ---")
            logger.debug(f"Identified Groups ({len(final_groups)}):")
            for group in final_groups:
                logger.debug(f"  - Name: {group.name}, Type: {group.interface_type.value}, Ports: {sorted(list(group.ports.keys()))}") # Sort keys for consistent logging
            logger.debug(f"Unassigned Ports ({len(still_unassigned_ports)}): {[p.name for p in still_unassigned_ports]}")
            logger.debug(f"--- End Interface Scan ---")

        return final_groups, still_unassigned_ports
