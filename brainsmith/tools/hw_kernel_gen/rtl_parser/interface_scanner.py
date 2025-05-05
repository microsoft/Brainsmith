############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

# interface_scanner.py
"""Identifies potential interfaces from port lists."""

import re
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
import logging # Added

# Assuming data structures are in these paths based on context
from brainsmith.tools.hw_kernel_gen.rtl_parser.data import Port, Direction
from brainsmith.tools.hw_kernel_gen.rtl_parser.interface_types import InterfaceType, PortGroup

# --- ADD Import from protocol_validator ---
from brainsmith.tools.hw_kernel_gen.rtl_parser.protocol_validator import (
    GLOBAL_SIGNALS,
    AXI_STREAM_SUFFIXES,
    AXI_LITE_SUFFIXES # Only need combined AXI_LITE for regex generation
)
# --- END Import ---

logger = logging.getLogger(__name__)

# Regex definitions remain here as they are specific to scanning logic
# --- MODIFIED: Adjusted regex to capture suffix *after* the underscore ---
# Dynamically build the regex from AXI_STREAM_SUFFIXES keys (imported, now without underscore)
AXI_STREAM_SPLIT_REGEX = re.compile(
    # Match prefix, optional _V_, then underscore, then capture one of the suffixes
    r"^(?P<prefix>.*?)(?:_V)?_(?P<suffix>(" + "|".join(re.escape(k) for k in AXI_STREAM_SUFFIXES.keys()) + r"))$"
)
# --- END MODIFICATION ---

# Dynamically build the regex from AXI_LITE_SUFFIXES keys (imported)
AXI_LITE_SPLIT_REGEX = re.compile(
    r"^(?P<prefix>.*?)(?:_V)?_(?P<signal>(" + "|".join(re.escape(k) for k in AXI_LITE_SUFFIXES.keys()) + r"))$"
)


class InterfaceScanner:
    """Identifies potential interfaces from port lists."""

    def __init__(self, debug: bool = False):
        self.debug = debug # Placeholder for potential future debugging logs

    def _is_global_signal(self, port_name: str) -> bool:
        """Check if a port name matches a known global signal."""
        return port_name in GLOBAL_SIGNALS

    def _get_axi_stream_parts(self, port_name: str) -> Optional[Tuple[str, str]]:
        """Extract prefix and suffix (without underscore) from potential AXI-Stream port name."""
        match = AXI_STREAM_SPLIT_REGEX.match(port_name)
        if match:
            prefix = match.group("prefix")
            suffix = match.group("suffix") # This suffix is now captured *without* the underscore
            # Ensure the extracted suffix is one we care about (redundant check, but safe)
            if suffix in AXI_STREAM_SUFFIXES: # Check against keys without underscore
                # Added check: Ensure prefix is not empty
                if prefix:
                    return prefix, suffix
                else:
                    logger.debug(f"Port '{port_name}' matched AXI stream suffix but has empty prefix. Ignoring.")
        return None

    def _get_axi_lite_parts(self, port_name: str) -> Optional[Tuple[str, str]]:
        """Extract prefix and generic signal name from potential AXI-Lite port name."""
        match = AXI_LITE_SPLIT_REGEX.match(port_name)
        if match:
            prefix = match.group("prefix")
            signal = match.group("signal")
            # Ensure prefix is not empty
            if prefix:
                return prefix, signal
            else:
                logger.debug(f"Port '{port_name}' matched AXI lite signal but has empty prefix. Ignoring.")
        return None

    def scan(self, ports: List[Port]) -> Tuple[List[PortGroup], List[Port]]:
        """
        Scans a list of ports and groups them into potential interfaces.

        Args:
            ports: A list of Port objects.

        Returns:
            A tuple containing:
                - A list of identified PortGroup objects.
                - A list of Port objects that were not assigned to any group.
        """
        identified_groups: Dict[Tuple[InterfaceType, str], PortGroup] = {}
        assigned_port_names: set[str] = set()
        unassigned_ports: List[Port] = []

        # 1. Identify Global Signals
        global_ports: Dict[str, Port] = {}
        for port in ports:
            if self._is_global_signal(port.name):
                global_ports[port.name] = port
                assigned_port_names.add(port.name)
        if global_ports:
            # Global signals form a single group named "global"
            group_key = (InterfaceType.GLOBAL_CONTROL, "global")
            identified_groups[group_key] = PortGroup(
                interface_type=InterfaceType.GLOBAL_CONTROL,
                name="global",
                ports=global_ports
            )
            logger.debug(f"Identified Global Control group with ports: {list(global_ports.keys())}")

        # 2. Identify AXI-Stream and AXI-Lite Interfaces
        axi_stream_groups: Dict[str, Dict[str, Port]] = defaultdict(dict)
        axi_lite_groups: Dict[str, Dict[str, Port]] = defaultdict(dict)

        for port in ports:
            if port.name in assigned_port_names:
                continue # Skip already assigned global ports

            # Check AXI-Stream
            stream_parts = self._get_axi_stream_parts(port.name)
            if stream_parts:
                prefix, suffix = stream_parts # suffix is now without underscore
                # Use suffix (without underscore) as the key within the group's port dictionary
                axi_stream_groups[prefix][suffix] = port # This should now work correctly
                assigned_port_names.add(port.name)
                logger.debug(f"Assigned '{port.name}' to potential AXI-Stream group '{prefix}' (suffix: {suffix})")
                continue # Move to next port

            # Check AXI-Lite
            lite_parts = self._get_axi_lite_parts(port.name)
            if lite_parts:
                prefix, signal = lite_parts
                # Use generic signal name as the key within the group's port dictionary
                axi_lite_groups[prefix][signal] = port
                assigned_port_names.add(port.name)
                logger.debug(f"Assigned '{port.name}' to potential AXI-Lite group '{prefix}' (signal: {signal})")
                continue # Move to next port

        # Create PortGroup objects for identified AXI streams
        for prefix, ports_dict in axi_stream_groups.items():
            group_key = (InterfaceType.AXI_STREAM, prefix)
            identified_groups[group_key] = PortGroup(
                interface_type=InterfaceType.AXI_STREAM,
                name=prefix,
                ports=ports_dict # ports_dict keys are now without underscore
            )
            logger.debug(f"Created AXI-Stream PortGroup '{prefix}' with signals: {list(ports_dict.keys())}")

        # Create PortGroup objects for identified AXI-Lite interfaces
        for prefix, ports_dict in axi_lite_groups.items():
            group_key = (InterfaceType.AXI_LITE, prefix)
            identified_groups[group_key] = PortGroup(
                interface_type=InterfaceType.AXI_LITE,
                name=prefix,
                ports=ports_dict
            )
            logger.debug(f"Created AXI-Lite PortGroup '{prefix}' with signals: {list(ports_dict.keys())}")

        # 3. Collect Unassigned Ports
        for port in ports:
            if port.name not in assigned_port_names:
                unassigned_ports.append(port)
                logger.debug(f"Port '{port.name}' remains unassigned.")

        return list(identified_groups.values()), unassigned_ports
