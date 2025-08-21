############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Scans and validates interface protocol requirements for ports.

Provides functionality to:
1. Scan ports and group them by interface patterns (scanning)
2. Validate if groups adhere to protocol rules (validation)

Protocol definitions (signal names, requirements) are defined as constants in this module.
"""

import re
import logging
from typing import Dict, Set, List, Tuple, Optional

from brainsmith.core.dataflow.types import Direction, ProtocolType, InterfaceType
from brainsmith.tools.kernel_integrator.metadata import InterfaceMetadata
from .types import Port, PortGroup

# --- Protocol Definitions ---
# Define known signal patterns based on RTL_Parser-Data-Analysis.md
# Keys are uppercase for case-insensitive matching
GLOBAL_SIGNAL_SUFFIXES = {
    "CLK": {"direction": Direction.INPUT, "required": True},
    "RST_N": {"direction": Direction.INPUT, "required": True},
    "CLK2X": {"direction": Direction.INPUT, "required": False},
}

# Suffixes for AXI-Stream signals (direction is slave, opposite for master)
# Keys are uppercase for case-insensitive matching
AXI_STREAM_SUFFIXES = {
    "TDATA": {"direction": Direction.INPUT, "required": True},
    "TVALID": {"direction": Direction.INPUT, "required": True},
    "TREADY": {"direction": Direction.OUTPUT, "required": True},
    "TLAST": {"direction": Direction.INPUT, "required": False}, # Optional
}

# Suffixes for AXI-Lite signals
# Keys are uppercase for case-insensitive matching
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

# Helper sets for channel identification
AXI_LITE_WRITE_SUFFIXES = {k: v for k, v in AXI_LITE_SUFFIXES.items() if k.startswith('AW') or k.startswith('W') or k.startswith('B')}
AXI_LITE_READ_SUFFIXES = {k: v for k, v in AXI_LITE_SUFFIXES.items() if k.startswith('AR') or k.startswith('R')}


logger = logging.getLogger(__name__)


class ProtocolScanner:
    """Scans ports for interface patterns and validates against protocol rules."""

    def __init__(self, debug: bool = False):
        """Initialize scanner state.

        Args:
            debug: Enable verbose debug logging (callers should also configure logging).

        Side Effects:
            Builds in-memory lookup structures (suffix dictionaries and compiled regex
            patterns) used for fast classification of ports during scanning.
        """
        self.debug = debug
        self.suffixes = {
            ProtocolType.CONTROL: GLOBAL_SIGNAL_SUFFIXES,
            ProtocolType.AXI_STREAM: AXI_STREAM_SUFFIXES,
            ProtocolType.AXI_LITE: AXI_LITE_SUFFIXES
        }
        
        # Create regex maps for each interface type
        self.regex_maps = {
            ProtocolType.CONTROL: self._generate_interface_regex(GLOBAL_SIGNAL_SUFFIXES),
            ProtocolType.AXI_STREAM: self._generate_interface_regex(AXI_STREAM_SUFFIXES),
            ProtocolType.AXI_LITE: self._generate_interface_regex(AXI_LITE_SUFFIXES)
        }
        
    @staticmethod
    def _generate_interface_regex(suffixes: Dict[str, Dict]) -> Dict[str, re.Pattern]:
        """
        Generates regex patterns for matching interface signals and maps them to canonical suffixes.

        This creates a mapping from regex pattern to canonical suffix, allowing direct retrieval
        of the correct case when a match is found.

        Args:
            suffixes (Dict[str, Dict]): Dictionary of signal suffixes and their properties.

        Returns:
            Dict[str, re.Pattern]: A dictionary mapping canonical suffix to a compiled regex pattern.
                                  The regex matches both case-insensitive suffixes and other variations.
        """
        regex_map = {}
        for canonical_suffix in suffixes.keys():
            # Create a case-insensitive pattern for this specific suffix
            pattern = re.compile(
                rf"^(?:(?P<prefix>.*?)_)?(?P<suffix>{re.escape(canonical_suffix)})$", 
                re.IGNORECASE
            )
            regex_map[canonical_suffix] = pattern
        return regex_map
    
    def scan(self, ports: List[Port]) -> Tuple[Dict[ProtocolType, Dict[str, InterfaceMetadata]], List[Port]]:
        """Classify raw `Port` objects into protocol interface candidate groups.

        Performs pattern matching of each port name against the compiled regex map for
        every supported protocol. A successful match yields a (protocol, prefix, suffix)
        triple where:
          * protocol: ProtocolType (CONTROL, AXI_STREAM, AXI_LITE)
          * prefix:   User / design specific identifier before the canonical suffix
          * suffix:   Canonical signal name (e.g. TDATA, AWADDR, clk)

        Ports with no match are collected as unassigned. After the full pass an error is
        raised if any unassigned ports remain (current contract: scan is strict).

        Args:
            ports: List of parsed RTL `Port` objects from a module.

        Returns:
            (interfaces_by_protocol, unassigned_ports)
                interfaces_by_protocol: dict keyed by ProtocolType -> dict[prefix] -> InterfaceMetadata
                    Each InterfaceMetadata.ports maps canonical suffix -> Port
                unassigned_ports: list of ports that did not match (always empty on success)

        Raises:
            ValueError: If one or more ports cannot be classified into a known protocol.
        """
        # Buckets for each protocol type
        interfaces_by_protocol: Dict[ProtocolType, Dict[str, InterfaceMetadata]] = {protocol: {} for protocol in self.suffixes}
        unassigned_ports: List[Port] = []

        for port in ports:
            port_assigned = False
            for protocol_type, regex_map in self.regex_maps.items():
                for protocol_suffix, regex in regex_map.items():
                    match = regex.match(port.name)
                    if not match:
                        continue
                    prefix = match.group("prefix") or "<NO_PREFIX>"
                    logger.debug("Matched '%s' with prefix '%s' and protocol suffix '%s'", port.name, prefix, protocol_suffix)

                    # Fetch / create interface metadata bucket for this prefix
                    if prefix not in interfaces_by_protocol[protocol_type]:
                        interfaces_by_protocol[protocol_type][prefix] = InterfaceMetadata(
                            name=prefix,
                            ports={}
                        )
                        logger.debug("Created new potential %s group '%s'", protocol_type, prefix)

                    # Record the port keyed by canonical suffix
                    interfaces_by_protocol[protocol_type][prefix].ports[protocol_suffix] = port
                    logger.debug("Assigned '%s' (suffix '%s') to %s group '%s'", port.name, protocol_suffix, protocol_type, prefix)
                    port_assigned = True
                    break  # Stop after first suffix match for this protocol
                if port_assigned:
                    break  # Stop checking other protocols

            if not port_assigned:
                unassigned_ports.append(port)
                logger.debug("Port '%s' did not match any known interface type regex and is unassigned", port.name)

        if unassigned_ports:
            unassigned_list = ", ".join(p.name for p in unassigned_ports)
            raise ValueError(f"Unassigned ports detected: {unassigned_list}")

        for protocol_type, interfaces in interfaces_by_protocol.items():
            logger.debug("Port groups for %s: %s", protocol_type, list(interfaces.keys()))

        return interfaces_by_protocol, unassigned_ports

    def check_signals(self, interface: InterfaceMetadata, protocol: ProtocolType):
        """Validate presence / absence of expected protocol signals for a group.

        Args:
            interface: Candidate interface (ports keyed by canonical suffix).
            protocol: ProtocolType the interface is assumed to implement.

        Returns:
            dict with protocol-specific metadata. For AXI-Lite returns keys:
                has_write (bool), has_read (bool).
            For other protocols returns an empty dict.

        Raises:
            ValueError: On missing required signals, unexpected extra signals, or
                        invalid partial channel composition (AXI-Lite only).
        """
        metadata = {}
        protocol_suffixes = self.suffixes[protocol]
        # Keys are already uppercase
        present_keys = set(interface.ports.keys())
        required_keys = {key for key, spec in protocol_suffixes.items() if spec["required"] is True}
        optional_keys = {key for key, spec in protocol_suffixes.items() if spec["required"] is False}
        missing = required_keys - present_keys
        unexpected = present_keys - required_keys - optional_keys

        # Special handling for AXI-Lite: support read-only, write-only, etc.
        if protocol == ProtocolType.AXI_LITE:   
            # Check for required signals in write/read channels
            write_missing = {sig for sig in missing if sig in AXI_LITE_WRITE_SUFFIXES}
            read_missing = {sig for sig in missing if sig in AXI_LITE_READ_SUFFIXES}
            has_write_channel = any(sig in interface.ports and AXI_LITE_WRITE_SUFFIXES[sig]['required'] for sig in AXI_LITE_WRITE_SUFFIXES)
            has_read_channel = any(sig in interface.ports and AXI_LITE_READ_SUFFIXES[sig]['required'] for sig in AXI_LITE_READ_SUFFIXES)
            if has_write_channel and write_missing:
                raise ValueError(f"AXI-Lite {interface.name}: Partial write interface, missing required signal(s): {write_missing}")
            if has_read_channel and read_missing:
                raise ValueError(f"AXI-Lite {interface.name}: Partial read interface, missing required signal(s): {read_missing}")
            if not has_write_channel and not has_read_channel:
                raise ValueError(f"AXI-Lite {interface.name}: Not enough valid signals for read or write, missing: {missing}")
            if unexpected:
                raise ValueError(f"AXI-Lite {interface.name}: Unexpected signal(s): {unexpected}")
            metadata['has_write'] = has_write_channel
            metadata['has_read'] = has_read_channel
        else:
            if missing:
                raise ValueError(f"{protocol.name} {interface.name}: Missing required signal(s): {missing}")
            if unexpected:
                raise ValueError(f"{protocol.name} {interface.name}: Unexpected signal(s): {unexpected}")

        return metadata

    def check_direction(self, interface: InterfaceMetadata, protocol: ProtocolType) -> int:
        """Determine overall direction alignment for an interface group.

        Args:
            interface: Candidate interface metadata.
            protocol: ProtocolType under evaluation.

        Returns:
            Direction.INPUT if all ports match expected directions, otherwise
            Direction.OUTPUT if all are inverted.

        Raises:
            ValueError: If a mixture of matching and inverted directions is detected.
        """
        alignment = {}
        protocol_suffixes = self.suffixes[protocol]
        for port_name, port in interface.ports.items():
            # Note: port_name here is already the canonical suffix (e.g., "CLK", "RST_N")
            # from interface.ports which maps suffix -> Port
            expected_direction = protocol_suffixes.get(port_name, {}).get("direction")
            alignment[port_name] = port.direction == expected_direction
        if all(alignment.values()):
            # All ports are aligned
            return Direction.INPUT
        elif not any(alignment.values()):
            # All ports are aligned but inverted
            return Direction.OUTPUT
        else:
            # Mixed alignment
            raise ValueError(f"{protocol.name}: Mixed directionality")
