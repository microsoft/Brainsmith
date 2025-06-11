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

import re
import logging
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

from brainsmith.dataflow.core.interface_types import InterfaceType
from .data import Port, PortGroup
from .protocol_validator import (
    GLOBAL_SIGNAL_SUFFIXES,
    AXI_STREAM_SUFFIXES,
    AXI_LITE_SUFFIXES
)

logger = logging.getLogger(__name__)


class InterfaceScanner:
    """Scans and groups ports into potential interfaces based on naming conventions."""

    def __init__(self, debug: bool = False):
        """Initializes the InterfaceScanner."""
        # Map protocol patterns to preliminary interface types 
        # Protocol validator will determine specific dataflow types later
        self.suffixes = {
            InterfaceType.CONTROL: GLOBAL_SIGNAL_SUFFIXES,  # Global control → CONTROL
            InterfaceType.INPUT: AXI_STREAM_SUFFIXES,       # AXI-Stream → INPUT (refined by validator)
            InterfaceType.CONFIG: AXI_LITE_SUFFIXES         # AXI-Lite → CONFIG
        }
        # Create regex maps for each interface type
        self.regex_maps = {
            InterfaceType.CONTROL: self._generate_interface_regex(GLOBAL_SIGNAL_SUFFIXES),
            InterfaceType.INPUT: self._generate_interface_regex(AXI_STREAM_SUFFIXES),
            InterfaceType.CONFIG: self._generate_interface_regex(AXI_LITE_SUFFIXES)
        }
        self.debug = debug

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
        port_groups = []
        temp_port_groups = {
            InterfaceType.CONTROL: defaultdict(dict),    # Global control → CONTROL
            InterfaceType.INPUT: defaultdict(dict),      # AXI-Stream → INPUT (refined by validator) 
            InterfaceType.CONFIG: defaultdict(dict)      # AXI-Lite → CONFIG
        }
        unassigned_ports = []

        for port in ports:
            port_assigned = False  # Flag to track if the port has been assigned
            # Check port name against each interface type regex map
            for interface_type, regex_map in self.regex_maps.items():
                # Try each canonical suffix regex until a match is found
                for canonical_suffix, regex in regex_map.items():
                    logger.debug(f"Checking port '{port.name}' against {interface_type} regex for '{canonical_suffix}'")
                    match = regex.match(port.name)
                    if match:
                        prefix = match.group("prefix")
                        logger.debug(f"Matched '{port.name}' with prefix '{prefix}' and canonical suffix '{canonical_suffix}'")
                        if not prefix:
                            prefix = "<NO_PREFIX>"
                            logger.debug(f"Port '{port.name}' has no prefix, using '<NO_PREFIX>'")
                        
                        # Group valid ports by their interface type and prefix, using canonical suffix as key
                        temp_port_groups[interface_type][prefix][canonical_suffix] = port
                        logger.debug(f"Assigned '{port.name}' to potential {interface_type} group with canonical suffix '{canonical_suffix}'")
                        port_assigned = True  # Mark port as assigned
                        break  # Skip checking other suffixes for this interface type
                
                if port_assigned:
                    break  # Skip checking other interface types if port is already assigned
            
            # If the port was not assigned to any interface type, add to unassigned
            if not port_assigned:
                unassigned_ports.append(port)
                logger.debug(f"Port '{port.name}' did not match any known interface type regex and is unassigned")

        # Create PortGroup objects from potential groups
        for interface_type, groups_dict in temp_port_groups.items():
            for prefix, ports_dict in groups_dict.items():
                port_groups.append(PortGroup(
                    interface_type=interface_type,
                    name=prefix,
                    ports=ports_dict
                ))
                logger.debug(f"Created {interface_type} PortGroup '{prefix}' with signals: {list(ports_dict.keys())}")

        logger.debug(f"Total PortGroups created: {len(port_groups)}")
        return port_groups, unassigned_ports
