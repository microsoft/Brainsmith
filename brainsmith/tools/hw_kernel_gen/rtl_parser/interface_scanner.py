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

from brainsmith.tools.hw_kernel_gen.rtl_parser.data import Port, InterfaceType, PortGroup
from brainsmith.tools.hw_kernel_gen.rtl_parser.protocol_validator import (
    GLOBAL_SIGNAL_SUFFIXES,
    AXI_STREAM_SUFFIXES,
    AXI_LITE_SUFFIXES
)

logger = logging.getLogger(__name__)


class InterfaceScanner:
    """Scans and groups ports into potential interfaces based on naming conventions."""

    def __init__(self, debug: bool = False):
        """Initializes the InterfaceScanner."""
        self.regex = {
            InterfaceType.GLOBAL_CONTROL: self._generate_interface_regex(GLOBAL_SIGNAL_SUFFIXES),
            InterfaceType.AXI_STREAM: self._generate_interface_regex(AXI_STREAM_SUFFIXES),
            InterfaceType.AXI_LITE: self._generate_interface_regex(AXI_LITE_SUFFIXES)
        }
        self.debug = debug

    @staticmethod
    def _generate_interface_regex(suffixes: Dict[str, Dict]) -> re.Pattern:
        """
        Generates a regex pattern for matching interface signals.

        This regex matches signals with an optional prefix and a required suffix.
        For example, it matches both "ap_clk" and "clk" if "clk" is a valid suffix.

        Args:
            suffixes (Dict[str, Dict]): Dictionary of signal suffixes and their properties.

        Returns:
            re.Pattern: A compiled regex pattern that matches signals with an optional prefix
            and a suffix from the provided suffixes.
        """
        return re.compile(
            rf"^(?:(?P<prefix>.*?)_)?(?P<suffix>({ '|'.join(re.escape(k) for k in suffixes.keys()) }))$",
            re.IGNORECASE
        )

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
            InterfaceType.GLOBAL_CONTROL: defaultdict(dict),
            InterfaceType.AXI_STREAM: defaultdict(dict),
            InterfaceType.AXI_LITE: defaultdict(dict)
        }
        unassigned_ports = []

        for port in ports:
            port_assigned = False  # Flag to track if the port has been assigned
            # Check port name against each interface type regex
            for interface_type, regex in self.regex.items():
                logger.debug(f"Checking port '{port.name}' against {interface_type} regex---------------")
                match = regex.match(port.name)
                if match:
                    prefix = match.group("prefix")
                    suffix = match.group("suffix").upper()
                    logger.warning(f"Matched '{port.name}' with prefix '{prefix}' and suffix '{suffix}'")
                    if not prefix:
                        prefix = "<NO_PREFIX>"
                        logger.debug(f"Port '{port.name}' has no prefix, using '<NO_PREFIX>'")
                    # Group valid ports by their interface type and prefix
                    temp_port_groups[interface_type][prefix][suffix] = port
                    logger.warning(f"Assigned '{port.name}' to potential {interface_type} group")
                    port_assigned = True  # Mark port as assigned
                    break  # Port assigned, no need to check other interface types
            
            # If the port was not assigned to any interface type, add to unassigned
            if not port_assigned:
                unassigned_ports.append(port)
                logger.warning(f"Port '{port.name}' did not match any known interface type regex and is unassigned")

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
