############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Coordinates interface identification and validation.

Uses InterfaceScanner to group ports based on naming conventions and
ProtocolValidator to check if the groups adhere to specific interface rules
(e.g., AXI-Stream, AXI-Lite). Returns validated Interface objects and
any ports that couldn't be assigned to a valid interface.
"""

import logging
from typing import List, Dict, Tuple

from brainsmith.tools.hw_kernel_gen.rtl_parser.data import Port, Interface, InterfaceType
from brainsmith.tools.hw_kernel_gen.rtl_parser.interface_scanner import InterfaceScanner
from brainsmith.tools.hw_kernel_gen.rtl_parser.protocol_validator import ProtocolValidator

logger = logging.getLogger(__name__)

class InterfaceBuilder:
    """Builds validated interface models by coordinating scanning and validation."""

    def __init__(self, debug: bool = False):
        """Initializes the InterfaceBuilder with scanner and validator instances."""
        self.debug = debug
        self.scanner = InterfaceScanner(debug=debug)
        self.validator = ProtocolValidator(debug=debug)

    def build_interfaces(self, ports: List[Port]) -> Tuple[Dict[str, Interface], List[Port]]:
        """
        Builds all valid interfaces from a port list.

        First, scans the ports to create potential PortGroups. Then, validates
        each group against protocol rules. Valid groups are converted to
        Interface objects.

        Args:
            ports: List of Port objects from the parsed module.

        Returns:
            A tuple containing:
            - Dictionary mapping interface names (e.g., "global", "in0", "config")
              to validated Interface objects.
            - List of ports that were not assigned to any valid interface.
        """
        identified_groups, remaining_ports_after_scan = self.scanner.scan(ports)
        validated_interfaces: Dict[str, Interface] = {}
        unassigned_ports: List[Port] = list(remaining_ports_after_scan) # Keep initialization

        # Keep original debug logging
        if self.debug:
            logger.debug(f"--- Groups received by InterfaceBuilder from Scanner ---")
            for group in identified_groups:
                logger.debug(f"  Scanner Group: Name='{group.name}', Type='{group.interface_type.value}', Ports={list(group.ports.keys())}")
            logger.debug(f"--- End Scanner Groups ---")

        for group in identified_groups:
            if self.debug:
                 logger.debug(f"Validating group '{group.name}' with type '{group.interface_type.value}' using ProtocolValidator.")

            validation_result = self.validator.validate(group)

            if self.debug:
                logger.debug(f"  Validation result for '{group.name}': Is Valid={validation_result.valid}, Reason='{validation_result.message}'")

            if validation_result.valid:
                # Create Interface object
                interface = Interface(
                    name=group.name,
                    type=group.interface_type,
                    ports=group.ports,
                    metadata=group.metadata,
                    validation_result=validation_result # Store the result
                )
                validated_interfaces[interface.name] = interface
                if self.debug:
                    logger.debug(f"Successfully validated and built interface: {interface.name} ({interface.type.value})")
            else:
                # Add ports from the failed group back to the unassigned list
                unassigned_ports.extend(group.ports.values())
                logger.warning(f"Validation failed for potential interface '{group.name}' ({group.interface_type.value}): {validation_result.message}")
                if self.debug:
                    logger.debug(f"Ports from failed group '{group.name}': {[p.name for p in group.ports.values()]}")

        # Sort unassigned ports alphabetically by name for consistent output
        unassigned_ports.sort(key=lambda p: p.name)

        # Final debug log for unassigned ports
        if self.debug:
            logger.debug(f"--- Final Unassigned Ports ({len(unassigned_ports)}) ---")
            for port in unassigned_ports:
                logger.debug(f"  - {port.name}")
            logger.debug(f"--- End Unassigned Ports ---")


        return validated_interfaces, unassigned_ports
