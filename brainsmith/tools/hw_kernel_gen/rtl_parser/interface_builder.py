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

# Updated import: Interface, PortGroup, InterfaceType, ValidationResult now come from data.py
from brainsmith.tools.hw_kernel_gen.rtl_parser.data import Port, Interface, InterfaceType # <<< Added InterfaceType here
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
        unassigned_ports: List[Port] = list(remaining_ports_after_scan)

        # --- REMOVED DEBUG LOG ---

        # Validate each potential interface
        for group in identified_groups:
            # --- REMOVED DEBUG LOG ---

            validation_result = self.validator.validate(group)

            # --- REMOVED DEBUG LOG ---

            if validation_result.valid:
                # Create the final Interface object
                interface = Interface(
                    name=group.name,
                    type=group.interface_type,
                    ports=group.ports,
                    validation_result=validation_result,
                    metadata=group.metadata # Pass along any metadata gathered during scanning/validation
                )
                validated_interfaces[interface.name] = interface
                if self.debug:
                    logger.debug(f"Successfully validated and built interface: {interface.name} ({interface.type.value})")
            else:
                # Validation failed, add these ports to the unassigned list
                unassigned_ports.extend(group.ports.values())
                logger.warning(f"Validation failed for potential interface '{group.name}' ({group.interface_type.value}): {validation_result.message}")
                if self.debug:
                    logger.debug(f"Ports from failed group '{group.name}': {[p.name for p in group.ports.values()]}")

        # Sort unassigned ports by name for consistency
        unassigned_ports.sort(key=lambda p: p.name)

        return validated_interfaces, unassigned_ports
