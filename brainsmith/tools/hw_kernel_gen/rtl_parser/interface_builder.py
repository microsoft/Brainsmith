"""Builds validated interface models from a list of ports."""

import logging
from typing import List, Dict, Tuple

from brainsmith.tools.hw_kernel_gen.rtl_parser.data import Port
from brainsmith.tools.hw_kernel_gen.rtl_parser.interface_types import Interface, PortGroup, InterfaceType, ValidationResult
from brainsmith.tools.hw_kernel_gen.rtl_parser.interface_scanner import InterfaceScanner
from brainsmith.tools.hw_kernel_gen.rtl_parser.protocol_validator import ProtocolValidator

logger = logging.getLogger(__name__)

class InterfaceBuilder:
    """Builds validated interface models from a list of ports."""

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.scanner = InterfaceScanner(debug=debug)
        self.validator = ProtocolValidator(debug=debug)

    def build_interfaces(self, ports: List[Port]) -> Tuple[Dict[str, Interface], List[Port]]:
        """
        Builds all valid interfaces from a port list.

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
        unassigned_ports: List[Port] = list(remaining_ports_after_scan) # Start with ports scanner couldn't group

        for group in identified_groups:
            if not group.name:
                logger.warning(f"Skipping group of type {group.interface_type} because it lacks a name.")
                unassigned_ports.extend(group.ports.values())
                continue

            validation_result = self.validator.validate(group)

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
