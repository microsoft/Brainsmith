############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Builds validated interface models from ports.

Uses ProtocolScanner to identify and validate port groups, then creates
Interface metadata objects. Returns validated Interface objects and any 
ports that couldn't be assigned to a valid interface.
"""

import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from brainsmith.core.dataflow.types import ProtocolType, InterfaceType
from brainsmith.tools.kernel_integrator.types.rtl import Port, PortGroup, Direction
from brainsmith.tools.kernel_integrator.types.metadata import InterfaceMetadata, AXIStreamMetadata, AXILiteMetadata, ControlMetadata
from .protocol_validator import ProtocolScanner

logger = logging.getLogger(__name__)

class InterfaceBuilder:
    """Builds validated interface models from ports."""

    def __init__(self, debug: bool = False):
        """Initializes the InterfaceBuilder."""
        self.debug = debug
        self.scanner = ProtocolScanner(debug=debug)

    def build_from_ports(self, ports: List[Port]) -> Dict[ProtocolType, List[InterfaceMetadata]]:
        # Stage 1: Scan ports to detect groups using regex patterns
        port_groups, unassigned_ports = self.scanner.scan(ports)

        # Fail fast if any ports could not be assigned to an interface
        if unassigned_ports:
            missing = ", ".join(p.name for p in unassigned_ports)
            raise ValueError(f"Unassigned ports after scanning: {missing}")

        # Process each protocol type
        validated_interfaces = defaultdict(list)
        for protocol_type, prefix_interfaces in port_groups.items():
            for prefix, interfaces in prefix_interfaces.items():
                if self.debug:
                    logger.debug(f"Validating group with prefix '{prefix}' and protocol '{protocol_type}'")

                validated_interfaces[protocol_type].extend(
                    self.build_interface_metadata(protocol_type, interfaces)
                )
        
        return validated_interfaces


    def build_interface_metadata(self, protocol: ProtocolType, interfaces: List[InterfaceMetadata]) -> List[InterfaceMetadata]:
        """Build specific metadata types for each interface based on protocol."""
        result = []
        for interface in interfaces:
            if protocol == ProtocolType.CONTROL:
                result.append(self.build_global_control(interface))
            elif protocol == ProtocolType.AXI_STREAM:
                result.append(self.build_axi_stream(interface))
            elif protocol == ProtocolType.AXI_LITE:
                result.append(self.build_axi_lite(interface))
            else:
                raise ValueError(f"Invalid protocol type: {protocol}")
        return result


    def build_global_control(self, interface: InterfaceMetadata) -> ControlMetadata:
        # Check against required & expected signals
        _ = self.scanner.check_signals(interface, ProtocolType.CONTROL)

        # Validate direction alignment (must match expected, not inverted or mixed)
        direction = self.scanner.check_direction(interface, ProtocolType.CONTROL)
        if direction != Direction.INPUT:
            raise ValueError(f"Control Interface {interface.name}: Invalid direction: {direction}")

        return ControlMetadata(name=interface.name, ports=interface.ports)


    def build_axi_stream(self,  interface: InterfaceMetadata) -> AXIStreamMetadata:
        """Validate an AXI-Stream interface group."""
        # Check against required & expected signals
        _ = self.scanner.check_signals(interface, ProtocolType.AXI_STREAM)

        # Validate direction alignment, determine input or output
        direction = self.scanner.check_direction(interface, ProtocolType.AXI_STREAM)
        if direction == Direction.INPUT:
            interface_type = InterfaceType.INPUT
        else:
            interface_type = InterfaceType.OUTPUT

        return AXIStreamMetadata(
            name=interface.name,
            ports=interface.ports,
            direction=direction,
            interface_type=interface_type
        )


    def build_axi_lite(self, interface: InterfaceMetadata) -> AXILiteMetadata:
        """Validate an AXI-Lite interface group."""
        # Check against required & expected signals
        metadata = self.scanner.check_signals(interface, ProtocolType.AXI_LITE)
        
        # Validate direction alignment
        direction = self.scanner.check_direction(interface, ProtocolType.AXI_LITE)
        if direction != Direction.INPUT:
            raise ValueError(f"AXI-Lite Interface {interface.name}: Invalid direction: {direction}")
        
        return AXILiteMetadata(
            name=interface.name,
            ports=interface.ports,
            has_write=metadata['has_write'],
            has_read=metadata['has_read']
        )
