############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Kernel metadata builder for SystemVerilog RTL parser.

This module handles:
- Building complete KernelMetadata objects from extracted components
- Interface detection and validation
- KernelMetadata assembly with proper interface organization
"""

import logging
from collections import defaultdict

from brainsmith.dataflow.types import InterfaceType, ProtocolType
from brainsmith.tools.kernel_integrator.metadata import (
    AXILiteMetadata,
    AXIStreamMetadata,
    ControlMetadata,
    InterfaceMetadata,
    KernelMetadata,
)

from .ast_parser import ASTParser
from .protocol_validator import ProtocolScanner
from .types import Direction, ParsedModule, Port

logger = logging.getLogger(__name__)


class KernelBuilder:
    """Builds KernelMetadata from extracted module components.

    This class handles:
    - Interface building and validation (via ProtocolScanner)
    - KernelMetadata assembly from extracted components
    """

    def __init__(self, ast_parser: ASTParser, debug: bool = False):
        """Initialize the kernel builder.

        Args:
            ast_parser: ASTParser instance for node traversal utilities.
            debug: Enable debug logging.
        """
        self.ast_parser = ast_parser
        self.scanner = ProtocolScanner(debug=debug)
        self.debug = debug

    def build(self, parsed_module: ParsedModule) -> KernelMetadata:
        """Build complete `KernelMetadata` from a ParsedModule.

        Steps:
            1. Build protocol-grouped interface metadata from ports.
            2. Enforce required invariants (exactly one control, at least one input & output stream).
            3. Assemble and return `KernelMetadata` object.

        Args:
            parsed_module: ParsedModule containing all extracted components.
            source_name: Optional source file path override (for diagnostics/meta).

        Returns:
            KernelMetadata: Fully assembled kernel description.

        Raises:
            ValueError: If protocol/interface invariants are violated.
        """
        # Extract components from ParsedModule
        module_name = parsed_module.name
        parameters = parsed_module.parameters
        ports = parsed_module.ports
        source_path = parsed_module.file_path

        # (1) Build interfaces organized by protocol type
        interfaces_by_protocol = self.scan_and_build_interfaces(ports)

        # (2) Organize / validate interface types
        control_interface = None
        input_interfaces = []
        output_interfaces = []
        config_interfaces = []

        # Control (exactly one)
        if ProtocolType.CONTROL in interfaces_by_protocol:
            control_candidates = interfaces_by_protocol[ProtocolType.CONTROL]
            if len(control_candidates) != 1:
                raise ValueError(
                    f"Expected exactly one control interface, found {len(control_candidates)}"
                )
            control_interface = control_candidates[0]
        else:
            raise ValueError("No control interface found")

        # AXI-Stream (must have >=1 input and >=1 output)
        if ProtocolType.AXI_STREAM in interfaces_by_protocol:
            for iface in interfaces_by_protocol[ProtocolType.AXI_STREAM]:
                if iface.interface_type == InterfaceType.INPUT:
                    input_interfaces.append(iface)
                elif iface.interface_type == InterfaceType.OUTPUT:
                    output_interfaces.append(iface)

        if not input_interfaces:
            raise ValueError("No AXI-Stream input interfaces found")
        if not output_interfaces:
            raise ValueError("No AXI-Stream output interfaces found")

        # AXI-Lite (optional configuration interfaces)
        if ProtocolType.AXI_LITE in interfaces_by_protocol:
            config_interfaces = interfaces_by_protocol[ProtocolType.AXI_LITE]

        # (3) Assemble metadata
        return KernelMetadata(
            name=module_name,
            source_file=str(source_path),
            control=control_interface,
            inputs=input_interfaces,
            outputs=output_interfaces,
            config=config_interfaces,
            parameters=parameters,
        )

    def scan_and_build_interfaces(
        self, ports: list[Port]
    ) -> dict[ProtocolType, list[InterfaceMetadata]]:
        """Scan ports for protocol patterns and build validated interface metadata.

        This method:
        1. Scans ports to detect protocol groups using regex patterns
        2. Validates each group against protocol requirements
        3. Builds appropriate metadata objects (ControlMetadata, AXIStreamMetadata, etc.)

        Args:
            ports: List of Port objects to organize into interfaces.

        Returns:
            Dictionary mapping ProtocolType to list of InterfaceMetadata objects.

        Raises:
            ValueError: If any ports cannot be assigned to a valid interface.
        """
        # Stage 1: Scan ports to detect groups using regex patterns
        port_groups, unassigned_ports = self.scanner.scan(ports)

        # Fail fast if any ports could not be assigned to an interface
        if unassigned_ports:
            missing = ", ".join(p.name for p in unassigned_ports)
            raise ValueError(f"Unassigned ports after scanning: {missing}")

        # Stage 2: Process each protocol type and build appropriate metadata
        validated_interfaces = defaultdict(list)
        for protocol_type, prefix_interfaces in port_groups.items():
            for prefix, interface in prefix_interfaces.items():
                if self.debug:
                    logger.debug(
                        f"Validating group with prefix '{prefix}' and protocol '{protocol_type}'"
                    )

                # Build appropriate metadata based on protocol type
                if protocol_type == ProtocolType.CONTROL:
                    metadata = self.build_global_control(interface)
                elif protocol_type == ProtocolType.AXI_STREAM:
                    metadata = self.build_axi_stream(interface)
                elif protocol_type == ProtocolType.AXI_LITE:
                    metadata = self.build_axi_lite(interface)
                else:
                    raise ValueError(f"Invalid protocol type: {protocol_type}")

                validated_interfaces[protocol_type].append(metadata)

        return validated_interfaces

    def build_global_control(self, interface: InterfaceMetadata) -> ControlMetadata:
        """Build and validate a control interface."""
        # Check against required & expected signals
        _ = self.scanner.check_signals(interface, ProtocolType.CONTROL)

        # Validate direction alignment (must match expected, not inverted or mixed)
        direction = self.scanner.check_direction(interface, ProtocolType.CONTROL)
        if direction != Direction.INPUT:
            raise ValueError(f"Control Interface {interface.name}: Invalid direction: {direction}")

        return ControlMetadata(name=interface.name, ports=interface.ports)

    def build_axi_stream(self, interface: InterfaceMetadata) -> AXIStreamMetadata:
        """Validate an AXI-Stream interface group."""
        # Check against required & expected signals
        _ = self.scanner.check_signals(interface, ProtocolType.AXI_STREAM)

        # Validate direction alignment
        direction = self.scanner.check_direction(interface, ProtocolType.AXI_STREAM)

        return AXIStreamMetadata(name=interface.name, ports=interface.ports, direction=direction)

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
            has_write=metadata["has_write"],
            has_read=metadata["has_read"],
        )
