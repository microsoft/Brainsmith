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

from brainsmith.core.dataflow.types import InterfaceType
from ..metadata import InterfaceMetadata
from brainsmith.tools.kernel_integrator.types.rtl import Port, ProtocolValidationResult, PortGroup
from .interface_scanner import InterfaceScanner
from .protocol_validator import ProtocolValidator

logger = logging.getLogger(__name__)

class InterfaceBuilder:
    """Builds validated interface models by coordinating scanning and validation."""

    def __init__(self, debug: bool = False):
        """Initializes the InterfaceBuilder with scanner and validator instances."""
        self.debug = debug
        self.scanner = InterfaceScanner(debug=debug)
        self.validator = ProtocolValidator(debug=debug)
        
        # Initialize counters for interface naming
        self._interface_counters = {
            InterfaceType.INPUT: 0,
            InterfaceType.OUTPUT: 0,
            InterfaceType.WEIGHT: 0,
            InterfaceType.CONFIG: 0,
            InterfaceType.CONTROL: 0
        }


    def build_interface_metadata(self, ports: List[Port]) -> Tuple[List[InterfaceMetadata], List[Port]]:
        """
        Directly build InterfaceMetadata objects from ports using existing components.
        
        This method leverages the existing InterfaceScanner and ProtocolValidator
        components to group and validate ports, then directly creates InterfaceMetadata
        objects.
        
        Args:
            ports: List of Port objects from RTL parsing
            
        Returns:
            Tuple of (interface_metadata_list, unassigned_ports)
        """
        # Stage 1: Port scanning using existing InterfaceScanner
        port_groups, unassigned_ports = self.scanner.scan(ports)
        
        if self.debug:
            logger.debug(f"--- Port Groups from Scanner ({len(port_groups)}) ---")
            for group in port_groups:
                logger.debug(f"  Group: Name='{group.name}', Type='{group.interface_type.value}', Ports={list(group.ports.keys())}")
            logger.debug(f"--- End Port Groups ---")
        
        # Stage 2: Protocol validation using existing ProtocolValidator
        validated_groups = []
        for group in port_groups:
            if self.debug:
                logger.debug(f"Validating group '{group.name}' with type '{group.interface_type.value}'")
                
            validation_result = self.validator.validate(group)
            
            if validation_result.valid:
                validated_groups.append(group)
                if self.debug:
                    logger.debug(f"  Group '{group.name}' validated successfully")
            else:
                # Add failed group ports back to unassigned
                unassigned_ports.extend(group.ports.values())
                if self.debug:
                    logger.debug(f"  Group '{group.name}' validation failed: {validation_result.message}")
        
        # Stage 3: Direct metadata creation with pragma application
        metadata_list = []
        for group in validated_groups:
            if self.debug:
                logger.debug(f"Creating InterfaceMetadata for group '{group.name}'")
                
            base_metadata = self._create_base_metadata(group)
            metadata_list.append(base_metadata)
            
            if self.debug:
                logger.debug(f"  Created InterfaceMetadata: {base_metadata.name} ({base_metadata.interface_type.value})")
        
        # Sort unassigned ports for consistent output
        unassigned_ports.sort(key=lambda p: p.name)
        
        if self.debug:
            logger.debug(f"--- Final InterfaceMetadata Results ---")
            logger.debug(f"  Created {len(metadata_list)} InterfaceMetadata objects")
            logger.debug(f"  {len(unassigned_ports)} unassigned ports")
            logger.debug(f"--- End Results ---")
        
        # Log interface name sanitization summary
        summary = []
        for interface_type, count in self._interface_counters.items():
            if count > 0:
                if interface_type == InterfaceType.CONTROL:
                    summary.append(f"{count} CONTROL (global)")
                else:
                    summary.append(f"{count} {interface_type.value.upper()}")
        
        if summary:
            logger.info(f"Interface name sanitization complete: {', '.join(summary)}")
        
        return metadata_list, unassigned_ports

    def _create_base_metadata(self, group: PortGroup) -> InterfaceMetadata:
        """
        Create base InterfaceMetadata from validated PortGroup with automatic parameter detection.
        
        The ProtocolValidator has already determined the correct interface_type
        and populated group.metadata with relevant information, so we can
        directly use this validated data.
        
        Args:
            group: Validated PortGroup from ProtocolValidator
            
        Returns:
            InterfaceMetadata: Base metadata with automatic parameter linkage
        """
        # Interface type has been correctly determined by ProtocolValidator
        interface_type = group.interface_type
        
        # NOTE: chunking_strategy is deprecated and no longer set by RTL parser
        # It will remain None and be handled by future AutoHWCustomOp refactoring
        
        # Extract description from validation metadata
        description = f"Interface {group.name} ({interface_type.value})"
        if 'direction' in group.metadata:
            description += f" - Direction: {group.metadata['direction']}"
        
        # Parameter linking is now handled in the parser's _apply_autolinking_to_kernel method
        # InterfaceBuilder just creates base metadata without parameter assumptions
        datatype_params = None
        bdim_param = None
        sdim_param = None
        
        # Update interface counters for naming
        if interface_type == InterfaceType.INPUT:
            self._interface_counters[InterfaceType.INPUT] += 1
        elif interface_type == InterfaceType.OUTPUT:
            self._interface_counters[InterfaceType.OUTPUT] += 1
        elif interface_type == InterfaceType.WEIGHT:
            self._interface_counters[InterfaceType.WEIGHT] += 1
        elif interface_type == InterfaceType.CONFIG:
            self._interface_counters[InterfaceType.CONFIG] += 1
        
        return InterfaceMetadata(
            name=group.name,
            interface_type=interface_type,
            description=description,
            datatype_metadata=None,  # Will be set by pragma application in parser
            bdim_params=[bdim_param] if bdim_param else None,  # Store as list
            sdim_params=[sdim_param] if sdim_param else None   # Store as list
        )
    
    

