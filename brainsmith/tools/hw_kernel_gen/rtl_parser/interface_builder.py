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

from brainsmith.dataflow.core.interface_types import InterfaceType
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata
from brainsmith.dataflow.core.block_chunking import DefaultChunkingStrategy
from .data import Port, ValidationResult, Pragma, PortGroup
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


    def build_interface_metadata(self, ports: List[Port], pragmas: List[Pragma]) -> Tuple[List[InterfaceMetadata], List[Port]]:
        """
        Directly build InterfaceMetadata objects from ports using existing components.
        
        This method leverages the existing InterfaceScanner and ProtocolValidator
        components to group and validate ports, then directly creates InterfaceMetadata
        objects with pragma application.
        
        Args:
            ports: List of Port objects from RTL parsing
            pragmas: List of Pragma objects for interface customization
            
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
        
        return metadata_list, unassigned_ports

    def _create_base_metadata(self, group: PortGroup) -> InterfaceMetadata:
        """
        Create base InterfaceMetadata from validated PortGroup.
        
        The ProtocolValidator has already determined the correct interface_type
        and populated group.metadata with relevant information, so we can
        directly use this validated data.
        
        Args:
            group: Validated PortGroup from ProtocolValidator
            
        Returns:
            InterfaceMetadata: Base metadata with no default datatypes and default chunking
        """
        # Interface type has been correctly determined by ProtocolValidator
        interface_type = group.interface_type
        
        # Start with no default datatype constraints - these should be specified by pragmas
        datatype_constraints = []
        
        # Create appropriate default chunking strategy based on interface type
        # This provides smart defaults when no BDIM pragma is specified
        from brainsmith.dataflow.core.block_chunking import BlockChunkingStrategy
        
        # For now, use simple defaults (will be refined based on actual tensor shape later)
        # The actual tensor shape is not available at RTL parsing time
        # Use only parameter names and ":" - NO magic numbers allowed
        if interface_type in [InterfaceType.INPUT, InterfaceType.OUTPUT]:
            # Default for activations: process rightmost dimensions
            chunking_strategy = BlockChunkingStrategy(block_shape=[":", ":"], rindex=0)
        elif interface_type == InterfaceType.WEIGHT:
            # Default for weights: use PE parameter for parameterizability
            chunking_strategy = BlockChunkingStrategy(block_shape=["PE"], rindex=0)
        else:
            # Default for others: full tensor
            chunking_strategy = BlockChunkingStrategy(block_shape=[":"], rindex=0)
        
        # Extract description from validation metadata
        description = f"Interface {group.name} ({interface_type.value})"
        if 'direction' in group.metadata:
            description += f" - Direction: {group.metadata['direction'].value}"
        
        return InterfaceMetadata(
            name=group.name,
            interface_type=interface_type,
            datatype_constraints=datatype_constraints,
            chunking_strategy=chunking_strategy,
            description=description
        )

