############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Coordinates interface identification and validation.

Groups ports based on naming conventions and uses ProtocolValidator to check 
if the groups adhere to specific interface rules (e.g., AXI-Stream, AXI-Lite).
Returns validated Interface objects and any ports that couldn't be assigned 
to a valid interface.
"""

import re
import logging
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

from brainsmith.core.dataflow.types import InterfaceType
from brainsmith.tools.kernel_integrator.types.metadata import InterfaceMetadata
from brainsmith.tools.kernel_integrator.types.rtl import Port, ProtocolValidationResult, PortGroup
from .protocol_validator import (
    ProtocolValidator,
    GLOBAL_SIGNAL_SUFFIXES,
    AXI_STREAM_SUFFIXES,
    AXI_LITE_SUFFIXES
)

logger = logging.getLogger(__name__)

class InterfaceBuilder:
    """Builds validated interface models by coordinating scanning and validation."""

    def __init__(self, debug: bool = False):
        """Initializes the InterfaceBuilder with integrated scanning and validation."""
        self.debug = debug
        self.validator = ProtocolValidator(debug=debug)
        
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
        
        # Initialize counters for interface naming
        self._interface_counters = {
            InterfaceType.INPUT: 0,
            InterfaceType.OUTPUT: 0,
            InterfaceType.WEIGHT: 0,
            InterfaceType.CONFIG: 0,
            InterfaceType.CONTROL: 0
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

    def _scan(self, ports: List[Port]) -> Tuple[List[PortGroup], List[Port]]:
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

    def build_interface_metadata(self, ports: List[Port]) -> Tuple[List[InterfaceMetadata], List[Port]]:
        """
        Directly build InterfaceMetadata objects from ports.
        
        This method groups and validates ports, then directly creates InterfaceMetadata
        objects.
        
        Args:
            ports: List of Port objects from RTL parsing
            
        Returns:
            Tuple of (interface_metadata_list, unassigned_ports)
        """
        # Stage 1: Port scanning using integrated scan method
        port_groups, unassigned_ports = self._scan(ports)
        
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
    
    

