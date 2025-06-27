############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Parameter linking service for automatic datatype parameter detection.

This module provides automatic linking of RTL parameters to datatype properties
based on naming conventions. It uses a naive prefix-based approach where
parameters with the same prefix are grouped together.
"""

import re
import logging
from typing import List, Dict, Set, Optional, Tuple, TYPE_CHECKING
from collections import defaultdict

from ..metadata import DatatypeMetadata, KernelMetadata
from .rtl_data import Parameter

logger = logging.getLogger(__name__)


# Suffix patterns for datatype properties
PROPERTY_SUFFIXES = [
    ("_WIDTH", "width"),
    ("_SIGNED", "signed"),
    ("_BIAS", "bias"),
    ("_FORMAT", "format"),
    ("_FRACTIONAL_WIDTH", "fractional_width"),
    ("_EXPONENT_WIDTH", "exponent_width"),
    ("_MANTISSA_WIDTH", "mantissa_width"),
]

# Special standalone parameter mappings (exact match)
# Currently empty - all parameters use prefix-based matching
STANDALONE_MAPPINGS = {}


class ParameterLinker:
    """Links RTL parameters to datatype properties using naive prefix grouping."""
    
    def __init__(self, enable_interface_linking: bool = True, 
                 enable_internal_linking: bool = True):
        """
        Initialize the parameter linker.
        
        Args:
            enable_interface_linking: Enable auto-linking for interface parameters
            enable_internal_linking: Enable auto-linking for internal parameters
        """
        self.enable_interface_linking = enable_interface_linking
        self.enable_internal_linking = enable_internal_linking
    
    def link_interface_parameters(self, interface_name: str, 
                                parameters: List[Parameter]) -> Optional[DatatypeMetadata]:
        """
        Link parameters to an interface based on naming patterns.
        
        Args:
            interface_name: Name of the interface
            parameters: List of all module parameters
            
        Returns:
            DatatypeMetadata with linked parameters, or None if no links found
        """
        if not self.enable_interface_linking:
            return None
        
        logger.debug(f"Attempting to auto-link parameters for interface '{interface_name}'")
        
        # Build parameter name lookup
        param_names = {p.name for p in parameters}
        
        # Try to find parameters matching interface patterns
        linked_params = {}
        
        for suffix, property_name in PROPERTY_SUFFIXES:
            # Construct expected parameter name
            expected_name = f"{interface_name}{suffix}"
            
            if expected_name in param_names:
                linked_params[property_name] = expected_name
                logger.debug(f"Auto-linked '{expected_name}' to {property_name} for interface '{interface_name}'")
        
        # Create DatatypeMetadata if we found any parameters
        if linked_params:
            return DatatypeMetadata(
                name=interface_name,
                width=linked_params.get("width"),
                signed=linked_params.get("signed"),
                format=linked_params.get("format"),
                bias=linked_params.get("bias"),
                fractional_width=linked_params.get("fractional_width"),
                exponent_width=linked_params.get("exponent_width"),
                mantissa_width=linked_params.get("mantissa_width"),
                description="Auto-linked from naming conventions"
            )
        
        return None
    
    def link_internal_parameters(self, parameters: List[Parameter], 
                                exclude_prefixes: Optional[List[str]] = None,
                                exclude_parameters: Optional[set] = None) -> List[DatatypeMetadata]:
        """
        Detect and link parameters for internal mechanisms using naive prefix grouping.
        
        This implementation:
        1. Extracts the prefix from parameters ending with recognized suffixes
        2. Groups parameters by common prefix
        3. Each prefix becomes a datatype name (e.g., THRESH_* → "THRESH", T_WIDTH → "T")
        
        Args:
            parameters: List of all module parameters
            exclude_prefixes: Optional list of prefixes to exclude (e.g., interface names)
            exclude_parameters: Optional set of parameter names to exclude (already claimed by pragmas)
            
        Returns:
            List of DatatypeMetadata for internal mechanisms
        """
        if not self.enable_internal_linking:
            return []
        
        logger.debug("Attempting to auto-link internal datatype parameters")
        
        # Track which parameters have been processed
        processed_params = set()
        
        # Group parameters by prefix
        prefix_groups = defaultdict(dict)
        
        for param in parameters:
            # Skip if already processed
            if param.name in processed_params:
                continue
            
            # Skip if parameter is explicitly excluded (already claimed by pragma)
            if exclude_parameters and param.name in exclude_parameters:
                logger.debug(f"Skipping '{param.name}' - parameter is excluded (claimed by pragma)")
                continue
            
            # Check for suffix patterns
            for suffix, property_name in PROPERTY_SUFFIXES:
                if param.name.endswith(suffix):
                    # Extract prefix
                    prefix = param.name[:-len(suffix)]
                    if prefix:  # Only if there's actually a prefix
                        # Check if prefix should be excluded
                        if exclude_prefixes and prefix in exclude_prefixes:
                            logger.debug(f"Skipping '{param.name}' - prefix '{prefix}' is excluded")
                            continue
                        prefix_groups[prefix][property_name] = param.name
                        processed_params.add(param.name)
                        logger.debug(f"Auto-linked '{param.name}' to {prefix}.{property_name}")
                        break
        
        # Create DatatypeMetadata objects
        internal_datatypes = []
        
        for prefix, properties in prefix_groups.items():
            # Create datatype with whatever properties we found
            dt_metadata = DatatypeMetadata(
                name=prefix,
                width=properties.get("width"),
                signed=properties.get("signed"),
                format=properties.get("format"),
                bias=properties.get("bias"),
                fractional_width=properties.get("fractional_width"),
                exponent_width=properties.get("exponent_width"),
                mantissa_width=properties.get("mantissa_width"),
                description=f"Auto-linked internal datatype from prefix '{prefix}'"
            )
            internal_datatypes.append(dt_metadata)
            logger.info(f"Created auto-linked internal datatype '{prefix}' with {len(properties)} parameters")
        
        return internal_datatypes
    
    def find_linked_parameters(self, datatype_metadata: DatatypeMetadata) -> Set[str]:
        """
        Find which parameters are linked by a DatatypeMetadata.
        
        Args:
            datatype_metadata: DatatypeMetadata to inspect
            
        Returns:
            Set of parameter names linked by this metadata
        """
        return set(datatype_metadata.get_all_parameters())
    
    def collect_dimension_parameters(self, interface_name: str, parameters: List[Parameter], 
                                   dimension_type: str) -> Tuple[Optional[str], Optional[List[str]]]:
        """
        Collect dimension parameters for an interface.
        
        This method supports both single parameter and indexed parameter conventions:
        - Single: <interface>_BDIM or <interface>_SDIM
        - Indexed: <interface>_BDIM0, <interface>_BDIM1, etc.
        
        For indexed parameters, gaps are filled with "1" (singleton dimensions).
        
        Args:
            interface_name: Name of the interface
            parameters: List of all module parameters
            dimension_type: Either "BDIM" or "SDIM"
            
        Returns:
            Tuple of (single_param_name, indexed_params_list)
            Only one will be non-None, based on what's found.
        """
        # Build parameter name lookup for efficiency
        param_names = {p.name for p in parameters}
        
        # Check for single parameter first (existing convention)
        single_name = f"{interface_name}_{dimension_type}"
        if single_name in param_names:
            logger.debug(f"Found single {dimension_type} parameter: {single_name}")
            return single_name, None
        
        # Check for indexed parameters
        indexed_params = {}
        # Support both uppercase and lowercase variants
        patterns = [
            re.compile(f"{interface_name}_{dimension_type}(\\d+)"),
            re.compile(f"{interface_name}_{dimension_type.lower()}(\\d+)")
        ]
        
        for param in parameters:
            for pattern in patterns:
                match = pattern.match(param.name)
                if match:
                    index = int(match.group(1))
                    indexed_params[index] = param.name
                    logger.debug(f"Found indexed {dimension_type} parameter: {param.name} at index {index}")
                    break
        
        if indexed_params:
            # Convert to list, handling gaps with singleton dimensions
            max_index = max(indexed_params.keys())
            result = []
            for i in range(max_index + 1):
                if i in indexed_params:
                    result.append(indexed_params[i])
                else:
                    # Gap in indices - treat as singleton dimension
                    result.append("1")
                    logger.debug(f"Gap at index {i} for {interface_name}_{dimension_type}, using singleton")
            
            logger.info(f"Collected indexed {dimension_type} parameters for '{interface_name}': {result}")
            return None, result
        
        # No parameters found
        logger.debug(f"No {dimension_type} parameters found for interface '{interface_name}'")
        return None, None
    
    def apply_to_kernel_metadata(self, kernel_metadata: 'KernelMetadata') -> None:
        """Apply all parameter linking to kernel metadata.
        
        This method handles:
        - Interface parameter linking for streaming interfaces
        - Creating default datatype metadata when needed
        - Setting default BDIM/SDIM parameters
        - Internal parameter linking
        - Managing the exposed parameters list
        
        Args:
            kernel_metadata: KernelMetadata to modify in place
        """
        logger.debug("Applying parameter linking to kernel metadata")
        
        # Apply interface auto-linking
        if self.enable_interface_linking:
            for interface in kernel_metadata.interfaces:
                # Auto-link all eligible interfaces (exclude control)
                if interface.interface_type.value in ['input', 'output', 'weight', 'config']:
                    
                    auto_linked_dt = self.link_interface_parameters(
                        interface.name, kernel_metadata.parameters
                    )
                    
                    if interface.datatype_metadata is None:
                        # No existing metadata - use auto-linked or create default
                        if auto_linked_dt:
                            interface.datatype_metadata = auto_linked_dt
                            logger.info(f"Auto-linked datatype parameters for interface '{interface.name}'")
                        else:
                            # No auto-linkable parameters found - create minimal metadata
                            interface.datatype_metadata = DatatypeMetadata(name=interface.name)
                            logger.debug(f"No datatype parameters found for interface '{interface.name}' - using minimal metadata")
                    elif auto_linked_dt:
                        # Merge auto-linked with existing metadata (auto-link fills gaps)
                        existing = interface.datatype_metadata
                        merged = existing.update(
                            width=existing.width or auto_linked_dt.width,
                            signed=existing.signed or auto_linked_dt.signed,
                            format=existing.format or auto_linked_dt.format,
                            bias=existing.bias or auto_linked_dt.bias,
                            fractional_width=existing.fractional_width or auto_linked_dt.fractional_width,
                            exponent_width=existing.exponent_width or auto_linked_dt.exponent_width,
                            mantissa_width=existing.mantissa_width or auto_linked_dt.mantissa_width
                        )
                        interface.datatype_metadata = merged
                        logger.info(f"Merged auto-linked parameters with existing metadata for interface '{interface.name}'")
                    
                    # Remove linked parameters from exposed list
                    if interface.datatype_metadata:
                        for param_name in interface.datatype_metadata.get_all_parameters():
                            if param_name in kernel_metadata.exposed_parameters:
                                kernel_metadata.exposed_parameters.remove(param_name)
                    
                    # Only auto-link BDIM if not already set by pragma
                    # BDIM applies to INPUT, OUTPUT, and WEIGHT interfaces
                    if not interface.bdim_params:
                        # Collect BDIM parameters using new method
                        single_bdim, indexed_bdims = self.collect_dimension_parameters(
                            interface.name, kernel_metadata.parameters, "BDIM"
                        )
                        
                        if indexed_bdims:
                            # Multi-dimensional case
                            interface.bdim_params = indexed_bdims
                            interface.block_shape = indexed_bdims  # For compatibility
                            # Remove all indexed params from exposed (except singletons)
                            for param in indexed_bdims:
                                if param != "1" and param in kernel_metadata.exposed_parameters:
                                    kernel_metadata.exposed_parameters.remove(param)
                                    logger.debug(f"Removed indexed BDIM parameter '{param}' from exposed list")
                        elif single_bdim:
                            # Single parameter case - store as list
                            interface.bdim_params = [single_bdim]
                            interface.block_shape = [single_bdim]  # For compatibility
                            if single_bdim in kernel_metadata.exposed_parameters:
                                kernel_metadata.exposed_parameters.remove(single_bdim)
                                logger.debug(f"Removed single BDIM parameter '{single_bdim}' from exposed list")
                        else:
                            # No matching BDIM parameters found - leave unset for validation
                            logger.debug(f"No BDIM parameters found for interface '{interface.name}'")
                    
                    # Only auto-link SDIM if not already set by pragma
                    # SDIM only applies to INPUT and WEIGHT interfaces (not OUTPUT)
                    if interface.interface_type.value in ['input', 'weight']:
                        if not interface.sdim_params:
                            # Collect SDIM parameters using new method
                            single_sdim, indexed_sdims = self.collect_dimension_parameters(
                                interface.name, kernel_metadata.parameters, "SDIM"
                            )
                            
                            if indexed_sdims:
                                # Multi-dimensional case
                                interface.sdim_params = indexed_sdims
                                # Remove all indexed params from exposed (except singletons)
                                for param in indexed_sdims:
                                    if param != "1" and param in kernel_metadata.exposed_parameters:
                                        kernel_metadata.exposed_parameters.remove(param)
                                        logger.debug(f"Removed indexed SDIM parameter '{param}' from exposed list")
                            elif single_sdim:
                                # Single parameter case - store as list
                                interface.sdim_params = [single_sdim]
                                if single_sdim in kernel_metadata.exposed_parameters:
                                    kernel_metadata.exposed_parameters.remove(single_sdim)
                                    logger.debug(f"Removed single SDIM parameter '{single_sdim}' from exposed list")
                            else:
                                # No matching SDIM parameters found - leave unset for validation
                                logger.debug(f"No SDIM parameters found for interface '{interface.name}'")
                    else:
                        logger.debug(f"Skipping SDIM auto-linking for interface '{interface.name}' "
                                   f"of type '{interface.interface_type.value}' (SDIM only applies to INPUT/WEIGHT)")
        
        # Apply internal auto-linking
        if self.enable_internal_linking:
            # Get interface names and already-linked parameters for exclusion
            interface_names = [iface.name for iface in kernel_metadata.interfaces]
            exclude_parameters = set()
            
            # Collect parameters already linked by pragmas
            for dt in kernel_metadata.internal_datatypes:
                exclude_parameters.update(dt.get_all_parameters())
            
            # Collect parameters already linked by interface datatypes
            for interface in kernel_metadata.interfaces:
                if interface.datatype_metadata:
                    exclude_parameters.update(interface.datatype_metadata.get_all_parameters())
            
            auto_linked_internals = self.link_internal_parameters(
                kernel_metadata.parameters,
                exclude_prefixes=interface_names,
                exclude_parameters=exclude_parameters
            )
            
            if auto_linked_internals:
                kernel_metadata.internal_datatypes.extend(auto_linked_internals)
                logger.info(f"Auto-linked {len(auto_linked_internals)} internal datatypes")
                
                # Remove auto-linked parameters from exposed list
                for dt in auto_linked_internals:
                    for param_name in dt.get_all_parameters():
                        if param_name in kernel_metadata.exposed_parameters:
                            kernel_metadata.exposed_parameters.remove(param_name)