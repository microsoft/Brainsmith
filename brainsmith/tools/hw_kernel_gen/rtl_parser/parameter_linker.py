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
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict

from .data import Parameter
from brainsmith.dataflow.core.datatype_metadata import DatatypeMetadata

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
    
    def merge_with_pragma_datatypes(self, auto_linked: List[DatatypeMetadata],
                                   pragma_defined: List[DatatypeMetadata]) -> List[DatatypeMetadata]:
        """
        Merge auto-linked datatypes with pragma-defined ones.
        
        Pragma-defined datatypes take precedence. If a pragma defines a property
        for a datatype, it overrides the auto-linked value.
        
        Args:
            auto_linked: List of auto-linked DatatypeMetadata
            pragma_defined: List of pragma-defined DatatypeMetadata
            
        Returns:
            Merged list of DatatypeMetadata
        """
        # Index auto-linked by name
        auto_linked_dict = {dt.name: dt for dt in auto_linked}
        
        # Index pragma-defined by name
        pragma_dict = {dt.name: dt for dt in pragma_defined}
        
        # Merge
        merged = {}
        
        # Start with auto-linked
        for name, auto_dt in auto_linked_dict.items():
            if name in pragma_dict:
                # Pragma exists - merge properties (pragma wins)
                pragma_dt = pragma_dict[name]
                merged[name] = DatatypeMetadata(
                    name=name,
                    width=pragma_dt.width or auto_dt.width,  # Pragma overrides if set
                    signed=pragma_dt.signed if pragma_dt.signed is not None else auto_dt.signed,
                    format=pragma_dt.format if pragma_dt.format is not None else auto_dt.format,
                    bias=pragma_dt.bias if pragma_dt.bias is not None else auto_dt.bias,
                    fractional_width=pragma_dt.fractional_width if pragma_dt.fractional_width is not None else auto_dt.fractional_width,
                    exponent_width=pragma_dt.exponent_width if pragma_dt.exponent_width is not None else auto_dt.exponent_width,
                    mantissa_width=pragma_dt.mantissa_width if pragma_dt.mantissa_width is not None else auto_dt.mantissa_width,
                    description=pragma_dt.description or auto_dt.description
                )
                logger.debug(f"Merged auto-linked and pragma-defined datatype '{name}'")
            else:
                # No pragma - keep auto-linked
                merged[name] = auto_dt
        
        # Add pragma-only datatypes
        for name, pragma_dt in pragma_dict.items():
            if name not in merged:
                merged[name] = pragma_dt
        
        return list(merged.values())