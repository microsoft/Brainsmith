############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Pragma processing for Kernel Integrator.

Handles high-level pragma operations such as grouping pragmas by type
and collecting internal datatype pragmas. The actual extraction and
validation of pragmas from AST is handled by ModuleExtractor.
"""

import logging
from typing import List, Optional, Dict

from brainsmith.tools.kernel_integrator.types.metadata import DatatypeMetadata
from brainsmith.tools.kernel_integrator.types.rtl import PragmaType
from .pragmas import Pragma

# Set up logger for this module
logger = logging.getLogger(__name__)

class PragmaHandler:
    """Handles high-level pragma operations and grouping.
    
    This class provides utilities for working with already-extracted pragmas,
    such as filtering by type and collecting internal datatype information.
    The actual pragma extraction from AST is handled by ModuleExtractor.
    """

    def __init__(self, debug: bool = False):
        """Initializes the PragmaHandler."""
        self.debug = debug
        self.pragmas: List[Pragma] = []  # List to store found pragmas

    def set_pragmas(self, pragmas: List[Pragma]) -> None:
        """Set the list of pragmas to work with.
        
        Args:
            pragmas: List of Pragma objects (typically from ModuleExtractor)
        """
        self.pragmas = pragmas

    def get_pragmas_by_type(self, pragma_type: PragmaType) -> List[Pragma]:
        """Get all pragmas of a specific type.
        
        Args:
            pragma_type: The PragmaType to filter by
            
        Returns:
            List of Pragma instances of the specified type
        """
        return [pragma for pragma in self.pragmas if pragma.type == pragma_type]

    def collect_internal_datatype_pragmas(self, interface_names: List[str]) -> List[DatatypeMetadata]:
        """
        Collect DATATYPE_PARAM pragmas that don't match any interface.
        
        These pragmas define datatype bindings for internal kernel mechanisms
        like accumulators, thresholds, etc.
        
        Args:
            interface_names: List of interface names to exclude
            
        Returns:
            List of DatatypeMetadata objects for internal mechanisms
        """
        # Get all DATATYPE_PARAM pragmas
        datatype_param_pragmas = self.get_pragmas_by_type(PragmaType.DATATYPE_PARAM)
        
        if not datatype_param_pragmas:
            logger.debug("No DATATYPE_PARAM pragmas found")
            return []
        
        # Group by target name (interface_name in pragma)
        internal_datatypes = {}
        
        for pragma in datatype_param_pragmas:
            target_name = pragma.parsed_data.get('interface_name')
            
            # Skip if it matches an interface
            if target_name in interface_names:
                continue
            
            # Create or update DatatypeMetadata for this internal mechanism
            if target_name not in internal_datatypes:
                internal_datatypes[target_name] = pragma.create_standalone_datatype()
            else:
                # Merge with existing metadata
                property_type = pragma.parsed_data['property_type']
                parameter_name = pragma.parsed_data['parameter_name']
                # Update the attribute directly on the existing datatype
                setattr(internal_datatypes[target_name], property_type, parameter_name)
        
        logger.info(f"Collected {len(internal_datatypes)} internal datatype bindings: {list(internal_datatypes.keys())}")
        return list(internal_datatypes.values())