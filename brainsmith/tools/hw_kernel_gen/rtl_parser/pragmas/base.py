############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Base classes for pragma implementations.

This module provides the base classes and common functionality for all
pragma types in the RTL parser.
"""

from dataclasses import dataclass, field
from typing import Dict, List
import logging

from ..data import PragmaType
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata

logger = logging.getLogger(__name__)


class PragmaError(Exception):
    """Custom exception for errors during pragma parsing or validation."""
    pass


@dataclass
class Pragma:
    """Brainsmith pragma representation.
    
    Pragmas are special comments that provide additional information to the
    Hardware Kernel Generator. They follow the format:
        // @brainsmith <type> <inputs...>
    
    Attributes:
        type: Pragma type identifier (using PragmaType enum)
        inputs: List of space-separated inputs
        line_number: Source line number for error reporting
        parsed_data: Optional processed data from pragma handler
    """
    type: PragmaType
    inputs: List[str]
    line_number: int
    parsed_data: Dict = field(init=False)  # Stores the result of _parse_inputs

    def __post_init__(self):
        try:
            self.parsed_data = self._parse_inputs()
        except PragmaError as e:
            logger.error(f"Error processing pragma {self.type.name} at line {self.line_number} with inputs {self.inputs}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing pragma {self.type.name} at line {self.line_number} with inputs {self.inputs}: {e}")
            # Wrap unexpected errors in PragmaError to ensure consistent error handling upstream
            raise PragmaError(f"Unexpected error during pragma {self.type.name} processing: {e}")

    def _parse_inputs(self) -> Dict:
        """
        Abstract method to parse pragma inputs.
        Subclasses must implement this method.
        """
        raise NotImplementedError(f"Pragma type {self.type.name} must implement _parse_inputs.")

    def apply_to_kernel(self, kernel: 'KernelMetadata') -> None:
        """
        Apply this pragma to kernel metadata.
        
        Subclasses must implement this method to modify the kernel metadata
        as appropriate for their pragma type.
        
        Args:
            kernel: KernelMetadata object to modify
        """
        raise NotImplementedError(f"Pragma type {self.type.name} must implement apply_to_kernel.")

    def __str__(self):
        return f"@brainsmith {self.type.value} " + " ".join(map(str, self.inputs))


@dataclass
class InterfacePragma(Pragma):
    """Base class for pragmas that modify interface metadata.
    
    This class provides common functionality for pragmas that target specific
    interfaces, including interface name matching and base application logic.
    """
    
    def apply_to_interface_by_name(self, interface_name: str, kernel: 'KernelMetadata') -> bool:
        """
        Find interface by name in KernelMetadata and apply pragma if found.
        
        This method centralizes the interface-finding logic that was previously
        duplicated across all interface pragma subclasses.
        
        Args:
            interface_name: Name of target interface
            kernel: KernelMetadata containing interfaces
            
        Returns:
            bool: True if interface was found and pragma applied, False otherwise
        """
        # Find the target interface
        for interface in kernel.interfaces:
            if interface.name == interface_name:
                # Apply pragma-specific logic via apply_to_interface
                updated_metadata = self.apply_to_interface(interface)
                
                # Update the interface in the kernel with the modified metadata
                # Since InterfaceMetadata objects are mutable, we update in place
                interface.datatype_constraints = updated_metadata.datatype_constraints
                interface.chunking_strategy = updated_metadata.chunking_strategy
                interface.datatype_metadata = updated_metadata.datatype_metadata
                interface.bdim_param = updated_metadata.bdim_param
                interface.sdim_param = updated_metadata.sdim_param
                interface.shape_params = updated_metadata.shape_params
                interface.interface_type = updated_metadata.interface_type
                interface.description = updated_metadata.description
                
                logger.debug(f"Applied {self.type.value} pragma to interface '{interface_name}'")
                return True
        
        logger.warning(f"{self.type.value} pragma target interface '{interface_name}' not found")
        return False
    
    def apply_to_kernel(self, kernel: 'KernelMetadata') -> None:
        """
        Apply interface pragma to kernel metadata.
        
        Default implementation handles single interface pragmas.
        Subclasses can override for more complex behavior (e.g., WeightPragma
        for multiple interfaces, DatatypeParamPragma for internal datatypes).
        """
        interface_name = self.parsed_data.get("interface_name")
        if interface_name:
            self.apply_to_interface_by_name(interface_name, kernel)

    def apply_to_interface(self, metadata: InterfaceMetadata) -> InterfaceMetadata:
        """
        Apply pragma effects to InterfaceMetadata.
        
        Subclasses must override this method to implement their specific effects.
        
        This method enables a clean chain-of-responsibility pattern where each
        pragma can modify the interface metadata independently and composably.
        
        Args:
            metadata: Current InterfaceMetadata to modify
            
        Returns:
            InterfaceMetadata: Modified metadata with pragma effects applied.
                              Should return a new InterfaceMetadata instance.
        """
        # Subclasses must override this method
        raise NotImplementedError(f"{self.__class__.__name__} must implement apply_to_interface()")


