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
from typing import Dict, List, Any
import logging

from ...metadata import InterfaceMetadata
from ..rtl_data import PragmaType

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
        inputs: Dict with 'raw', 'positional', and 'named' arguments
        line_number: Source line number for error reporting
        parsed_data: Optional processed data from pragma handler
    """
    type: PragmaType
    inputs: Dict[str, Any]
    line_number: int
    parsed_data: Dict = field(init=False)  # Stores the result of _parse_inputs

    def __post_init__(self):
        # For backward compatibility - convert list to dict format
        if isinstance(self.inputs, list):
            self.inputs = {
                'raw': self.inputs,
                'positional': self.inputs,
                'named': {}
            }
        
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
        # Use raw inputs if available, otherwise fall back to positional
        raw_inputs = self.inputs.get('raw', self.inputs.get('positional', []))
        return f"@brainsmith {self.type.value} " + " ".join(map(str, raw_inputs))


@dataclass
class InterfacePragma(Pragma):
    """Base class for pragmas that modify interface metadata.
    
    This class provides common functionality for pragmas that target specific
    interfaces, including interface name matching and base application logic.
    """
    
    def apply_to_kernel(self, kernel: 'KernelMetadata') -> None:
        """
        Apply interface pragma to kernel metadata.
        
        Default implementation finds interface by name and applies pragma.
        Subclasses can override for more complex behavior (e.g., WeightPragma
        for multiple interfaces, DatatypeParamPragma for internal datatypes).
        
        Args:
            kernel: KernelMetadata object containing interfaces to modify
        """
        interface_name = self.parsed_data.get("interface_name")
        if not interface_name:
            return
            
        # Find and apply to interface
        for interface in kernel.interfaces:
            if interface.name == interface_name:
                self.apply_to_interface(interface)
                logger.debug(f"Applied {self.type.value} pragma to interface '{interface_name}'")
                return
                
        logger.warning(f"{self.type.value} pragma target interface '{interface_name}' not found")

    def apply_to_interface(self, metadata: InterfaceMetadata) -> None:
        """
        Apply pragma effects to InterfaceMetadata in-place.
        
        Subclasses must override this method to implement their specific effects.
        
        This method enables a clean chain-of-responsibility pattern where each
        pragma can modify the interface metadata independently and composably.
        
        Args:
            metadata: InterfaceMetadata object to modify in-place
        """
        # Subclasses must override this method
        raise NotImplementedError(f"{self.__class__.__name__} must implement apply_to_interface()")


