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
from typing import Dict, List, Any, Optional
import logging

from brainsmith.tools.kernel_integrator.types.metadata import KernelMetadata, InterfaceMetadata
from brainsmith.tools.kernel_integrator.types.rtl import PragmaType

logger = logging.getLogger(__name__)


class PragmaError(Exception):
    """Custom exception for errors during pragma parsing or validation."""
    pass


@dataclass
class Pragma:
    """Brainsmith pragma representation.
    
    Pragmas are special comments that provide additional information to the
    Kernel Integrator. They follow the format:
        // @brainsmith <type> <inputs...>
    
    Attributes:
        type: Pragma type identifier (using PragmaType enum)
        inputs: Dict with 'raw', 'positional', and 'named' arguments
        parsed_data: Optional processed data from pragma handler
    """
    type: PragmaType
    inputs: Dict[str, Any]
    parsed_data: Dict = field(init=False)  # Stores the result of _parse_inputs

    def _parse_inputs(self) -> Dict:
        """
        Abstract method to parse pragma inputs.
        Subclasses must implement this method.
        """
        raise NotImplementedError(f"Pragma type {self.type.name} must implement _parse_inputs.")

    def apply_to_kernel(self, kernel: KernelMetadata) -> None:
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
    """Base class providing utilities for interface-related pragmas.
    
    This class provides helper methods for pragmas that work with interfaces,
    but does not enforce any particular pattern. Subclasses implement
    apply_to_kernel directly and can use the provided utilities as needed.
    """
    
    def find_interface(self, kernel: KernelMetadata, interface_name: str) -> Optional[InterfaceMetadata]:
        """Find an interface by name across all interface types in the kernel.
        
        Args:
            kernel: KernelMetadata to search
            interface_name: Name of the interface to find
            
        Returns:
            The interface if found, None otherwise
        """
        # Check stream interfaces (inputs/outputs)
        for interface in kernel.inputs + kernel.outputs:
            if interface.name == interface_name:
                return interface
        
        # Check config interfaces
        for interface in kernel.config:
            if interface.name == interface_name:
                return interface
        
        # Check control interface
        if kernel.control and kernel.control.name == interface_name:
            return kernel.control
            
        return None


