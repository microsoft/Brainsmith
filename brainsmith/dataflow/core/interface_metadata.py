"""
Interface metadata classes for enhanced AutoHWCustomOp architecture.

This module provides the object-oriented metadata system that replaces
static dictionaries in generated AutoHWCustomOp classes.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from .interface_types import InterfaceType
from .block_chunking import default_chunking
from .qonnx_types import DatatypeConstraintGroup, validate_datatype_against_constraints, BaseDataType


@dataclass
class InterfaceMetadata:
    """
    Interface metadata with QONNX constraint groups - no default datatypes.
    
    This class encapsulates all information needed to create and configure
    a dataflow interface with runtime datatype validation.
    """
    name: str
    interface_type: InterfaceType
    datatype_constraints: List[DatatypeConstraintGroup] = field(default_factory=list)
    chunking_strategy: Any = field(default_factory=default_chunking)  # Can be BlockChunkingStrategy or DefaultChunkingStrategy
    default_layout: Optional[str] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        """Validate metadata parameters."""
        if not self.name:
            raise ValueError("Interface name cannot be empty")
        
        # Allow empty datatype_constraints - these can be populated by pragmas
    
    def validates_datatype(self, datatype: BaseDataType) -> bool:
        """
        Check if a QONNX datatype satisfies constraint groups.
        
        Args:
            datatype: QONNX BaseDataType instance to validate
            
        Returns:
            bool: True if datatype satisfies at least one constraint group
        """
        if not self.datatype_constraints:
            return True  # No constraints = allow anything
        return validate_datatype_against_constraints(datatype, self.datatype_constraints)
    
    def get_constraint_description(self) -> str:
        """
        Get human-readable description of constraints.
        
        Returns:
            str: Description like "UINT8-16, INT4-8" or "No datatype constraints"
        """
        if not self.datatype_constraints:
            return "No datatype constraints"
        
        descriptions = []
        for group in self.datatype_constraints:
            if group.min_width == group.max_width:
                descriptions.append(f"{group.base_type}{group.min_width}")
            else:
                descriptions.append(f"{group.base_type}{group.min_width}-{group.max_width}")
        return ", ".join(descriptions)


@dataclass
class InterfaceMetadataCollection:
    """
    Collection of interface metadata for an AutoHWCustomOp.
    
    Provides convenient access patterns and validation.
    """
    interfaces: List[InterfaceMetadata]
    
    def __post_init__(self):
        """Validate the collection."""
        interface_names = [iface.name for iface in self.interfaces]
        if len(interface_names) != len(set(interface_names)):
            raise ValueError("Duplicate interface names found")
    
    def get_by_name(self, name: str) -> Optional[InterfaceMetadata]:
        """Get interface metadata by name."""
        for iface in self.interfaces:
            if iface.name == name:
                return iface
        return None
    
    def get_by_type(self, interface_type: InterfaceType) -> List[InterfaceMetadata]:
        """Get all interfaces of a specific type."""
        return [iface for iface in self.interfaces if iface.interface_type == interface_type]
    
    def get_input_interfaces(self) -> List[InterfaceMetadata]:
        """Get all input interfaces."""
        return self.get_by_type(InterfaceType.INPUT)
    
    def get_output_interfaces(self) -> List[InterfaceMetadata]:
        """Get all output interfaces."""
        return self.get_by_type(InterfaceType.OUTPUT)
    
    def get_weight_interfaces(self) -> List[InterfaceMetadata]:
        """Get all weight interfaces."""
        return self.get_by_type(InterfaceType.WEIGHT)
    
    def get_config_interfaces(self) -> List[InterfaceMetadata]:
        """Get all config interfaces."""
        return self.get_by_type(InterfaceType.CONFIG)
    
    def interface_names(self) -> List[str]:
        """Get list of all interface names."""
        return [iface.name for iface in self.interfaces]
    
    def validate_datatype_for_interface(self, interface_name: str, datatype_string: str) -> bool:
        """Validate a datatype for a specific interface."""
        iface = self.get_by_name(interface_name)
        if not iface:
            raise KeyError(f"Interface '{interface_name}' not found")
        return iface.validates_datatype(datatype_string)
