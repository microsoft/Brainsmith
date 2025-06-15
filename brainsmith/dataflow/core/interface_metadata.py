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
    
    # NEW: Optional datatype parameter mappings
    datatype_params: Optional[Dict[str, str]] = None
    """
    Optional mapping of datatype properties to RTL parameters.
    If None, defaults to {clean_interface_name}_WIDTH, {clean_interface_name}_SIGNED pattern.
    
    Example: {"width": "INPUT0_WIDTH", "signed": "SIGNED_INPUT0"}
    """
    
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
    
    def get_datatype_parameter_name(self, property_type: str) -> str:
        """
        Get RTL parameter name for a datatype property.
        
        Args:
            property_type: 'width', 'signed', 'format', 'bias', 'fractional_width'
            
        Returns:
            RTL parameter name (e.g., 'INPUT0_WIDTH', 'SIGNED_INPUT0')
        """
        if self.datatype_params and property_type in self.datatype_params:
            return self.datatype_params[property_type]
        
        # Default naming convention
        clean_name = self._get_clean_interface_name()
        if property_type == 'width':
            return f"{clean_name}_WIDTH"
        elif property_type == 'signed':
            return f"SIGNED_{clean_name}"
        elif property_type == 'format':
            return f"{clean_name}_FORMAT"
        elif property_type == 'bias':
            return f"{clean_name}_BIAS"
        elif property_type == 'fractional_width':
            return f"{clean_name}_FRACTIONAL_WIDTH"
        else:
            return f"{clean_name}_{property_type.upper()}"
    
    def _get_clean_interface_name(self) -> str:
        """Extract clean name from interface for parameter generation."""
        # Remove common prefixes/suffixes: s_axis_input0 -> INPUT0
        clean = self.name
        for prefix in ['s_axis_', 'm_axis_', 'axis_']:
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
                break
        for suffix in ['_tdata', '_tvalid', '_tready']:
            if clean.endswith(suffix):
                clean = clean[:-len(suffix)]
                break
        return clean.upper()


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
