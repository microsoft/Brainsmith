"""
Interface metadata classes for enhanced AutoHWCustomOp architecture.

This module provides the object-oriented metadata system that replaces
static dictionaries in generated AutoHWCustomOp classes.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from .interface_types import InterfaceType
from .block_chunking import ChunkingStrategy, default_chunking


@dataclass
class DataTypeConstraint:
    """
    Constraint specification for interface datatypes.
    
    Replaces nested dictionary specifications with proper type safety.
    """
    finn_type: str
    bit_width: int
    signed: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    def __post_init__(self):
        """Validate constraint parameters."""
        if self.bit_width <= 0:
            raise ValueError(f"Bit width must be positive, got {self.bit_width}")
        
        if self.min_value is not None and self.max_value is not None:
            if self.min_value > self.max_value:
                raise ValueError(f"min_value ({self.min_value}) must be <= max_value ({self.max_value})")
    
    def validates(self, datatype_string: str) -> bool:
        """
        Check if a datatype string satisfies this constraint.
        
        Args:
            datatype_string: FINN datatype string (e.g., "UINT8", "INT16")
            
        Returns:
            bool: True if datatype satisfies constraint
        """
        # Basic validation - exact match for now
        # TODO: Add more sophisticated validation for bit width ranges, etc.
        return datatype_string == self.finn_type
    
    @classmethod
    def from_dict(cls, constraint_dict: Dict[str, Any]) -> 'DataTypeConstraint':
        """Create DataTypeConstraint from legacy dictionary format."""
        return cls(
            finn_type=constraint_dict.get("finn_type", "UINT8"),
            bit_width=constraint_dict.get("bit_width", 8),
            signed=constraint_dict.get("signed", False),
            min_value=constraint_dict.get("min_value"),
            max_value=constraint_dict.get("max_value")
        )


@dataclass
class InterfaceMetadata:
    """
    Metadata specification for a dataflow interface.
    
    This class encapsulates all information needed to create and configure
    a dataflow interface with pure computational properties and chunking strategy.
    """
    name: str
    interface_type: InterfaceType
    allowed_datatypes: List[DataTypeConstraint]
    chunking_strategy: ChunkingStrategy = field(default_factory=default_chunking)
    default_layout: Optional[str] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        """Validate metadata parameters."""
        if not self.name:
            raise ValueError("Interface name cannot be empty")
        
        if not self.allowed_datatypes:
            raise ValueError(f"Interface {self.name} must have at least one allowed datatype")
    
    def get_default_datatype(self) -> DataTypeConstraint:
        """Get the default (first) allowed datatype."""
        return self.allowed_datatypes[0]
    
    def validates_datatype(self, datatype_string: str) -> bool:
        """
        Check if a datatype string is allowed for this interface.
        
        Args:
            datatype_string: FINN datatype string
            
        Returns:
            bool: True if datatype is allowed
        """
        return any(constraint.validates(datatype_string) for constraint in self.allowed_datatypes)
    
    
    @classmethod
    def from_dict(cls, interface_dict: Dict[str, Any]) -> 'InterfaceMetadata':
        """Create InterfaceMetadata from legacy dictionary format."""
        # Convert interface type
        iface_type_str = interface_dict.get("interface_type", "INPUT")
        if isinstance(iface_type_str, str):
            interface_type = InterfaceType(iface_type_str.upper())
        else:
            interface_type = iface_type_str
        
        # Convert datatype constraints
        dtype_constraints = []
        dtype_specs = interface_dict.get("allowed_datatypes", [])
        
        if isinstance(dtype_specs, dict):
            # Legacy nested dictionary format
            for dtype_name, constraint_dict in dtype_specs.items():
                constraint_dict["finn_type"] = dtype_name
                dtype_constraints.append(DataTypeConstraint.from_dict(constraint_dict))
        elif isinstance(dtype_specs, list):
            # List of constraint dictionaries
            for constraint_dict in dtype_specs:
                dtype_constraints.append(DataTypeConstraint.from_dict(constraint_dict))
        
        if not dtype_constraints:
            # Default constraint if none specified
            dtype_constraints = [DataTypeConstraint(finn_type="UINT8", bit_width=8)]
        
        return cls(
            name=interface_dict["name"],
            interface_type=interface_type,
            allowed_datatypes=dtype_constraints,
            chunking_strategy=interface_dict.get("chunking_strategy", default_chunking()),
            default_layout=interface_dict.get("default_layout"),
            description=interface_dict.get("description")
        )


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