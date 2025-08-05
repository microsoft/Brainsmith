"""
Metadata types for higher-level kernel representation.

This module contains types that represent parsed and processed kernel
information at a higher abstraction level than raw RTL.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from brainsmith.core.dataflow.types import InterfaceType
from brainsmith.core.dataflow.constraint_types import DatatypeConstraintGroup

from .core import DatatypeSpec, DimensionSpec
from .rtl import Port, Parameter


@dataclass
class InterfaceMetadata:
    """Metadata for a single interface.
    
    Represents a complete interface with its type, datatype, dimensions,
    and associated ports and parameters.
    """
    type: InterfaceType
    name: str
    datatype: DatatypeSpec
    dimensions: DimensionSpec
    ports: List[Port]
    parameters: Dict[str, Parameter] = field(default_factory=dict)
    
    # Constraint information
    datatype_constraints: Optional[DatatypeConstraintGroup] = None
    
    # Pragma-derived metadata
    is_weight: bool = False
    weight_file: Optional[str] = None
    relationships: Dict[str, str] = field(default_factory=dict)
    
    # Computed properties
    @property
    def width(self) -> Optional[int]:
        """Total interface width in bits.
        
        Returns None if dimensions contain symbolic parameters.
        """
        total_elements = self.dimensions.total_elements
        if total_elements is None:
            return None
        return self.datatype.bit_width * total_elements
    
    @property
    def has_axi_stream(self) -> bool:
        """Check if interface has AXI-Stream protocol ports."""
        required_suffixes = {'valid', 'ready', 'data'}
        port_suffixes = {p.name.split('_')[-1] for p in self.ports}
        return required_suffixes.issubset(port_suffixes)
    
    @property
    def has_axi_lite(self) -> bool:
        """Check if interface has AXI-Lite protocol ports."""
        # AXI-Lite has many signals, just check for key ones
        required_patterns = {'awaddr', 'wdata', 'araddr', 'rdata'}
        port_names = {p.name for p in self.ports}
        matches = sum(1 for pattern in required_patterns if any(pattern in name for name in port_names))
        return matches >= 2
    
    def get_port(self, suffix: str) -> Optional[Port]:
        """Get port by suffix (e.g., 'valid', 'ready', 'data')."""
        for port in self.ports:
            if port.name.endswith(suffix):
                return port
        return None
    
    def get_parameter_names(self) -> Set[str]:
        """Get all parameter names used in this interface."""
        param_names = set(self.parameters.keys())
        param_names.update(self.dimensions.get_parameters())
        return param_names


@dataclass
class KernelMetadata:
    """Complete kernel metadata.
    
    Represents all information about a kernel needed for code generation,
    including interfaces, parameters, and relationships.
    """
    module_name: str
    interfaces: Dict[str, InterfaceMetadata]
    global_parameters: Dict[str, Parameter]
    
    # File information
    source_file: str
    
    # Pragma-derived metadata  
    top_module: Optional[str] = None
    exposed_parameters: Set[str] = field(default_factory=set)
    derived_parameters: Dict[str, str] = field(default_factory=dict)
    axi_lite_parameters: Set[str] = field(default_factory=set)
    
    # Methods for specific queries
    def get_interface(self, interface_type: InterfaceType) -> Optional[InterfaceMetadata]:
        """Get first interface of given type."""
        for interface in self.interfaces.values():
            if interface.type == interface_type:
                return interface
        return None
    
    def get_interfaces_by_type(self, interface_type: InterfaceType) -> List[InterfaceMetadata]:
        """Get all interfaces of given type."""
        return [i for i in self.interfaces.values() if i.type == interface_type]
    
    def get_input_interface(self) -> Optional[InterfaceMetadata]:
        """Get the primary input interface."""
        return self.get_interface(InterfaceType.INPUT)
    
    def get_output_interface(self) -> Optional[InterfaceMetadata]:
        """Get the primary output interface."""
        return self.get_interface(InterfaceType.OUTPUT)
    
    def get_weight_interfaces(self) -> List[InterfaceMetadata]:
        """Get all weight interfaces."""
        return self.get_interfaces_by_type(InterfaceType.WEIGHT)
    
    def get_config_interface(self) -> Optional[InterfaceMetadata]:
        """Get the configuration interface."""
        return self.get_interface(InterfaceType.CONFIG)
    
    def get_all_parameters(self) -> Dict[str, Parameter]:
        """Get all parameters (global and interface-specific)."""
        all_params = dict(self.global_parameters)
        for interface in self.interfaces.values():
            all_params.update(interface.parameters)
        return all_params
    
    def get_exposed_parameters(self) -> Dict[str, Parameter]:
        """Get only exposed parameters."""
        all_params = self.get_all_parameters()
        return {name: param for name, param in all_params.items() if name in self.exposed_parameters}
    
    def validate(self) -> bool:
        """Basic validation of kernel metadata.
        
        Returns:
            True if metadata is valid, False otherwise
        """
        # Must have at least input and output
        if not self.get_input_interface() or not self.get_output_interface():
            return False
            
        # All exposed parameters must exist
        all_param_names = set(self.get_all_parameters().keys())
        if not self.exposed_parameters.issubset(all_param_names):
            return False
            
        # All derived parameters must reference existing parameters
        for derived, expr in self.derived_parameters.items():
            # Simple check - more sophisticated expression parsing could be added
            if derived in all_param_names:
                return False  # Derived param shadows existing param
                
        return True