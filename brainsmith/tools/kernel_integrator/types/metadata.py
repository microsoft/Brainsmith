"""
Metadata types for higher-level kernel representation.

This module contains types that represent parsed and processed kernel
information at a higher abstraction level than raw RTL.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union

from brainsmith.core.dataflow.types import InterfaceType
from brainsmith.core.dataflow.constraint_types import DatatypeConstraintGroup

from .rtl import Port, Parameter


@dataclass
class InterfaceMetadata:
    """Metadata for a single interface.
    
    Represents a complete interface with its type, datatype, dimensions,
    and associated ports and parameters.
    """
    name: str
    interface_type: InterfaceType
    datatype_constraints: List[DatatypeConstraintGroup] = field(default_factory=list)
    description: Optional[str] = None
    
    # Parameter linkage mappings
    datatype_metadata: Optional['DatatypeMetadata'] = None
    bdim_params: Optional[List[str]] = None
    sdim_params: Optional[List[str]] = None
    
    # Shape expressions for new tiling system
    bdim_shape: Optional[List] = None
    sdim_shape: Optional[List] = None
    
    # Additional attributes from original
    allowed_datatypes: Optional[List] = None
    chunking_strategy: Optional[str] = None
    ports: List[Port] = field(default_factory=list)
    parameters: Dict[str, Parameter] = field(default_factory=dict)
    
    # Pragma-derived metadata
    is_weight: bool = False
    weight_file: Optional[str] = None
    relationships: Dict[str, str] = field(default_factory=dict)
    
    # Keep minimal helper methods
    
    def get_dimension_params(self) -> List[str]:
        """Get all parameter names used in dimensions."""
        params = set()
        for dim_list in [self.bdim_shape, self.sdim_shape]:
            if dim_list:
                for d in dim_list:
                    if isinstance(d, str) and d not in ['*', '1']:
                        params.add(d)
        return sorted(params)
    
    def get_port(self, suffix: str) -> Optional[Port]:
        """Get port by suffix (e.g., 'valid', 'ready', 'data')."""
        for port in self.ports:
            if port.name.endswith(suffix):
                return port
        return None


@dataclass
class KernelMetadata:
    """Complete kernel metadata.
    
    Represents all information about a kernel needed for code generation,
    including interfaces, parameters, and relationships.
    """
    # Core attributes matching original structure
    name: str
    interfaces: List[InterfaceMetadata]
    parameters: List[Parameter]
    source_file: str
    
    # Pragma-derived metadata  
    exposed_parameters: List[str] = field(default_factory=list)
    internal_datatypes: List['DatatypeMetadata'] = field(default_factory=list)
    linked_parameters: Dict[str, Dict[str, str]] = field(default_factory=dict)
    relationships: List = field(default_factory=list)  # List of DimensionRelationship
    
    # Additional attributes
    top_module: Optional[str] = None
    derived_parameters: Dict[str, str] = field(default_factory=dict)
    axi_lite_parameters: Set[str] = field(default_factory=set)
    
    # Parser state tracking (temporary)
    pragmas: List = field(default_factory=list)
    parsing_warnings: List = field(default_factory=list)
    
    # Categorized interface properties for efficient access
    @property
    def input_interfaces(self) -> List[InterfaceMetadata]:
        """Get all input interfaces."""
        return [i for i in self.interfaces if i.interface_type == InterfaceType.INPUT]
    
    @property
    def output_interfaces(self) -> List[InterfaceMetadata]:
        """Get all output interfaces."""
        return [i for i in self.interfaces if i.interface_type == InterfaceType.OUTPUT]
    
    @property
    def weight_interfaces(self) -> List[InterfaceMetadata]:
        """Get all weight interfaces."""
        return [i for i in self.interfaces if i.interface_type == InterfaceType.WEIGHT]
    
    @property
    def config_interfaces(self) -> List[InterfaceMetadata]:
        """Get all config interfaces."""
        return [i for i in self.interfaces if i.interface_type == InterfaceType.CONFIG]
    
    @property
    def control_interfaces(self) -> List[InterfaceMetadata]:
        """Get all control interfaces."""
        return [i for i in self.interfaces if i.interface_type == InterfaceType.CONTROL]
    
    # Convenience flags
    @property
    def has_inputs(self) -> bool:
        """Check if kernel has input interfaces."""
        return len(self.input_interfaces) > 0
    
    @property
    def has_outputs(self) -> bool:
        """Check if kernel has output interfaces."""
        return len(self.output_interfaces) > 0
    
    @property
    def has_weights(self) -> bool:
        """Check if kernel has weight interfaces."""
        return len(self.weight_interfaces) > 0
    
    @property
    def has_config(self) -> bool:
        """Check if kernel has config interfaces."""
        return len(self.config_interfaces) > 0
    
    # Simple transformations as properties
    @property
    def class_name(self) -> str:
        """Get PascalCase class name from module name."""
        from brainsmith.tools.kernel_integrator.utils import pascal_case
        return pascal_case(self.name)
    
    @property
    def required_attributes(self) -> List[str]:
        """Get parameter names without defaults (required for node creation)."""
        return [p.name for p in self.parameters if p.default_value is None]
    
    # Methods for specific queries (kept for backward compatibility)
    def get_interface(self, interface_type: InterfaceType) -> Optional[InterfaceMetadata]:
        """Get first interface of given type.
        
        .. deprecated:: 
            Use the specific interface properties instead (e.g., input_interfaces[0])
        """
        for interface in self.interfaces:
            if interface.interface_type == interface_type:
                return interface
        return None
    
    def get_interfaces_by_type(self, interface_type: InterfaceType) -> List[InterfaceMetadata]:
        """Get all interfaces of given type.
        
        .. deprecated::
            Use the specific interface properties instead (e.g., input_interfaces)
        """
        return [i for i in self.interfaces if i.interface_type == interface_type]
    
    def get_input_interface(self) -> Optional[InterfaceMetadata]:
        """Get the primary input interface."""
        interfaces = self.input_interfaces
        return interfaces[0] if interfaces else None
    
    def get_output_interface(self) -> Optional[InterfaceMetadata]:
        """Get the primary output interface."""
        interfaces = self.output_interfaces
        return interfaces[0] if interfaces else None
    
    def get_weight_interfaces(self) -> List[InterfaceMetadata]:
        """Get all weight interfaces."""
        return self.weight_interfaces
    
    def get_config_interface(self) -> Optional[InterfaceMetadata]:
        """Get the configuration interface."""
        interfaces = self.config_interfaces
        return interfaces[0] if interfaces else None
    
    def get_all_parameters(self) -> Dict[str, Parameter]:
        """Get all parameters as a dict."""
        return {p.name: p for p in self.parameters}
    
    def get_exposed_parameters(self) -> Dict[str, Parameter]:
        """Get only exposed parameters."""
        all_params = self.get_all_parameters()
        return {name: param for name, param in all_params.items() if name in self.exposed_parameters}
    
    @property
    def module_name(self) -> str:
        """Get the actual module name for instantiation."""
        return self.top_module if self.top_module else self.name
    
    @property
    def has_datatype_parameters(self) -> bool:
        """Check if kernel has any datatype parameters."""
        from .rtl import ParameterCategory
        return any(p.category == ParameterCategory.DATATYPE for p in self.parameters)
    
    @property
    def parameters_by_interface(self) -> Dict[str, List[Parameter]]:
        """Group parameters by their associated interface."""
        from collections import defaultdict
        groups = defaultdict(list)
        for param in self.parameters:
            if param.interface_name:
                groups[param.interface_name].append(param)
        return dict(groups)
    
    def validate(self) -> List[str]:
        """Validate kernel metadata.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Must have at least input and output
        if not self.has_inputs:
            errors.append("No input interface found")
        if not self.has_outputs:
            errors.append("No output interface found")
            
        # Validate Global Control interface exists
        if not self.control_interfaces:
            errors.append(f"Module '{self.name}' is missing a valid Global Control interface (ap_clk, ap_rst_n)")
            
        # All exposed parameters must exist
        all_param_names = set(self.get_all_parameters().keys())
        exposed_set = set(self.exposed_parameters)
        if not exposed_set.issubset(all_param_names):
            missing = exposed_set - all_param_names
            errors.append(f"Exposed parameters not found: {missing}")
            
        # All derived parameters must reference existing parameters
        for derived, expr in self.derived_parameters.items():
            # Simple check - more sophisticated expression parsing could be added
            if derived in all_param_names:
                errors.append(f"Derived parameter '{derived}' shadows existing parameter")
                
        return errors


@dataclass
class DatatypeMetadata:
    """
    Explicit binding between RTL parameters and datatype properties.
    
    This class provides a structured way to map RTL parameter names to
    specific datatype properties used in code generation. All properties
    are optional RTL parameter names, allowing flexible datatype definitions.
    
    Attributes:
        name: Identifier for this datatype (e.g., "in", "accumulator", "threshold")
        width: RTL parameter name for bit width
        signed: RTL parameter name for signedness
        format: RTL parameter name for format type (e.g., "fixed", "float")
        bias: RTL parameter name for bias value
        fractional_width: RTL parameter name for fractional width (fixed-point)
        exponent_width: RTL parameter name for exponent width (floating-point)
        mantissa_width: RTL parameter name for mantissa width (floating-point)
        description: Human-readable description
    """
    name: str  # Required - identifier for this datatype
    width: Optional[str] = None
    signed: Optional[str] = None
    format: Optional[str] = None
    bias: Optional[str] = None
    fractional_width: Optional[str] = None
    exponent_width: Optional[str] = None
    mantissa_width: Optional[str] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        """Validate metadata parameters."""
        if not self.name:
            raise ValueError("DatatypeMetadata name cannot be empty")
    
    def get_all_parameters(self) -> List[str]:
        """
        Get list of all RTL parameter names referenced by this metadata.
        
        Returns:
            List of parameter names (non-None values only)
        """
        params = []
        
        # Add all non-None parameters
        if self.width is not None:
            params.append(self.width)
        if self.signed is not None:
            params.append(self.signed)
        if self.format is not None:
            params.append(self.format)
        if self.bias is not None:
            params.append(self.bias)
        if self.fractional_width is not None:
            params.append(self.fractional_width)
        if self.exponent_width is not None:
            params.append(self.exponent_width)
        if self.mantissa_width is not None:
            params.append(self.mantissa_width)
            
        return params