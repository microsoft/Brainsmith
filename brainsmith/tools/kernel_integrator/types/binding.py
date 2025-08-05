"""
Binding types for code generation.

This module contains types that represent the binding between
RTL parameters and generated code attributes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from enum import Enum

from brainsmith.core.dataflow.types import InterfaceType


class ParameterSource(Enum):
    """Source of a parameter value."""
    INTERFACE = "interface"      # From interface metadata
    GLOBAL = "global"           # From global parameters
    DERIVED = "derived"         # Computed from other parameters
    DEFAULT = "default"         # Default value
    USER = "user"              # User-provided


class ParameterCategory(Enum):
    """Category of parameter for organization."""
    SHAPE = "shape"            # BDIM/SDIM parameters
    DATATYPE = "datatype"      # Datatype-related parameters
    ALGORITHM = "algorithm"    # Algorithm-specific parameters
    PERFORMANCE = "performance" # Performance tuning parameters
    OTHER = "other"           # Uncategorized


@dataclass
class IOSpec:
    """Specification for operator I/O.
    
    Maps between Python operator interface and RTL interface.
    """
    python_name: str        # Name in Python operator (e.g., "X", "Y") 
    cpp_name: str          # Name in C++ (e.g., "in0", "out")
    interface_type: InterfaceType
    interface_name: str    # Actual RTL interface name
    index: int = 0        # For multiple interfaces of same type
    
    @property
    def is_input(self) -> bool:
        """Check if this is an input."""
        return self.interface_type in [InterfaceType.INPUT, InterfaceType.WEIGHT]
    
    @property
    def is_output(self) -> bool:
        """Check if this is an output."""
        return self.interface_type == InterfaceType.OUTPUT


@dataclass
class AttributeBinding:
    """Binding for operator attribute.
    
    Maps RTL parameter to Python/C++ operator attribute.
    """
    python_name: str       # Python attribute name
    cpp_name: str         # C++ attribute name
    param_name: str       # RTL parameter name
    python_type: str      # Python type (e.g., "int", "bool")
    cpp_type: str        # C++ type (e.g., "unsigned", "bool")
    default_value: Optional[str] = None
    
    # Metadata
    category: ParameterCategory = ParameterCategory.OTHER
    source: ParameterSource = ParameterSource.GLOBAL
    description: Optional[str] = None
    
    # Validation
    min_value: Optional[int] = None
    max_value: Optional[int] = None
    allowed_values: Optional[List[Any]] = None
    
    def get_python_default(self) -> str:
        """Get default value formatted for Python."""
        if self.default_value is None:
            return "None"
        elif self.python_type == "bool":
            return "True" if self.default_value.lower() in ["true", "1"] else "False"
        elif self.python_type == "str":
            return f'"{self.default_value}"'
        else:
            return self.default_value
    
    def get_cpp_default(self) -> str:
        """Get default value formatted for C++."""
        if self.default_value is None:
            return "0"
        elif self.cpp_type == "bool":
            return "true" if self.default_value.lower() in ["true", "1"] else "false"
        else:
            return self.default_value


@dataclass
class InterfaceBinding:
    """Binding for interface-specific parameters."""
    interface_name: str
    interface_type: InterfaceType
    shape_params: List[str] = field(default_factory=list)    # BDIM/SDIM params
    datatype_params: List[str] = field(default_factory=list) # Datatype params
    
    def get_all_params(self) -> Set[str]:
        """Get all parameter names for this interface."""
        return set(self.shape_params + self.datatype_params)


@dataclass
class CodegenBinding:
    """Complete binding specification.
    
    Contains all information needed to generate code bindings between
    RTL parameters and operator attributes.
    """
    class_name: str
    finn_op_type: str
    io_specs: List[IOSpec] = field(default_factory=list)
    attributes: List[AttributeBinding] = field(default_factory=list)
    interface_bindings: Dict[str, InterfaceBinding] = field(default_factory=dict)
    
    # Categorized access
    def get_io_spec(self, interface_type: InterfaceType) -> Optional[IOSpec]:
        """Get IO spec for interface type."""
        for spec in self.io_specs:
            if spec.interface_type == interface_type:
                return spec
        return None
    
    def get_input_specs(self) -> List[IOSpec]:
        """Get all input IO specs."""
        return [s for s in self.io_specs if s.is_input]
    
    def get_output_specs(self) -> List[IOSpec]:
        """Get all output IO specs."""
        return [s for s in self.io_specs if s.is_output]
    
    def get_attribute(self, param_name: str) -> Optional[AttributeBinding]:
        """Get attribute binding by parameter name."""
        for attr in self.attributes:
            if attr.param_name == param_name:
                return attr
        return None
    
    def get_attributes_by_category(self, category: ParameterCategory) -> List[AttributeBinding]:
        """Get all attributes of a specific category."""
        return [a for a in self.attributes if a.category == category]
    
    def get_required_attributes(self) -> List[AttributeBinding]:
        """Get attributes without default values."""
        return [a for a in self.attributes if a.default_value is None]
    
    def get_shape_attributes(self) -> List[AttributeBinding]:
        """Get shape-related attributes."""
        return self.get_attributes_by_category(ParameterCategory.SHAPE)
    
    def get_datatype_attributes(self) -> List[AttributeBinding]:
        """Get datatype-related attributes."""
        return self.get_attributes_by_category(ParameterCategory.DATATYPE)
    
    def validate(self) -> bool:
        """Validate binding completeness.
        
        Returns:
            True if binding is valid and complete
        """
        # Must have at least input and output
        has_input = any(s.is_input for s in self.io_specs)
        has_output = any(s.is_output for s in self.io_specs)
        
        if not has_input or not has_output:
            return False
            
        # All IO specs must have unique Python names
        python_names = [s.python_name for s in self.io_specs]
        if len(python_names) != len(set(python_names)):
            return False
            
        # All attributes must have unique Python names  
        attr_names = [a.python_name for a in self.attributes]
        if len(attr_names) != len(set(attr_names)):
            return False
            
        return True