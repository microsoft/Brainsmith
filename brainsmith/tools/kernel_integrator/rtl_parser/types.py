"""
RTL types for parsing and representation.

This module contains types specific to RTL parsing, including
SystemVerilog constructs and validation results.
"""

from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Iterator, TYPE_CHECKING
from enum import Enum

from brainsmith.core.dataflow.types import Direction, ProtocolType, InterfaceType

if TYPE_CHECKING:
    from .pragmas import Pragma


class PragmaType(Enum):
    """Valid pragma types recognized by the parser."""
    TOP_MODULE = "top_module"          # Specify the top module if multiple exist
    DATATYPE_CONSTRAINT = "datatype_constraint"  # Restrict datatype for an interface
    DERIVED_PARAMETER = "derived_parameter" # Link module param to python function
    WEIGHT = "weight"                  # Specify interface as a weight
    BDIM = "bdim"                      # Override block dimensions for an interface
    SDIM = "sdim"                      # Override stream dimensions for an interface
    DATATYPE = "datatype"              # Map interface datatype properties to RTL parameters
    ALIAS = "alias"                    # Expose RTL parameter with different name in nodeattr
    AXILITE_PARAM = "axilite_param"    # Mark parameter as AXI-Lite configuration related
    RELATIONSHIP = "relationship"      # Define relationships between interfaces

@dataclass
class Port:
    """SystemVerilog port representation.
    
    Attributes:
        name: Port identifier
        direction: Port direction (input/output/inout)
        width: Bit width expression (preserved as string)
        description: Optional documentation from RTL comments
    """
    name: str
    direction: Direction
    width: str = "1"  # Default to single bit
    description: Optional[str] = None
    
    # Legacy compatibility
    array_bounds: Optional[List[int]] = field(default=None, init=False)

    def __post_init__(self):
        """Validate port attributes, converting string direction to Enum if needed."""
        # Convert string direction to enum if needed
        if isinstance(self.direction, str):
            self.direction = Direction(self.direction.lower())
    
    @property
    def total_width(self) -> Optional[int]:
        """Calculate total width if parseable as integer.
        
        Returns:
            Integer width if parseable, None for complex expressions.
        """
        try:
            # Simple case - just a number
            return int(self.width)
        except (ValueError, TypeError):
            # Complex expression - return None instead of lying
            return None


PortGroup = Dict[str, Port]  # Alias for easier type hints


@dataclass
class Parameter:
    """RTL parameter representation.
    
    Simple parameter object that represents a SystemVerilog parameter.
    The parameter's role and usage is determined by its location in the
    data structure, not by fields on the parameter itself.
    
    Attributes:
        name: RTL parameter identifier
        rtl_type: SystemVerilog type (e.g., "integer", "logic", etc.)
        default_value: Default value from RTL (as string)
        line_number: Source location for error reporting
        kernel_value: Optional value for special cases:
                     - For ALIAS parameters: the nodeattr name (e.g., "parallelism_factor")
                     - For DERIVED parameters: the Python expression (e.g., "self.get_nodeattr('PE') * 2")
    """
    # Identity
    name: str
    rtl_type: Optional[str] = None  # SystemVerilog type
    
    # Values
    default_value: Optional[str] = None  # Raw string value from RTL
    kernel_value: Optional[str] = None  # For ALIAS names or DERIVED expressions
    
    # Metadata
    line_number: Optional[int] = None
    
    @property
    def nodeattr_name(self) -> str:
        """Get the node attribute name for this parameter.
        
        Returns kernel_value if set (ALIAS case), otherwise the parameter name.
        """
        return self.kernel_value if self.kernel_value else self.name
    
    @property
    def template_var(self) -> str:
        """Template substitution variable."""
        return f"${self.name.upper()}$"
    
    @property
    def template_param_name(self) -> str:
        """Template substitution name (e.g., $PE$)."""
        return f"${self.name.upper()}$"
    
    @property
    def resolved_default(self) -> Optional[Any]:
        """Get resolved default value (parsed from RTL)."""
        return self._parse_value(self.default_value)
    
    def _parse_value(self, value: Optional[str]) -> Optional[Any]:
        """Parse RTL string value to Python type."""
        if not value:
            return None
        try:
            # Handle SystemVerilog formats
            if value.startswith("'b"):
                return int(value[2:], 2)
            elif value.startswith("'h"):
                return int(value[2:], 16)
            elif value.startswith("'d"):
                return int(value[2:])
            else:
                return int(value)
        except (ValueError, TypeError):
            return value  # Return as string if not parseable
    
    def get_numeric_value(self) -> Optional[int]:
        """Legacy method - try to parse parameter value as integer.
        
        Returns:
            Integer value or None if not parseable
        """
        return self._parse_value(self.default_value) if isinstance(self._parse_value(self.default_value), int) else None
    
    @property
    def needs_nodeattr(self) -> bool:
        """Check if this parameter needs to be exposed as a node attribute.
        
        Parameters don't need node attributes if:
        - They have a kernel_value (alias or derived expression)
        - They are localparams (compile-time constants)
        
        Returns:
            True if parameter needs a node attribute, False otherwise
        """
        # Parameters with kernel_value are either aliased or derived
        if self.kernel_value:
            return False
        
        # Localparams are compile-time constants
        if self.rtl_type and 'localparam' in self.rtl_type.lower():
            return False
            
        return True
    
    def is_string_type(self) -> bool:
        """Check if this parameter should be typed as a string in nodeattr.
        
        Returns True if the default value is a string literal (wrapped in quotes).
        This is used to determine whether to use 's' or 'i' type in nodeattr.
        
        Returns:
            True if parameter has a string literal default value, False otherwise
        """
        if not self.default_value:
            return False
        
        val = self.default_value.strip()
        # Check for double quotes or single quotes
        if (val.startswith('"') and val.endswith('"')) or \
           (val.startswith("'") and val.endswith("'")):
            return True
        
        return False


@dataclass
class ParsedModule:
    """Parsed RTL module representation.
    
    Contains all information extracted from a SystemVerilog module.
    """
    name: str
    ports: List[Port]
    parameters: List[Parameter]
    pragmas: List['Pragma']
    file_path: str = "<string>"
    line_number: int = 0
    
    def get_port(self, name: str) -> Optional[Port]:
        """Get a port by name."""
        for port in self.ports:
            if port.name == name:
                return port
        return None
    
    def get_parameter(self, name: str) -> Optional[Parameter]:
        """Get a parameter by name."""
        for param in self.parameters:
            if param.name == name:
                return param
        return None
    
    def get_input_ports(self) -> List[Port]:
        """Get all input ports."""
        return [p for p in self.ports if p.direction == Direction.INPUT]
    
    def get_output_ports(self) -> List[Port]:
        """Get all output ports."""
        return [p for p in self.ports if p.direction == Direction.OUTPUT]