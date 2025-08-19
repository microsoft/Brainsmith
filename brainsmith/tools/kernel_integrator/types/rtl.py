"""
RTL types for parsing and representation.

This module contains types specific to RTL parsing, including
SystemVerilog constructs and validation results.
"""

from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Iterator
from enum import Enum

from brainsmith.core.dataflow.types import  Direction, ProtocolType, InterfaceType


class PragmaType(Enum):
    """Valid pragma types recognized by the parser."""
    TOP_MODULE = "top_module"          # Specify the top module if multiple exist
    DATATYPE = "datatype"              # Restrict datatype for an interface
    DERIVED_PARAMETER = "derived_parameter" # Link module param to python function
    WEIGHT = "weight"                  # Specify interface as a weight
    BDIM = "bdim"                      # Override block dimensions for an interface
    SDIM = "sdim"                      # Override stream dimensions for an interface
    DATATYPE_PARAM = "datatype_param"  # Map interface datatype properties to RTL parameters
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


@dataclass
class PortGroup(MutableMapping[str, Port]):
    name: Optional[str] = None
    interface_type: Optional[InterfaceType] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    ports: Dict[str, Port] = field(default_factory=dict)

    # MutableMapping methods
    def __getitem__(self, key: str) -> Port: return self.ports[key]
    def __setitem__(self, key: str, value: Port) -> None: self.ports[key] = value
    def __delitem__(self, key: str) -> None: del self.ports[key]
    def __iter__(self) -> Iterator[str]: return iter(self.ports)
    def __len__(self) -> int: return len(self.ports)

    # Helpers
    def add(self, suffix: str, port: Port) -> None:
        self.ports[suffix] = port

    def get_port(self, suffix: str) -> Optional[Port]:
        return self.ports.get(suffix)

PortGroup = Dict[str, Port]  # Alias for easier type hints

@dataclass
class Interface:
    name: str
    ports: PortGroup = field(default_factory=dict)

    def add(self, prefix: str, suffix: str, port: Port) -> None:
        self.ports.setdefault(prefix, {})[suffix] = port

    def get(self, prefix: str, suffix: str) -> Optional[Port]:
        return self.ports.get(prefix, {}).get(suffix)


@dataclass
class Parameter:
    """Unified parameter representation for kernel integration.
    
    Combines RTL parsing, template generation, and code generation needs.
    Replaces: Parameter, ParameterDefinition, AttributeBinding, ParameterBinding.
    
    Attributes:
        name: RTL parameter identifier
        rtl_type: SystemVerilog type (e.g., "integer", "logic", etc.)
        default_value: Default value from RTL (as string)
        line_number: Source location for error reporting
        source_type: How parameter gets its value (enum)
        source_detail: Detailed source information (dict)
        interface_name: Which interface owns this parameter (if applicable)
    """
    # Identity
    name: str
    rtl_type: Optional[str] = None  # SystemVerilog type
    
    # Values
    default_value: Optional[str] = None  # Raw string value from RTL
    
    # Metadata
    line_number: Optional[int] = None
    
    # Enhanced source information
    source_detail: Dict[str, Any] = field(default_factory=dict)
    # Examples:
    # NODEATTR_ALIAS: {"nodeattr_name": "parallelism_factor"}
    # INTERFACE_DATATYPE: {"interface": "input0", "property": "width"}
    # DERIVED: {"expression": "self.get_nodeattr('PE') * 2"}
    # INTERFACE_SHAPE: {"interface": "input0", "dimension": 0, "shape_type": "bdim"}
    
    # Relationships
    interface_name: Optional[str] = None  # Which interface owns this
    
    @property
    def nodeattr_name(self) -> str:
        """Get the node attribute name for this parameter."""
        if self.source_type == SourceType.NODEATTR_ALIAS:
            return self.source_detail.get("nodeattr_name", self.name)
        return self.name
    
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


@dataclass
class ParsedModule:
    """Parsed RTL module representation.
    
    Contains all information extracted from a SystemVerilog module.
    """
    name: str
    ports: List[Port]
    parameters: List[Parameter]
    file_path: str
    line_number: int
    
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
