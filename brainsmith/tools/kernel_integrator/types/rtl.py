"""
RTL types for parsing and representation.

This module contains types specific to RTL parsing, including
SystemVerilog constructs and validation results.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from enum import Enum


class PortDirection(Enum):
    """Direction of RTL ports."""
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"

if TYPE_CHECKING:
    from brainsmith.core.dataflow.types import InterfaceType


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
    direction: PortDirection
    width: str = "1"  # Default to single bit
    description: Optional[str] = None
    
    # Legacy compatibility
    array_bounds: Optional[List[int]] = field(default=None, init=False)

    def __post_init__(self):
        """Validate port attributes, converting string direction to Enum if needed."""
        # Convert string direction to enum if needed
        if isinstance(self.direction, str):
            self.direction = PortDirection(self.direction.lower())
        
        # Legacy compatibility - try to parse array bounds from width
        self.array_bounds = None  # Not currently parsed
    
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
    
    def is_array(self) -> bool:
        """Check if this port is an array."""
        return self.array_bounds is not None and len(self.array_bounds) > 0


@dataclass
class Parameter:
    """Unified parameter representation for kernel integration.
    
    Combines RTL parsing, template generation, and code generation needs.
    Replaces: Parameter, ParameterDefinition, AttributeBinding, ParameterBinding.
    
    Attributes:
        name: RTL parameter identifier
        param_type: SystemVerilog type (legacy name, maps to rtl_type)
        default_value: Default value from RTL (as string)
        line_number: Source location for error reporting
        is_exposed: Whether parameter is available to user
        source: How parameter gets its value ("rtl", "derived", "alias", "axilite")
        source_ref: Reference for derived/alias (expression or target parameter)
    """
    # Identity
    name: str
    param_type: Optional[str] = None  # SystemVerilog type (legacy field name)
    
    # Values
    default_value: Optional[str] = None  # Raw string value from RTL
    
    # Metadata
    line_number: Optional[int] = None
    
    # Exposure & Policy
    is_exposed: bool = True  # Available to user
    
    # Binding info
    source: str = "rtl"  # "rtl", "derived", "alias", "axilite"
    source_ref: Optional[str] = None  # For derived/alias: expression/target
    
    @property
    def rtl_type(self) -> Optional[str]:
        """Modern name for param_type."""
        return self.param_type
    
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
class PortGroup:
    """Represents a group of related ports potentially forming an interface.

    This is an intermediate structure created by the InterfaceScanner based on
    naming conventions, before protocol validation.
    """
    interface_type: "InterfaceType"  # Forward reference since imported from dataflow
    name: Optional[str] = None # e.g., "in0" for AXI-Stream, "config" for AXI-Lite
    ports: Dict[str, Port] = field(default_factory=dict) # Maps signal suffix (e.g., TDATA) or full name to Port object
    metadata: Dict[str, Any] = field(default_factory=dict) # e.g., data width for AXI

    def add_port(self, port: Port, key: Optional[str] = None) -> None:
        """Adds a port to the group, using a specific key or the port name.

        If a key (e.g., signal suffix like 'TDATA') is provided, it's used.
        Otherwise, the full port name is used as the key.
        Warns when overriding existing keys.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        if key is None:
            key = port.name
        if key in self.ports:
            logger.warning(f"Overwriting port key '{key}' in PortGroup '{self.name}'")
        self.ports[key] = port
    
    def get_port(self, suffix: str) -> Optional[Port]:
        """Get a port by suffix (e.g., 'valid', 'ready', 'data')."""
        # First try direct key lookup
        if suffix in self.ports:
            return self.ports[suffix]
        # Then try finding by port name suffix
        for key, port in self.ports.items():
            if port.name.endswith(suffix):
                return port
        return None
    
    @property
    def port_count(self) -> int:
        """Number of ports in this group."""
        return len(self.ports)


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
        return [p for p in self.ports if p.direction == PortDirection.INPUT]
    
    def get_output_ports(self) -> List[Port]:
        """Get all output ports."""
        return [p for p in self.ports if p.direction == PortDirection.OUTPUT]


@dataclass
class ValidationError:
    """Single validation error.
    
    Represents an error found during validation with severity and location info.
    """
    severity: str  # "error" or "warning"
    message: str
    location: Optional[str] = None
    
    def is_error(self) -> bool:
        """Check if this is an error (vs warning)."""
        return self.severity == "error"
    
    def is_warning(self) -> bool:
        """Check if this is a warning."""
        return self.severity == "warning"


@dataclass
class ValidationResult:
    """Result of validation operations.
    
    Contains all errors and warnings found during validation.
    """
    is_valid: bool = True
    errors: List[ValidationError] = field(default_factory=list)
    
    def add_error(self, message: str, location: Optional[str] = None) -> None:
        """Add an error to the validation result."""
        self.errors.append(ValidationError("error", message, location))
        self.is_valid = False
    
    def add_warning(self, message: str, location: Optional[str] = None) -> None:
        """Add a warning to the validation result."""
        self.errors.append(ValidationError("warning", message, location))
    
    def get_errors(self) -> List[ValidationError]:
        """Get only errors (not warnings)."""
        return [e for e in self.errors if e.is_error()]
    
    def get_warnings(self) -> List[ValidationError]:
        """Get only warnings."""
        return [e for e in self.errors if e.is_warning()]
    
    @property
    def error_count(self) -> int:
        """Number of errors."""
        return len(self.get_errors())
    
    @property
    def warning_count(self) -> int:
        """Number of warnings."""
        return len(self.get_warnings())
    
    def merge(self, other: "ValidationResult") -> None:
        """Merge another validation result into this one."""
        self.errors.extend(other.errors)
        self.is_valid = self.is_valid and other.is_valid


@dataclass
class ProtocolValidationResult:
    """Represents the result of a protocol validation check.
    
    Simple validation result for RTL protocol validation.
    """
    valid: bool
    message: Optional[str] = None