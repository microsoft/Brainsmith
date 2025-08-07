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


class SourceType(Enum):
    """How a parameter gets its value during code generation."""
    RTL = "rtl"                    # Direct from RTL (exposed)
    NODEATTR_ALIAS = "alias"       # Aliased node attribute
    DERIVED = "derived"            # Computed from expression
    INTERFACE_DATATYPE = "if_dtype" # From interface datatype property
    INTERFACE_SHAPE = "if_shape"   # From interface shape (BDIM/SDIM)
    INTERNAL_DATATYPE = "int_dtype" # From internal datatype
    AXILITE = "axilite"           # From AXI-Lite interface
    CONSTANT = "constant"          # Fixed constant value


class ParameterCategory(Enum):
    """Semantic category of parameter."""
    ALGORITHM = "algorithm"         # Core algorithm parameter
    DATATYPE = "datatype"          # Datatype-related (width, signed)
    SHAPE = "shape"                # Shape-related (BDIM, SDIM)
    CONTROL = "control"            # Control/config (AXI-Lite)
    INTERNAL = "internal"          # Internal mechanism

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
        rtl_type: SystemVerilog type (e.g., "integer", "logic", etc.)
        default_value: Default value from RTL (as string)
        line_number: Source location for error reporting
        source_type: How parameter gets its value (enum)
        source_detail: Detailed source information (dict)
        category: Semantic category of parameter
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
    source_type: SourceType = SourceType.RTL
    source_detail: Dict[str, Any] = field(default_factory=dict)
    # Examples:
    # NODEATTR_ALIAS: {"nodeattr_name": "parallelism_factor"}
    # INTERFACE_DATATYPE: {"interface": "input0", "property": "width"}
    # DERIVED: {"expression": "self.get_nodeattr('PE') * 2"}
    # INTERFACE_SHAPE: {"interface": "input0", "dimension": 0, "shape_type": "bdim"}
    
    # Semantic categorization
    category: ParameterCategory = ParameterCategory.ALGORITHM
    
    # Relationships
    interface_name: Optional[str] = None  # Which interface owns this
    
    @property
    def is_exposed(self) -> bool:
        """Check if exposed as node attribute."""
        return self.source_type == SourceType.RTL
    
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