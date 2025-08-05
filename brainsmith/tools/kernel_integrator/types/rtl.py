"""
RTL types for parsing and representation.

This module contains types specific to RTL parsing, including
SystemVerilog constructs and validation results.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

from .core import PortDirection


class PragmaType(Enum):
    """Types of pragmas recognized by the RTL parser."""
    INTERFACE = "INTERFACE"
    TOP = "TOP"
    DATATYPE = "DATATYPE"
    WEIGHT = "WEIGHT"
    PARAMETER_DATATYPE = "PARAMETER_DATATYPE"
    ALIAS = "ALIAS"
    DERIVED_PARAMETER = "DERIVED_PARAMETER"
    AXI_LITE_PARAMETER = "AXI_LITE_PARAMETER"
    BDIM = "BDIM"
    SDIM = "SDIM"
    RELATIONSHIP = "RELATIONSHIP"
    MODULE = "MODULE"
    UNKNOWN = "UNKNOWN"


@dataclass
class Port:
    """Single RTL port definition.
    
    Represents a SystemVerilog port with its properties.
    """
    name: str
    direction: PortDirection
    width: int
    array_bounds: Optional[List[int]] = None
    
    @property
    def total_width(self) -> int:
        """Calculate total width including array dimensions."""
        if self.array_bounds:
            array_size = 1
            for bound in self.array_bounds:
                array_size *= bound
            return self.width * array_size
        return self.width
    
    def is_array(self) -> bool:
        """Check if this port is an array."""
        return self.array_bounds is not None and len(self.array_bounds) > 0


@dataclass
class Parameter:
    """RTL parameter definition.
    
    Represents a SystemVerilog parameter or localparam.
    """
    name: str
    value: str
    is_local: bool = False
    
    def get_numeric_value(self) -> Optional[int]:
        """Try to parse parameter value as integer.
        
        Returns:
            Integer value or None if not parseable
        """
        try:
            # Handle common SystemVerilog number formats
            if self.value.startswith("'b"):
                return int(self.value[2:], 2)
            elif self.value.startswith("'h"):
                return int(self.value[2:], 16)
            elif self.value.startswith("'d"):
                return int(self.value[2:])
            else:
                return int(self.value)
        except (ValueError, TypeError):
            return None


@dataclass
class PortGroup:
    """Group of related ports that may form an interface.
    
    Used during interface detection to group ports by naming patterns.
    """
    prefix: str
    ports: List[Port] = field(default_factory=list)
    
    def add_port(self, port: Port) -> None:
        """Add a port to this group."""
        self.ports.append(port)
    
    def get_port(self, suffix: str) -> Optional[Port]:
        """Get a port by suffix (e.g., 'valid', 'ready', 'data')."""
        for port in self.ports:
            if port.name == f"{self.prefix}_{suffix}" or port.name == f"{self.prefix}{suffix}":
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