############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
RTL Parser Data Structures.

This module contains data structures specific to RTL parsing, including
SystemVerilog port and parameter representations, pragma types, and 
validation results.

Classes and Enums:
- PragmaType: Valid pragma types recognized by the parser
- Parameter: SystemVerilog parameter representation
- Port: SystemVerilog port representation
- PortGroup: Grouped ports forming potential interfaces
- ProtocolValidationResult: Protocol validation results
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any
import logging

# Import shared types from main data module
from ..data import InterfaceType

# Temporary: Define InterfaceDirection locally to avoid import issues
from enum import Enum as _Enum
class InterfaceDirection(_Enum):
    INPUT = "input"
    OUTPUT = "output"

# Set up logger for this module
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class PortDirection(Enum):
    """RTL port directions (includes bidirectional).
    
    This enum is specific to RTL ports which can be bidirectional (INOUT).
    For dataflow interfaces, use InterfaceDirection which only has INPUT/OUTPUT.
    """
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"
    
    def to_interface_direction(self) -> Optional[InterfaceDirection]:
        """Convert to dataflow interface direction.
        
        Returns:
            InterfaceDirection for INPUT/OUTPUT, None for INOUT
        """
        if self == PortDirection.INPUT:
            return InterfaceDirection.INPUT
        elif self == PortDirection.OUTPUT:
            return InterfaceDirection.OUTPUT
        else:
            return None  # INOUT has no dataflow equivalent

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


# ============================================================================
# RTL DATA STRUCTURES
# ============================================================================

@dataclass
class Parameter:
    """SystemVerilog parameter representation.
    
    Attributes:
        name: Parameter identifier
        param_type: Parameter datatype (e.g., "int", "logic", "derived")
        default_value: Default value if specified
        description: Optional documentation from RTL comments
        template_param_name: Name used in the wrapper template (e.g., $NAME$).
    """
    name: str
    param_type: Optional[str] = None  # Parameter datatype (can be None for typeless parameters)
    default_value: Optional[str] = None
    description: Optional[str] = None
    template_param_name: str = field(init=False)  # Computed template parameter name

    def __post_init__(self):
        """Validate parameter attributes after initialization."""
        if not self.name.isidentifier():
            raise ValueError(f"Invalid parameter name: {self.name}")
        self.template_param_name = f"${self.name.upper()}$"


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

    def __post_init__(self):
        """Validate port attributes, converting string direction to Enum if needed."""
        if not self.name.isidentifier():
            raise ValueError(f"Invalid port name: {self.name}")
        if not isinstance(self.direction, PortDirection):
            if isinstance(self.direction, str):
                try:
                    self.direction = PortDirection(self.direction.lower())
                except ValueError:
                    raise ValueError(f"Invalid port direction string: {self.direction}")
            else:
                raise ValueError(f"Invalid port direction type: {type(self.direction)}")


@dataclass
class PortGroup:
    """Represents a group of related ports potentially forming an interface.

    This is an intermediate structure created by the InterfaceScanner based on
    naming conventions, before protocol validation.
    """
    interface_type: InterfaceType
    name: Optional[str] = None # e.g., "in0" for AXI-Stream, "config" for AXI-Lite
    ports: Dict[str, Port] = field(default_factory=dict) # Maps signal suffix (e.g., TDATA) or full name to Port object
    metadata: Dict[str, Any] = field(default_factory=dict) # e.g., data width for AXI

    def add_port(self, port: Port, key: Optional[str] = None) -> None:
        """Adds a port to the group, using a specific key or the port name.

        If a key (e.g., signal suffix like 'TDATA') is provided, it's used.
        Otherwise, the full port name is used as the key.
        Warns when overriding existing keys.
        """
        if key is None:
            key = port.name
        if key in self.ports:
            logger.warning(f"Overwriting port key '{key}' in PortGroup '{self.name}'")
        self.ports[key] = port


@dataclass
class ProtocolValidationResult:
    """Represents the result of a protocol validation check.
    
    Simple validation result for RTL protocol validation.
    """
    valid: bool
    message: Optional[str] = None


# Module exports
__all__ = [
    # Enums
    "PortDirection",
    "PragmaType",
    
    # RTL Data Structures  
    "Parameter",
    "Port", 
    "PortGroup",
    "ProtocolValidationResult",
]