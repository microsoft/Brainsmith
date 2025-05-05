from __future__ import annotations
############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Data structures for RTL Parser.

This module defines the core data structures used by the RTL Parser to represent
parsed SystemVerilog modules, their components (ports, parameters, pragmas),
and the identified hardware interfaces (Global Control, AXI-Stream, AXI-Lite).

Includes:
- Enums for Port Direction and Interface Type.
- Dataclasses for Parameter, Port, Pragma, ValidationResult, PortGroup, Interface, and HWKernel.

Each class uses Python's dataclass decorator for clean initialization and
representation, along with type hints for better IDE support and runtime
validation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any

# --- Enums ---

class Direction(Enum):
    """Port direction enumeration."""
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"

class InterfaceType(Enum):
    """Enumeration of supported interface types."""
    GLOBAL_CONTROL = "global"
    AXI_STREAM = "axis"
    AXI_LITE = "axilite"
    UNKNOWN = "unknown" # For ports not part of a recognized interface

class PragmaType(Enum):
    """Valid pragma types recognized by the parser."""
    TOP_MODULE = "top_module"          # Specify the top module if multiple exist
    DATATYPE = "datatype"              # Restrict datatype for an interface
    DERIVED_PARAMETER = "derived_parameter" # Link module param to python function

# --- Simple Data Structures ---

@dataclass
class ValidationResult:
    """Represents the result of a protocol validation check."""
    valid: bool
    message: Optional[str] = None

@dataclass
class Parameter:
    """SystemVerilog parameter representation.
    
    Attributes:
        name: Parameter identifier
        param_type: Parameter datatype (e.g., "int", "logic")
        default_value: Default value if specified
        description: Optional documentation from RTL comments
    """
    name: str
    param_type: str
    default_value: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self):
        """Validate parameter attributes after initialization."""
        if not self.name.isidentifier():
            raise ValueError(f"Invalid parameter name: {self.name}")
        # Parameter type can be None for typeless parameters

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

    def __post_init__(self):
        """Validate port attributes, converting string direction to Enum if needed."""
        if not self.name.isidentifier():
            raise ValueError(f"Invalid port name: {self.name}")
        if not isinstance(self.direction, Direction):
            if isinstance(self.direction, str):
                try:
                    self.direction = Direction(self.direction.lower())
                except ValueError:
                    raise ValueError(f"Invalid port direction string: {self.direction}")
            else:
                raise ValueError(f"Invalid port direction type: {type(self.direction)}")

# --- Intermediate Structures ---

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
        Currently overrides existing keys without warning.
        """
        if key is None:
            key = port.name
        # if key in self.ports:
            # logger.warning(f"Overwriting port key '{key}' in PortGroup '{self.name}'")
        self.ports[key] = port

# --- Validated/Complex Structures ---

@dataclass
class Interface:
    """Represents a fully validated and identified interface.

    Created by the InterfaceBuilder after a PortGroup successfully passes
    validation by the ProtocolValidator.
    """
    name: str # e.g., "global", "in0", "config"
    type: InterfaceType
    ports: Dict[str, Port] # Maps signal suffix/name to Port object
    validation_result: ValidationResult
    metadata: Dict[str, Any] = field(default_factory=dict) # e.g., data width, address width

# --- Pragma Structure ---

@dataclass
class Pragma:
    """Brainsmith pragma representation.
    
    Pragmas are special comments that provide additional information to the
    Hardware Kernel Generator. They follow the format:
        // @brainsmith <type> <inputs...>
    
    Attributes:
        type: Pragma type identifier (using PragmaType enum)
        inputs: List of space-separated inputs
        line_number: Source line number for error reporting
        processed_data: Optional processed data from pragma handler
    """
    type: PragmaType
    inputs: List[str]
    line_number: int
    processed_data: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate pragma attributes after initialization."""
        if not self.type:
            raise ValueError("Pragma type cannot be empty")
        if not isinstance(self.inputs, list):
            raise ValueError("Pragma inputs must be a list")

# --- Top-Level Structure ---

@dataclass
class HWKernel:
    """Top-level representation of a parsed hardware kernel.
    
    This structure holds the consolidated information extracted from an RTL file,
    focusing on a single target module (often specified by a pragma).
    
    Attributes:
        name: Kernel (module) name
        parameters: List of parameters
        interfaces: Dictionary of detected interfaces (e.g., AXI-Lite, AXI-Stream)
        pragmas: List of Brainsmith pragmas found
        metadata: Optional dictionary for additional info (e.g., source file)
    """
    name: str
    parameters: List[Parameter] = field(default_factory=list)
    interfaces: Dict[str, Interface] = field(default_factory=dict)
    pragmas: List[Pragma] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate HWKernel attributes after initialization."""
        if not self.name:
            raise ValueError("HWKernel name cannot be empty")
        # Add more validation as needed
