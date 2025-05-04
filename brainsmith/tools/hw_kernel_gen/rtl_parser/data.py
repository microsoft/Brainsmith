"""Data structures for RTL Parser.

This module defines the core data structures used by the RTL Parser to represent
parsed SystemVerilog modules and their components.

Each class uses Python's dataclass decorator for clean initialization and
representation, along with type hints for better IDE support and runtime
validation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any # Added Any for metadata flexibility

# Import Interface type for HWKernel
from brainsmith.tools.hw_kernel_gen.rtl_parser.interface_types import Interface, InterfaceType # Added InterfaceType

class Direction(Enum):
    """Port direction enumeration."""
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"

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
        """Validate parameter attributes."""
        if not self.name.isidentifier():
            raise ValueError(f"Invalid parameter name: {self.name}")
        if not self.param_type:
            raise ValueError("Parameter type cannot be empty")

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
        """Validate port attributes."""
        if not self.name.isidentifier():
            raise ValueError(f"Invalid port name: {self.name}")
        if not isinstance(self.direction, Direction):
            if isinstance(self.direction, str):
                try:
                    self.direction = Direction(self.direction.lower())
                except ValueError:
                    raise ValueError(f"Invalid port direction: {self.direction}")
            else:
                raise ValueError(f"Invalid port direction type: {type(self.direction)}")

@dataclass
class Pragma:
    """Brainsmith pragma representation.
    
    Pragmas are special comments that provide additional information to the
    Hardware Kernel Generator. They follow the format:
        // @brainsmith <type> <inputs...>
    
    Attributes:
        type: Pragma type identifier
        inputs: List of space-separated inputs
        line_number: Source line number for error reporting
        processed_data: Optional processed data from pragma handler
    """
    type: str
    inputs: List[str]
    line_number: int
    processed_data: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate pragma attributes."""
        if not self.type:
            raise ValueError("Pragma type cannot be empty")
        if not isinstance(self.inputs, list):
            raise ValueError("Pragma inputs must be a list")

@dataclass
class ModuleSummary:
    """Represents the parsed information for a single SystemVerilog module.

    Attributes:
        name: The name of the module.
        ports: A list of Port objects representing the module's ports.
        parameters: A dictionary of parameters found in the module.
        interfaces: A dictionary mapping interface names to Interface objects.
    """
    name: str
    ports: List[Port] = field(default_factory=list)
    parameters: Dict[str, Parameter] = field(default_factory=dict) # Changed to Dict[str, Parameter]
    interfaces: Dict[str, Interface] = field(default_factory=dict)

@dataclass
class HWKernel:
    """Top-level representation of a parsed hardware kernel.
    
    This structure holds the consolidated information extracted from an RTL file,
    focusing on a single target module (often specified by a pragma).
    
    Attributes:
        name: Kernel (module) name
        ports: List of ports
        parameters: List of parameters
        interfaces: Dictionary of detected interfaces (e.g., AXI-Lite, AXI-Stream)
        pragmas: List of Brainsmith pragmas found
        metadata: Optional dictionary for additional info (e.g., source file)
    """
    name: str
    ports: List[Port] = field(default_factory=list)
    parameters: List[Parameter] = field(default_factory=list)
    interfaces: Dict[str, Interface] = field(default_factory=dict)
    pragmas: List[Pragma] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate HWKernel attributes."""
        if not self.name:
            raise ValueError("HWKernel name cannot be empty")
        # Add more validation as needed