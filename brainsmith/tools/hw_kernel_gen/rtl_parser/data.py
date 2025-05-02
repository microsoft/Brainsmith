"""Data structures for RTL Parser.

This module defines the core data structures used by the RTL Parser to represent
parsed SystemVerilog modules and their components.

Each class uses Python's dataclass decorator for clean initialization and
representation, along with type hints for better IDE support and runtime
validation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional

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
class HWKernel:
    """Top-level representation of a parsed hardware kernel.
    
    This class represents all information extracted from a SystemVerilog module
    that will be used to generate a FINN-compatible hardware kernel.
    
    Attributes:
        name: Module name
        parameters: List of module parameters
        ports: List of module ports
        pragmas: List of Brainsmith pragmas
        metadata: Additional metadata extracted during parsing
    """
    name: str
    parameters: List[Parameter] = field(default_factory=list)
    ports: List[Port] = field(default_factory=list)
    pragmas: List[Pragma] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate kernel attributes."""
        if not self.name.isidentifier():
            raise ValueError(f"Invalid kernel name: {self.name}")
        
        # Ensure unique parameter names
        param_names = [p.name for p in self.parameters]
        if len(param_names) != len(set(param_names)):
            raise ValueError("Duplicate parameter names found")
        
        # Ensure unique port names
        port_names = [p.name for p in self.ports]
        if len(port_names) != len(set(port_names)):
            raise ValueError("Duplicate port names found")

    def add_parameter(self, parameter: Parameter) -> None:
        """Add a parameter to the kernel.
        
        Args:
            parameter: Parameter instance to add
            
        Raises:
            ValueError: If parameter with same name already exists
        """
        if any(p.name == parameter.name for p in self.parameters):
            raise ValueError(f"Parameter {parameter.name} already exists")
        self.parameters.append(parameter)

    def add_port(self, port: Port) -> None:
        """Add a port to the kernel.
        
        Args:
            port: Port instance to add
            
        Raises:
            ValueError: If port with same name already exists
        """
        if any(p.name == port.name for p in self.ports):
            raise ValueError(f"Port {port.name} already exists")
        self.ports.append(port)

    def add_pragma(self, pragma: Pragma) -> None:
        """Add a pragma to the kernel.
        
        Args:
            pragma: Pragma instance to add
        """
        self.pragmas.append(pragma)