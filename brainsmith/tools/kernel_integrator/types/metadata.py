"""
Metadata types for higher-level kernel representation.

This module contains types that represent parsed and processed kernel
information at a higher abstraction level than raw RTL.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union, Iterator
from collections.abc import MutableMapping

from brainsmith.core.dataflow.types import Direction, InterfaceType
from brainsmith.core.dataflow.constraint_types import DatatypeConstraintGroup

from .rtl import Port, PortGroup, Parameter, Direction


@dataclass
class InterfaceMetadata(MutableMapping[str, Port]):
    """Base metadata for all interfaces."""
    name: str
    ports: Dict[str, Port]
    description: Optional[str]

    def add_port(self, port: Port):
        """Add a port to the interface."""
        self.ports[port.name] = port

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





    
@dataclass
class AXIStreamMetadata(InterfaceMetadata):
    """Metadata for a AXI-Stream interface."""
    # interface_type is determined by direction: INPUT or OUTPUT
    direction: Direction
    is_weight: bool = False

    # Owned RTL Parameters
    bdim_params: List[Parameter] = field(default_factory=list)
    sdim_params: List[Parameter] = field(default_factory=list)
    dtype_params: List[Parameter] = field(default_factory=list)

    # Shape expressions for tiling system
    bdim_shape: Optional[List] = None
    sdim_shape: Optional[List] = None

    # TAFK TODO: Fix/add these
    relationships: Dict[str, str] = field(default_factory=dict)
    datatype_constraints: DatatypeConstraintGroup = field(default_factory=DatatypeConstraintGroup)
    
    @property
    def interface_type(self) -> InterfaceType:
        """Interface type based on direction."""
        if self.is_weight:
            return InterfaceType.WEIGHT
        return InterfaceType.INPUT if self.direction == Direction.INPUT else InterfaceType.OUTPUT

@dataclass
class AXILiteMetadata(InterfaceMetadata):
    """Metadata for an AXI-Lite interface."""
    interface_type: InterfaceType = InterfaceType.CONFIG
    has_write: bool = True  # Default to true, can be overridden
    has_read: bool = True   # Default to true, can be overridden

    # Owned RTL Parameters
    enable_param: Optional[Parameter] = None
    data_width_param: Optional[Parameter] = None
    addr_width_param: Optional[Parameter] = None

    @property
    def is_read_only(self) -> bool:
        """Check if this AXI-Lite interface is read-only."""
        return not self.has_write and self.has_read

    @property
    def is_write_only(self) -> bool:
        """Check if this AXI-Lite interface is write-only."""
        return self.has_write and not self.has_read
    
@dataclass
class ControlMetadata(InterfaceMetadata):
    """Metadata for a Control interface."""
    interface_type: InterfaceType = InterfaceType.CONTROL


@dataclass
class KernelMetadata:
    """Complete kernel metadata.
    
    Represents all information about a kernel needed for code generation,
    including interfaces, parameters, and relationships.
    """
    # Core attributes matching original structure
    name: str # Module/Kernel name
    source_file: str
    # Interface metadata (required)
    control: ControlMetadata
    # Optional fields with defaults
    parameters: List[Parameter] = field(default_factory=list)
    inputs: List[AXIStreamMetadata] = field(default_factory=list)
    outputs: List[AXIStreamMetadata] = field(default_factory=list)
    config: List[AXILiteMetadata] = field(default_factory=list)

    # Simple transformations as properties
    @property
    def class_name(self) -> str:
        """Get PascalCase class name from module name."""
        from brainsmith.tools.kernel_integrator.utils import pascal_case
        return pascal_case(self.name)
    



    def validate(self) -> List[str]:
        """Validate kernel metadata.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Must have at least input and output
        if len(self.inputs) < 1:
            errors.append("No input interface found")
        if len(self.outputs) < 1:
            errors.append("No output interface found")

        # TAFK TODO: Flesh out

        return errors
