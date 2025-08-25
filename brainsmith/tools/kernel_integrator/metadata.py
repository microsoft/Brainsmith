"""
Metadata types for higher-level kernel representation.

This module contains types that represent parsed and processed kernel
information at a higher abstraction level than raw RTL.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterator, Tuple, Any
from collections.abc import MutableMapping

from brainsmith.core.dataflow.types import Direction, InterfaceType
from brainsmith.core.dataflow.constraint_types import DatatypeConstraintGroup

from .rtl_parser.types import Port, Parameter


@dataclass
class DataflowMetadata:
    """Shared metadata for dataflow properties of interfaces.
    
    Contains all attributes related to dataflow semantics that can apply
    to both AXI-Stream and AXI-Lite interfaces when used for data transfer.
    """
    is_weight: bool = False
    bdim_params: List[Parameter] = field(default_factory=list)
    sdim_params: List[Parameter] = field(default_factory=list)
    bdim_shape: Optional[List] = None
    sdim_shape: Optional[List] = None
    datatype_constraints: List[DatatypeConstraintGroup] = field(default_factory=list)
    relationships: Dict[str, str] = field(default_factory=dict)
    
    def has_shape_params(self) -> bool:
        """Check if interface has BDIM or SDIM parameters."""
        return bool(self.bdim_shape or self.sdim_shape)
    
    def get_shape_params(self) -> List[Parameter]:
        """Get all shape-related parameters."""
        return self.bdim_params + self.sdim_params
    
    def get_all_params(self) -> List[Parameter]:
        """Get all parameters (shape params) in consistent order."""
        return self.bdim_params + self.sdim_params


@dataclass
class DatatypeParameters:
    """Container for datatype-related parameters.
    
    Each property is optional and can hold at most one Parameter.
    This ensures no duplicate properties and provides structured access.
    """
    width: Optional[Parameter] = None
    signed: Optional[Parameter] = None
    bias: Optional[Parameter] = None
    format: Optional[Parameter] = None
    fractional_width: Optional[Parameter] = None
    exponent_width: Optional[Parameter] = None
    mantissa_width: Optional[Parameter] = None
    
    def has_any(self) -> bool:
        """Check if any datatype parameters are set."""
        return any([
            self.width,
            self.signed,
            self.bias,
            self.format,
            self.fractional_width,
            self.exponent_width,
            self.mantissa_width
        ])


@dataclass  
class InterfaceMetadata(MutableMapping[str, Port]):
    """Base metadata for all interfaces."""
    name: str
    ports: Dict[str, Port]
    compiler_name: Optional[str] = None  # Standardized name for compiler export

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
    
    def _get_signal(self, suffix: str) -> Optional[Port]:
        """Get signal by suffix (case-insensitive)."""
        return self.ports.get(suffix.upper())
    
    def has_parameters(self) -> bool:
        """Check if this interface has any parameters."""
        return False  # Base implementation - override in subclasses
    
    def supports_dataflow(self) -> bool:
        """Check if this interface type can have dataflow properties."""
        # Will be overridden by AXIStreamMetadata and AXILiteMetadata
        return False

    
@dataclass
class AXIStreamMetadata(InterfaceMetadata):
    """Metadata for a AXI-Stream interface."""
    # interface_type is determined by direction: INPUT or OUTPUT
    direction: Direction = field(default=Direction.INPUT)  # Will be set during parsing
    dtype_params: Optional[DatatypeParameters] = None
    dataflow: Optional[DataflowMetadata] = None
    
    @property
    def interface_type(self) -> InterfaceType:
        """Interface type based on direction."""
        if self.dataflow and self.dataflow.is_weight:
            return InterfaceType.WEIGHT
        return InterfaceType.INPUT if self.direction == Direction.INPUT else InterfaceType.OUTPUT
    
    # Delegation properties for backward compatibility
    @property
    def is_weight(self) -> bool:
        """Check if this interface is marked as weight."""
        return self.dataflow.is_weight if self.dataflow else False
    
    @property
    def bdim_params(self) -> List[Parameter]:
        """Get block dimension parameters."""
        return self.dataflow.bdim_params if self.dataflow else []
    
    @property
    def sdim_params(self) -> List[Parameter]:
        """Get stream dimension parameters."""
        return self.dataflow.sdim_params if self.dataflow else []
    
    @property
    def bdim_shape(self) -> Optional[List]:
        """Get block dimension shape expression."""
        return self.dataflow.bdim_shape if self.dataflow else None
    
    @property
    def sdim_shape(self) -> Optional[List]:
        """Get stream dimension shape expression."""
        return self.dataflow.sdim_shape if self.dataflow else None
    
    @property
    def datatype_constraints(self) -> List[DatatypeConstraintGroup]:
        """Get datatype constraints."""
        return self.dataflow.datatype_constraints if self.dataflow else []
    
    @property
    def relationships(self) -> Dict[str, str]:
        """Get relationships."""
        return self.dataflow.relationships if self.dataflow else {}
    
    # Signal role properties
    @property
    def tdata(self) -> Optional[Port]:
        """Get TDATA signal port."""
        return self._get_signal("TDATA")
    
    @property
    def tvalid(self) -> Optional[Port]:
        """Get TVALID signal port."""
        return self._get_signal("TVALID")
    
    @property
    def tready(self) -> Optional[Port]:
        """Get TREADY signal port."""
        return self._get_signal("TREADY")
    
    @property
    def tlast(self) -> Optional[Port]:
        """Get TLAST signal port."""
        return self._get_signal("TLAST")
    
    def has_parameters(self) -> bool:
        """Check if this interface has any parameters."""
        has_dataflow_params = self.dataflow and len(self.dataflow.get_all_params()) > 0
        return bool(has_dataflow_params or 
                   (self.dtype_params and self.dtype_params.has_any()))
    
    @property
    def has_shape_params(self) -> bool:
        """Check if interface has BDIM or SDIM parameters."""
        return self.dataflow.has_shape_params() if self.dataflow else False
    
    def get_all_params(self) -> List[Parameter]:
        """Get all parameters for this interface in consistent order."""
        params = []
        
        # Add dataflow parameters
        if self.dataflow:
            params.extend(self.dataflow.get_all_params())
        
        # Add non-None dtype params in consistent order
        if self.dtype_params:
            if self.dtype_params.width:
                params.append(self.dtype_params.width)
            if self.dtype_params.signed:
                params.append(self.dtype_params.signed)
            if self.dtype_params.bias:
                params.append(self.dtype_params.bias)
            if self.dtype_params.format:
                params.append(self.dtype_params.format)
            if self.dtype_params.fractional_width:
                params.append(self.dtype_params.fractional_width)
            if self.dtype_params.exponent_width:
                params.append(self.dtype_params.exponent_width)
            if self.dtype_params.mantissa_width:
                params.append(self.dtype_params.mantissa_width)
        
        return params
    
    def supports_dataflow(self) -> bool:
        """AXI-Stream interfaces support dataflow properties."""
        return True


@dataclass
class AXILiteMetadata(InterfaceMetadata):
    """Metadata for an AXI-Lite interface."""
    has_write: bool = True  # Default to true, can be overridden
    has_read: bool = True   # Default to true, can be overridden

    # Owned RTL Parameters
    enable_param: Optional[Parameter] = None
    data_width_param: Optional[Parameter] = None
    addr_width_param: Optional[Parameter] = None
    dtype_params: Optional[DatatypeParameters] = None
    dataflow: Optional[DataflowMetadata] = None
    
    @property
    def interface_type(self) -> InterfaceType:
        """Interface type based on whether it's marked as weight."""
        if self.dataflow and self.dataflow.is_weight:
            return InterfaceType.WEIGHT
        return InterfaceType.CONFIG

    # Delegation properties for dataflow metadata
    @property
    def is_weight(self) -> bool:
        """Check if this interface is marked as weight."""
        return self.dataflow.is_weight if self.dataflow else False
    
    @property
    def bdim_params(self) -> List[Parameter]:
        """Get block dimension parameters."""
        return self.dataflow.bdim_params if self.dataflow else []
    
    @property
    def sdim_params(self) -> List[Parameter]:
        """Get stream dimension parameters."""
        return self.dataflow.sdim_params if self.dataflow else []
    
    @property
    def bdim_shape(self) -> Optional[List]:
        """Get block dimension shape expression."""
        return self.dataflow.bdim_shape if self.dataflow else None
    
    @property
    def sdim_shape(self) -> Optional[List]:
        """Get stream dimension shape expression."""
        return self.dataflow.sdim_shape if self.dataflow else None
    
    @property
    def datatype_constraints(self) -> List[DatatypeConstraintGroup]:
        """Get datatype constraints."""
        return self.dataflow.datatype_constraints if self.dataflow else []
    
    @property
    def relationships(self) -> Dict[str, str]:
        """Get relationships."""
        return self.dataflow.relationships if self.dataflow else {}
    
    @property
    def is_read_only(self) -> bool:
        """Check if this AXI-Lite interface is read-only."""
        return not self.has_write and self.has_read

    @property
    def is_write_only(self) -> bool:
        """Check if this AXI-Lite interface is write-only."""
        return self.has_write and not self.has_read
    
    # Write Address Channel
    @property
    def awaddr(self) -> Optional[Port]:
        """Get AWADDR signal port."""
        return self._get_signal("AWADDR")
    
    @property
    def awprot(self) -> Optional[Port]:
        """Get AWPROT signal port."""
        return self._get_signal("AWPROT")
    
    @property
    def awvalid(self) -> Optional[Port]:
        """Get AWVALID signal port."""
        return self._get_signal("AWVALID")
    
    @property
    def awready(self) -> Optional[Port]:
        """Get AWREADY signal port."""
        return self._get_signal("AWREADY")
    
    # Write Data Channel
    @property
    def wdata(self) -> Optional[Port]:
        """Get WDATA signal port."""
        return self._get_signal("WDATA")
    
    @property
    def wstrb(self) -> Optional[Port]:
        """Get WSTRB signal port."""
        return self._get_signal("WSTRB")
    
    @property
    def wvalid(self) -> Optional[Port]:
        """Get WVALID signal port."""
        return self._get_signal("WVALID")
    
    @property
    def wready(self) -> Optional[Port]:
        """Get WREADY signal port."""
        return self._get_signal("WREADY")
    
    # Write Response Channel
    @property
    def bresp(self) -> Optional[Port]:
        """Get BRESP signal port."""
        return self._get_signal("BRESP")
    
    @property
    def bvalid(self) -> Optional[Port]:
        """Get BVALID signal port."""
        return self._get_signal("BVALID")
    
    @property
    def bready(self) -> Optional[Port]:
        """Get BREADY signal port."""
        return self._get_signal("BREADY")
    
    # Read Address Channel
    @property
    def araddr(self) -> Optional[Port]:
        """Get ARADDR signal port."""
        return self._get_signal("ARADDR")
    
    @property
    def arprot(self) -> Optional[Port]:
        """Get ARPROT signal port."""
        return self._get_signal("ARPROT")
    
    @property
    def arvalid(self) -> Optional[Port]:
        """Get ARVALID signal port."""
        return self._get_signal("ARVALID")
    
    @property
    def arready(self) -> Optional[Port]:
        """Get ARREADY signal port."""
        return self._get_signal("ARREADY")
    
    # Read Data Channel
    @property
    def rdata(self) -> Optional[Port]:
        """Get RDATA signal port."""
        return self._get_signal("RDATA")
    
    @property
    def rresp(self) -> Optional[Port]:
        """Get RRESP signal port."""
        return self._get_signal("RRESP")
    
    @property
    def rvalid(self) -> Optional[Port]:
        """Get RVALID signal port."""
        return self._get_signal("RVALID")
    
    @property
    def rready(self) -> Optional[Port]:
        """Get RREADY signal port."""
        return self._get_signal("RREADY")
    
    def has_parameters(self) -> bool:
        """Check if this interface has any parameters."""
        has_dataflow_params = self.dataflow and len(self.dataflow.get_all_params()) > 0
        return bool(has_dataflow_params or
                   self.enable_param or self.data_width_param or 
                   self.addr_width_param or 
                   (self.dtype_params and self.dtype_params.has_any()))
    
    def get_all_params(self) -> List[Parameter]:
        """Get all parameters for this interface in consistent order."""
        params = []
        
        # Add dataflow parameters
        if self.dataflow:
            params.extend(self.dataflow.get_all_params())
        
        # Add control parameters
        if self.enable_param:
            params.append(self.enable_param)
        if self.data_width_param:
            params.append(self.data_width_param)
        if self.addr_width_param:
            params.append(self.addr_width_param)
        
        # Add non-None dtype params in consistent order
        if self.dtype_params:
            if self.dtype_params.width:
                params.append(self.dtype_params.width)
            if self.dtype_params.signed:
                params.append(self.dtype_params.signed)
            # Other dtype params less common for AXI-Lite
        
        return params
    
    def supports_dataflow(self) -> bool:
        """AXI-Lite interfaces support dataflow properties when used as weights."""
        return True


@dataclass
class ControlMetadata(InterfaceMetadata):
    """Metadata for a Control interface."""
    interface_type: InterfaceType = InterfaceType.CONTROL
    
    # Signal role properties
    @property
    def clk(self) -> Optional[Port]:
        """Get CLK signal port."""
        return self._get_signal("CLK")
    
    @property
    def rst_n(self) -> Optional[Port]:
        """Get RST_N signal port."""
        return self._get_signal("RST_N")
    
    @property
    def clk2x(self) -> Optional[Port]:
        """Get CLK2X signal port."""
        return self._get_signal("CLK2X")


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
    linked_parameters: List[Parameter] = field(default_factory=list)
    inputs: List[AXIStreamMetadata] = field(default_factory=list)
    outputs: List[AXIStreamMetadata] = field(default_factory=list)
    config: List[AXILiteMetadata] = field(default_factory=list)

    # Simple transformations as properties
    @property
    def class_name(self) -> str:
        """Get PascalCase class name from module name."""
        return pascal_case(self.name)
    
    @property
    def file_name(self) -> str:
        """Get snake_case file name from module name."""
        return snake_case(self.name)
    
    # Navigation helpers
    @property
    def stream_interfaces(self) -> List[AXIStreamMetadata]:
        """Get all AXI-Stream interfaces (inputs + outputs)."""
        return self.inputs + self.outputs
    
    # Convenience flags
    @property
    def has_weights(self) -> bool:
        """Check if any input interface is marked as a weight."""
        return any(i.is_weight for i in self.inputs)
    
    @property
    def has_bdim_params(self) -> bool:
        """Check if any stream interface has BDIM parameters."""
        return any(iface.bdim_shape for iface in self.stream_interfaces)
    
    @property
    def has_sdim_params(self) -> bool:
        """Check if any stream interface has SDIM parameters."""
        return any(iface.sdim_shape for iface in self.stream_interfaces)
    
    @property
    def has_interface_params(self) -> bool:
        """Check if any interface has parameters."""
        # Check stream interfaces
        for iface in self.stream_interfaces:
            if iface.has_parameters():
                return True
        # Check config interfaces
        for iface in self.config:
            if iface.has_parameters():
                return True
        return False
    
    @property
    def has_axilite_enable_params(self) -> bool:
        """Check if any config interface has enable parameter."""
        return any(iface.enable_param for iface in self.config)
    
    @property
    def interfaces(self) -> List[InterfaceMetadata]:
        """Get all interfaces in a single list."""
        interfaces = []
        interfaces.append(self.control)
        interfaces.extend(self.inputs)
        interfaces.extend(self.outputs) 
        interfaces.extend(self.config)
        return interfaces
    
    # Collection methods
    def get_all_bdim_params(self) -> List[str]:
        """Get all unique BDIM parameter names from all interfaces."""
        params = []
        seen = set()
        for iface in self.stream_interfaces:
            if iface.bdim_shape:
                for param in iface.bdim_shape:
                    if param not in seen:
                        params.append(param)
                        seen.add(param)
        return params
    
    def get_all_sdim_params(self) -> List[str]:
        """Get all unique SDIM parameter names from all interfaces."""
        params = []
        seen = set()
        for iface in self.stream_interfaces:
            if iface.sdim_shape:
                for param in iface.sdim_shape:
                    if param not in seen:
                        params.append(param)
                        seen.add(param)
        return params
    
    def get_nodeattr_types(self) -> Dict[str, Tuple[str, bool, Any]]:
        """Build complete nodeattr types dictionary for FINN.
        
        Returns:
            Dictionary mapping attribute names to (type, required, default) tuples.
        """
        attrs = {}
        
        # Interface datatype attributes
        for iface in self.inputs:
            attrs[f"{iface.compiler_name}DataType"] = ('s', True, "")
        for iface in self.config:
            attrs[f"{iface.compiler_name}DataType"] = ('s', True, "")
        for iface in self.outputs:
            attrs[f"{iface.compiler_name}DataType"] = ('s', True, "")
        
        # BDIM shape parameters
        for param_name in self.get_all_bdim_params():
            attrs[param_name] = ('i', True, 0)
        
        # SDIM shape parameters
        for param_name in self.get_all_sdim_params():
            attrs[param_name] = ('i', True, 0)
        
        # Runtime writeable weights if config interface exists
        if self.config:
            attrs["runtime_writeable_weights"] = ('b', False, True)
        
        return attrs
    

# Utility functions


def pascal_case(name: str) -> str:
    """
    Convert snake_case or kebab-case to PascalCase.
    
    Args:
        name: String to convert (e.g., "my_module_name" or "my-module-name")
        
    Returns:
        PascalCase string (e.g., "MyModuleName")
        
    Examples:
        >>> pascal_case("thresholding_axi")
        "ThresholdingAxi"
        >>> pascal_case("matrix-multiply")
        "MatrixMultiply"
        >>> pascal_case("my_custom_op")
        "MyCustomOp"
    """
    # Replace hyphens with underscores
    name = name.replace('-', '_')
    
    # Split on underscores and capitalize each part
    parts = name.split('_')
    return ''.join(word.capitalize() for word in parts if word)


def snake_case(name: str) -> str:
    """
    Convert PascalCase or kebab-case to snake_case.
    
    Args:
        name: String to convert (e.g., "MyModuleName" or "my-module-name")
        
    Returns:
        snake_case string (e.g., "my_module_name")
        
    Examples:
        >>> snake_case("ThresholdingAxi")
        "thresholding_axi"
        >>> snake_case("MatrixMultiply")
        "matrix_multiply"
    """
    # Replace hyphens with underscores
    name = name.replace('-', '_')
    
    # Insert underscores before capitals and convert to lowercase
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
