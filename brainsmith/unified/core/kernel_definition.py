############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""
Kernel Definition System

Provides the core data structures for defining hardware kernels in the
unified framework.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set
from enum import Enum

from brainsmith.dataflow.core.interface_types import InterfaceType
from brainsmith.core.dataflow import Shape


class ProtocolType(Enum):
    """Supported interface protocols."""
    AXI_STREAM = "axi_stream"
    AXI_LITE = "axi_lite"
    AXI_MM = "axi_mm"
    BRAM = "bram"
    CUSTOM = "custom"


@dataclass
class DatatypeConstraint:
    """Datatype constraint specification."""
    base_type: str      # "INT", "UINT", "FIXED", "FLOAT", "BIPOLAR", "TERNARY"
    min_width: int      # Minimum bit width (inclusive)
    max_width: int      # Maximum bit width (inclusive)
    
    def __post_init__(self):
        """Validate constraint parameters."""
        if self.min_width <= 0:
            raise ValueError(f"min_width must be positive, got {self.min_width}")
        if self.max_width < self.min_width:
            raise ValueError(f"max_width ({self.max_width}) must be >= min_width ({self.min_width})")
        
        valid_base_types = ["INT", "UINT", "FIXED", "FLOAT", "BIPOLAR", "TERNARY", "BINARY"]
        if self.base_type not in valid_base_types:
            raise ValueError(f"Invalid base_type '{self.base_type}'. Must be one of {valid_base_types}")
    
    def to_qonnx_constraint(self):
        """Convert to QONNX constraint format for compatibility."""
        from brainsmith.dataflow.core.qonnx_types import DatatypeConstraintGroup
        return DatatypeConstraintGroup(self.base_type, self.min_width, self.max_width)


@dataclass
class InterfaceDefinition:
    """
    Complete definition of a hardware interface.
    This captures all static information about an interface.
    """
    # Basic identification (required fields first)
    name: str
    type: InterfaceType
    
    # Protocol information (optional fields with defaults)
    protocol: ProtocolType = ProtocolType.AXI_STREAM
    protocol_config: Dict[str, Any] = field(default_factory=dict)
    
    # Datatype constraints
    datatype_constraints: List[DatatypeConstraint] = field(default_factory=list)
    
    # Parameter linkage
    parameter_links: Dict[str, str] = field(default_factory=dict)
    # Common links: {'BDIM': 'param_name', 'SDIM': 'param_name', 'datatype': 'param_name'}
    
    # Metadata for code generation and analysis
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def supports_datatype(self, datatype_str: str) -> bool:
        """Check if interface supports given datatype."""
        if not self.datatype_constraints:
            return True  # No constraints = support all
        
        # TODO: Implement datatype checking logic
        return True


@dataclass
class PerformanceModel:
    """
    Performance model for a kernel.
    Captures how performance scales with configuration.
    """
    # Base latency (cycles) for minimal configuration
    base_latency: int
    
    # Throughput scaling factors
    # Maps interface name to scaling factor
    throughput_scaling: Dict[str, float] = field(default_factory=dict)
    
    # Pipeline depth as function of parameters
    pipeline_depth_formula: Optional[str] = None
    
    # Additional performance metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceModel:
    """
    Resource utilization model for a kernel.
    Captures how resources scale with configuration.
    """
    # Base resource usage (minimal configuration)
    base_luts: int = 0
    base_ffs: int = 0
    base_brams: int = 0
    base_dsps: int = 0
    base_urams: int = 0
    
    # Scaling factors for parallelism
    # Maps interface name to resource scaling
    lut_scaling: Dict[str, float] = field(default_factory=dict)
    ff_scaling: Dict[str, float] = field(default_factory=dict)
    bram_scaling: Dict[str, float] = field(default_factory=dict)
    dsp_scaling: Dict[str, float] = field(default_factory=dict)
    
    # Additional resource metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KernelDefinition:
    """
    Complete definition of a hardware kernel.
    This is the single source of truth for kernel behavior.
    """
    # Required fields first
    name: str
    interfaces: List[InterfaceDefinition]
    
    # Optional fields with defaults
    version: str = "1.0"
    description: str = ""
    
    # Behavioral constraints (from pragmas or manual)
    constraints: List[Any] = field(default_factory=list)
    
    # RTL metadata
    rtl_parameters: Dict[str, Any] = field(default_factory=dict)
    exposed_parameters: Set[str] = field(default_factory=set)
    
    # Parameter defaults and ranges
    parameter_defaults: Dict[str, Any] = field(default_factory=dict)
    parameter_ranges: Dict[str, tuple] = field(default_factory=dict)
    
    # Performance and resource models
    performance_model: Optional[PerformanceModel] = None
    resource_model: Optional[ResourceModel] = None
    
    # Code generation metadata
    template_file: Optional[str] = None
    support_files: List[str] = field(default_factory=list)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_interface(self, name: str) -> Optional[InterfaceDefinition]:
        """Get interface by name."""
        for intf in self.interfaces:
            if intf.name == name:
                return intf
        return None
    
    def get_interfaces_by_type(self, intf_type: InterfaceType) -> List[InterfaceDefinition]:
        """Get all interfaces of given type."""
        return [intf for intf in self.interfaces if intf.type == intf_type]
    
    def validate(self) -> List[str]:
        """Validate kernel definition consistency."""
        errors = []
        
        # Check interface names are unique
        names = [intf.name for intf in self.interfaces]
        if len(names) != len(set(names)):
            errors.append("Duplicate interface names found")
        
        # Check exposed parameters exist in rtl_parameters
        for param in self.exposed_parameters:
            if param not in self.rtl_parameters:
                errors.append(f"Exposed parameter '{param}' not found in rtl_parameters")
        
        # Check parameter defaults match exposed parameters
        for param in self.parameter_defaults:
            if param not in self.exposed_parameters:
                errors.append(f"Default provided for non-exposed parameter '{param}'")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'interfaces': [
                {
                    'name': intf.name,
                    'type': intf.type.value,
                    'protocol': intf.protocol.value,
                    'protocol_config': intf.protocol_config,
                    'datatype_constraints': [
                        {
                            'base_type': c.base_type,
                            'min_width': c.min_width,
                            'max_width': c.max_width
                        } for c in intf.datatype_constraints
                    ],
                    'parameter_links': intf.parameter_links,
                    'metadata': intf.metadata
                } for intf in self.interfaces
            ],
            'rtl_parameters': self.rtl_parameters,
            'exposed_parameters': list(self.exposed_parameters),
            'parameter_defaults': self.parameter_defaults,
            'parameter_ranges': self.parameter_ranges,
            'metadata': self.metadata
        }