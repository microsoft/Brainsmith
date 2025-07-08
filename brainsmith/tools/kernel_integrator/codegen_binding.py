############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""
Unified RTL parameter binding system for code generation.

This module provides a comprehensive system for describing how RTL parameters
are bound to various sources (node attributes, interface properties, etc.)
during code generation for AutoHWCustomOp and RTL backends.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any


class SourceType(Enum):
    """How a parameter gets its value during code generation."""
    NODEATTR = "nodeattr"           # Direct from node attribute
    NODEATTR_ALIAS = "alias"        # From aliased node attribute
    DERIVED = "derived"             # Computed from expression
    INTERFACE_DATATYPE = "if_dtype" # From interface datatype property
    INTERFACE_SHAPE = "if_shape"    # From interface shape (BDIM/SDIM)
    INTERNAL_DATATYPE = "int_dtype" # From internal datatype
    CONSTANT = "constant"           # Fixed constant value


class ParameterCategory(Enum):
    """What kind of parameter this is."""
    ALGORITHM = "algorithm"         # Core algorithm parameter
    DATATYPE = "datatype"          # Datatype-related (width, signed)
    SHAPE = "shape"                # Shape-related (BDIM, SDIM)
    CONTROL = "control"            # Control/config (AXI-Lite)
    INTERNAL = "internal"          # Internal mechanism


@dataclass
class ParameterSource:
    """Describes how a parameter gets its value."""
    
    type: SourceType
    """Type of source (nodeattr, derived, interface, etc.)"""
    
    # Source-specific fields
    nodeattr_name: Optional[str] = None
    """For NODEATTR/NODEATTR_ALIAS: name of the node attribute"""
    
    expression: Optional[str] = None
    """For DERIVED: Python expression to compute value"""
    
    interface_name: Optional[str] = None
    """For INTERFACE_*: which interface this relates to"""
    
    property_name: Optional[str] = None
    """For INTERFACE_*/INTERNAL_*: which property (width, signed, etc.)"""
    
    dimension_index: Optional[int] = None
    """For INTERFACE_SHAPE: which dimension index (for BDIM/SDIM)"""
    
    constant_value: Optional[Any] = None
    """For CONSTANT: the fixed value"""
    
    def __post_init__(self):
        """Validate source configuration."""
        if self.type == SourceType.NODEATTR and not self.nodeattr_name:
            # For direct nodeattr, name can be inferred from parameter name
            pass
        elif self.type == SourceType.NODEATTR_ALIAS and not self.nodeattr_name:
            raise ValueError("NODEATTR_ALIAS source requires nodeattr_name")
        elif self.type == SourceType.DERIVED and not self.expression:
            raise ValueError("DERIVED source requires expression")
        elif self.type in [SourceType.INTERFACE_DATATYPE, SourceType.INTERFACE_SHAPE]:
            if not self.interface_name:
                raise ValueError(f"{self.type.value} source requires interface_name")
        elif self.type == SourceType.INTERNAL_DATATYPE:
            if not self.interface_name:  # Using interface_name field for internal name
                raise ValueError("INTERNAL_DATATYPE source requires name")
        elif self.type == SourceType.CONSTANT and self.constant_value is None:
            raise ValueError("CONSTANT source requires constant_value")


@dataclass
class ParameterBinding:
    """Complete binding information for a single RTL parameter."""
    
    name: str
    """RTL parameter name"""
    
    source: ParameterSource
    """How this parameter gets its value"""
    
    category: ParameterCategory
    """What kind of parameter this is"""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional binding metadata (e.g., default values, constraints)"""
    
    def is_exposed(self) -> bool:
        """Check if this parameter should be exposed as a node attribute."""
        return self.source.type in [SourceType.NODEATTR, SourceType.NODEATTR_ALIAS]


@dataclass
class InterfaceBinding:
    """Parameter bindings for a specific interface."""
    
    interface_name: str
    """Name of the interface (compiler name)"""
    
    # Datatype parameters
    datatype_params: Dict[str, str] = field(default_factory=dict)
    """Maps property name to RTL parameter (e.g., 'width' -> 'INPUT_WIDTH')"""
    
    # Shape parameters
    bdim_params: List[str] = field(default_factory=list)
    """RTL parameters for block dimensions"""
    
    sdim_params: List[str] = field(default_factory=list)
    """RTL parameters for stream dimensions"""
    
    def get_all_parameters(self) -> List[str]:
        """Get all RTL parameters for this interface."""
        params = list(self.datatype_params.values())
        params.extend(p for p in self.bdim_params if p != '1')
        params.extend(p for p in self.sdim_params if p != '1')
        return params
    
    def get_datatype_parameter(self, property_name: str) -> Optional[str]:
        """Get RTL parameter name for a datatype property."""
        return self.datatype_params.get(property_name)
    
    def get_shape_parameters(self, shape_type: str) -> List[str]:
        """Get shape parameters by type ('bdim' or 'sdim')."""
        if shape_type.lower() == 'bdim':
            return [p for p in self.bdim_params if p != '1']
        elif shape_type.lower() == 'sdim':
            return [p for p in self.sdim_params if p != '1']
        return []


@dataclass
class InternalBinding:
    """Parameter bindings for internal mechanisms."""
    
    name: str
    """Name of internal mechanism (e.g., 'accumulator', 'threshold')"""
    
    datatype_params: Dict[str, str] = field(default_factory=dict)
    """Maps property name to RTL parameter"""
    
    def get_all_parameters(self) -> List[str]:
        """Get all RTL parameters for this internal mechanism."""
        return list(self.datatype_params.values())


@dataclass
class CodegenBinding:
    """Unified RTL parameter binding information for code generation.
    
    This single object captures all parameter linkage information needed
    to generate AutoHWCustomOp subclasses and RTL backends.
    """
    
    # Core parameter sets
    exposed_parameters: List[str] = field(default_factory=list)
    """RTL parameters exposed as FINN node attributes"""
    
    hidden_parameters: List[str] = field(default_factory=list)
    """RTL parameters handled internally (not exposed as nodeattr)"""
    
    # Parameter mappings (param_name -> binding info)
    parameter_bindings: Dict[str, ParameterBinding] = field(default_factory=dict)
    """Complete binding information for each RTL parameter"""
    
    # Interface-specific bindings (interface_name -> interface bindings)
    interface_bindings: Dict[str, InterfaceBinding] = field(default_factory=dict)
    """Parameter bindings specific to each interface"""
    
    # Internal mechanism bindings (name -> internal bindings)
    internal_bindings: Dict[str, InternalBinding] = field(default_factory=dict)
    """Parameter bindings for internal mechanisms (accumulator, threshold, etc.)"""
    
    def add_parameter_binding(self, param_name: str, source: ParameterSource, 
                            category: ParameterCategory, **metadata) -> None:
        """Add a parameter binding."""
        binding = ParameterBinding(
            name=param_name,
            source=source,
            category=category,
            metadata=metadata
        )
        self.parameter_bindings[param_name] = binding
        
        # Update exposed/hidden lists
        if binding.is_exposed():
            if param_name not in self.exposed_parameters:
                self.exposed_parameters.append(param_name)
        else:
            if param_name not in self.hidden_parameters:
                self.hidden_parameters.append(param_name)
    
    def add_interface_binding(self, interface_name: str, 
                            datatype_params: Optional[Dict[str, str]] = None,
                            bdim_params: Optional[List[str]] = None,
                            sdim_params: Optional[List[str]] = None) -> None:
        """Add or update interface binding."""
        if interface_name not in self.interface_bindings:
            self.interface_bindings[interface_name] = InterfaceBinding(interface_name)
        
        binding = self.interface_bindings[interface_name]
        
        if datatype_params:
            binding.datatype_params.update(datatype_params)
        if bdim_params is not None:
            binding.bdim_params = bdim_params
        if sdim_params is not None:
            binding.sdim_params = sdim_params
    
    def add_internal_binding(self, name: str, 
                           datatype_params: Optional[Dict[str, str]] = None) -> None:
        """Add or update internal mechanism binding."""
        if name not in self.internal_bindings:
            self.internal_bindings[name] = InternalBinding(name)
        
        if datatype_params:
            self.internal_bindings[name].datatype_params.update(datatype_params)
    
    def get_parameter_source(self, param_name: str) -> Optional[ParameterSource]:
        """Get the source/computation method for a parameter."""
        if param_name in self.parameter_bindings:
            return self.parameter_bindings[param_name].source
        return None
    
    def get_interface_parameters(self, interface_name: str) -> List[str]:
        """Get all parameters linked to a specific interface."""
        if interface_name in self.interface_bindings:
            return self.interface_bindings[interface_name].get_all_parameters()
        return []
    
    def get_nodeattr_parameters(self) -> Dict[str, str]:
        """Get all parameters that should be node attributes.
        
        Returns:
            Dict mapping RTL parameter name to node attribute name
        """
        result = {}
        for param_name, binding in self.parameter_bindings.items():
            if binding.source.type == SourceType.NODEATTR:
                # Direct mapping
                result[param_name] = param_name
            elif binding.source.type == SourceType.NODEATTR_ALIAS:
                # Aliased mapping
                result[param_name] = binding.source.nodeattr_name
        return result
    
    def get_derived_parameters(self) -> Dict[str, str]:
        """Get all derived parameters with their expressions.
        
        Returns:
            Dict mapping parameter name to Python expression
        """
        result = {}
        for param_name, binding in self.parameter_bindings.items():
            if binding.source.type == SourceType.DERIVED:
                result[param_name] = binding.source.expression
        return result
    
    def validate(self) -> List[str]:
        """Validate the codegen binding configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check that all exposed parameters have proper bindings
        for param in self.exposed_parameters:
            if param not in self.parameter_bindings:
                errors.append(f"Exposed parameter '{param}' has no binding information")
        
        # Check interface bindings reference valid parameters
        for interface_name, interface_binding in self.interface_bindings.items():
            for param in interface_binding.get_all_parameters():
                if param not in self.parameter_bindings:
                    errors.append(
                        f"Interface '{interface_name}' references parameter '{param}' "
                        f"which has no binding information"
                    )
        
        # Check internal bindings reference valid parameters
        for internal_name, internal_binding in self.internal_bindings.items():
            for param in internal_binding.get_all_parameters():
                if param not in self.parameter_bindings:
                    errors.append(
                        f"Internal mechanism '{internal_name}' references parameter '{param}' "
                        f"which has no binding information"
                    )
        
        return errors