"""
Template context data structures and builders for the Hardware Kernel Generator.

This module provides centralized template context building to eliminate code
duplication across generators and ensure consistent data structures.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from abc import ABC, abstractmethod
import re

from .errors import CodeGenerationError, ValidationError


@dataclass
class BaseContext:
    """Base class for all template contexts."""
    
    # Metadata
    context_type: str = field(init=False)
    generated_timestamp: str = field(default="")
    generator_version: str = field(default="1.0.0")
    
    # Common data
    module_name: str = ""
    file_name: str = ""
    class_name: str = ""
    
    # Template control
    include_debug_info: bool = False
    include_documentation: bool = True
    include_type_hints: bool = True
    
    def __post_init__(self):
        """Initialize context-specific data after creation."""
        self.context_type = self.__class__.__name__
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for template rendering."""
        return asdict(self)
    
    def validate(self) -> None:
        """Validate context data."""
        if not self.module_name:
            raise ValidationError(
                "Module name is required for template context",
                validation_type="template_context",
                suggestion="Set module_name in context data"
            )
        
        if not self.file_name:
            raise ValidationError(
                "File name is required for template context",
                validation_type="template_context",
                suggestion="Set file_name in context data"
            )


@dataclass
class InterfaceInfo:
    """Information about a hardware interface."""
    
    name: str
    direction: str  # "input", "output", "inout"
    width: Optional[int] = None
    type: str = "wire"
    description: str = ""
    is_clock: bool = False
    is_reset: bool = False
    is_control: bool = False
    
    # AXI-specific fields
    is_axi: bool = False
    axi_type: Optional[str] = None  # "s_axis", "m_axis", "s_axi", "m_axi"
    axi_signals: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize derived fields."""
        # Detect special signal types
        name_lower = self.name.lower()
        
        if 'clk' in name_lower or 'clock' in name_lower:
            self.is_clock = True
        
        if 'rst' in name_lower or 'reset' in name_lower:
            self.is_reset = True
            
        if any(ctrl in name_lower for ctrl in ['ap_', 'ctrl_', 'control_']):
            self.is_control = True
        
        # Detect AXI interfaces
        if any(axi in name_lower for axi in ['axis', 'axi']):
            self.is_axi = True
            if 's_axis' in name_lower:
                self.axi_type = "s_axis"
            elif 'm_axis' in name_lower:
                self.axi_type = "m_axis"
            elif 's_axi' in name_lower:
                self.axi_type = "s_axi"
            elif 'm_axi' in name_lower:
                self.axi_type = "m_axi"


@dataclass
class ParameterInfo:
    """Information about a module parameter."""
    
    name: str
    value: Any
    type: str = "int"
    description: str = ""
    is_configurable: bool = True
    
    def __post_init__(self):
        """Initialize derived fields."""
        # Try to infer type from value if not specified
        if self.type == "int" and self.value is not None:
            if isinstance(self.value, bool):
                self.type = "bool"
            elif isinstance(self.value, str):
                self.type = "string"
            elif isinstance(self.value, float):
                self.type = "float"


@dataclass
class HWCustomOpContext(BaseContext):
    """Template context for HW Custom Op generation."""
    
    # RTL information
    rtl_file_path: str = ""
    top_module_name: str = ""
    
    # Interfaces
    interfaces: List[InterfaceInfo] = field(default_factory=list)
    input_interfaces: List[InterfaceInfo] = field(default_factory=list)
    output_interfaces: List[InterfaceInfo] = field(default_factory=list)
    
    # Parameters
    parameters: List[ParameterInfo] = field(default_factory=list)
    
    # FINN-specific
    finn_datatype: str = "float32"
    input_shape: List[int] = field(default_factory=list)
    output_shape: List[int] = field(default_factory=list)
    
    # Code generation options
    generate_wrapper: bool = True
    generate_testbench: bool = False
    
    # Tensor dimension information
    tdims: List[str] = field(default_factory=list)
    tensor_configs: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize derived fields and validate data."""
        super().__post_init__()
        
        # Derive class name from module name if not set
        if not self.class_name and self.top_module_name:
            # Convert snake_case to CamelCase
            self.class_name = ''.join(word.capitalize() for word in self.top_module_name.split('_'))
        
        # Separate interfaces by direction
        self.input_interfaces = [iface for iface in self.interfaces if iface.direction == "input"]
        self.output_interfaces = [iface for iface in self.interfaces if iface.direction == "output"]
        
        # Set default file name if not provided
        if not self.file_name and self.module_name:
            self.file_name = f"{self.module_name}.py"
    
    def get_axi_interfaces(self) -> Dict[str, List[InterfaceInfo]]:
        """Get AXI interfaces grouped by type."""
        axi_interfaces = {}
        for iface in self.interfaces:
            if iface.is_axi and iface.axi_type:
                if iface.axi_type not in axi_interfaces:
                    axi_interfaces[iface.axi_type] = []
                axi_interfaces[iface.axi_type].append(iface)
        return axi_interfaces
    
    def get_control_interfaces(self) -> List[InterfaceInfo]:
        """Get control interfaces (clock, reset, ap_*)."""
        return [iface for iface in self.interfaces if iface.is_control or iface.is_clock or iface.is_reset]
    
    def get_data_interfaces(self) -> List[InterfaceInfo]:
        """Get data interfaces (non-control)."""
        return [iface for iface in self.interfaces if not (iface.is_control or iface.is_clock or iface.is_reset)]
    
    def validate(self) -> None:
        """Validate HW Custom Op context."""
        super().validate()
        
        if not self.top_module_name:
            raise ValidationError(
                "Top module name is required for HW Custom Op",
                validation_type="hw_custom_op_context",
                suggestion="Set top_module_name from RTL analysis"
            )
        
        if not self.interfaces:
            raise ValidationError(
                "At least one interface is required",
                validation_type="hw_custom_op_context",
                suggestion="Ensure RTL has detectable interfaces"
            )
        
        # Validate required control signals
        has_clock = any(iface.is_clock for iface in self.interfaces)
        has_reset = any(iface.is_reset for iface in self.interfaces)
        
        if not has_clock:
            raise ValidationError(
                "Clock signal not found in interfaces",
                validation_type="hw_custom_op_context",
                suggestion="Ensure RTL has ap_clk or similar clock signal"
            )
        
        if not has_reset:
            raise ValidationError(
                "Reset signal not found in interfaces",
                validation_type="hw_custom_op_context",
                suggestion="Ensure RTL has ap_rst_n or similar reset signal"
            )


@dataclass
class RTLBackendContext(BaseContext):
    """Template context for RTL backend generation."""
    
    # RTL files
    rtl_files: List[str] = field(default_factory=list)
    main_rtl_file: str = ""
    
    # Build configuration
    synthesis_tool: str = "vivado"
    target_device: str = ""
    clock_frequency: float = 100.0  # MHz
    
    # RTL analysis results
    modules: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # Backend-specific
    backend_type: str = "hls"
    optimization_level: str = "2"
    
    def __post_init__(self):
        """Initialize derived fields."""
        super().__post_init__()
        
        # Set default file name
        if not self.file_name and self.module_name:
            self.file_name = f"{self.module_name}_backend.py"
    
    def validate(self) -> None:
        """Validate RTL backend context."""
        super().validate()
        
        if not self.rtl_files:
            raise ValidationError(
                "At least one RTL file is required",
                validation_type="rtl_backend_context",
                suggestion="Provide RTL files for backend generation"
            )


class TemplateContextBuilder:
    """Builder for creating template contexts from analysis data."""
    
    def __init__(self, config=None):
        """Initialize context builder with configuration."""
        self.config = config
        self._context_cache = {}
    
    def build_hw_custom_op_context(self, analysis_data: Dict[str, Any], **kwargs) -> HWCustomOpContext:
        """Build HW Custom Op context from analysis data."""
        # Create cache key
        cache_key = f"hw_custom_op_{hash(str(analysis_data))}"
        
        if cache_key in self._context_cache:
            return self._context_cache[cache_key]
        
        try:
            # Extract basic information
            context = HWCustomOpContext(
                module_name=analysis_data.get('module_name', ''),
                top_module_name=analysis_data.get('top_module', ''),
                rtl_file_path=str(analysis_data.get('rtl_file', '')),
                **kwargs
            )
            
            # Build interfaces
            context.interfaces = self._build_interfaces(analysis_data.get('interfaces', []))
            
            # Build parameters
            context.parameters = self._build_parameters(analysis_data.get('parameters', {}))
            
            # Extract FINN-specific data
            if 'finn_config' in analysis_data:
                finn_config = analysis_data['finn_config']
                context.finn_datatype = finn_config.get('datatype', 'float32')
                context.input_shape = finn_config.get('input_shape', [])
                context.output_shape = finn_config.get('output_shape', [])
            
            # Extract tensor dimension data
            if 'tdims' in analysis_data:
                context.tdims = analysis_data['tdims']
            
            if 'tensor_configs' in analysis_data:
                context.tensor_configs = analysis_data['tensor_configs']
            
            # Apply configuration overrides
            if self.config:
                context.include_debug_info = self.config.generation.include_debug_info
                context.include_documentation = self.config.generation.include_documentation
                context.include_type_hints = self.config.generation.include_type_hints
            
            # Validate and cache
            context.validate()
            self._context_cache[cache_key] = context
            
            return context
            
        except Exception as e:
            raise CodeGenerationError(
                f"Failed to build HW Custom Op context: {e}",
                generator_type="hw_custom_op",
                suggestion="Check analysis data format and completeness"
            )
    
    def build_rtl_backend_context(self, analysis_data: Dict[str, Any], **kwargs) -> RTLBackendContext:
        """Build RTL backend context from analysis data."""
        # Create cache key
        cache_key = f"rtl_backend_{hash(str(analysis_data))}"
        
        if cache_key in self._context_cache:
            return self._context_cache[cache_key]
        
        try:
            context = RTLBackendContext(
                module_name=analysis_data.get('module_name', ''),
                rtl_files=analysis_data.get('rtl_files', []),
                main_rtl_file=analysis_data.get('main_rtl_file', ''),
                **kwargs
            )
            
            # Extract backend configuration
            if 'backend_config' in analysis_data:
                backend_config = analysis_data['backend_config']
                context.synthesis_tool = backend_config.get('synthesis_tool', 'vivado')
                context.target_device = backend_config.get('target_device', '')
                context.clock_frequency = backend_config.get('clock_frequency', 100.0)
                context.backend_type = backend_config.get('backend_type', 'hls')
                context.optimization_level = backend_config.get('optimization_level', '2')
            
            # Extract module information
            context.modules = analysis_data.get('modules', [])
            context.dependencies = analysis_data.get('dependencies', [])
            
            # Apply configuration overrides
            if self.config:
                context.include_debug_info = self.config.generation.include_debug_info
                context.include_documentation = self.config.generation.include_documentation
                context.include_type_hints = self.config.generation.include_type_hints
            
            # Validate and cache
            context.validate()
            self._context_cache[cache_key] = context
            
            return context
            
        except Exception as e:
            raise CodeGenerationError(
                f"Failed to build RTL backend context: {e}",
                generator_type="rtl_backend",
                suggestion="Check analysis data format and completeness"
            )
    
    def _build_interfaces(self, interface_data: List[Dict[str, Any]]) -> List[InterfaceInfo]:
        """Build interface info objects from analysis data."""
        interfaces = []
        
        for iface_data in interface_data:
            interface = InterfaceInfo(
                name=iface_data.get('name', ''),
                direction=iface_data.get('direction', ''),
                width=iface_data.get('width'),
                type=iface_data.get('type', 'wire'),
                description=iface_data.get('description', '')
            )
            interfaces.append(interface)
        
        return interfaces
    
    def _build_parameters(self, parameter_data: Dict[str, Any]) -> List[ParameterInfo]:
        """Build parameter info objects from analysis data."""
        parameters = []
        
        for name, value in parameter_data.items():
            if isinstance(value, dict):
                # Extended parameter format
                parameter = ParameterInfo(
                    name=name,
                    value=value.get('value'),
                    type=value.get('type', 'int'),
                    description=value.get('description', ''),
                    is_configurable=value.get('configurable', True)
                )
            else:
                # Simple parameter format
                parameter = ParameterInfo(name=name, value=value)
            
            parameters.append(parameter)
        
        return parameters
    
    def clear_cache(self) -> None:
        """Clear the context cache."""
        self._context_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_size': len(self._context_cache),
            'cached_contexts': list(self._context_cache.keys())
        }


def create_context_builder(config=None) -> TemplateContextBuilder:
    """Factory function to create a template context builder."""
    return TemplateContextBuilder(config)