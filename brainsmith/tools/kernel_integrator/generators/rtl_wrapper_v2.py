"""RTL Wrapper generator with direct KernelMetadata support."""
from typing import Dict, Any, List, Set

from .base_v2 import GeneratorBase
from ..types.metadata import KernelMetadata
from ..types.rtl import Parameter
from brainsmith.core.dataflow.types import InterfaceType


class RTLWrapperGeneratorV2(GeneratorBase):
    """Generates RTL wrapper Verilog files."""
    
    @property
    def name(self) -> str:
        return "rtl_wrapper"
    
    @property
    def template_file(self) -> str:
        return "rtl_wrapper_minimal_v2.v.j2"
    
    @property
    def output_pattern(self) -> str:
        return "{kernel_name}_wrapper.v"
    
    def _get_specific_vars(self, metadata: KernelMetadata) -> Dict[str, Any]:
        """Get RTL Wrapper-specific template variables."""
        return {
            'categorized_parameters': self._categorize_parameters(metadata),
            'generation_timestamp': self._get_timestamp(),
        }
    
    def _categorize_parameters(self, metadata: KernelMetadata) -> Dict[str, Any]:
        """Categorize parameters for organized RTL wrapper generation."""
        parameter_definitions = metadata.parameters
        
        # Collect internal datatype parameters
        internal_datatype_params = set()
        internal_datatype_groups = {}
        for dt_meta in metadata.internal_datatypes:
            dt_params = dt_meta.get_all_parameters()
            internal_datatype_params.update(dt_params)
            if dt_params:
                internal_datatype_groups[dt_meta.name] = [
                    param for param in parameter_definitions 
                    if param.name in dt_params
                ]
        
        # Collect interface-specific parameters grouped by interface
        interface_parameter_groups = {}
        for interface in metadata.interfaces:
            interface_params = set()
            
            # Collect datatype parameters
            if hasattr(interface, 'datatype_metadata') and interface.datatype_metadata:
                dt_meta = interface.datatype_metadata
                interface_params.update(dt_meta.get_all_parameters())
            
            # Collect shape parameters (BDIM/SDIM)
            if hasattr(interface, 'bdim_params') and interface.bdim_params:
                for param in interface.bdim_params:
                    if param != '1':  # Skip singleton dimensions
                        interface_params.add(param)
            if hasattr(interface, 'sdim_params') and interface.sdim_params:
                for param in interface.sdim_params:
                    if param != '1':  # Skip singleton dimensions
                        interface_params.add(param)
            
            if interface_params:
                # Sort parameters: BDIM first, SDIM second, then datatype params
                sorted_params = []
                
                # Add BDIM parameters first
                if hasattr(interface, 'bdim_params') and interface.bdim_params:
                    for bdim_param in interface.bdim_params:
                        if bdim_param != '1':
                            bdim_params = [p for p in parameter_definitions if p.name == bdim_param]
                            sorted_params.extend(bdim_params)
                
                # Add SDIM parameters second
                if hasattr(interface, 'sdim_params') and interface.sdim_params:
                    for sdim_param in interface.sdim_params:
                        if sdim_param != '1':
                            sdim_params = [p for p in parameter_definitions if p.name == sdim_param]
                            sorted_params.extend(sdim_params)
                
                # Add datatype parameters
                if hasattr(interface, 'datatype_metadata') and interface.datatype_metadata:
                    dt_meta = interface.datatype_metadata
                    # Exclude shape params already added
                    shape_params = set()
                    if hasattr(interface, 'bdim_params') and interface.bdim_params:
                        shape_params.update(p for p in interface.bdim_params if p != '1')
                    if hasattr(interface, 'sdim_params') and interface.sdim_params:
                        shape_params.update(p for p in interface.sdim_params if p != '1')
                    
                    for param_name in dt_meta.get_all_parameters():
                        if param_name not in shape_params:
                            dt_params = [p for p in parameter_definitions if p.name == param_name]
                            sorted_params.extend(dt_params)
                
                interface_parameter_groups[interface.name] = {
                    'interface': interface,
                    'parameters': sorted_params
                }
        
        # Identify AXI-Lite configuration parameters
        axilite_params = []
        axilite_param_names = set(metadata.linked_parameters.get("axilite", {}).keys())
        
        # Collect all interface-linked parameters
        interface_linked_params = set()
        for group_data in interface_parameter_groups.values():
            interface_linked_params.update(p.name for p in group_data['parameters'])
        
        for param in parameter_definitions:
            if param.name in axilite_param_names:
                if param.name not in interface_linked_params and param.name not in internal_datatype_params:
                    axilite_params.append(param)
        
        # General algorithm parameters (everything else)
        excluded_params = internal_datatype_params | interface_linked_params | {p.name for p in axilite_params}
        general_params = [
            param for param in parameter_definitions 
            if param.name not in excluded_params
        ]
        
        # Organize interface parameters by interface type
        organized_interface_params = []
        
        # Input interfaces first
        for interface in metadata.interfaces:
            if (interface.interface_type == InterfaceType.INPUT and 
                interface.name in interface_parameter_groups):
                organized_interface_params.append(interface_parameter_groups[interface.name])
        
        # Output interfaces second  
        for interface in metadata.interfaces:
            if (interface.interface_type == InterfaceType.OUTPUT and 
                interface.name in interface_parameter_groups):
                organized_interface_params.append(interface_parameter_groups[interface.name])
        
        # Weight interfaces third
        for interface in metadata.interfaces:
            if (interface.interface_type == InterfaceType.WEIGHT and 
                interface.name in interface_parameter_groups):
                organized_interface_params.append(interface_parameter_groups[interface.name])
        
        # Convert parameters to dicts with template_param_name
        def add_template_name(param):
            return {
                'name': param.name,
                'template_param_name': param.template_param_name,  # Use the parameter's own template_param_name
                'default_value': param.default_value,
                'rtl_type': param.rtl_type
            }
        
        # Convert all parameter lists
        general_params_with_template = [add_template_name(p) for p in general_params]
        axilite_params_with_template = [add_template_name(p) for p in axilite_params]
        
        # Convert internal datatype groups
        internal_groups_with_template = {}
        for dt_name, params in internal_datatype_groups.items():
            internal_groups_with_template[dt_name] = [add_template_name(p) for p in params]
        
        # Convert interface parameter groups
        interface_groups_with_template = []
        for group in organized_interface_params:
            interface_groups_with_template.append({
                'interface': group['interface'],
                'parameters': [add_template_name(p) for p in group['parameters']]
            })
        
        return {
            'general_parameters': general_params_with_template,
            'axilite_parameters': axilite_params_with_template,
            'internal_datatype_groups': internal_groups_with_template,
            'interface_parameter_groups': interface_groups_with_template
        }
    
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for generation tracking."""
        from datetime import datetime
        return datetime.now().isoformat()
