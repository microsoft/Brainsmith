"""
Template context generator for KernelMetadata.

Generates complete template context for Jinja2 templates from KernelMetadata.
Updated to work with the new unified KernelMetadata structure.
"""

from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path

from brainsmith.core.dataflow.types import InterfaceType
from brainsmith.tools.kernel_integrator.types.metadata import KernelMetadata, InterfaceMetadata
from brainsmith.tools.kernel_integrator.types.rtl import Parameter
from .template_context import TemplateContext


class TemplateContextGenerator:
    """Generates template context from KernelMetadata for Jinja2 templates."""
    
    
    @staticmethod
    def generate_template_context(kernel_metadata: KernelMetadata) -> TemplateContext:
        """Generate enhanced TemplateContext for Phase 2 code generation."""
        generator = TemplateContextGenerator()
        
        # Extract datatype parameter mappings
        datatype_mappings = generator._extract_datatype_parameter_mappings(kernel_metadata)
        
        # Create the enhanced TemplateContext
        template_context = TemplateContext(
            kernel_metadata=kernel_metadata,  # Pass the entire KernelMetadata
            # Active transformations still used by templates
            datatype_linked_params=datatype_mappings.get('datatype_linked_params', []),
            categorized_parameters=generator._categorize_parameters(kernel_metadata, kernel_metadata.parameters, datatype_mappings),
            shape_nodeattrs=generator._extract_shape_nodeattrs(kernel_metadata)
        )
        
        return template_context
    
    def _get_class_name(self, module_name: str) -> str:
        """Generate Python class name from module name."""
        from ..utils import pascal_case
        return pascal_case(module_name)
    
    def _get_dataflow_interfaces(self, kernel_metadata: KernelMetadata) -> List:
        """Get all dataflow interfaces (INPUT, OUTPUT, WEIGHT)."""
        return kernel_metadata.input_interfaces + kernel_metadata.output_interfaces + kernel_metadata.weight_interfaces
    
    
    
    def _requires_resource_estimation(self, kernel_metadata: KernelMetadata) -> bool:
        """Check if resource estimation is needed."""
        # Enable for complex kernels or when multiple interfaces exist
        return len(kernel_metadata.interfaces) > 2
    
    def _requires_verification(self, kernel_metadata: KernelMetadata) -> bool:
        """Check if verification is needed."""
        # Enable for kernels with weight interfaces
        return kernel_metadata.has_weights
    
    
    
    
    @staticmethod
    def _extract_datatype_parameter_mappings(kernel_metadata: KernelMetadata) -> Dict[str, Any]:
        """
        Extract datatype-linked parameter names for categorization.
        
        Returns:
            Dictionary with:
            - datatype_linked_params: List of parameter names that are datatype-linked
        """
        datatype_linked_params = set()
        
        # Extract from interface metadata (populated by pragma processing)
        for interface in kernel_metadata.interfaces:
            if hasattr(interface, 'datatype_metadata') and interface.datatype_metadata:
                dt_meta = interface.datatype_metadata
                # Collect all datatype-linked parameter names
                param_names = dt_meta.get_all_parameters()
                datatype_linked_params.update(param_names)
        
        return {
            'datatype_linked_params': list(datatype_linked_params)
        }
    
    @staticmethod
    def _extract_shape_nodeattrs(kernel_metadata: KernelMetadata) -> List[Dict[str, str]]:
        """
        Extract unique parameter names from BDIM/SDIM SHAPE expressions for HWCustomOp node attributes.
        
        This collects all parameter names found in bdim_shape and sdim_shape fields across
        all interfaces and creates node attribute definitions for the HWCustomOp template.
        
        Returns:
            List of dictionaries with 'name' and 'source_comment' for each unique parameter
        """
        shape_params = {}  # Dict to collect unique params with their sources
        
        # Scan all interfaces for SHAPE parameters
        for interface in kernel_metadata.interfaces:
            interface_name = interface.name
            
            # Extract from BDIM shape expressions
            if hasattr(interface, 'bdim_shape') and interface.bdim_shape:
                for element in interface.bdim_shape:
                    if isinstance(element, str) and element != ":" and element.isidentifier():
                        # This is a parameter name (not singleton 1 or full slice ":")
                        if element not in shape_params:
                            shape_params[element] = []
                        shape_params[element].append(f"BDIM: {interface_name}")
            
            # Extract from SDIM shape expressions  
            if hasattr(interface, 'sdim_shape') and interface.sdim_shape:
                for element in interface.sdim_shape:
                    if isinstance(element, str) and element != ":" and element.isidentifier():
                        # This is a parameter name (not singleton 1 or full slice ":")
                        if element not in shape_params:
                            shape_params[element] = []
                        shape_params[element].append(f"SDIM: {interface_name}")
        
        # Convert to list format for template
        nodeattrs = []
        for param_name, sources in shape_params.items():
            nodeattrs.append({
                'name': param_name,
                'source_comment': ', '.join(sources)
            })
        
        # Sort by parameter name for consistent output
        nodeattrs.sort(key=lambda x: x['name'])
        
        return nodeattrs
    
    def _categorize_parameters(self, kernel_metadata: KernelMetadata, 
                             parameter_definitions: List[Parameter],
                             datatype_mappings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Categorize parameters for organized RTL wrapper generation.
        
        Args:
            kernel_metadata: Kernel metadata with interfaces and datatypes
            parameter_definitions: All parameter definitions
            datatype_mappings: Datatype parameter mapping information
            
        Returns:
            Dictionary with categorized parameter groups
        """
        # Get sets of parameters that are linked to specific purposes
        datatype_linked_params = set(datatype_mappings.get('datatype_linked_params', []))
        
        # Collect internal datatype parameters
        internal_datatype_params = set()
        internal_datatype_groups = {}
        for dt_meta in kernel_metadata.internal_datatypes:
            dt_params = dt_meta.get_all_parameters()
            internal_datatype_params.update(dt_params)
            if dt_params:
                internal_datatype_groups[dt_meta.name] = [
                    param for param in parameter_definitions 
                    if param.name in dt_params
                ]
        
        # Collect interface-specific parameters grouped by interface
        interface_parameter_groups = {}
        for interface in kernel_metadata.interfaces:
            interface_params = set()
            
            # Collect datatype parameters
            if hasattr(interface, 'datatype_metadata') and interface.datatype_metadata:
                dt_meta = interface.datatype_metadata
                interface_params.update(dt_meta.get_all_parameters())
            
            # Collect shape parameters (BDIM/SDIM) - now lists
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
                
                # Add BDIM parameters first (now a list)
                if hasattr(interface, 'bdim_params') and interface.bdim_params:
                    for bdim_param in interface.bdim_params:
                        if bdim_param != '1':  # Skip singleton dimensions
                            bdim_params = [p for p in parameter_definitions if p.name == bdim_param]
                            sorted_params.extend(bdim_params)
                
                # Add SDIM parameters second (now a list)
                if hasattr(interface, 'sdim_params') and interface.sdim_params:
                    for sdim_param in interface.sdim_params:
                        if sdim_param != '1':  # Skip singleton dimensions
                            sdim_params = [p for p in parameter_definitions if p.name == sdim_param]
                            sorted_params.extend(sdim_params)
                
                # Add datatype parameters in their defined order
                if hasattr(interface, 'datatype_metadata') and interface.datatype_metadata:
                    dt_meta = interface.datatype_metadata
                    # Get parameters in the order they're defined in datatype metadata
                    # Collect all BDIM/SDIM params to exclude
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
        
        # Identify AXI-Lite configuration parameters from linked_parameters
        axilite_params = []
        axilite_param_names = set(kernel_metadata.linked_parameters.get("axilite", {}).keys())
        
        for param in parameter_definitions:
            if param.name in axilite_param_names:
                if param.name not in datatype_linked_params and param.name not in internal_datatype_params:
                    axilite_params.append(param)
        
        # Collect all interface-linked parameters (including BDIM/SDIM)
        interface_linked_params = set()
        for group_data in interface_parameter_groups.values():
            interface_linked_params.update(p.name for p in group_data['parameters'])
        
        # General algorithm parameters (everything else, in original order)
        excluded_params = datatype_linked_params | internal_datatype_params | interface_linked_params | {p.name for p in axilite_params}
        general_params = [
            param for param in parameter_definitions 
            if param.name not in excluded_params
        ]
        
        # Organize interface parameters by interface type and order
        organized_interface_params = []
        
        # Input interfaces first
        for interface in kernel_metadata.interfaces:
            if (interface.interface_type.value == 'input' and 
                interface.name in interface_parameter_groups):
                organized_interface_params.append(interface_parameter_groups[interface.name])
        
        # Output interfaces second  
        for interface in kernel_metadata.interfaces:
            if (interface.interface_type.value == 'output' and 
                interface.name in interface_parameter_groups):
                organized_interface_params.append(interface_parameter_groups[interface.name])
        
        # Weight interfaces third
        for interface in kernel_metadata.interfaces:
            if (interface.interface_type.value == 'weight' and 
                interface.name in interface_parameter_groups):
                organized_interface_params.append(interface_parameter_groups[interface.name])
        
        return {
            'general_parameters': general_params,
            'axilite_parameters': axilite_params,
            'internal_datatype_groups': internal_datatype_groups,
            'interface_parameter_groups': organized_interface_params
        }