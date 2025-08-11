"""HWCustomOp generator with direct KernelMetadata support."""
from typing import Dict, Any, List

from .base_v2 import GeneratorBase
from ..types.metadata import KernelMetadata
from ..types.rtl import ParameterCategory


class HWCustomOpGeneratorV2(GeneratorBase):
    """Generates HWCustomOp subclasses for FINN integration."""
    
    @property
    def name(self) -> str:
        return "hw_custom_op"
    
    @property
    def template_file(self) -> str:
        return "hw_custom_op_v2.py.j2"
    
    @property
    def output_pattern(self) -> str:
        return "{kernel_name}.py"
    
    def _get_specific_vars(self, metadata: KernelMetadata) -> Dict[str, Any]:
        """Get HWCustomOp-specific template variables."""
        return {
            # Extracted attributes
            'explicit_datatype_attrs': self._extract_datatype_attrs(metadata),
            'shape_nodeattrs': self._extract_shape_nodeattrs(metadata),
            'has_datatype_params': self._has_datatype_params(metadata),
            
            # Additional metadata
            'relationships': metadata.relationships,
            'class_name': metadata.class_name,
            'source_file': metadata.source_file,
            
            # Verification flags
            'verification_required': len(metadata.exposed_parameters) > 0,
            'required_attributes': metadata.exposed_parameters,
            
            # Import datetime for timestamp
            'generation_timestamp': self._get_timestamp()
        }
    
    def _extract_datatype_attrs(self, metadata: KernelMetadata) -> List[Dict[str, Any]]:
        """Extract interface datatype attributes for HWCustomOp nodeattrs."""
        datatype_attrs = []
        
        # Check which interfaces have datatype parameters
        interfaces_with_datatypes = set()
        for param in metadata.parameters:
            if param.category == ParameterCategory.DATATYPE and param.interface_name:
                interfaces_with_datatypes.add(param.interface_name)
        
        # Generate datatype attributes for each interface
        for interface_name in interfaces_with_datatypes:
            attr_name = f"{interface_name}DataType"
            datatype_attrs.append({
                "name": attr_name,
                "interface_name": interface_name,
                "attr_spec": ("s", True, "")  # (type, required, default)
            })
        
        return datatype_attrs
    
    def _extract_shape_nodeattrs(self, metadata: KernelMetadata) -> List[Dict[str, str]]:
        """Extract SHAPE parameter node attributes from interfaces."""
        shape_params = {}  # Dict to collect unique params with their sources
        
        # Scan all interfaces for SHAPE parameters
        for interface in metadata.interfaces:
            interface_name = interface.name
            
            # Extract from BDIM shape expressions
            if hasattr(interface, 'bdim_shape') and interface.bdim_shape:
                for element in interface.bdim_shape:
                    if isinstance(element, str) and element != ":" and element.isidentifier():
                        if element not in shape_params:
                            shape_params[element] = []
                        shape_params[element].append(f"BDIM: {interface_name}")
            
            # Extract from SDIM shape expressions  
            if hasattr(interface, 'sdim_shape') and interface.sdim_shape:
                for element in interface.sdim_shape:
                    if isinstance(element, str) and element != ":" and element.isidentifier():
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
        
        return sorted(nodeattrs, key=lambda x: x['name'])
    
    def _has_datatype_params(self, metadata: KernelMetadata) -> bool:
        """Check if kernel has any datatype parameters."""
        for param in metadata.parameters:
            if param.category == ParameterCategory.DATATYPE:
                return True
        return False
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for generation tracking."""
        from datetime import datetime
        return datetime.now().isoformat()