"""RTL Backend generator with direct KernelMetadata support."""
from typing import Dict, Any, List

from .base_v2 import GeneratorBase
from ..types.metadata import KernelMetadata
from ..types.rtl import ParameterCategory, SourceType


class RTLBackendGeneratorV2(GeneratorBase):
    """Generates RTL Backend subclasses for FINN integration."""
    
    @property
    def name(self) -> str:
        return "rtl_backend"
    
    @property
    def template_file(self) -> str:
        return "rtl_backend_v2.py.j2"
    
    @property
    def output_pattern(self) -> str:
        return "{kernel_name}_rtl.py"
    
    def _get_specific_vars(self, metadata: KernelMetadata) -> Dict[str, Any]:
        """Get RTL Backend-specific template variables."""
        return {
            # Only the generated code assignments
            'explicit_parameter_assignments': self._generate_assignments(metadata),
            'generation_timestamp': self._get_timestamp(),
        }
    
    
    def _generate_assignments(self, metadata: KernelMetadata) -> List[Dict[str, str]]:
        """Generate explicit parameter assignments for code generation."""
        assignments = []
        
        # Generate assignment for each parameter based on its source type and category
        for param in metadata.parameters:
            param_name = param.name
            
            # Handle shape parameters (BDIM/SDIM) via KernelModel
            if param.category == ParameterCategory.SHAPE:
                assignment = self._create_shape_assignment(param)
                if assignment:
                    assignments.append(assignment)
            
            # Handle algorithm parameters via node attributes
            elif param.category == ParameterCategory.ALGORITHM:
                if param.source_type == SourceType.RTL:
                    assignment = {
                        'template_var': param_name,
                        'assignment': f'str(self.get_nodeattr("{param_name}"))',
                        'comment': f'Algorithm parameter {param_name}'
                    }
                    assignments.append(assignment)
                elif param.source_type == SourceType.NODEATTR_ALIAS:
                    alias_name = param.source_detail.get("nodeattr_name", param_name)
                    assignment = {
                        'template_var': param_name,
                        'assignment': f'str(self.get_nodeattr("{alias_name}"))',
                        'comment': f'Aliased parameter {param_name} from {alias_name}'
                    }
                    assignments.append(assignment)
            
            # Handle datatype parameters via KernelModel
            elif param.category == ParameterCategory.DATATYPE:
                assignment = self._create_datatype_assignment(param)
                if assignment:
                    assignments.append(assignment)
            
            # Handle internal datatype parameters
            elif param.category == ParameterCategory.INTERNAL:
                assignment = self._create_datatype_assignment(param)
                if assignment:
                    assignments.append(assignment)
        
        return assignments
    
    def _create_shape_assignment(self, param) -> Dict[str, str]:
        """Create assignment for shape parameters."""
        param_name = param.name
        
        if param.source_type == SourceType.INTERFACE_SHAPE:
            interface_name = param.interface_name
            dimension_index = param.source_detail.get("dimension", 0)
            shape_type = param.source_detail.get("shape_type", "")
            
            if shape_type == "bdim" or "BDIM" in param_name:
                return {
                    'template_var': param_name,
                    'assignment': f'str(self._get_interface_bdim("{interface_name}", {dimension_index}))',
                    'comment': f'Block dimension from KernelModel for {interface_name}'
                }
            elif shape_type == "sdim" or "SDIM" in param_name:
                return {
                    'template_var': param_name,
                    'assignment': f'str(self._get_interface_sdim("{interface_name}", {dimension_index}))',
                    'comment': f'Stream dimension from KernelModel for {interface_name}'
                }
        elif param.source_type == SourceType.RTL:
            # Shape parameter exposed as node attribute
            return {
                'template_var': param_name,
                'assignment': f'str(self.get_nodeattr("{param_name}"))',
                'comment': f'Shape parameter {param_name}'
            }
        return None
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for generation tracking."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _create_datatype_assignment(self, param) -> Dict[str, str]:
        """Create assignment for datatype parameters."""
        param_name = param.name
        
        if param.source_type == SourceType.INTERFACE_DATATYPE:
            interface_name = param.interface_name
            # V1 uses "property", not "detail" - fixing to match parser output
            detail = param.source_detail.get("property", "WIDTH").upper()
            
            if detail == "WIDTH":
                return {
                    'template_var': param_name,
                    'assignment': f'str(self._get_interface_datatype_width("{interface_name}"))',
                    'comment': f'Datatype width from KernelModel for {interface_name}'
                }
            elif detail == "NAME":
                return {
                    'template_var': param_name,
                    'assignment': f'self._get_interface_datatype_name("{interface_name}")',
                    'comment': f'Datatype name from KernelModel for {interface_name}'
                }
            elif detail == "SIGNED":
                return {
                    'template_var': param_name,
                    'assignment': f'"1" if self._get_interface_datatype_is_signed("{interface_name}") else "0"',
                    'comment': f'Datatype signedness from KernelModel for {interface_name}'
                }
        elif param.source_type == SourceType.INTERNAL_DATATYPE:
            # Internal datatype parameter (e.g., T_WIDTH for threshold)
            dt_name = param.source_detail.get("interface", "")  # Actually datatype name
            property_name = param.source_detail.get("property", "").upper()
            
            if property_name == "WIDTH":
                return {
                    'template_var': param_name,
                    'assignment': f'str(DataType[self.get_nodeattr("{dt_name}DataType")].bitwidth())',
                    'comment': f'Internal datatype {dt_name} width'
                }
            elif property_name == "SIGNED":
                return {
                    'template_var': param_name,
                    'assignment': f'"1" if DataType[self.get_nodeattr("{dt_name}DataType")].signed() else "0"',
                    'comment': f'Internal datatype {dt_name} signedness'
                }
        elif param.source_type == SourceType.RTL:
            # Datatype parameter exposed as node attribute
            return {
                'template_var': param_name,
                'assignment': f'str(self.get_nodeattr("{param_name}"))',
                'comment': f'Datatype parameter {param_name}'
            }
        return None
    
