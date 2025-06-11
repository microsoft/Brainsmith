"""
Template context generator for ParsedKernelData.

Generates complete template context for Jinja2 templates from ParsedKernelData.
Moved from ParsedKernelData to keep it as a simple data container.
"""

from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path

from brainsmith.dataflow.core.interface_types import InterfaceType
from ..rtl_parser.data import ParsedKernelData, Interface, Parameter, TemplateDatatype, SimpleKernel


class TemplateContextGenerator:
    """Generates template context from ParsedKernelData for Jinja2 templates."""
    
    @staticmethod
    def generate_context(parsed_data: ParsedKernelData) -> Dict[str, Any]:
        """Generate complete template context for Jinja2 templates."""
        generator = TemplateContextGenerator()
        
        # Core kernel metadata
        context = {
            "kernel_name": parsed_data.name,
            "class_name": generator._get_class_name(parsed_data.name),
            "source_file": str(parsed_data.source_file),
            "generation_timestamp": datetime.now().isoformat(),
            
            # All interfaces (direct reuse of existing Interface objects)
            "interfaces": list(parsed_data.interfaces.values()),
            "interfaces_list": list(parsed_data.interfaces.values()),  # RTL wrapper compatibility
            
            # Interface categorization using existing InterfaceType enum
            "input_interfaces": generator._get_interfaces_by_type(parsed_data, InterfaceType.INPUT),
            "output_interfaces": generator._get_interfaces_by_type(parsed_data, InterfaceType.OUTPUT),
            "weight_interfaces": generator._get_interfaces_by_type(parsed_data, InterfaceType.WEIGHT),
            "config_interfaces": generator._get_interfaces_by_type(parsed_data, InterfaceType.CONFIG),
            "control_interfaces": generator._get_interfaces_by_type(parsed_data, InterfaceType.CONTROL),
            "dataflow_interfaces": generator._get_dataflow_interfaces(parsed_data),
            
            # RTL parameters (direct reuse with existing template_param_name)
            "rtl_parameters": [
                {
                    "name": param.name,
                    "param_type": param.param_type or "int",
                    "default_value": param.default_value or 0,
                    "template_param_name": param.template_param_name
                }
                for param in parsed_data.parameters
            ],
            
            # Template boolean flags
            "has_inputs": len(generator._get_interfaces_by_type(parsed_data, InterfaceType.INPUT)) > 0,
            "has_outputs": len(generator._get_interfaces_by_type(parsed_data, InterfaceType.OUTPUT)) > 0,
            "has_weights": len(generator._get_interfaces_by_type(parsed_data, InterfaceType.WEIGHT)) > 0,
            
            # Interface counts
            "input_interfaces_count": len(generator._get_interfaces_by_type(parsed_data, InterfaceType.INPUT)),
            "output_interfaces_count": len(generator._get_interfaces_by_type(parsed_data, InterfaceType.OUTPUT)),
            "weight_interfaces_count": len(generator._get_interfaces_by_type(parsed_data, InterfaceType.WEIGHT)),
            
            # Kernel analysis
            "kernel_complexity": generator._estimate_complexity(parsed_data),
            "kernel_type": generator._infer_kernel_type(parsed_data),
            "resource_estimation_required": generator._requires_resource_estimation(parsed_data),
            "verification_required": generator._requires_verification(parsed_data),
            
            # Template enums and utilities
            "InterfaceType": InterfaceType,  # Direct enum access
            
            # Kernel object for RTL wrapper template compatibility  
            "kernel": SimpleKernel(parsed_data.name, parsed_data.parameters),
            
            # Summary statistics
            "dataflow_model_summary": {
                "num_interfaces": len(parsed_data.interfaces),
                "input_count": len(generator._get_interfaces_by_type(parsed_data, InterfaceType.INPUT)),
                "output_count": len(generator._get_interfaces_by_type(parsed_data, InterfaceType.OUTPUT)),
                "weight_count": len(generator._get_interfaces_by_type(parsed_data, InterfaceType.WEIGHT)),
            }
        }
        
        return context
    
    def _get_class_name(self, module_name: str) -> str:
        """Generate Python class name from module name."""
        # Convert snake_case or kebab-case to PascalCase
        parts = module_name.replace('-', '_').split('_')
        return ''.join(word.capitalize() for word in parts)
    
    def _get_interfaces_by_type(self, parsed_data: ParsedKernelData, interface_type: InterfaceType) -> List[Interface]:
        """Get interfaces matching specific InterfaceType."""
        return [iface for iface in parsed_data.interfaces.values() if iface.type == interface_type]
    
    def _get_dataflow_interfaces(self, parsed_data: ParsedKernelData) -> List[Interface]:
        """Get all dataflow interfaces (INPUT, OUTPUT, WEIGHT)."""
        return [iface for iface in parsed_data.interfaces.values() 
                if iface.type in [InterfaceType.INPUT, InterfaceType.OUTPUT, InterfaceType.WEIGHT]]
    
    def _estimate_complexity(self, parsed_data: ParsedKernelData) -> str:
        """Estimate kernel complexity for resource calculations."""
        interface_count = len(parsed_data.interfaces)
        param_count = len(parsed_data.parameters)
        
        if interface_count <= 2 and param_count <= 3:
            return 'low'
        elif interface_count <= 4 and param_count <= 6:
            return 'medium'
        else:
            return 'high'
    
    def _infer_kernel_type(self, parsed_data: ParsedKernelData) -> str:
        """Infer kernel type from name for resource estimation."""
        name_lower = parsed_data.name.lower()
        if any(term in name_lower for term in ['matmul', 'gemm', 'dot']):
            return 'matmul'
        elif any(term in name_lower for term in ['conv', 'convolution']):
            return 'conv'
        elif any(term in name_lower for term in ['threshold', 'compare']):
            return 'threshold'
        elif any(term in name_lower for term in ['norm', 'batch', 'layer']):
            return 'norm'
        else:
            return 'generic'
    
    def _requires_resource_estimation(self, parsed_data: ParsedKernelData) -> bool:
        """Check if resource estimation is needed."""
        # Enable for complex kernels or when multiple interfaces exist
        return len(parsed_data.interfaces) > 2 or self._estimate_complexity(parsed_data) != 'low'
    
    def _requires_verification(self, parsed_data: ParsedKernelData) -> bool:
        """Check if verification is needed."""
        # Enable for kernels with weight interfaces or high complexity
        has_weights = len(self._get_interfaces_by_type(parsed_data, InterfaceType.WEIGHT)) > 0
        return has_weights or self._estimate_complexity(parsed_data) == 'high'
    
    def _get_datatype_info(self, interface: Interface) -> TemplateDatatype:
        """Extract datatype info from interface metadata for templates."""
        constraints = interface.metadata.get("datatype_constraints", {})
        base_types = constraints.get("base_types", ["UINT"])
        min_bits = constraints.get("min_bitwidth", 8)
        max_bits = constraints.get("max_bitwidth", 32)
        
        # Create template-compatible datatype object
        return TemplateDatatype(
            name=base_types[0],
            value=base_types[0], 
            finn_type=f"{base_types[0]}{min_bits}",
            bitwidth=min_bits,
            bit_width=min_bits,  # Template compatibility
            signed=base_types[0] == "INT",
            base_type=base_types[0],
            min_bits=min_bits,
            max_bits=max_bits
        )