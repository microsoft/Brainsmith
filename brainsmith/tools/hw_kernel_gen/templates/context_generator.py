"""
Template context generator for KernelMetadata.

Generates complete template context for Jinja2 templates from KernelMetadata.
Updated to work with the new unified KernelMetadata structure.
"""

from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path

from brainsmith.dataflow.core.interface_types import InterfaceType
from brainsmith.dataflow.core.kernel_metadata import KernelMetadata
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata
from ..rtl_parser.data import Parameter
from ..parameter_config.parameter_defaults import (
    is_parameter_whitelisted,
    get_default_value,
    PARAMETER_DEFAULTS
)
from .template_context import TemplateContext, ParameterDefinition
# TODO: Need to implement TemplateDatatype, SimpleKernel or replace with proper classes
# from ..rtl_parser.data import TemplateDatatype, SimpleKernel


class TemplateContextGenerator:
    """Generates template context from KernelMetadata for Jinja2 templates."""
    
    @staticmethod
    def generate_context(kernel_metadata: KernelMetadata) -> Dict[str, Any]:
        """Generate complete template context for Jinja2 templates."""
        # First generate the enhanced TemplateContext for Phase 2
        template_ctx = TemplateContextGenerator.generate_template_context(kernel_metadata)
        
        # Then convert to dictionary format for backward compatibility
        return TemplateContextGenerator._template_context_to_dict(template_ctx)
    
    @staticmethod
    def generate_template_context(kernel_metadata: KernelMetadata) -> TemplateContext:
        """Generate enhanced TemplateContext for Phase 2 code generation."""
        generator = TemplateContextGenerator()
        
        # Extract parallelism and algorithm parameters from RTL analysis
        parallelism_info = generator._analyze_parallelism_parameters(kernel_metadata)
        algorithm_info = generator._infer_algorithm_parameters(kernel_metadata)
        
        # Create enhanced parameter definitions with whitelist info
        parameter_definitions = []
        whitelisted_defaults = {}
        required_attributes = []
        
        for param in kernel_metadata.parameters:
            is_whitelisted = is_parameter_whitelisted(param.name)
            
            # Determine if parameter is required (no default or not whitelisted)
            has_rtl_default = param.default_value is not None
            is_required = not has_rtl_default or not is_whitelisted
            
            # Get effective default value
            if is_whitelisted and has_rtl_default:
                default_value = int(param.default_value)
                whitelisted_defaults[param.name] = default_value
            elif is_whitelisted and not has_rtl_default:
                # Use system default for whitelisted params without RTL default
                default_value = get_default_value(param.name)
                whitelisted_defaults[param.name] = default_value
            else:
                default_value = None
                required_attributes.append(param.name)
            
            param_def = ParameterDefinition(
                name=param.name,
                param_type=param.param_type,
                default_value=default_value,
                description=param.description,
                line_number=0,  # Parameter object doesn't have line_number
                template_param_name=param.template_param_name,
                is_whitelisted=is_whitelisted,
                is_required=is_required
            )
            parameter_definitions.append(param_def)
        
        # Categorize interfaces
        input_interfaces = generator._get_interfaces_by_type(kernel_metadata, InterfaceType.INPUT)
        output_interfaces = generator._get_interfaces_by_type(kernel_metadata, InterfaceType.OUTPUT)
        weight_interfaces = generator._get_interfaces_by_type(kernel_metadata, InterfaceType.WEIGHT)
        config_interfaces = generator._get_interfaces_by_type(kernel_metadata, InterfaceType.CONFIG)
        control_interfaces = generator._get_interfaces_by_type(kernel_metadata, InterfaceType.CONTROL)
        
        # Create the enhanced TemplateContext
        template_context = TemplateContext(
            module_name=kernel_metadata.name,
            class_name=generator._get_class_name(kernel_metadata.name),
            source_file=kernel_metadata.source_file,
            interface_metadata=kernel_metadata.interfaces,
            parameter_definitions=parameter_definitions,
            whitelisted_defaults=whitelisted_defaults,
            required_attributes=required_attributes,
            input_interfaces=input_interfaces,
            output_interfaces=output_interfaces,
            weight_interfaces=weight_interfaces,
            config_interfaces=config_interfaces,
            control_interfaces=control_interfaces,
            parallelism_info=parallelism_info,
            algorithm_info=algorithm_info,
            node_attributes=generator._generate_node_attributes(kernel_metadata, parallelism_info, algorithm_info),
            # Note: Datatype, shape, and stream width methods removed - handled by AutoHWCustomOp parent class
            resource_estimation_methods=generator._generate_resource_estimation_methods(kernel_metadata, parallelism_info),
            has_inputs=len(input_interfaces) > 0,
            has_outputs=len(output_interfaces) > 0,
            has_weights=len(weight_interfaces) > 0,
            has_config=len(config_interfaces) > 0,
            kernel_complexity=generator._estimate_complexity(kernel_metadata),
            kernel_type=generator._infer_kernel_type(kernel_metadata)
        )
        
        return template_context
    
    @staticmethod
    def _template_context_to_dict(template_ctx: TemplateContext) -> Dict[str, Any]:
        """Convert TemplateContext to dictionary format for backward compatibility."""
        # Enhanced RTL parameters with whitelist info
        rtl_parameters = []
        for param in template_ctx.parameter_definitions:
            rtl_parameters.append({
                "name": param.name,
                "param_type": param.param_type or "int",
                "default_value": param.default_value if param.default_value is not None else 0,
                "template_param_name": param.template_param_name,
                "is_whitelisted": param.is_whitelisted,
                "is_required": param.is_required
            })
        
        # Get dataflow interfaces
        dataflow_interfaces = [
            iface for iface in template_ctx.interface_metadata 
            if iface.interface_type in [InterfaceType.INPUT, InterfaceType.OUTPUT, InterfaceType.WEIGHT]
        ]
        
        # Start with the basic context
        context = {
            "kernel_name": template_ctx.module_name,
            "class_name": template_ctx.class_name,
            "source_file": str(template_ctx.source_file),
            "generation_timestamp": datetime.now().isoformat(),
            
            # Interface metadata
            "interface_metadata": template_ctx.interface_metadata,
            "interfaces_list": template_ctx.interface_metadata,  # Compatibility
            
            # Categorized interfaces
            "input_interfaces": template_ctx.input_interfaces,
            "output_interfaces": template_ctx.output_interfaces,
            "weight_interfaces": template_ctx.weight_interfaces,
            "config_interfaces": template_ctx.config_interfaces,
            "control_interfaces": template_ctx.control_interfaces,
            "dataflow_interfaces": dataflow_interfaces,
            
            # Enhanced RTL parameters with Phase 2 info
            "rtl_parameters": rtl_parameters,
            "parameter_definitions": template_ctx.parameter_definitions,
            "whitelisted_defaults": template_ctx.whitelisted_defaults,
            "required_attributes": template_ctx.required_attributes,
            
            # AutoHWCustomOp-specific enhancements
            "node_attributes": template_ctx.node_attributes,
            "parallelism_info": template_ctx.parallelism_info,
            "algorithm_info": template_ctx.algorithm_info,
            # Note: datatype_mappings, shape_calculation_methods, stream_width_methods removed
            "resource_estimation_methods": template_ctx.resource_estimation_methods,
            
            # Template boolean flags
            "has_inputs": template_ctx.has_inputs,
            "has_outputs": template_ctx.has_outputs,
            "has_weights": template_ctx.has_weights,
            "has_config": template_ctx.has_config,
            
            # Interface counts
            "input_interfaces_count": len(template_ctx.input_interfaces),
            "output_interfaces_count": len(template_ctx.output_interfaces),
            "weight_interfaces_count": len(template_ctx.weight_interfaces),
            
            # Kernel analysis
            "kernel_complexity": template_ctx.kernel_complexity,
            "kernel_type": template_ctx.kernel_type,
            "resource_estimation_required": len(template_ctx.interface_metadata) > 2 or template_ctx.kernel_complexity != 'low',
            "verification_required": template_ctx.has_weights or template_ctx.kernel_complexity == 'high',
            
            # Template enums and utilities
            "InterfaceType": InterfaceType,  # Direct enum access
            
            # Kernel object for RTL wrapper template compatibility  
            "kernel": {
                "name": template_ctx.module_name,
                "parameters": [
                    Parameter(
                        name=p.name,
                        param_type=p.param_type,
                        default_value=str(p.default_value) if p.default_value is not None else None,
                        description=p.description
                        # Note: template_param_name is computed automatically, line_number not supported
                    ) for p in template_ctx.parameter_definitions
                ]
            },
            
            # Summary statistics
            "dataflow_model_summary": {
                "num_interfaces": len(template_ctx.interface_metadata),
                "input_count": len(template_ctx.input_interfaces),
                "output_count": len(template_ctx.output_interfaces),
                "weight_count": len(template_ctx.weight_interfaces),
            }
        }
        
        return context
    
    def _get_class_name(self, module_name: str) -> str:
        """Generate Python class name from module name."""
        # Convert snake_case or kebab-case to PascalCase
        parts = module_name.replace('-', '_').split('_')
        return ''.join(word.capitalize() for word in parts)
    
    def _get_interfaces_by_type(self, kernel_metadata: KernelMetadata, interface_type: InterfaceType) -> List:
        """Get interfaces matching specific InterfaceType."""
        return [iface for iface in kernel_metadata.interfaces if iface.interface_type == interface_type]
    
    def _get_dataflow_interfaces(self, kernel_metadata: KernelMetadata) -> List:
        """Get all dataflow interfaces (INPUT, OUTPUT, WEIGHT)."""
        return [iface for iface in kernel_metadata.interfaces 
                if iface.interface_type in [InterfaceType.INPUT, InterfaceType.OUTPUT, InterfaceType.WEIGHT]]
    
    def _estimate_complexity(self, kernel_metadata: KernelMetadata) -> str:
        """Estimate kernel complexity for resource calculations."""
        interface_count = len(kernel_metadata.interfaces)
        param_count = len(kernel_metadata.parameters)
        
        if interface_count <= 2 and param_count <= 3:
            return 'low'
        elif interface_count <= 4 and param_count <= 6:
            return 'medium'
        else:
            return 'high'
    
    def _infer_kernel_type(self, kernel_metadata: KernelMetadata) -> str:
        """Infer kernel type from name for resource estimation."""
        name_lower = kernel_metadata.name.lower()
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
    
    def _requires_resource_estimation(self, kernel_metadata: KernelMetadata) -> bool:
        """Check if resource estimation is needed."""
        # Enable for complex kernels or when multiple interfaces exist
        return len(kernel_metadata.interfaces) > 2 or self._estimate_complexity(kernel_metadata) != 'low'
    
    def _requires_verification(self, kernel_metadata: KernelMetadata) -> bool:
        """Check if verification is needed."""
        # Enable for kernels with weight interfaces or high complexity
        has_weights = len(self._get_interfaces_by_type(kernel_metadata, InterfaceType.WEIGHT)) > 0
        return has_weights or self._estimate_complexity(kernel_metadata) == 'high'
    
    def _get_datatype_info(self, interface: InterfaceMetadata):
        """Extract datatype info from InterfaceMetadata for templates."""
        # Get the first (primary) datatype constraint
        if interface.allowed_datatypes:
            constraint = interface.allowed_datatypes[0]
            base_type = "INT" if constraint.signed else "UINT"
            
            # Create template-compatible datatype object
            return {
                "name": constraint.finn_type,
                "value": constraint.finn_type, 
                "finn_type": constraint.finn_type,
                "bitwidth": constraint.bit_width,
                "bit_width": constraint.bit_width,  # Template compatibility
                "signed": constraint.signed,
                "base_type": base_type,
                "min_bits": constraint.bit_width,
                "max_bits": constraint.bit_width
            }
        else:
            # Default fallback (should not happen with proper metadata)
            return {
                "name": "UINT8",
                "value": "UINT8", 
                "finn_type": "UINT8",
                "bitwidth": 8,
                "bit_width": 8,
                "signed": False,
                "base_type": "UINT",
                "min_bits": 8,
                "max_bits": 8
            }
    
    def _analyze_parallelism_parameters(self, kernel_metadata: KernelMetadata) -> Dict[str, Any]:
        """Extract PE/SIMD equivalent parallelism from RTL analysis."""
        parallelism = {
            "inferred_pe": 1,
            "inferred_simd": 1,
            "inferred_channels": None,
            "inferred_width": None,
            "parallel_elements": []
        }
        
        # Look for PE parameter directly in RTL parameters
        for param in kernel_metadata.parameters:
            if param.name.upper() == "PE":
                parallelism["inferred_pe"] = int(param.default_value) if param.default_value else 1
            elif param.name.upper() in ["SIMD", "WIDTH"]:
                parallelism["inferred_simd"] = int(param.default_value) if param.default_value else 1
            elif param.name.upper() in ["C", "CHANNELS", "NUM_CHANNELS"]:
                parallelism["inferred_channels"] = int(param.default_value) if param.default_value else 1
            elif param.name.upper() in ["W", "WIDTH", "DATA_WIDTH"]:
                parallelism["inferred_width"] = int(param.default_value) if param.default_value else 1
        
        # Analyze port arrays for parallel processing patterns
        for interface in kernel_metadata.interfaces:
            if interface.interface_type in [InterfaceType.INPUT, InterfaceType.OUTPUT]:
                # Look for array ports indicating parallelism
                if "[" in interface.name or "_" in interface.name:
                    parallelism["parallel_elements"].append({
                        "interface": interface.name,
                        "type": interface.interface_type.value,
                        "inferred_width": 1  # TODO: Parse actual width from port declaration
                    })
        
        return parallelism
    
    def _infer_algorithm_parameters(self, kernel_metadata: KernelMetadata) -> Dict[str, Any]:
        """Extract algorithm-specific parameters from RTL analysis."""
        algorithm = {
            "type": self._infer_kernel_type(kernel_metadata),
            "parameters": {},
            "constraints": []
        }
        
        # Map RTL parameters to algorithm parameters based on kernel type
        kernel_type = algorithm["type"]
        
        if kernel_type == "threshold":
            # Thresholding-specific parameters
            for param in kernel_metadata.parameters:
                if param.name.upper() in ["N", "NUM_STEPS"]:
                    algorithm["parameters"]["numSteps"] = int(param.default_value) if param.default_value else 1
                elif param.name.upper() in ["BIAS"]:
                    algorithm["parameters"]["ActVal"] = int(param.default_value) if param.default_value else 0
                elif param.name.upper() in ["SIGNED"]:
                    algorithm["parameters"]["signed_input"] = bool(int(param.default_value)) if param.default_value else False
        elif kernel_type == "matmul":
            # Matrix multiplication parameters
            for param in kernel_metadata.parameters:
                if param.name.upper() in ["M", "ROWS"]:
                    algorithm["parameters"]["rows"] = int(param.default_value) if param.default_value else 1
                elif param.name.upper() in ["N", "COLS"]:
                    algorithm["parameters"]["cols"] = int(param.default_value) if param.default_value else 1
        elif kernel_type == "conv":
            # Convolution parameters
            for param in kernel_metadata.parameters:
                if param.name.upper() in ["K", "KERNEL_SIZE"]:
                    algorithm["parameters"]["kernel_size"] = int(param.default_value) if param.default_value else 1
                elif param.name.upper() in ["S", "STRIDE"]:
                    algorithm["parameters"]["stride"] = int(param.default_value) if param.default_value else 1
        
        return algorithm
    
    def _generate_node_attributes(self, kernel_metadata: KernelMetadata, parallelism_info: Dict, algorithm_info: Dict) -> Dict[str, Any]:
        """Generate node attribute definitions for HWCustomOp."""
        node_attrs = {}
        
        # Hardware-specific attributes
        if parallelism_info["inferred_pe"] > 1:
            node_attrs["PE"] = ("i", True, parallelism_info["inferred_pe"])
        if parallelism_info["inferred_channels"]:
            node_attrs["NumChannels"] = ("i", True, parallelism_info["inferred_channels"])
        
        # Data type specifications (will be filled at runtime)
        input_interfaces = self._get_interfaces_by_type(kernel_metadata, InterfaceType.INPUT)
        output_interfaces = self._get_interfaces_by_type(kernel_metadata, InterfaceType.OUTPUT)
        weight_interfaces = self._get_interfaces_by_type(kernel_metadata, InterfaceType.WEIGHT)
        
        if input_interfaces:
            node_attrs["inputDataType"] = ("s", True, "")
        if output_interfaces:
            node_attrs["outputDataType"] = ("s", True, "")
        if weight_interfaces:
            node_attrs["weightDataType"] = ("s", True, "")
        
        # Algorithm-specific parameters
        for param_name, param_value in algorithm_info["parameters"].items():
            if isinstance(param_value, bool):
                node_attrs[param_name] = ("i", False, 1 if param_value else 0, {0, 1})
            elif isinstance(param_value, int):
                node_attrs[param_name] = ("i", True if param_value > 0 else False, param_value)
            else:
                node_attrs[param_name] = ("s", False, str(param_value))
        
        # Default runtime attributes
        node_attrs["runtime_writeable_weights"] = ("i", False, 0, {0, 1})
        node_attrs["numInputVectors"] = ("ints", False, [1])
        
        return node_attrs
    
    # Note: _generate_datatype_mappings removed - handled by AutoHWCustomOp parent class
    
    # Note: _generate_shape_calculation_methods removed - handled by AutoHWCustomOp parent class
    
    # Note: _generate_stream_width_methods removed - handled by AutoHWCustomOp parent class
    
    def _generate_resource_estimation_methods(self, kernel_metadata: KernelMetadata, parallelism_info: Dict) -> Dict[str, Any]:
        """Generate simplified resource estimation stubs for template."""
        # Simplified - only basic resource estimation stubs that can't be computed generically
        methods = {
            "bram_estimation": "return 1",
            "lut_estimation": "return 2000", 
            "dsp_estimation": "return 0"
        }
        # Note: get_exp_cycles and calc_tmem handled by AutoHWCustomOp parent class
        return methods