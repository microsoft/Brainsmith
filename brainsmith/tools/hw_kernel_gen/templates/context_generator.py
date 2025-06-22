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
        
        # Then convert to dictionary format for backward compatibility with datatype mapping extraction
        return TemplateContextGenerator._template_context_to_dict(template_ctx, kernel_metadata)
    
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
        
        # Extract datatype parameter mappings
        datatype_mappings = generator._extract_datatype_parameter_mappings(kernel_metadata)
        
        # Create the enhanced TemplateContext
        template_context = TemplateContext(
            module_name=kernel_metadata.name,
            class_name=generator._get_class_name(kernel_metadata.name),
            source_file=kernel_metadata.source_file,
            interface_metadata=kernel_metadata.interfaces,
            parameter_definitions=parameter_definitions,
            exposed_parameters=kernel_metadata.exposed_parameters,
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
            kernel_type=generator._infer_kernel_type(kernel_metadata),
            # Add datatype parameter information
            datatype_linked_params=datatype_mappings.get('datatype_linked_params', []),
            datatype_param_mappings=datatype_mappings.get('datatype_param_mappings', {}),
            interface_datatype_attributes=datatype_mappings.get('interface_datatype_attributes', []),
            datatype_derivation_methods=datatype_mappings.get('datatype_derivation_methods', {}),
            # Add linked parameter data
            linked_parameters=kernel_metadata.linked_parameters,
            # Add internal datatypes
            internal_datatypes=kernel_metadata.internal_datatypes,
            # Generate template-time parameter assignments
            datatype_parameter_assignments=generator._generate_datatype_parameter_assignments(kernel_metadata),
            # Add categorized parameters for organized RTL wrapper generation
            categorized_parameters=generator._categorize_parameters(kernel_metadata, parameter_definitions, datatype_mappings)
        )
        
        return template_context
    
    @staticmethod
    def _template_context_to_dict(template_ctx: TemplateContext, kernel_metadata: KernelMetadata = None) -> Dict[str, Any]:
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
        
        # Get dataflow interfaces with enhanced datatype parameter info
        dataflow_interfaces = []
        for iface in template_ctx.interface_metadata:
            if iface.interface_type in [InterfaceType.INPUT, InterfaceType.OUTPUT, InterfaceType.WEIGHT]:
                # Add datatype parameter generation to interface
                interface_dict = {
                    'name': iface.name,
                    'compiler_name': iface.compiler_name,
                    'interface_type': iface.interface_type,
                    'datatype_constraints': iface.datatype_constraints,
                    'chunking_strategy': iface.chunking_strategy,
                    'description': iface.description,
                    'datatype_metadata': iface.datatype_metadata,
                    # Generate parameter names for common properties
                    'width_param': iface.get_datatype_parameter_name('width'),
                    'signed_param': iface.get_datatype_parameter_name('signed'),
                    'format_param': iface.get_datatype_parameter_name('format'),
                    'bias_param': iface.get_datatype_parameter_name('bias'),
                    'fractional_width_param': iface.get_datatype_parameter_name('fractional_width')
                }
                dataflow_interfaces.append(interface_dict)
        
        # Use datatype information from TemplateContext if available, otherwise extract from kernel_metadata
        datatype_mappings = {}
        if hasattr(template_ctx, 'datatype_linked_params'):
            # Use datatype information from TemplateContext
            datatype_mappings = {
                'datatype_linked_params': template_ctx.datatype_linked_params,
                'datatype_param_mappings': template_ctx.datatype_param_mappings,
                'interface_datatype_attributes': template_ctx.interface_datatype_attributes,
                'datatype_derivation_methods': template_ctx.datatype_derivation_methods
            }
        elif kernel_metadata:
            # Fallback: extract from kernel_metadata 
            datatype_mappings = TemplateContextGenerator._extract_datatype_parameter_mappings(kernel_metadata)
        
        # Collect all datatype-linked parameters to exclude from node attributes
        datatype_linked_params = set(datatype_mappings.get('datatype_linked_params', []))
        if not datatype_linked_params:
            # Fallback: extract from interface metadata
            for iface in template_ctx.interface_metadata:
                if hasattr(iface, 'datatype_params') and iface.datatype_params:
                    datatype_linked_params.update(iface.datatype_params.values())
        
        # Start with the basic context
        context = {
            "kernel_name": template_ctx.module_name,
            "class_name": template_ctx.class_name,
            "source_file": str(template_ctx.source_file),
            "generation_timestamp": datetime.now().isoformat(),
            "datatype_linked_params": list(datatype_linked_params),
            
            # Interface metadata
            "interface_metadata": template_ctx.interface_metadata,
            "interfaces_list": template_ctx.interface_metadata,  # Compatibility
            
            # Categorized interfaces with datatype parameter info
            "input_interfaces": TemplateContextGenerator()._enhance_interfaces_with_datatype_params(template_ctx.input_interfaces),
            "output_interfaces": TemplateContextGenerator()._enhance_interfaces_with_datatype_params(template_ctx.output_interfaces),
            "weight_interfaces": TemplateContextGenerator()._enhance_interfaces_with_datatype_params(template_ctx.weight_interfaces),
            "config_interfaces": TemplateContextGenerator()._enhance_interfaces_with_datatype_params(template_ctx.config_interfaces),
            "control_interfaces": TemplateContextGenerator()._enhance_interfaces_with_datatype_params(template_ctx.control_interfaces),
            "dataflow_interfaces": dataflow_interfaces,
            
            # Enhanced RTL parameters with Phase 2 info
            "rtl_parameters": rtl_parameters,
            "parameter_definitions": template_ctx.parameter_definitions,
            "exposed_parameters": template_ctx.exposed_parameters,
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
        
        # Add datatype parameter mappings to context
        context.update(datatype_mappings)
        
        # Add linked parameter data to context
        context["linked_parameters"] = template_ctx.linked_parameters
        context["parameter_aliases"] = template_ctx.linked_parameters.get("aliases", {})
        context["derived_parameters"] = template_ctx.linked_parameters.get("derived", {})
        context["axilite_parameters"] = template_ctx.linked_parameters.get("axilite", {})
        
        # Add internal datatypes to context
        context["internal_datatypes"] = template_ctx.internal_datatypes
        
        # Add template-time parameter assignments to context
        context["datatype_parameter_assignments"] = template_ctx.datatype_parameter_assignments
        
        # Add categorized parameters for organized RTL wrapper generation
        context["categorized_parameters"] = template_ctx.categorized_parameters
        
        # Add CONFIG interface detection and control parameter mapping
        if kernel_metadata:
            config_interfaces_list = TemplateContextGenerator()._get_interfaces_by_type(kernel_metadata, InterfaceType.CONFIG)
            context["has_config_interface"] = len(config_interfaces_list) > 0
            
            # Create interface -> control parameter mapping
            interface_control_params = {}
            for param_name, interface_name in context["axilite_parameters"].items():
                interface_control_params[interface_name] = param_name
            context["interface_control_params"] = interface_control_params
            
            # For backward compatibility - use first config interface's control param
            if config_interfaces_list:
                first_interface = config_interfaces_list[0].name
                context["config_interface_name"] = first_interface
                context["axilite_control_param"] = interface_control_params.get(first_interface, "axilite_config")
            else:
                context["config_interface_name"] = None
                context["axilite_control_param"] = "axilite_config"
        else:
            # Fallback: extract from template_ctx interfaces
            config_interfaces_list = [iface for iface in template_ctx.interface_metadata 
                                    if iface.interface_type == InterfaceType.CONFIG]
            context["has_config_interface"] = len(config_interfaces_list) > 0
            context["config_interface_name"] = config_interfaces_list[0].name if config_interfaces_list else None
            context["axilite_control_param"] = "axilite_config"  # Default fallback
            context["interface_control_params"] = {}
        
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
        
        # Data type specifications now handled by interface-specific datatype attributes
        # (e.g., input0DataType, output0DataType) based on compiler_name for consistency
        # Generic inputDataType/outputDataType/weightDataType removed to avoid redundancy
        
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
    
    def _enhance_interfaces_with_datatype_params(self, interfaces: List[InterfaceMetadata]) -> List[Dict[str, Any]]:
        """Enhance interface list with datatype parameter information for templates."""
        enhanced_interfaces = []
        for iface in interfaces:
            interface_dict = {
                'name': iface.name,
                'compiler_name': iface.compiler_name,
                'interface_type': iface.interface_type,
                'datatype_constraints': iface.datatype_constraints,
                'chunking_strategy': iface.chunking_strategy,
                'description': iface.description,
                'datatype_metadata': iface.datatype_metadata,
                # Generate parameter names for common properties
                'width_param': iface.get_datatype_parameter_name('width'),
                'signed_param': iface.get_datatype_parameter_name('signed'),
                'format_param': iface.get_datatype_parameter_name('format'),
                'bias_param': iface.get_datatype_parameter_name('bias'),
                'fractional_width_param': iface.get_datatype_parameter_name('fractional_width')
            }
            enhanced_interfaces.append(interface_dict)
        return enhanced_interfaces
    
    @staticmethod
    def _extract_datatype_parameter_mappings(kernel_metadata: KernelMetadata) -> Dict[str, Any]:
        """
        Extract datatype parameter mappings from DATATYPE_PARAM pragmas.
        
        Returns:
            Dictionary with datatype parameter mapping information for templates:
            - datatype_linked_params: List of parameter names that are datatype-linked
            - datatype_param_mappings: Dict mapping interface names to parameter mappings
            - interface_datatype_attributes: List of interface datatype attribute definitions
            - datatype_derivation_methods: Dict of methods to generate for RTLBackend
        """
        datatype_linked_params = set()
        datatype_param_mappings = {}
        interface_datatype_attributes = []
        datatype_derivation_methods = {}
        
        # Extract from interface metadata (populated by pragma processing)
        for interface in kernel_metadata.interfaces:
            if hasattr(interface, 'datatype_metadata') and interface.datatype_metadata:
                dt_meta = interface.datatype_metadata
                # Collect all datatype-linked parameter names
                param_names = dt_meta.get_all_parameters()
                datatype_linked_params.update(param_names)
                
                # Store interface-to-parameter mappings (convert DatatypeMetadata to dict)
                param_mapping = {}
                if dt_meta.width:
                    param_mapping['width'] = dt_meta.width
                if dt_meta.signed:
                    param_mapping['signed'] = dt_meta.signed
                if dt_meta.format:
                    param_mapping['format'] = dt_meta.format
                if dt_meta.bias:
                    param_mapping['bias'] = dt_meta.bias
                if dt_meta.fractional_width:
                    param_mapping['fractional_width'] = dt_meta.fractional_width
                if dt_meta.exponent_width:
                    param_mapping['exponent_width'] = dt_meta.exponent_width
                if dt_meta.mantissa_width:
                    param_mapping['mantissa_width'] = dt_meta.mantissa_width
                    
                datatype_param_mappings[interface.name] = param_mapping
                
                # Generate datatype derivation methods for RTLBackend
                for property_type, parameter_name in param_mapping.items():
                    derivation_method = TemplateContextGenerator._generate_datatype_derivation_method(
                        parameter_name, interface.name, property_type
                    )
                    datatype_derivation_methods[parameter_name] = derivation_method
                
                # Generate interface datatype attribute for AutoHWCustomOp
                if interface.interface_type in [InterfaceType.INPUT, InterfaceType.OUTPUT, InterfaceType.WEIGHT]:
                    # Use compiler_name for consistent naming (e.g., input0DataType instead of s_axis_input0DataType)
                    compiler_name = getattr(interface, 'compiler_name', interface.name)
                    datatype_attr_name = f"{compiler_name}DataType"
                    default_datatype = "INT8"  # Default fallback
                    
                    interface_datatype_attributes.append({
                        'name': datatype_attr_name,
                        'interface_name': interface.name,
                        'compiler_name': compiler_name,
                        'interface_type': interface.interface_type.value,
                        'default_datatype': default_datatype,
                        'attr_spec': ("s", False, default_datatype)
                    })
        
        return {
            'datatype_linked_params': list(datatype_linked_params),
            'datatype_param_mappings': datatype_param_mappings,
            'interface_datatype_attributes': interface_datatype_attributes,
            'datatype_derivation_methods': datatype_derivation_methods
        }
    
    @staticmethod
    def _generate_datatype_parameter_assignments(kernel_metadata: KernelMetadata) -> List[Dict[str, str]]:
        """
        Generate template-time parameter assignments for datatype parameters.
        
        This replaces the complex runtime logic in the RTL backend template with
        simple assignment statements generated at template time.
        
        Returns:
            List of parameter assignment dictionaries with 'template_var', 'source', and 'comment'
        """
        assignments = []
        
        # Interface datatype parameter assignments
        for interface in kernel_metadata.interfaces:
            if hasattr(interface, 'datatype_metadata') and interface.datatype_metadata:
                dt_meta = interface.datatype_metadata
                compiler_name = getattr(interface, 'compiler_name', interface.name)
                datatype_attr_name = f"{compiler_name}DataType"
                
                # Generate assignments for each datatype parameter
                for param_name in dt_meta.get_all_parameters():
                    if dt_meta.width and param_name == dt_meta.width:
                        assignments.append({
                            'template_var': f'${param_name.upper()}$',
                            'source': f'str(DataType[self.get_nodeattr("{datatype_attr_name}")].bitwidth())',
                            'comment': f'Interface {interface.name} width parameter'
                        })
                    elif dt_meta.signed and param_name == dt_meta.signed:
                        assignments.append({
                            'template_var': f'${param_name.upper()}$',
                            'source': f'str(1 if DataType[self.get_nodeattr("{datatype_attr_name}")].signed() else 0)',
                            'comment': f'Interface {interface.name} signed parameter'
                        })
                    elif dt_meta.format and param_name == dt_meta.format:
                        assignments.append({
                            'template_var': f'${param_name.upper()}$',
                            'source': f'("FIXED" if DataType[self.get_nodeattr("{datatype_attr_name}")].is_fixed() else ("FLOAT" if DataType[self.get_nodeattr("{datatype_attr_name}")].is_float() else ("INT" if DataType[self.get_nodeattr("{datatype_attr_name}")].signed() else "UINT")))',
                            'comment': f'Interface {interface.name} format parameter'
                        })
                    elif dt_meta.fractional_width and param_name == dt_meta.fractional_width:
                        assignments.append({
                            'template_var': f'${param_name.upper()}$',
                            'source': f'str(DataType[self.get_nodeattr("{datatype_attr_name}")].get_fractional_width() if hasattr(DataType[self.get_nodeattr("{datatype_attr_name}")], "get_fractional_width") else 0)',
                            'comment': f'Interface {interface.name} fractional width parameter'
                        })
        
        # Internal datatype parameter assignments  
        for dt_meta in kernel_metadata.internal_datatypes:
            datatype_attr_name = f"{dt_meta.name}DataType"
            
            # Generate assignments for each internal datatype parameter
            for param_name in dt_meta.get_all_parameters():
                if dt_meta.width and param_name == dt_meta.width:
                    assignments.append({
                        'template_var': f'${param_name.upper()}$',
                        'source': f'str(DataType[self.get_nodeattr("{datatype_attr_name}")].bitwidth())',
                        'comment': f'Internal {dt_meta.name} width parameter'
                    })
                elif dt_meta.signed and param_name == dt_meta.signed:
                    assignments.append({
                        'template_var': f'${param_name.upper()}$',
                        'source': f'str(1 if DataType[self.get_nodeattr("{datatype_attr_name}")].signed() else 0)',
                        'comment': f'Internal {dt_meta.name} signed parameter'
                    })
                elif dt_meta.format and param_name == dt_meta.format:
                    assignments.append({
                        'template_var': f'${param_name.upper()}$',
                        'source': f'("FIXED" if DataType[self.get_nodeattr("{datatype_attr_name}")].is_fixed() else ("FLOAT" if DataType[self.get_nodeattr("{datatype_attr_name}")].is_float() else ("INT" if DataType[self.get_nodeattr("{datatype_attr_name}")].signed() else "UINT")))',
                        'comment': f'Internal {dt_meta.name} format parameter'
                    })
                elif dt_meta.fractional_width and param_name == dt_meta.fractional_width:
                    assignments.append({
                        'template_var': f'${param_name.upper()}$',
                        'source': f'str(DataType[self.get_nodeattr("{datatype_attr_name}")].get_fractional_width() if hasattr(DataType[self.get_nodeattr("{datatype_attr_name}")], "get_fractional_width") else 0)',
                        'comment': f'Internal {dt_meta.name} fractional width parameter'
                    })
        
        return assignments
    
    @staticmethod
    def _generate_datatype_derivation_method(parameter_name: str, interface_name: str, property_type: str) -> str:
        """
        Generate Python code for datatype derivation method in RTLBackend.
        
        Args:
            parameter_name: RTL parameter name (e.g., 'WI')
            interface_name: Interface name (e.g., 's_axis')
            property_type: Property type ('width', 'signed', 'format', 'bias', 'fractional_width')
            
        Returns:
            Python code string for the derivation method
        """
        datatype_attr_name = f"{interface_name}DataType"
        
        if property_type == 'width':
            return f"""
    def _derive_{parameter_name}(self):
        \"\"\"Derive {parameter_name} from {interface_name} datatype width.\"\"\"
        interface_dt = self.get_nodeattr("{datatype_attr_name}")
        return DataType[interface_dt].bitwidth()"""
        
        elif property_type == 'signed':
            return f"""
    def _derive_{parameter_name}(self):
        \"\"\"Derive {parameter_name} from {interface_name} datatype signedness.\"\"\"
        interface_dt = self.get_nodeattr("{datatype_attr_name}")
        return 1 if DataType[interface_dt].signed() else 0"""
        
        elif property_type == 'format':
            return f"""
    def _derive_{parameter_name}(self):
        \"\"\"Derive {parameter_name} from {interface_name} datatype format.\"\"\"
        interface_dt = self.get_nodeattr("{datatype_attr_name}")
        # Format mapping: floating-point vs fixed-point
        return 1 if 'FLOAT' in interface_dt else 0"""
        
        elif property_type == 'bias':
            return f"""
    def _derive_{parameter_name}(self):
        \"\"\"Derive {parameter_name} from {interface_name} datatype bias.\"\"\"
        # Default bias value - could be extended with datatype-specific logic
        return 0"""
        
        elif property_type == 'fractional_width':
            return f"""
    def _derive_{parameter_name}(self):
        \"\"\"Derive {parameter_name} from {interface_name} datatype fractional width.\"\"\"
        interface_dt = self.get_nodeattr("{datatype_attr_name}")
        # Extract fractional width for fixed-point datatypes
        if hasattr(DataType[interface_dt], 'fractional_width'):
            return DataType[interface_dt].fractional_width()
        return 0"""
        
        else:
            # Fallback for unknown property types
            return f"""
    def _derive_{parameter_name}(self):
        \"\"\"Derive {parameter_name} from {interface_name} datatype (property: {property_type}).\"\"\"
        # TODO: Implement derivation for property type '{property_type}'
        return 1"""
    
    def _categorize_parameters(self, kernel_metadata: KernelMetadata, 
                             parameter_definitions: List[ParameterDefinition],
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
            
            # Collect shape parameters (BDIM/SDIM)
            if hasattr(interface, 'bdim_param') and interface.bdim_param:
                interface_params.add(interface.bdim_param)
            if hasattr(interface, 'sdim_param') and interface.sdim_param:
                interface_params.add(interface.sdim_param)
            
            if interface_params:
                # Sort parameters: BDIM first, SDIM second, then datatype params
                sorted_params = []
                
                # Add BDIM parameter first
                if hasattr(interface, 'bdim_param') and interface.bdim_param:
                    bdim_params = [p for p in parameter_definitions if p.name == interface.bdim_param]
                    sorted_params.extend(bdim_params)
                
                # Add SDIM parameter second
                if hasattr(interface, 'sdim_param') and interface.sdim_param:
                    sdim_params = [p for p in parameter_definitions if p.name == interface.sdim_param]
                    sorted_params.extend(sdim_params)
                
                # Add datatype parameters in their defined order
                if hasattr(interface, 'datatype_metadata') and interface.datatype_metadata:
                    dt_meta = interface.datatype_metadata
                    # Get parameters in the order they're defined in datatype metadata
                    for param_name in dt_meta.get_all_parameters():
                        if param_name not in [interface.bdim_param if hasattr(interface, 'bdim_param') else None,
                                             interface.sdim_param if hasattr(interface, 'sdim_param') else None]:
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