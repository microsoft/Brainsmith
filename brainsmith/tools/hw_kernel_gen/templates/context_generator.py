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
from ..rtl_parser.data import Interface, Parameter, TemplateDatatype, SimpleKernel


class TemplateContextGenerator:
    """Generates template context from KernelMetadata for Jinja2 templates."""
    
    @staticmethod
    def generate_context(kernel_metadata: KernelMetadata) -> Dict[str, Any]:
        """Generate complete template context for Jinja2 templates."""
        generator = TemplateContextGenerator()
        
        # Extract parallelism and algorithm parameters from RTL analysis
        parallelism_info = generator._analyze_parallelism_parameters(kernel_metadata)
        algorithm_info = generator._infer_algorithm_parameters(kernel_metadata)
        node_attrs = generator._generate_node_attributes(kernel_metadata, parallelism_info, algorithm_info)
        
        # Core kernel metadata
        context = {
            "kernel_name": kernel_metadata.name,
            "class_name": generator._get_class_name(kernel_metadata.name),
            "source_file": str(kernel_metadata.source_file),
            "generation_timestamp": datetime.now().isoformat(),
            
            # Interface metadata (new format - List[InterfaceMetadata])
            "interface_metadata": kernel_metadata.interfaces,
            "interfaces_list": kernel_metadata.interfaces,  # Compatibility
            
            # Legacy interface categorization (will need conversion)
            "input_interfaces": generator._get_interfaces_by_type(kernel_metadata, InterfaceType.INPUT),
            "output_interfaces": generator._get_interfaces_by_type(kernel_metadata, InterfaceType.OUTPUT),
            "weight_interfaces": generator._get_interfaces_by_type(kernel_metadata, InterfaceType.WEIGHT),
            "config_interfaces": generator._get_interfaces_by_type(kernel_metadata, InterfaceType.CONFIG),
            "control_interfaces": generator._get_interfaces_by_type(kernel_metadata, InterfaceType.CONTROL),
            "dataflow_interfaces": generator._get_dataflow_interfaces(kernel_metadata),
            
            # RTL parameters (direct reuse with existing template_param_name)
            "rtl_parameters": [
                {
                    "name": param.name,
                    "param_type": param.param_type or "int",
                    "default_value": param.default_value or 0,
                    "template_param_name": param.template_param_name
                }
                for param in kernel_metadata.parameters
            ],
            
            # AutoHWCustomOp-specific enhancements
            "node_attributes": node_attrs,
            "parallelism_info": parallelism_info,
            "algorithm_info": algorithm_info,
            "datatype_mappings": generator._generate_datatype_mappings(kernel_metadata),
            "shape_calculation_methods": generator._generate_shape_calculation_methods(kernel_metadata, parallelism_info),
            "stream_width_methods": generator._generate_stream_width_methods(kernel_metadata, parallelism_info),
            "resource_estimation_methods": generator._generate_resource_estimation_methods(kernel_metadata, parallelism_info),
            
            # Template boolean flags
            "has_inputs": len(generator._get_interfaces_by_type(kernel_metadata, InterfaceType.INPUT)) > 0,
            "has_outputs": len(generator._get_interfaces_by_type(kernel_metadata, InterfaceType.OUTPUT)) > 0,
            "has_weights": len(generator._get_interfaces_by_type(kernel_metadata, InterfaceType.WEIGHT)) > 0,
            
            # Interface counts
            "input_interfaces_count": len(generator._get_interfaces_by_type(kernel_metadata, InterfaceType.INPUT)),
            "output_interfaces_count": len(generator._get_interfaces_by_type(kernel_metadata, InterfaceType.OUTPUT)),
            "weight_interfaces_count": len(generator._get_interfaces_by_type(kernel_metadata, InterfaceType.WEIGHT)),
            
            # Kernel analysis
            "kernel_complexity": generator._estimate_complexity(kernel_metadata),
            "kernel_type": generator._infer_kernel_type(kernel_metadata),
            "resource_estimation_required": generator._requires_resource_estimation(kernel_metadata),
            "verification_required": generator._requires_verification(kernel_metadata),
            
            # Template enums and utilities
            "InterfaceType": InterfaceType,  # Direct enum access
            
            # Kernel object for RTL wrapper template compatibility  
            "kernel": SimpleKernel(kernel_metadata.name, kernel_metadata.parameters),
            
            # Summary statistics
            "dataflow_model_summary": {
                "num_interfaces": len(kernel_metadata.interfaces),
                "input_count": len(generator._get_interfaces_by_type(kernel_metadata, InterfaceType.INPUT)),
                "output_count": len(generator._get_interfaces_by_type(kernel_metadata, InterfaceType.OUTPUT)),
                "weight_count": len(generator._get_interfaces_by_type(kernel_metadata, InterfaceType.WEIGHT)),
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
            if interface.type in [InterfaceType.INPUT, InterfaceType.OUTPUT]:
                # Look for array ports indicating parallelism
                if "[" in interface.name or "_" in interface.name:
                    parallelism["parallel_elements"].append({
                        "interface": interface.name,
                        "type": interface.type.value,
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
    
    def _generate_datatype_mappings(self, kernel_metadata: KernelMetadata) -> Dict[str, Any]:
        """Generate datatype mapping methods for template."""
        mappings = {
            "input_methods": [],
            "output_methods": [],
            "weight_methods": []
        }
        
        input_interfaces = self._get_interfaces_by_type(kernel_metadata, InterfaceType.INPUT)
        for i, interface in enumerate(input_interfaces):
            mappings["input_methods"].append({
                "index": i,
                "interface_name": interface.name,
                "method_body": f'return DataType[self.get_nodeattr("inputDataType")]'
            })
        
        output_interfaces = self._get_interfaces_by_type(kernel_metadata, InterfaceType.OUTPUT)
        for i, interface in enumerate(output_interfaces):
            mappings["output_methods"].append({
                "index": i,
                "interface_name": interface.name,
                "method_body": f'return DataType[self.get_nodeattr("outputDataType")]'
            })
        
        weight_interfaces = self._get_interfaces_by_type(kernel_metadata, InterfaceType.WEIGHT)
        if weight_interfaces:
            mappings["weight_methods"].append({
                "method_body": 'return DataType[self.get_nodeattr("weightDataType")]'
            })
        
        return mappings
    
    def _generate_shape_calculation_methods(self, kernel_metadata: KernelMetadata, parallelism_info: Dict) -> Dict[str, Any]:
        """Generate shape calculation methods for template."""
        methods = {
            "normal_input_shape": None,
            "normal_output_shape": None,
            "folded_input_shape": None,
            "folded_output_shape": None
        }
        
        # Generate based on interface analysis and parallelism
        if parallelism_info["inferred_channels"]:
            methods["normal_input_shape"] = f"""
        ich = self.get_nodeattr("NumChannels")
        vecs = list(self.get_nodeattr("numInputVectors"))
        return tuple(vecs + [ich])"""
            
            methods["folded_input_shape"] = f"""
        pe = self.get_nodeattr("PE")
        fold = self.calc_tmem()
        vecs = list(self.get_nodeattr("numInputVectors"))
        return tuple(vecs + [fold, pe])"""
        else:
            # Default simple shape calculations
            methods["normal_input_shape"] = """
        vecs = list(self.get_nodeattr("numInputVectors"))
        return tuple(vecs + [1])"""
            
            methods["folded_input_shape"] = """
        vecs = list(self.get_nodeattr("numInputVectors"))
        return tuple(vecs + [1, 1])"""
        
        # Output shapes typically match input shapes for most operations
        methods["normal_output_shape"] = "return self.get_normal_input_shape()"
        methods["folded_output_shape"] = "return self.get_folded_input_shape()"
        
        return methods
    
    def _generate_stream_width_methods(self, kernel_metadata: KernelMetadata, parallelism_info: Dict) -> Dict[str, Any]:
        """Generate stream width calculation methods for template."""
        methods = {
            "instream_width": None,
            "outstream_width": None,
            "weightstream_width": None
        }
        
        # Input stream width
        methods["instream_width"] = f"""
        i_bits = self.get_input_datatype().bitwidth()
        pe = self.get_nodeattr("PE") if self.get_nodeattr("PE") else {parallelism_info["inferred_pe"]}
        return i_bits * pe"""
        
        # Output stream width
        methods["outstream_width"] = f"""
        o_bits = self.get_output_datatype().bitwidth()
        pe = self.get_nodeattr("PE") if self.get_nodeattr("PE") else {parallelism_info["inferred_pe"]}
        return o_bits * pe"""
        
        # Weight stream width (if weights exist)
        if self._get_interfaces_by_type(kernel_metadata, InterfaceType.WEIGHT):
            methods["weightstream_width"] = """
        pe = self.get_nodeattr("PE")
        wp = self.get_weight_datatype().bitwidth()
        return pe * wp"""
        
        return methods
    
    def _generate_resource_estimation_methods(self, kernel_metadata: KernelMetadata, parallelism_info: Dict) -> Dict[str, Any]:
        """Generate resource estimation methods for template."""
        methods = {
            "get_exp_cycles": None,
            "calc_tmem": None,
            "bram_estimation": None,
            "lut_estimation": None,
            "dsp_estimation": None
        }
        
        # Cycle estimation based on kernel type
        kernel_type = self._infer_kernel_type(kernel_metadata)
        if kernel_type == "threshold":
            methods["get_exp_cycles"] = "return np.prod(self.get_folded_output_shape()[:-1])"
        else:
            methods["get_exp_cycles"] = "return np.prod(self.get_folded_output_shape()[:-1])"
        
        # TMEM calculation (if channels and PE are available)
        if parallelism_info["inferred_channels"] and parallelism_info["inferred_pe"]:
            methods["calc_tmem"] = """
        num_channels = self.get_nodeattr("NumChannels")
        pe = self.get_nodeattr("PE")
        return num_channels // pe"""
        
        # Resource estimation based on kernel type
        if kernel_type == "threshold":
            methods["bram_estimation"] = "return 1  # Minimal BRAM for thresholding"
            methods["lut_estimation"] = "return 3000 * self.get_nodeattr('PE', 1)"
            methods["dsp_estimation"] = "return 0  # No DSPs for thresholding"
        elif kernel_type == "matmul":
            methods["bram_estimation"] = "return self.get_nodeattr('PE', 1) * 2"
            methods["lut_estimation"] = "return 5000 * self.get_nodeattr('PE', 1)"
            methods["dsp_estimation"] = "return self.get_nodeattr('PE', 1)"
        else:
            methods["bram_estimation"] = "return 1"
            methods["lut_estimation"] = "return 2000"
            methods["dsp_estimation"] = "return 0"
        
        return methods