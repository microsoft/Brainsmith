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
        
        # Extract parallelism and algorithm parameters from RTL analysis
        parallelism_info = generator._analyze_parallelism_parameters(parsed_data)
        algorithm_info = generator._infer_algorithm_parameters(parsed_data)
        node_attrs = generator._generate_node_attributes(parsed_data, parallelism_info, algorithm_info)
        
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
            
            # AutoHWCustomOp-specific enhancements
            "node_attributes": node_attrs,
            "parallelism_info": parallelism_info,
            "algorithm_info": algorithm_info,
            "datatype_mappings": generator._generate_datatype_mappings(parsed_data),
            "shape_calculation_methods": generator._generate_shape_calculation_methods(parsed_data, parallelism_info),
            "stream_width_methods": generator._generate_stream_width_methods(parsed_data, parallelism_info),
            "resource_estimation_methods": generator._generate_resource_estimation_methods(parsed_data, parallelism_info),
            
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
    
    def _analyze_parallelism_parameters(self, parsed_data: ParsedKernelData) -> Dict[str, Any]:
        """Extract PE/SIMD equivalent parallelism from RTL analysis."""
        parallelism = {
            "inferred_pe": 1,
            "inferred_simd": 1,
            "inferred_channels": None,
            "inferred_width": None,
            "parallel_elements": []
        }
        
        # Look for PE parameter directly in RTL parameters
        for param in parsed_data.parameters:
            if param.name.upper() == "PE":
                parallelism["inferred_pe"] = int(param.default_value) if param.default_value else 1
            elif param.name.upper() in ["SIMD", "WIDTH"]:
                parallelism["inferred_simd"] = int(param.default_value) if param.default_value else 1
            elif param.name.upper() in ["C", "CHANNELS", "NUM_CHANNELS"]:
                parallelism["inferred_channels"] = int(param.default_value) if param.default_value else 1
            elif param.name.upper() in ["W", "WIDTH", "DATA_WIDTH"]:
                parallelism["inferred_width"] = int(param.default_value) if param.default_value else 1
        
        # Analyze port arrays for parallel processing patterns
        for interface in parsed_data.interfaces.values():
            if interface.type in [InterfaceType.INPUT, InterfaceType.OUTPUT]:
                # Look for array ports indicating parallelism
                if "[" in interface.name or "_" in interface.name:
                    parallelism["parallel_elements"].append({
                        "interface": interface.name,
                        "type": interface.type.value,
                        "inferred_width": 1  # TODO: Parse actual width from port declaration
                    })
        
        return parallelism
    
    def _infer_algorithm_parameters(self, parsed_data: ParsedKernelData) -> Dict[str, Any]:
        """Extract algorithm-specific parameters from RTL analysis."""
        algorithm = {
            "type": self._infer_kernel_type(parsed_data),
            "parameters": {},
            "constraints": []
        }
        
        # Map RTL parameters to algorithm parameters based on kernel type
        kernel_type = algorithm["type"]
        
        if kernel_type == "threshold":
            # Thresholding-specific parameters
            for param in parsed_data.parameters:
                if param.name.upper() in ["N", "NUM_STEPS"]:
                    algorithm["parameters"]["numSteps"] = int(param.default_value) if param.default_value else 1
                elif param.name.upper() in ["BIAS"]:
                    algorithm["parameters"]["ActVal"] = int(param.default_value) if param.default_value else 0
                elif param.name.upper() in ["SIGNED"]:
                    algorithm["parameters"]["signed_input"] = bool(int(param.default_value)) if param.default_value else False
        elif kernel_type == "matmul":
            # Matrix multiplication parameters
            for param in parsed_data.parameters:
                if param.name.upper() in ["M", "ROWS"]:
                    algorithm["parameters"]["rows"] = int(param.default_value) if param.default_value else 1
                elif param.name.upper() in ["N", "COLS"]:
                    algorithm["parameters"]["cols"] = int(param.default_value) if param.default_value else 1
        elif kernel_type == "conv":
            # Convolution parameters
            for param in parsed_data.parameters:
                if param.name.upper() in ["K", "KERNEL_SIZE"]:
                    algorithm["parameters"]["kernel_size"] = int(param.default_value) if param.default_value else 1
                elif param.name.upper() in ["S", "STRIDE"]:
                    algorithm["parameters"]["stride"] = int(param.default_value) if param.default_value else 1
        
        return algorithm
    
    def _generate_node_attributes(self, parsed_data: ParsedKernelData, parallelism_info: Dict, algorithm_info: Dict) -> Dict[str, Any]:
        """Generate node attribute definitions for HWCustomOp."""
        node_attrs = {}
        
        # Hardware-specific attributes
        if parallelism_info["inferred_pe"] > 1:
            node_attrs["PE"] = ("i", True, parallelism_info["inferred_pe"])
        if parallelism_info["inferred_channels"]:
            node_attrs["NumChannels"] = ("i", True, parallelism_info["inferred_channels"])
        
        # Data type specifications (will be filled at runtime)
        input_interfaces = self._get_interfaces_by_type(parsed_data, InterfaceType.INPUT)
        output_interfaces = self._get_interfaces_by_type(parsed_data, InterfaceType.OUTPUT)
        weight_interfaces = self._get_interfaces_by_type(parsed_data, InterfaceType.WEIGHT)
        
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
    
    def _generate_datatype_mappings(self, parsed_data: ParsedKernelData) -> Dict[str, Any]:
        """Generate datatype mapping methods for template."""
        mappings = {
            "input_methods": [],
            "output_methods": [],
            "weight_methods": []
        }
        
        input_interfaces = self._get_interfaces_by_type(parsed_data, InterfaceType.INPUT)
        for i, interface in enumerate(input_interfaces):
            mappings["input_methods"].append({
                "index": i,
                "interface_name": interface.name,
                "method_body": f'return DataType[self.get_nodeattr("inputDataType")]'
            })
        
        output_interfaces = self._get_interfaces_by_type(parsed_data, InterfaceType.OUTPUT)
        for i, interface in enumerate(output_interfaces):
            mappings["output_methods"].append({
                "index": i,
                "interface_name": interface.name,
                "method_body": f'return DataType[self.get_nodeattr("outputDataType")]'
            })
        
        weight_interfaces = self._get_interfaces_by_type(parsed_data, InterfaceType.WEIGHT)
        if weight_interfaces:
            mappings["weight_methods"].append({
                "method_body": 'return DataType[self.get_nodeattr("weightDataType")]'
            })
        
        return mappings
    
    def _generate_shape_calculation_methods(self, parsed_data: ParsedKernelData, parallelism_info: Dict) -> Dict[str, Any]:
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
    
    def _generate_stream_width_methods(self, parsed_data: ParsedKernelData, parallelism_info: Dict) -> Dict[str, Any]:
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
        if self._get_interfaces_by_type(parsed_data, InterfaceType.WEIGHT):
            methods["weightstream_width"] = """
        pe = self.get_nodeattr("PE")
        wp = self.get_weight_datatype().bitwidth()
        return pe * wp"""
        
        return methods
    
    def _generate_resource_estimation_methods(self, parsed_data: ParsedKernelData, parallelism_info: Dict) -> Dict[str, Any]:
        """Generate resource estimation methods for template."""
        methods = {
            "get_exp_cycles": None,
            "calc_tmem": None,
            "bram_estimation": None,
            "lut_estimation": None,
            "dsp_estimation": None
        }
        
        # Cycle estimation based on kernel type
        kernel_type = self._infer_kernel_type(parsed_data)
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