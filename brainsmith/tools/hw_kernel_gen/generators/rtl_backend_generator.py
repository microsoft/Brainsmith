"""
RTL Backend generator for HWKG.

Generates AutoRTLBackend subclasses with operation-specific customizations
for complete FINN RTLBackend compatibility.
"""

from typing import Dict, List
import re

try:
    from .base import GeneratorBase
    from ..templates.template_context import TemplateContext
except ImportError:
    # Handle case when imported directly
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    sys.path.insert(0, str(current_dir.parent))
    from base import GeneratorBase
    from templates.template_context import TemplateContext


class RTLBackendGenerator(GeneratorBase):
    """Generates AutoRTLBackend subclasses with operation-specific customizations."""
    
    name = "rtl_backend"
    template_file = "rtl_backend.py.j2"
    output_pattern = "{kernel_name}_rtl.py"
    
    def process_context(self, context: TemplateContext) -> Dict:
        """
        Process context for RTL backend generation.
        
        Analyzes the RTL operation to determine:
        - Base kernel class mapping
        - Complexity level and required mixins
        - Template variable mappings
        - Resource estimation formulas
        - finn-rtllib module mapping
        
        Args:
            context: Full template context from TemplateContextGenerator
            
        Returns:
            Context dictionary for RTL backend template rendering
        """
        # Start with base context
        rtl_context = self.context_to_dict(context)
        
        # Extract datatype parameter information directly from TemplateContext
        datatype_linked_params = context.datatype_linked_params
        datatype_param_mappings = context.datatype_param_mappings
        interface_datatype_attributes = context.interface_datatype_attributes
        
        # Add datatype information to RTL context
        rtl_context["datatype_linked_params"] = datatype_linked_params
        rtl_context["datatype_param_mappings"] = datatype_param_mappings  
        rtl_context["interface_datatype_attributes"] = interface_datatype_attributes
        
        # Determine base kernel class from operation patterns
        base_kernel_class = self._determine_base_kernel_class(context.module_name)
        rtl_context["base_kernel_class"] = base_kernel_class
        
        # Determine complexity level and required features
        complexity_info = self._analyze_complexity_level(context)
        rtl_context.update(complexity_info)
        
        # Map to finn-rtllib module
        rtl_context["finn_rtllib_module"] = self._map_to_finn_rtllib_module(context.module_name)
        
        # Generate template variables mapping
        rtl_context["template_variables"] = self._generate_template_variables(context, datatype_linked_params)
        
        # Determine supporting RTL files
        rtl_context["supporting_rtl_files"] = self._determine_supporting_files(context.module_name)
        
        # Always generate pass stubs for resource estimation for now
        # TODO: Implement sophisticated resource estimation when we have more data
        rtl_context.update({
            "has_lut_formula": False,
            "has_bram_formula": False, 
            "has_dsp_formula": False,
            "lut_estimation_formula": "",
            "bram_estimation_formula": "",
            "dsp_estimation_formula": ""
        })
        
        # Add operation description from comments or pragmas
        rtl_context["operation_description"] = self._extract_operation_description(context)
        
        return rtl_context
    
    def _determine_base_kernel_class(self, module_name: str) -> str:
        """
        Determine base kernel class from module name patterns.
        
        Maps RTL module names to their corresponding FINN kernel classes.
        """
        name_lower = module_name.lower()
        
        # Matrix/vector operations
        if any(pattern in name_lower for pattern in ["mvu", "mvau", "matrix_vector"]):
            return "MVAU"
        elif any(pattern in name_lower for pattern in ["vvu", "vvau", "vector_vector"]):
            return "VVAU"
        elif any(pattern in name_lower for pattern in ["dynmvu", "dynamic_mvu"]):
            return "MVAU"  # Dynamic MVU extends MVAU
        
        # Activation and thresholding
        elif any(pattern in name_lower for pattern in ["threshold", "thres", "activation"]):
            return "Thresholding"
        
        # Convolution operations
        elif any(pattern in name_lower for pattern in ["conv", "convolution", "input_gen"]):
            return "ConvolutionInputGenerator"
        
        # Padding operations
        elif any(pattern in name_lower for pattern in ["pad", "padding", "fmpad"]):
            return "FMPadding"
        
        # FIFO and streaming
        elif any(pattern in name_lower for pattern in ["fifo", "queue", "streaming_fifo"]):
            return "StreamingFIFO"
        
        # Data width conversion
        elif any(pattern in name_lower for pattern in ["dwc", "datawidth", "width_conv"]):
            return "StreamingDataWidthConverter"
        
        # Memory streaming
        elif any(pattern in name_lower for pattern in ["memstream", "memory_stream"]):
            return "StreamingFIFO"  # Use FIFO as base for memory streaming
        
        # Default to None for unknown operations
        return None
    
    def _analyze_complexity_level(self, context: TemplateContext) -> Dict:
        """
        Analyze complexity level and determine required features.
        
        Based on the RTLBackend findings:
        - Low: 149-188 lines, standard patterns
        - Medium: 242-356 lines, custom execution OR implementation styles
        - High: 516-982 lines, advanced memory OR dynamic config
        """
        complexity_info = {
            "complexity_level": "low",
            "has_implementation_styles": False,
            "has_advanced_memory": False,
            "has_dynamic_config": False,
            "has_custom_execution": False
        }
        
        # Count interfaces and parameters to assess complexity
        interface_count = len(context.input_interfaces) + len(context.output_interfaces) + len(context.config_interfaces)
        parameter_count = len(context.parameter_definitions)
        
        module_name_lower = context.module_name.lower()
        
        # High complexity indicators
        if any(pattern in module_name_lower for pattern in [
            "threshold", "convolution", "input_gen"
        ]):
            complexity_info["complexity_level"] = "high"
            complexity_info["has_advanced_memory"] = True
            if "convolution" in module_name_lower or "input_gen" in module_name_lower:
                complexity_info["has_dynamic_config"] = True
        
        # Medium complexity indicators
        elif any(pattern in module_name_lower for pattern in [
            "fifo", "mvu", "vvu", "dynmvu"
        ]):
            complexity_info["complexity_level"] = "medium"
            if "fifo" in module_name_lower:
                complexity_info["has_implementation_styles"] = True
            if any(pattern in module_name_lower for pattern in ["mvu", "vvu", "dynmvu"]):
                complexity_info["has_custom_execution"] = True
        
        # Override based on interface/parameter complexity
        if interface_count > 4 or parameter_count > 8:
            if complexity_info["complexity_level"] == "low":
                complexity_info["complexity_level"] = "medium"
        
        if interface_count > 6 or parameter_count > 12:
            complexity_info["complexity_level"] = "high"
            complexity_info["has_advanced_memory"] = True
        
        return complexity_info
    
    def _map_to_finn_rtllib_module(self, module_name: str) -> str:
        """
        Map module name to finn-rtllib directory name.
        
        Returns the directory name within finn-rtllib that contains
        the template and supporting files for this operation.
        """
        name_lower = module_name.lower()
        
        # Direct mappings to known finn-rtllib modules
        if any(pattern in name_lower for pattern in ["mvu", "mvau", "matrix_vector"]):
            return "mvu"
        elif any(pattern in name_lower for pattern in ["vvu", "vvau", "vector_vector"]):
            return "vvu"
        elif any(pattern in name_lower for pattern in ["threshold", "thres"]):
            return "thresholding"
        elif any(pattern in name_lower for pattern in ["conv", "input_gen"]):
            return "convolution_input_generator"
        elif any(pattern in name_lower for pattern in ["pad", "fmpad"]):
            return "padding"
        elif any(pattern in name_lower for pattern in ["fifo", "streaming_fifo"]):
            return "fifo"
        elif any(pattern in name_lower for pattern in ["dwc", "datawidth"]):
            return "dwc"
        elif any(pattern in name_lower for pattern in ["memstream"]):
            return "memstream"
        
        # Fallback: use module name with common suffixes removed
        clean_name = re.sub(r'_(axi|wrapper|top|rtl)$', '', name_lower)
        return clean_name
    
    def _generate_template_variables(self, context: TemplateContext, datatype_linked_params: List[str]) -> Dict[str, str]:
        """
        Generate template variable mappings from RTL parameters to template placeholders.
        
        Maps RTL parameters and pragmas to the template variable format used
        by finn-rtllib templates.
        """
        variables = {}
        
        # Generate template variables from RTL parameters
        # Only algorithm parameters are accessed via nodeattrs now
        # Datatype-linked parameters are computed on-demand in prepare_codegen_rtl_values
        for param in context.parameter_definitions:
            param_name = param.name
            
            # Only include algorithm parameters in template variables
            if param_name not in datatype_linked_params:
                # Algorithm parameter - use nodeattr access
                access_method = f"self.get_nodeattr(\"{param_name}\")"
                
                # Map specific parameter names to template variables
                if param_name in ["PE", "pe", "Pe"]:
                    variables["PE"] = access_method
                elif param_name in ["SIMD", "simd", "Simd"]:
                    variables["SIMD"] = access_method
                elif param_name in ["DEPTH", "depth"]:
                    variables["DEPTH"] = access_method
                
                # Generic mapping: map any algorithm parameter to its name as template variable
                variables[param_name] = access_method
        
        # Add stream width variables (these are still computed dynamically)
        variables["IBITS"] = "self.get_instream_width()"
        variables["OBITS"] = "self.get_outstream_width()"
        
        return variables
    
    def _determine_supporting_files(self, module_name: str) -> List[str]:
        """
        Determine supporting RTL files based on module patterns.
        
        Returns list of .sv/.v files that should be included from finn-rtllib.
        """
        name_lower = module_name.lower()
        
        # Known file patterns for different operations
        if any(pattern in name_lower for pattern in ["mvu", "mvau"]):
            return ["mvu_axi.sv", "mvu.sv", "mvu_dsp.sv"]
        elif any(pattern in name_lower for pattern in ["vvu", "vvau"]):
            return ["vvu_axi.sv", "vvu.sv", "vvu_dsp.sv"]
        elif any(pattern in name_lower for pattern in ["threshold"]):
            return ["thresholding_axi.sv", "thresholding.sv"]
        elif any(pattern in name_lower for pattern in ["fifo"]):
            return ["fifo_axi.sv", "Q_srl.v"]
        elif any(pattern in name_lower for pattern in ["dwc", "datawidth"]):
            return ["dwc_axi.sv", "dwc.sv"]
        elif any(pattern in name_lower for pattern in ["memstream"]):
            return ["memstream_axi.sv", "memstream.sv"]
        
        # Default: try to find files with similar names
        return [f"{name_lower}_axi.sv", f"{name_lower}.sv"]
    
    
    def _extract_operation_description(self, context: TemplateContext) -> str:
        """
        Extract operation description from RTL comments or pragmas.
        
        Returns a brief description of what the operation does.
        """
        module_name_lower = context.module_name.lower()
        
        # Default descriptions based on operation patterns
        if any(pattern in module_name_lower for pattern in ["mvu", "mvau"]):
            return "Matrix-Vector multiplication with activation unit for neural network layers."
        elif any(pattern in module_name_lower for pattern in ["vvu", "vvau"]):
            return "Vector-Vector multiplication with activation unit for element-wise operations."
        elif any(pattern in module_name_lower for pattern in ["threshold"]):
            return "Multi-threshold activation function with configurable thresholds and output encoding."
        elif any(pattern in module_name_lower for pattern in ["conv", "input_gen"]):
            return "Convolution input generator with sliding window and padding support."
        elif any(pattern in module_name_lower for pattern in ["pad", "fmpad"]):
            return "Feature map padding operation for convolution boundary handling."
        elif any(pattern in module_name_lower for pattern in ["fifo"]):
            return "Streaming FIFO buffer with configurable depth and AXI-Stream interface."
        elif any(pattern in module_name_lower for pattern in ["dwc", "datawidth"]):
            return "Data width converter for AXI-Stream width adaptation."
        elif any(pattern in module_name_lower for pattern in ["memstream"]):
            return "Memory streaming interface with configurable access patterns."
        
        return f"Hardware accelerator operation: {context.module_name}"