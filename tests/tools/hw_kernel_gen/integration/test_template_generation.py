############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# End-to-end template generation tests
############################################################################

import pytest
from pathlib import Path
import tempfile
import os

from brainsmith.tools.hw_kernel_gen.rtl_parser.parser import RTLParser
from brainsmith.tools.hw_kernel_gen.templates.context_generator import TemplateContextGenerator
from brainsmith.dataflow.core.interface_types import InterfaceType


class TestTemplateGeneration:
    """End-to-end template generation tests."""
    
    def test_template_context_generation(self):
        """Test template context generation from KernelMetadata."""
        parser = RTLParser(debug=False)
        
        # Use a realistic module with pragmas including BDIM
        systemverilog_code = """
// @brainsmith datatype in0_V_data_V INT 8 8
// @brainsmith weight weights_V_data_V
// @brainsmith bdim in0_V_data_V [PE]
// @brainsmith bdim weights_V_data_V [PE] RINDEX=0
// @brainsmith bdim out0_V_data_V [CHANNELS,PE] RINDEX=1
module test_kernel #(
    parameter PE = 4,
    parameter CHANNELS = 16
)(
    input ap_clk,
    input ap_rst_n,
    
    input [31:0] in0_V_data_V_TDATA,
    input in0_V_data_V_TVALID,
    output in0_V_data_V_TREADY,
    
    input [7:0] weights_V_data_V_TDATA,
    input weights_V_data_V_TVALID,
    output weights_V_data_V_TREADY,
    
    output [31:0] out0_V_data_V_TDATA,
    output out0_V_data_V_TVALID,
    input out0_V_data_V_TREADY
);
    // Module implementation...
endmodule
"""
        
        # Parse the SystemVerilog
        kernel_metadata = parser.parse(systemverilog_code, source_name="test_kernel")
        
        # Generate template context
        context = TemplateContextGenerator.generate_context(kernel_metadata)
        
        # Verify basic context structure
        assert "kernel_name" in context
        assert "class_name" in context
        assert "interface_metadata" in context
        assert "rtl_parameters" in context
        assert "node_attributes" in context
        
        # Verify BDIM pragma effects on interface metadata
        interfaces_with_custom_chunking = [
            iface for iface in kernel_metadata.interfaces 
            if hasattr(iface.chunking_strategy, 'block_shape') and 
               iface.chunking_strategy.block_shape != [":"]
        ]
        assert len(interfaces_with_custom_chunking) >= 3, "BDIM pragmas should create custom chunking strategies"
        
        # Verify kernel information
        assert context["kernel_name"] == "test_kernel"
        assert context["class_name"] == "TestKernel"  # PascalCase conversion
        
        # Verify interface categorization
        assert "input_interfaces" in context
        assert "output_interfaces" in context
        assert "weight_interfaces" in context
        assert "control_interfaces" in context
        
        # Check interface counts
        assert len(context["input_interfaces"]) >= 1  # Should have input interface
        assert len(context["output_interfaces"]) >= 1  # Should have output interface
        assert len(context["weight_interfaces"]) >= 1  # Should have weight interface from pragma
        assert len(context["control_interfaces"]) >= 1  # Should have control interface
        
        # Verify RTL parameters (used in BDIM pragmas)
        assert len(context["rtl_parameters"]) == 2  # PE and CHANNELS
        param_names = {param["name"] for param in context["rtl_parameters"]}
        assert "PE" in param_names
        assert "CHANNELS" in param_names
        
        # Verify block chunking strategies reference these parameters
        chunking_params_used = set()
        for iface in kernel_metadata.interfaces:
            if hasattr(iface.chunking_strategy, 'block_shape'):
                for shape_elem in iface.chunking_strategy.block_shape:
                    if shape_elem != ":" and shape_elem.isidentifier():
                        chunking_params_used.add(shape_elem)
        
        assert "PE" in chunking_params_used, "PE parameter should be used in block chunking"
        assert "CHANNELS" in chunking_params_used, "CHANNELS parameter should be used in block chunking"
        
        # Verify template flags
        assert context["has_inputs"] is True
        assert context["has_outputs"] is True  
        assert context["has_weights"] is True
        
        # Verify node attributes for HWCustomOp generation
        node_attrs = context["node_attributes"]
        assert "PE" in node_attrs
        assert "inputDataType" in node_attrs
        assert "outputDataType" in node_attrs
        assert "weightDataType" in node_attrs
        
        print("✅ Template context generation successful:")
        print(f"  - Kernel: {context['kernel_name']}")
        print(f"  - Interfaces: {len(context['interface_metadata'])}")
        print(f"  - Parameters: {len(context['rtl_parameters'])}")
        print(f"  - Node attributes: {len(node_attrs)}")
        print(f"  - Custom chunking interfaces: {len(interfaces_with_custom_chunking)}")
        print(f"  - Chunking params used: {sorted(chunking_params_used)}")
    
    def test_datatype_mapping_generation(self):
        """Test datatype mapping methods generation."""
        parser = RTLParser(debug=False)
        
        systemverilog_code = """
// @brainsmith datatype in0_V_data_V INT 4 8
// @brainsmith datatype out0_V_data_V UINT 8 16
// @brainsmith bdim in0_V_data_V [WIDTH]
// @brainsmith bdim out0_V_data_V [:] RINDEX=0
module datatype_test #(
    parameter WIDTH = 32
)(
    input ap_clk,
    input ap_rst_n,
    
    input [31:0] in0_V_data_V_TDATA,
    input in0_V_data_V_TVALID,
    output in0_V_data_V_TREADY,
    
    output [31:0] out0_V_data_V_TDATA,
    output out0_V_data_V_TVALID,
    input out0_V_data_V_TREADY
);
endmodule
"""
        
        kernel_metadata = parser.parse(systemverilog_code, source_name="datatype_test")
        context = TemplateContextGenerator.generate_context(kernel_metadata)
        
        # Verify datatype mappings
        datatype_mappings = context["datatype_mappings"]
        assert "input_methods" in datatype_mappings
        assert "output_methods" in datatype_mappings
        
        # Should have methods for each interface
        assert len(datatype_mappings["input_methods"]) >= 1
        assert len(datatype_mappings["output_methods"]) >= 1
        
        # Verify block chunking information is available in context
        chunking_info = context.get("block_chunking_info", {})
        interface_chunking = [iface for iface in kernel_metadata.interfaces 
                             if hasattr(iface.chunking_strategy, 'block_shape')]
        assert len(interface_chunking) >= 2, "BDIM pragmas should create chunking strategies"
        
        # Verify method structure
        input_method = datatype_mappings["input_methods"][0]
        assert "index" in input_method
        assert "interface_name" in input_method
        assert "method_body" in input_method
        
        print("✅ Datatype mapping generation successful:")
        print(f"  - Input methods: {len(datatype_mappings['input_methods'])}")
        print(f"  - Output methods: {len(datatype_mappings['output_methods'])}")
    
    def test_parallelism_analysis(self):
        """Test parallelism parameter analysis."""
        parser = RTLParser(debug=False)
        
        systemverilog_code = """
// @brainsmith bdim in0_V_data_V [SIMD,PE]
// @brainsmith bdim out0_V_data_V [CHANNELS] RINDEX=0
module parallelism_test #(
    parameter PE = 8,
    parameter SIMD = 4,
    parameter CHANNELS = 64
)(
    input ap_clk,
    input ap_rst_n,
    
    input [31:0] in0_V_data_V_TDATA,
    input in0_V_data_V_TVALID,
    output in0_V_data_V_TREADY,
    
    output [31:0] out0_V_data_V_TDATA,
    output out0_V_data_V_TVALID,
    input out0_V_data_V_TREADY
);
endmodule
"""
        
        kernel_metadata = parser.parse(systemverilog_code, source_name="parallelism_test")
        context = TemplateContextGenerator.generate_context(kernel_metadata)
        
        # Verify parallelism analysis
        parallelism_info = context["parallelism_info"]
        assert "inferred_pe" in parallelism_info
        assert "inferred_simd" in parallelism_info
        assert "inferred_channels" in parallelism_info
        
        # Should detect PE parameter
        assert parallelism_info["inferred_pe"] == 8
        assert parallelism_info["inferred_simd"] == 4
        assert parallelism_info["inferred_channels"] == 64
        
        # Verify parallelism parameters are used in block chunking
        block_chunking_params = set()
        for iface in kernel_metadata.interfaces:
            if hasattr(iface.chunking_strategy, 'block_shape'):
                for shape_elem in iface.chunking_strategy.block_shape:
                    if shape_elem in ["PE", "SIMD", "CHANNELS"]:
                        block_chunking_params.add(shape_elem)
        
        assert len(block_chunking_params) >= 2, "Parallelism parameters should be used in block chunking"
        
        print("✅ Parallelism analysis successful:")
        print(f"  - PE: {parallelism_info['inferred_pe']}")
        print(f"  - SIMD: {parallelism_info['inferred_simd']}")
        print(f"  - Channels: {parallelism_info['inferred_channels']}")
    
    def test_algorithm_parameter_inference(self):
        """Test algorithm parameter inference."""
        parser = RTLParser(debug=False)
        
        systemverilog_code = """
// @brainsmith bdim in0_V_data_V [N] RINDEX=0
// @brainsmith bdim out0_V_data_V [:] RINDEX=0
module thresholding_kernel #(
    parameter N = 16,
    parameter BIAS = 128,
    parameter SIGNED = 1
)(
    input ap_clk,
    input ap_rst_n,
    
    input [31:0] in0_V_data_V_TDATA,
    input in0_V_data_V_TVALID,
    output in0_V_data_V_TREADY,
    
    output [31:0] out0_V_data_V_TDATA,
    output out0_V_data_V_TVALID,
    input out0_V_data_V_TREADY
);
endmodule
"""
        
        kernel_metadata = parser.parse(systemverilog_code, source_name="thresholding_kernel")
        context = TemplateContextGenerator.generate_context(kernel_metadata)
        
        # Verify algorithm inference
        algorithm_info = context["algorithm_info"]
        assert "type" in algorithm_info
        assert "parameters" in algorithm_info
        
        # Should infer thresholding algorithm from name
        assert algorithm_info["type"] == "threshold"
        
        # Should map parameters correctly
        params = algorithm_info["parameters"]
        assert "numSteps" in params
        assert "ActVal" in params
        assert "signed_input" in params
        
        assert params["numSteps"] == 16  # From N parameter
        assert params["ActVal"] == 128   # From BIAS parameter
        assert params["signed_input"] is True  # From SIGNED parameter
        
        # Verify algorithm parameters are used in block chunking
        algo_chunking_interfaces = [
            iface for iface in kernel_metadata.interfaces 
            if hasattr(iface.chunking_strategy, 'block_shape') and 
               any(elem == "N" for elem in iface.chunking_strategy.block_shape)
        ]
        assert len(algo_chunking_interfaces) >= 1, "Algorithm parameter N should be used in chunking"
        
        print("✅ Algorithm parameter inference successful:")
        print(f"  - Type: {algorithm_info['type']}")
        print(f"  - Parameters: {len(params)}")
    
    def test_resource_estimation_methods(self):
        """Test resource estimation method generation."""
        parser = RTLParser(debug=False)
        
        systemverilog_code = """
// @brainsmith bdim in0_V_data_V [PE]
// @brainsmith bdim out0_V_data_V [PE]
module resource_test #(
    parameter PE = 4
)(
    input ap_clk,
    input ap_rst_n,
    
    input [31:0] in0_V_data_V_TDATA,
    input in0_V_data_V_TVALID,
    output in0_V_data_V_TREADY,
    
    output [31:0] out0_V_data_V_TDATA,
    output out0_V_data_V_TVALID,
    input out0_V_data_V_TREADY
);
endmodule
"""
        
        kernel_metadata = parser.parse(systemverilog_code, source_name="resource_test")
        context = TemplateContextGenerator.generate_context(kernel_metadata)
        
        # Verify resource estimation methods
        resource_methods = context["resource_estimation_methods"]
        assert "get_exp_cycles" in resource_methods
        assert "bram_estimation" in resource_methods
        assert "lut_estimation" in resource_methods
        assert "dsp_estimation" in resource_methods
        
        # Methods should be non-empty strings
        assert resource_methods["get_exp_cycles"]
        assert resource_methods["bram_estimation"]
        assert resource_methods["lut_estimation"]
        assert resource_methods["dsp_estimation"]
        
        # Verify resource methods account for block chunking
        pe_chunking_interfaces = [
            iface for iface in kernel_metadata.interfaces 
            if hasattr(iface.chunking_strategy, 'block_shape') and 
               "PE" in iface.chunking_strategy.block_shape
        ]
        assert len(pe_chunking_interfaces) >= 2, "PE parameter should be used in block chunking for resource estimation"
        
        print("✅ Resource estimation methods generated successfully")
    
    def test_stream_width_methods(self):
        """Test stream width calculation method generation."""
        parser = RTLParser(debug=False)
        
        systemverilog_code = """
// @brainsmith weight weights_V_data_V
// @brainsmith bdim in0_V_data_V [PE]
// @brainsmith bdim weights_V_data_V [PE] RINDEX=0
// @brainsmith bdim out0_V_data_V [PE]
module stream_width_test #(
    parameter PE = 8
)(
    input ap_clk,
    input ap_rst_n,
    
    input [31:0] in0_V_data_V_TDATA,
    input in0_V_data_V_TVALID,
    output in0_V_data_V_TREADY,
    
    input [7:0] weights_V_data_V_TDATA,
    input weights_V_data_V_TVALID,
    output weights_V_data_V_TREADY,
    
    output [31:0] out0_V_data_V_TDATA,
    output out0_V_data_V_TVALID,
    input out0_V_data_V_TREADY
);
endmodule
"""
        
        kernel_metadata = parser.parse(systemverilog_code, source_name="stream_width_test")
        context = TemplateContextGenerator.generate_context(kernel_metadata)
        
        # Verify stream width methods
        stream_methods = context["stream_width_methods"]
        assert "instream_width" in stream_methods
        assert "outstream_width" in stream_methods
        assert "weightstream_width" in stream_methods  # Should exist due to weight interface
        
        # Methods should reference PE parameter
        assert "PE" in stream_methods["instream_width"]
        assert "PE" in stream_methods["outstream_width"]
        assert "pe" in stream_methods["weightstream_width"]
        
        # Verify stream width methods consider block chunking
        pe_chunking_count = sum(1 for iface in kernel_metadata.interfaces 
                               if hasattr(iface.chunking_strategy, 'block_shape') and 
                                  "PE" in iface.chunking_strategy.block_shape)
        assert pe_chunking_count >= 3, "PE parameter should be used in block chunking for all stream interfaces"
        
        print("✅ Stream width methods generated successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])