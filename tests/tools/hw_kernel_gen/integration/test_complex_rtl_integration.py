############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Comprehensive integration tests for complex SystemVerilog files
############################################################################

import pytest
from pathlib import Path

from brainsmith.tools.hw_kernel_gen.rtl_parser.parser import RTLParser
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata
from brainsmith.dataflow.core.interface_types import InterfaceType


class TestComplexRTLIntegration:
    """Integration tests with complex SystemVerilog files."""
    
    def test_thresholding_axi_file_parsing(self):
        """Test parsing of thresholding_axi.sv file if available."""
        parser = RTLParser(debug=True)
        
        # Complex thresholding module with multiple interfaces and BDIM pragmas
        systemverilog_code = """
// @brainsmith datatype in0_V_data_V INT 8 8  
// @brainsmith weight weights_V_data_V
// @brainsmith bdim in0_V_data_V [N,PE] RINDEX=0
// @brainsmith bdim weights_V_data_V [PE] RINDEX=0
// @brainsmith bdim out0_V_data_V [N] RINDEX=0
module thresholding_axi #(
    parameter N = 8,
    parameter BIAS = 0,
    parameter PE = 1
)(
    input ap_clk,
    input ap_rst_n,
    
    // Input stream
    input [31:0] in0_V_data_V_TDATA,
    input in0_V_data_V_TVALID,
    output in0_V_data_V_TREADY,
    
    // Weight stream
    input [7:0] weights_V_data_V_TDATA,
    input weights_V_data_V_TVALID,
    output weights_V_data_V_TREADY,
    
    // Output stream
    output [31:0] out0_V_data_V_TDATA,
    output out0_V_data_V_TVALID,
    input out0_V_data_V_TREADY,
    
    // AXI-Lite configuration
    input [11:0] s_axi_control_AWADDR,
    input s_axi_control_AWVALID,
    output s_axi_control_AWREADY,
    input [31:0] s_axi_control_WDATA,
    input [3:0] s_axi_control_WSTRB,
    input s_axi_control_WVALID,
    output s_axi_control_WREADY,
    output [1:0] s_axi_control_BRESP,
    output s_axi_control_BVALID,
    input s_axi_control_BREADY,
    input [11:0] s_axi_control_ARADDR,
    input s_axi_control_ARVALID,
    output s_axi_control_ARREADY,
    output [31:0] s_axi_control_RDATA,
    output [1:0] s_axi_control_RRESP,
    output s_axi_control_RVALID,
    input s_axi_control_RREADY
);
    // Module implementation...
endmodule
"""
        
        try:
            kernel_metadata = parser.parse(systemverilog_code, source_name="thresholding_axi_test")
            
            # Verify basic structure
            assert kernel_metadata.name == "thresholding_axi"
            assert len(kernel_metadata.parameters) == 3  # N, BIAS, PE
            assert len(kernel_metadata.pragmas) == 5  # DATATYPE, WEIGHT, 3 BDIM
            assert len(kernel_metadata.interfaces) >= 4  # Control, Input, Weight, Output, Config
            
            # Verify parameter extraction
            param_names = {param.name for param in kernel_metadata.parameters}
            assert "N" in param_names
            assert "BIAS" in param_names  
            assert "PE" in param_names
            
            # Verify interface types
            interface_types = {iface.interface_type for iface in kernel_metadata.interfaces}
            assert InterfaceType.CONTROL in interface_types
            assert InterfaceType.INPUT in interface_types
            assert InterfaceType.WEIGHT in interface_types  # Should be overridden by pragma
            assert InterfaceType.OUTPUT in interface_types
            assert InterfaceType.CONFIG in interface_types
            
            # Verify pragma application
            weight_interface = next((iface for iface in kernel_metadata.interfaces 
                                   if iface.interface_type == InterfaceType.WEIGHT), None)
            assert weight_interface is not None, "WEIGHT pragma should create weight interface"
            assert "weights" in weight_interface.name
            
            # Verify datatype pragma application
            input_interface = next((iface for iface in kernel_metadata.interfaces 
                                  if "in0_V_data_V" in iface.name), None)
            assert input_interface is not None, "Should find input interface"
            
            # Check for INT8 datatype from pragma
            has_int8 = any(dt.finn_type == "INT8" and dt.signed for dt in input_interface.allowed_datatypes)
            assert has_int8, "DATATYPE pragma should apply INT8 to input interface"
            
            # Verify BDIM pragma effects on chunking strategies
            interfaces_with_chunking = [
                iface for iface in kernel_metadata.interfaces 
                if hasattr(iface.chunking_strategy, 'block_shape') and 
                   any(param in iface.chunking_strategy.block_shape for param in ["N", "PE"])
            ]
            assert len(interfaces_with_chunking) >= 3, "BDIM pragmas should create block chunking strategies"
            
            # Verify specific chunking parameter usage
            chunking_params_used = set()
            for iface in kernel_metadata.interfaces:
                if hasattr(iface.chunking_strategy, 'block_shape'):
                    chunking_params_used.update(iface.chunking_strategy.block_shape)
            
            assert "N" in chunking_params_used, "N parameter should be used in block chunking"
            assert "PE" in chunking_params_used, "PE parameter should be used in block chunking"
            
            print(f"✅ Complex thresholding file parsed successfully:")
            print(f"  - Module: {kernel_metadata.name}")
            print(f"  - Parameters: {len(kernel_metadata.parameters)}")
            print(f"  - Interfaces: {len(kernel_metadata.interfaces)}")
            print(f"  - Pragmas: {len(kernel_metadata.pragmas)}")
            print(f"  - Custom chunking interfaces: {len(interfaces_with_chunking)}")
            print(f"  - Chunking params: {sorted(chunking_params_used - {':'})}")
            
        except Exception as e:
            pytest.fail(f"Complex RTL parsing failed: {e}")
    
    def test_multiple_axi_stream_interfaces(self):
        """Test parsing module with multiple AXI-Stream interfaces."""
        parser = RTLParser(debug=False)
        
        systemverilog_code = """
// @brainsmith bdim s_axis_0 [CHANNELS] RINDEX=0
// @brainsmith bdim s_axis_1 [WIDTH] RINDEX=0
// @brainsmith bdim m_axis_0 [CHANNELS,WIDTH] RINDEX=1
// @brainsmith bdim m_axis_1 [:] RINDEX=0
module multi_stream_processor #(
    parameter CHANNELS = 16,
    parameter WIDTH = 32
)(
    input ap_clk,
    input ap_rst_n,
    
    // First input stream
    input [31:0] s_axis_0_tdata,
    input s_axis_0_tvalid,
    output s_axis_0_tready,
    
    // Second input stream  
    input [31:0] s_axis_1_tdata,
    input s_axis_1_tvalid,
    output s_axis_1_tready,
    
    // First output stream
    output [31:0] m_axis_0_tdata,
    output m_axis_0_tvalid,
    input m_axis_0_tready,
    
    // Second output stream
    output [31:0] m_axis_1_tdata,
    output m_axis_1_tvalid,
    input m_axis_1_tready
);
    // Implementation...
endmodule
"""
        
        kernel_metadata = parser.parse(systemverilog_code, source_name="multi_stream_test")
        
        # Verify multiple streams are detected
        input_interfaces = [iface for iface in kernel_metadata.interfaces 
                          if iface.interface_type == InterfaceType.INPUT]
        output_interfaces = [iface for iface in kernel_metadata.interfaces 
                           if iface.interface_type == InterfaceType.OUTPUT]
        
        assert len(input_interfaces) == 2, f"Expected 2 input interfaces, got {len(input_interfaces)}"
        assert len(output_interfaces) == 2, f"Expected 2 output interfaces, got {len(output_interfaces)}"
        
        # Verify interface names are distinct
        input_names = {iface.name for iface in input_interfaces}
        output_names = {iface.name for iface in output_interfaces}
        
        assert len(input_names) == 2, "Input interface names should be unique"
        assert len(output_names) == 2, "Output interface names should be unique"
        
        # Verify BDIM pragma applications for different interface chunking patterns
        chunking_strategies = {}
        for iface in kernel_metadata.interfaces:
            if hasattr(iface.chunking_strategy, 'block_shape'):
                chunking_strategies[iface.name] = {
                    'block_shape': iface.chunking_strategy.block_shape,
                    'rindex': iface.chunking_strategy.rindex
                }
        
        # Should have at least 4 interfaces with custom chunking (from BDIM pragmas)
        assert len(chunking_strategies) >= 4, "BDIM pragmas should create chunking strategies"
        
        # Verify parameter usage in chunking
        all_chunking_params = set()
        for strategy in chunking_strategies.values():
            all_chunking_params.update(param for param in strategy['block_shape'] if param != ":")
        
        assert "CHANNELS" in all_chunking_params, "CHANNELS parameter should be used in chunking"
        assert "WIDTH" in all_chunking_params, "WIDTH parameter should be used in chunking"
        
        print(f"✅ Multiple stream interfaces detected:")
        print(f"  - Input interfaces: {input_names}")
        print(f"  - Output interfaces: {output_names}")
        print(f"  - Chunking strategies: {len(chunking_strategies)}")
        print(f"  - Chunking params used: {sorted(all_chunking_params)}")
    
    def test_edge_case_interface_patterns(self):
        """Test edge cases in interface pattern recognition."""
        parser = RTLParser(debug=False)
        
        systemverilog_code = """
// @brainsmith bdim data_in [SIZE] RINDEX=0
// @brainsmith bdim result [:] RINDEX=0
module edge_case_module #(
    parameter SIZE = 64
)(
    input ap_clk,
    input ap_rst_n,
    
    // Non-standard AXI naming
    input [63:0] data_in_TDATA,
    input data_in_TVALID,
    output data_in_TREADY,
    
    // Minimal AXI-Lite
    input [7:0] cfg_awaddr,
    input cfg_awvalid,
    output cfg_awready,
    input [31:0] cfg_wdata,
    input cfg_wvalid,
    output cfg_wready,
    
    // Single direction AXI-Stream (missing some signals)
    output [31:0] result_tdata,
    output result_tvalid,
    input result_tready,
    
    // Standalone signals (should be unassigned)
    input enable,
    output [7:0] status,
    input [3:0] mode
);
    // Implementation...
endmodule
"""
        
        kernel_metadata = parser.parse(systemverilog_code, source_name="edge_case_test")
        
        # Should detect some interfaces despite non-standard naming
        assert len(kernel_metadata.interfaces) >= 2, "Should detect at least control and some data interfaces"
        
        # Verify control interface
        control_interfaces = [iface for iface in kernel_metadata.interfaces 
                            if iface.interface_type == InterfaceType.CONTROL]
        assert len(control_interfaces) >= 1, "Should detect global control interface"
        
        # Verify BDIM pragmas work with edge case interfaces
        edge_chunking_interfaces = [
            iface for iface in kernel_metadata.interfaces 
            if hasattr(iface.chunking_strategy, 'block_shape') and 
               "SIZE" in iface.chunking_strategy.block_shape
        ]
        
        # Print detected interfaces for analysis
        print(f"✅ Edge case parsing completed:")
        for iface in kernel_metadata.interfaces:
            chunking_info = ""
            if hasattr(iface.chunking_strategy, 'block_shape'):
                chunking_info = f" [chunking: {iface.chunking_strategy.block_shape}]"
            print(f"  - {iface.name}: {iface.interface_type.value}{chunking_info}")
        print(f"  - SIZE parameter used in chunking: {len(edge_chunking_interfaces)} interfaces")
    
    def test_performance_with_large_module(self):
        """Test performance with a module containing many interfaces."""
        parser = RTLParser(debug=False)
        
        # Generate a module with many interfaces programmatically
        ports = []
        ports.extend([
            "    input ap_clk,",
            "    input ap_rst_n,"
        ])
        
        # Generate BDIM pragmas for multiple interfaces
        bdim_pragmas = []
        for i in range(4):  # Add pragmas for first 4 interfaces to test performance
            bdim_pragmas.append(f"// @brainsmith bdim s_axis_{i} [NUM_CHANNELS] RINDEX=0")
            bdim_pragmas.append(f"// @brainsmith bdim m_axis_{i} [DATA_WIDTH] RINDEX=0")
        
        # Generate multiple AXI-Stream interfaces
        for i in range(8):
            ports.extend([
                f"    input [31:0] s_axis_{i}_tdata,",
                f"    input s_axis_{i}_tvalid,",
                f"    output s_axis_{i}_tready,"
            ])
        
        for i in range(8):
            ports.extend([
                f"    output [31:0] m_axis_{i}_tdata,",
                f"    output m_axis_{i}_tvalid,",
                f"    input m_axis_{i}_tready,"
            ])
        
        # Remove trailing comma from last port
        if ports:
            ports[-1] = ports[-1].rstrip(',')
        
        # Add pragmas to the code string after generating ports
        
        systemverilog_code = f"""
{chr(10).join(bdim_pragmas)}
module large_interface_module #(
    parameter NUM_CHANNELS = 128,
    parameter DATA_WIDTH = 32,
    parameter FIFO_DEPTH = 512
)(
{chr(10).join(ports)}
);
    // Large module implementation...
endmodule
"""
        
        import time
        start_time = time.time()
        
        kernel_metadata = parser.parse(systemverilog_code, source_name="large_module_test")
        
        end_time = time.time()
        parse_time = end_time - start_time
        
        # Verify parsing completed successfully
        assert kernel_metadata.name == "large_interface_module"
        assert len(kernel_metadata.interfaces) >= 16, "Should detect many interfaces"
        
        # Performance should be reasonable (under 2 seconds for this size)
        assert parse_time < 2.0, f"Parsing took too long: {parse_time:.2f}s"
        
        # Verify BDIM pragmas were processed for performance test
        chunked_interfaces = [
            iface for iface in kernel_metadata.interfaces 
            if hasattr(iface.chunking_strategy, 'block_shape') and 
               any(param in iface.chunking_strategy.block_shape 
                   for param in ["NUM_CHANNELS", "DATA_WIDTH"])
        ]
        
        # Should have some chunked interfaces from the BDIM pragmas we added
        assert len(chunked_interfaces) >= 2, "BDIM pragmas should be processed even in large modules"
        
        print(f"✅ Large module performance test:")
        print(f"  - Interfaces detected: {len(kernel_metadata.interfaces)}")
        print(f"  - Parse time: {parse_time:.3f}s")
        print(f"  - Parameters: {len(kernel_metadata.parameters)}")
        print(f"  - Chunked interfaces: {len(chunked_interfaces)}")
        print(f"  - Total pragmas: {len(kernel_metadata.pragmas)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])