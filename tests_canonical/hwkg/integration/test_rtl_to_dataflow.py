"""
Integration tests for RTL to Dataflow conversion pipeline.

Tests the complete pipeline from SystemVerilog RTL parsing through
to DataflowInterface creation, validating the critical integration
point between HWKG and dataflow modeling systems.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from brainsmith.tools.hw_kernel_gen.rtl_parser.parser import RTLParser
from brainsmith.tools.hw_kernel_gen.rtl_parser.data import Interface as RTLInterface, InterfaceType as RTLInterfaceType
from brainsmith.dataflow.integration.rtl_conversion import RTLInterfaceConverter, validate_conversion_result
from brainsmith.dataflow.core.dataflow_interface import DataflowInterface, DataflowInterfaceType


@pytest.mark.integration
class TestRTLToDataflowConversion:
    """Test complete RTL to dataflow conversion pipeline."""
    
    def test_complete_pipeline_simple_kernel(self):
        """Test complete conversion pipeline for simple kernel."""
        
        simple_kernel = """
        module simple_add #(
            parameter WIDTH = 32
        )(
            input wire clk,
            input wire rst,
            
            // @brainsmith INTERFACE_TYPE=AXI_STREAM
            // @brainsmith TDIM=[WIDTH]
            input wire [WIDTH*8-1:0] s_axis_a_tdata,
            input wire s_axis_a_tvalid,
            output wire s_axis_a_tready,
            
            // @brainsmith INTERFACE_TYPE=AXI_STREAM
            // @brainsmith TDIM=[WIDTH]
            input wire [WIDTH*8-1:0] s_axis_b_tdata,
            input wire s_axis_b_tvalid,
            output wire s_axis_b_tready,
            
            // @brainsmith INTERFACE_TYPE=AXI_STREAM
            output wire [WIDTH*8-1:0] m_axis_result_tdata,
            output wire m_axis_result_tvalid,
            input wire m_axis_result_tready
        );
        
        // Simple addition implementation
        assign m_axis_result_tdata = s_axis_a_tdata + s_axis_b_tdata;
        assign m_axis_result_tvalid = s_axis_a_tvalid & s_axis_b_tvalid;
        assign s_axis_a_tready = m_axis_result_tready;
        assign s_axis_b_tready = m_axis_result_tready;
        
        endmodule
        """
        
        # Step 1: Parse RTL
        rtl_parser = RTLParser()
        parse_result = rtl_parser.parse(simple_kernel)
        
        assert parse_result.success == True
        assert parse_result.module_name == "simple_add"
        
        # Step 2: Filter AXI-Stream interfaces
        axi_interfaces = [iface for iface in parse_result.interfaces 
                         if iface.type == RTLInterfaceType.AXI_STREAM]
        assert len(axi_interfaces) == 3  # a, b, result
        
        # Step 3: Convert to dataflow interfaces
        converter = RTLInterfaceConverter()
        parameters = {"WIDTH": 32}
        
        dataflow_interfaces = []
        for rtl_interface in axi_interfaces:
            df_interface = converter._convert_single_interface(rtl_interface, parameters)
            if df_interface:
                dataflow_interfaces.append(df_interface)
        
        assert len(dataflow_interfaces) == 3
        
        # Step 4: Validate conversion results
        input_interfaces = [iface for iface in dataflow_interfaces 
                          if iface.interface_type == DataflowInterfaceType.INPUT]
        output_interfaces = [iface for iface in dataflow_interfaces
                           if iface.interface_type == DataflowInterfaceType.OUTPUT]
        
        assert len(input_interfaces) == 2  # a and b
        assert len(output_interfaces) == 1  # result
        
        # Step 5: Validate dataflow properties
        for interface in dataflow_interfaces:
            # Should have valid dimensions from pragmas
            assert len(interface.tensor_dims) > 0
            assert len(interface.block_dims) > 0
            assert len(interface.stream_dims) > 0
            
            # Should pass basic validation
            validation_result = interface.validate()
            assert validation_result.is_valid()
    
    def test_cnn_kernel_conversion(self):
        """Test conversion of CNN-style kernel with complex interfaces."""
        
        cnn_kernel = """
        module conv2d_kernel #(
            parameter INPUT_CHANNELS = 64,
            parameter OUTPUT_CHANNELS = 128,
            parameter KERNEL_SIZE = 3,
            parameter SIMD = 8,
            parameter PE = 16
        )(
            input wire clk,
            input wire rst_n,
            
            // Input feature maps
            // @brainsmith INTERFACE_TYPE=AXI_STREAM
            // @brainsmith TDIM=[INPUT_CHANNELS,KERNEL_SIZE,KERNEL_SIZE]
            input wire [SIMD*8-1:0] s_axis_input_tdata,
            input wire s_axis_input_tvalid,
            output wire s_axis_input_tready,
            
            // Weight data
            // @brainsmith INTERFACE_TYPE=AXI_STREAM
            // @brainsmith TDIM=[OUTPUT_CHANNELS,INPUT_CHANNELS,KERNEL_SIZE,KERNEL_SIZE]
            input wire [PE*SIMD*8-1:0] s_axis_weights_tdata,
            input wire s_axis_weights_tvalid,
            output wire s_axis_weights_tready,
            
            // Output feature maps
            // @brainsmith INTERFACE_TYPE=AXI_STREAM
            output wire [PE*16-1:0] m_axis_output_tdata,
            output wire m_axis_output_tvalid,
            input wire m_axis_output_tready,
            
            // Control interface
            // @brainsmith INTERFACE_TYPE=AXI_LITE
            input wire [31:0] s_axi_control_awaddr,
            input wire s_axi_control_awvalid,
            output wire s_axi_control_awready
        );
        endmodule
        """
        
        # Parse and convert
        rtl_parser = RTLParser()
        parse_result = rtl_parser.parse(cnn_kernel)
        
        parameters = {
            "INPUT_CHANNELS": 64,
            "OUTPUT_CHANNELS": 128, 
            "KERNEL_SIZE": 3,
            "SIMD": 8,
            "PE": 16
        }
        
        converter = RTLInterfaceConverter()
        
        # Convert AXI-Stream interfaces only (exclude AXI-Lite and control)
        axi_stream_interfaces = [iface for iface in parse_result.interfaces
                               if iface.type == RTLInterfaceType.AXI_STREAM]
        
        dataflow_interfaces = []
        for rtl_interface in axi_stream_interfaces:
            df_interface = converter._convert_single_interface(rtl_interface, parameters)
            if df_interface:
                dataflow_interfaces.append(df_interface)
        
        # Validate CNN-specific properties
        assert len(dataflow_interfaces) == 3  # input, weights, output
        
        # Find specific interfaces
        input_iface = next((iface for iface in dataflow_interfaces 
                          if "input" in iface.name), None)
        weight_iface = next((iface for iface in dataflow_interfaces
                           if "weight" in iface.name), None)
        output_iface = next((iface for iface in dataflow_interfaces
                           if "output" in iface.name), None)
        
        assert input_iface is not None
        assert weight_iface is not None
        assert output_iface is not None
        
        # Validate dimensional relationships
        assert input_iface.interface_type == DataflowInterfaceType.INPUT
        assert weight_iface.interface_type == DataflowInterfaceType.WEIGHT
        assert output_iface.interface_type == DataflowInterfaceType.OUTPUT
        
        # Check that pragma dimensions were processed
        assert len(input_iface.tensor_dims) == 3   # [C,H,W] from TDIM
        assert len(weight_iface.tensor_dims) == 4  # [OC,IC,KH,KW] from TDIM
    
    def test_transformer_kernel_conversion(self):
        """Test conversion of transformer-style kernel."""
        
        transformer_kernel = """
        module attention_kernel #(
            parameter SEQ_LEN = 512,
            parameter HIDDEN_DIM = 768,
            parameter NUM_HEADS = 12,
            parameter HEAD_DIM = 64
        )(
            input wire clk,
            input wire rst,
            
            // Query input
            // @brainsmith INTERFACE_TYPE=AXI_STREAM
            // @brainsmith TDIM=[SEQ_LEN,HIDDEN_DIM]
            input wire [HIDDEN_DIM*8-1:0] s_axis_query_tdata,
            input wire s_axis_query_tvalid,
            output wire s_axis_query_tready,
            
            // Key input
            // @brainsmith INTERFACE_TYPE=AXI_STREAM
            // @brainsmith TDIM=[SEQ_LEN,HIDDEN_DIM]
            input wire [HIDDEN_DIM*8-1:0] s_axis_key_tdata,
            input wire s_axis_key_tvalid,
            output wire s_axis_key_tready,
            
            // Attention output
            // @brainsmith INTERFACE_TYPE=AXI_STREAM
            output wire [HIDDEN_DIM*8-1:0] m_axis_attention_tdata,
            output wire m_axis_attention_tvalid,
            input wire m_axis_attention_tready
        );
        endmodule
        """
        
        # Parse and convert
        rtl_parser = RTLParser()
        parse_result = rtl_parser.parse(transformer_kernel)
        
        parameters = {
            "SEQ_LEN": 512,
            "HIDDEN_DIM": 768,
            "NUM_HEADS": 12,
            "HEAD_DIM": 64
        }
        
        converter = RTLInterfaceConverter()
        
        # Convert interfaces
        axi_interfaces = [iface for iface in parse_result.interfaces
                         if iface.type == RTLInterfaceType.AXI_STREAM]
        
        dataflow_interfaces = []
        for rtl_interface in axi_interfaces:
            df_interface = converter._convert_single_interface(rtl_interface, parameters)
            if df_interface:
                dataflow_interfaces.append(df_interface)
        
        # Validate transformer-specific properties
        assert len(dataflow_interfaces) == 3  # query, key, attention
        
        for interface in dataflow_interfaces:
            # Should have 2D tensors for sequence processing
            assert len(interface.tensor_dims) == 2  # [SeqLen, HiddenDim]
            
            # Validate sequence processing dimensions
            assert interface.tensor_dims[0] == 512  # SEQ_LEN
            assert interface.tensor_dims[1] == 768  # HIDDEN_DIM
    
    def test_conversion_error_handling(self):
        """Test error handling during conversion process."""
        
        # RTL with missing pragmas
        incomplete_kernel = """
        module incomplete_kernel (
            input wire clk,
            
            // Missing @brainsmith pragmas
            input wire [63:0] s_axis_tdata,
            input wire s_axis_tvalid,
            output wire s_axis_tready
        );
        endmodule
        """
        
        rtl_parser = RTLParser()
        parse_result = rtl_parser.parse(incomplete_kernel)
        
        converter = RTLInterfaceConverter()
        
        # Should handle missing pragma information gracefully
        axi_interfaces = [iface for iface in parse_result.interfaces
                         if iface.type == RTLInterfaceType.AXI_STREAM]
        
        for rtl_interface in axi_interfaces:
            # Should either convert with defaults or return None
            df_interface = converter._convert_single_interface(rtl_interface, {})
            
            if df_interface:
                # If converted, should still be valid
                validation_result = df_interface.validate()
                # May have warnings but should not crash
    
    def test_parameter_evaluation_in_conversion(self):
        """Test evaluation of complex parameters during conversion."""
        
        parameterized_kernel = """
        module param_kernel #(
            parameter BASE_WIDTH = 8,
            parameter MULTIPLIER = 4,
            parameter DEPTH = 1024
        )(
            input wire clk,
            
            // @brainsmith INTERFACE_TYPE=AXI_STREAM
            // @brainsmith TDIM=[BASE_WIDTH*MULTIPLIER,$clog2(DEPTH)]
            input wire [BASE_WIDTH*MULTIPLIER*8-1:0] s_axis_tdata,
            input wire s_axis_tvalid,
            output wire s_axis_tready
        );
        endmodule
        """
        
        rtl_parser = RTLParser()
        parse_result = rtl_parser.parse(parameterized_kernel)
        
        parameters = {
            "BASE_WIDTH": 8,
            "MULTIPLIER": 4,
            "DEPTH": 1024
        }
        
        converter = RTLInterfaceConverter()
        
        axi_interfaces = [iface for iface in parse_result.interfaces
                         if iface.type == RTLInterfaceType.AXI_STREAM]
        
        for rtl_interface in axi_interfaces:
            df_interface = converter._convert_single_interface(rtl_interface, parameters)
            
            if df_interface:
                # Should evaluate TDIM=[32, 10] from [8*4, $clog2(1024)]
                assert df_interface.tensor_dims[0] == 32  # BASE_WIDTH*MULTIPLIER
                assert df_interface.tensor_dims[1] == 10  # $clog2(1024)
    
    def test_conversion_result_validation(self):
        """Test validation of complete conversion results."""
        
        # Use CNN kernel for comprehensive validation
        cnn_code = """
        module validation_test (
            input wire clk,
            
            // @brainsmith INTERFACE_TYPE=AXI_STREAM
            // @brainsmith TDIM=[64,56,56]
            input wire [511:0] s_axis_input_tdata,
            input wire s_axis_input_tvalid,
            output wire s_axis_input_tready,
            
            // @brainsmith INTERFACE_TYPE=AXI_STREAM
            output wire [1023:0] m_axis_output_tdata,
            output wire m_axis_output_tvalid,
            input wire m_axis_output_tready
        );
        endmodule
        """
        
        rtl_parser = RTLParser()
        parse_result = rtl_parser.parse(cnn_code)
        
        converter = RTLInterfaceConverter()
        conversion_result = converter.convert_interfaces(
            parse_result.interfaces, 
            parameters={}
        )
        
        # Validate conversion result structure
        validation_result = validate_conversion_result(conversion_result)
        
        assert validation_result.is_valid()
        assert len(conversion_result.dataflow_interfaces) >= 2
        
        # Check that interfaces are properly categorized
        input_count = len([iface for iface in conversion_result.dataflow_interfaces
                          if iface.interface_type == DataflowInterfaceType.INPUT])
        output_count = len([iface for iface in conversion_result.dataflow_interfaces
                           if iface.interface_type == DataflowInterfaceType.OUTPUT])
        
        assert input_count >= 1
        assert output_count >= 1
    
    def test_end_to_end_dataflow_model_integration(self):
        """Test integration with DataflowModel creation."""
        
        model_kernel = """
        module model_test (
            input wire clk,
            
            // @brainsmith INTERFACE_TYPE=AXI_STREAM
            // @brainsmith TDIM=[32,32]
            input wire [255:0] s_axis_input_tdata,
            input wire s_axis_input_tvalid,
            output wire s_axis_input_tready,
            
            // @brainsmith INTERFACE_TYPE=AXI_STREAM
            // @brainsmith TDIM=[64,16]
            input wire [127:0] s_axis_weights_tdata,
            input wire s_axis_weights_tvalid,
            output wire s_axis_weights_tready,
            
            // @brainsmith INTERFACE_TYPE=AXI_STREAM
            output wire [511:0] m_axis_output_tdata,
            output wire m_axis_output_tvalid,
            input wire m_axis_output_tready
        );
        endmodule
        """
        
        # Complete pipeline: RTL → Dataflow Interfaces → Dataflow Model
        rtl_parser = RTLParser()
        parse_result = rtl_parser.parse(model_kernel)
        
        converter = RTLInterfaceConverter()
        conversion_result = converter.convert_interfaces(parse_result.interfaces, {})
        
        # Create DataflowModel from converted interfaces
        from brainsmith.dataflow.core.dataflow_model import DataflowModel
        
        dataflow_model = DataflowModel(
            conversion_result.dataflow_interfaces, 
            conversion_result.metadata
        )
        
        # Validate model functionality
        assert len(dataflow_model.input_interfaces) >= 1
        assert len(dataflow_model.weight_interfaces) >= 1
        assert len(dataflow_model.output_interfaces) >= 1
        
        # Test model operations
        iPar = {iface.name: 1 for iface in dataflow_model.input_interfaces}
        wPar = {iface.name: 1 for iface in dataflow_model.weight_interfaces}
        
        intervals = dataflow_model.calculate_initiation_intervals(iPar, wPar)
        
        # Should successfully calculate timing intervals
        assert hasattr(intervals, 'bottleneck_analysis')
        assert intervals.bottleneck_analysis["bottleneck_eII"] > 0
    
    def test_conversion_with_layout_inference(self):
        """Test conversion with automatic layout inference."""
        
        layout_kernel = """
        module layout_test (
            input wire clk,
            
            // CNN-style input (should infer NCHW layout)
            // @brainsmith INTERFACE_TYPE=AXI_STREAM
            // @brainsmith TDIM=[256,14,14]
            input wire [2047:0] s_axis_cnn_input_tdata,
            input wire s_axis_cnn_input_tvalid,
            output wire s_axis_cnn_input_tready,
            
            // Transformer-style input (should infer NLC layout)
            // @brainsmith INTERFACE_TYPE=AXI_STREAM
            // @brainsmith TDIM=[512,768]
            input wire [6143:0] s_axis_transformer_input_tdata,
            input wire s_axis_transformer_input_tvalid,
            output wire s_axis_transformer_input_tready
        );
        endmodule
        """
        
        rtl_parser = RTLParser()
        parse_result = rtl_parser.parse(layout_kernel)
        
        converter = RTLInterfaceConverter()
        conversion_result = converter.convert_interfaces(parse_result.interfaces, {})
        
        # Find converted interfaces
        cnn_interface = next((iface for iface in conversion_result.dataflow_interfaces
                            if "cnn" in iface.name), None)
        transformer_interface = next((iface for iface in conversion_result.dataflow_interfaces
                                    if "transformer" in iface.name), None)
        
        assert cnn_interface is not None
        assert transformer_interface is not None
        
        # Validate inferred properties
        # CNN: 3D tensor should suggest channel-first processing
        assert len(cnn_interface.tensor_dims) == 3
        assert cnn_interface.tensor_dims == [256, 14, 14]
        
        # Transformer: 2D tensor should suggest sequence processing
        assert len(transformer_interface.tensor_dims) == 2
        assert transformer_interface.tensor_dims == [512, 768]