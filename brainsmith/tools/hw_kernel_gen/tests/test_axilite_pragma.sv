// Test module for AXILITE_PARAM pragma

// @brainsmith top_module test_axilite
module test_axilite #(
    parameter INPUT_WIDTH = 8,
    parameter OUTPUT_WIDTH = 8,
    // @brainsmith axilite_param s_axilite_config MY_CUSTOM_CONFIG
    parameter MY_CUSTOM_CONFIG = 1,
    parameter USE_AXILITE = 1,
    parameter SOME_OTHER_PARAM = 42
) (
    input wire ap_clk,
    input wire ap_rst_n,
    
    // Input interface
    input wire [INPUT_WIDTH-1:0] s_axis_input_TDATA,
    input wire s_axis_input_TVALID,
    output wire s_axis_input_TREADY,
    
    // Output interface  
    output wire [OUTPUT_WIDTH-1:0] m_axis_output_TDATA,
    output wire m_axis_output_TVALID,
    input wire m_axis_output_TREADY,
    
    // AXI-Lite config interface
    // @brainsmith datatype s_axilite_config UINT32 CONFIG
    input wire s_axilite_config_AWVALID,
    output wire s_axilite_config_AWREADY,
    input wire [31:0] s_axilite_config_AWADDR
);

// Dummy implementation
assign s_axis_input_TREADY = 1'b1;
assign m_axis_output_TDATA = s_axis_input_TDATA;
assign m_axis_output_TVALID = s_axis_input_TVALID;
assign s_axilite_config_AWREADY = 1'b1;

endmodule