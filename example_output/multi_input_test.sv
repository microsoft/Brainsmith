// Test RTL file with multiple interfaces and DATATYPE_PARAM pragmas
// @brainsmith DATATYPE_PARAM s_axis_input0 width INPUT0_WIDTH
// @brainsmith DATATYPE_PARAM s_axis_input0 signed SIGNED_INPUT0
// @brainsmith DATATYPE_PARAM s_axis_input1 width INPUT1_WIDTH  
// @brainsmith DATATYPE_PARAM s_axis_input1 signed SIGNED_INPUT1
// @brainsmith DATATYPE_PARAM m_axis_output0 width OUTPUT_WIDTH
// @brainsmith DATATYPE_PARAM m_axis_output0 signed SIGNED_OUTPUT

module multi_input_add #(
    parameter INPUT0_WIDTH = 8,
    parameter SIGNED_INPUT0 = 0,
    parameter INPUT1_WIDTH = 8,
    parameter SIGNED_INPUT1 = 0,
    parameter OUTPUT_WIDTH = 8,
    parameter SIGNED_OUTPUT = 0,
    parameter ALGORITHM_PARAM = 16,  // This should be in node attributes
    parameter PE = 1                  // This should be in node attributes
) (
    input ap_clk,
    input ap_rst_n,
    
    input [INPUT0_WIDTH-1:0] s_axis_input0_tdata,
    input s_axis_input0_tvalid,
    output s_axis_input0_tready,
    
    input [INPUT1_WIDTH-1:0] s_axis_input1_tdata,
    input s_axis_input1_tvalid,
    output s_axis_input1_tready,
    
    output [OUTPUT_WIDTH-1:0] m_axis_output0_tdata,
    output m_axis_output0_tvalid,
    input m_axis_output0_tready
);

// Simple addition logic
assign m_axis_output0_tdata = s_axis_input0_tdata + s_axis_input1_tdata;
assign m_axis_output0_tvalid = s_axis_input0_tvalid & s_axis_input1_tvalid;
assign s_axis_input0_tready = m_axis_output0_tready;
assign s_axis_input1_tready = m_axis_output0_tready;

endmodule