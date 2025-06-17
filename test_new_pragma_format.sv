// Test file for new BDIM/SDIM pragma format

// @brainsmith BDIM s_axis_input0 INPUT0_BDIM SHAPE=[C,PE] RINDEX=0
// @brainsmith SDIM s_axis_input0 INPUT0_SDIM
// @brainsmith BDIM m_axis_output0 OUTPUT0_BDIM
// @brainsmith SDIM m_axis_output0 OUTPUT0_SDIM
// @brainsmith DATATYPE_PARAM s_axis_input0 width INPUT0_WIDTH
// @brainsmith DATATYPE_PARAM s_axis_input0 signed SIGNED_INPUT0
// @brainsmith DATATYPE_PARAM m_axis_output0 width OUTPUT0_WIDTH

module test_new_format #(
    parameter INPUT0_WIDTH = 8,
    parameter SIGNED_INPUT0 = 0,
    parameter OUTPUT0_WIDTH = 8,
    parameter INPUT0_BDIM = 16,
    parameter INPUT0_SDIM = 4,
    parameter OUTPUT0_BDIM = 8,
    parameter OUTPUT0_SDIM = 2,
    parameter C = 64,
    parameter PE = 4
)(
    input wire ap_clk,
    input wire ap_rst_n,
    
    input wire [INPUT0_WIDTH-1:0] s_axis_input0_tdata,
    input wire s_axis_input0_tvalid,
    output wire s_axis_input0_tready,
    
    output wire [OUTPUT0_WIDTH-1:0] m_axis_output0_tdata,
    output wire m_axis_output0_tvalid,
    input wire m_axis_output0_tready
);

// Simple operation
assign m_axis_output0_tdata = s_axis_input0_tdata;
assign m_axis_output0_tvalid = s_axis_input0_tvalid;
assign s_axis_input0_tready = m_axis_output0_tready;

endmodule