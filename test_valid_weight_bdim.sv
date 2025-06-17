// Test file with VALID BDIM/SDIM pragma on WEIGHT interface

// @brainsmith WEIGHT weights_V
// @brainsmith BDIM weights_V WEIGHT_BDIM SHAPE=[WGT_SIZE] RINDEX=0
// @brainsmith SDIM weights_V WEIGHT_SDIM

module test_valid_weight_bdim #(
    parameter WEIGHT_BDIM = 64,
    parameter WEIGHT_SDIM = 8,
    parameter WGT_SIZE = 32
)(
    input wire ap_clk,
    input wire ap_rst_n,
    
    input wire [31:0] weights_V_tdata,
    input wire weights_V_tvalid,
    output wire weights_V_tready,
    
    output wire [7:0] m_axis_output0_tdata,
    output wire m_axis_output0_tvalid,
    input wire m_axis_output0_tready
);

assign m_axis_output0_tdata = weights_V_tdata[7:0];
assign m_axis_output0_tvalid = weights_V_tvalid;
assign weights_V_tready = m_axis_output0_tready;

endmodule