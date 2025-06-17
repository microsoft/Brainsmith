// Test file with INVALID BDIM pragma on OUTPUT interface

// @brainsmith BDIM m_axis_output0 OUTPUT0_BDIM

module test_invalid_output_bdim #(
    parameter OUTPUT0_BDIM = 8
)(
    input wire ap_clk,
    input wire ap_rst_n,
    
    output wire [7:0] m_axis_output0_tdata,
    output wire m_axis_output0_tvalid,
    input wire m_axis_output0_tready
);

assign m_axis_output0_tdata = 8'h00;
assign m_axis_output0_tvalid = 1'b1;

endmodule