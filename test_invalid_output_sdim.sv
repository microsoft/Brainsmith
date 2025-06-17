// Test file with INVALID SDIM pragma on OUTPUT interface

// @brainsmith SDIM m_axis_output0 OUTPUT0_SDIM

module test_invalid_output_sdim #(
    parameter OUTPUT0_SDIM = 2
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