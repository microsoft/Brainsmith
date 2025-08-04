////////////////////////////////////////////////////////////////////////////
// Module missing global control interface
//
// This fixture demonstrates a common error case where the module
// lacks the required ap_clk and ap_rst_n signals.
// Should fail strict validation with clear error message.
////////////////////////////////////////////////////////////////////////////

module missing_control #(
    parameter integer DATA_WIDTH = 32,
    parameter integer INPUT_BDIM = 16,
    parameter integer INPUT_SDIM = 256,
    parameter integer OUTPUT_BDIM = 16
) (
    // Missing: input wire ap_clk,
    // Missing: input wire ap_rst_n,
    
    // Only has regular clock/reset (not recognized as global control)
    input wire clk,
    input wire reset,
    
    // Data interfaces are present
    input wire [DATA_WIDTH-1:0] s_axis_input_tdata,
    input wire s_axis_input_tvalid,
    output wire s_axis_input_tready,
    
    output wire [DATA_WIDTH-1:0] m_axis_output_tdata,
    output wire m_axis_output_tvalid,
    input wire m_axis_output_tready
);

    // Simple passthrough
    assign m_axis_output_tdata = s_axis_input_tdata;
    assign m_axis_output_tvalid = s_axis_input_tvalid;
    assign s_axis_input_tready = m_axis_output_tready;

endmodule