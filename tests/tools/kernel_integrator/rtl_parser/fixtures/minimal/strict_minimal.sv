////////////////////////////////////////////////////////////////////////////
// Minimal SystemVerilog module that passes strict validation
//
// This fixture provides the absolute minimum required for a valid
// hardware kernel in strict mode:
// - Global control interface (ap_clk, ap_rst_n)
// - At least one input interface with BDIM and SDIM parameters
// - At least one output interface with BDIM parameter
////////////////////////////////////////////////////////////////////////////

module strict_minimal #(
    // Input interface parameters
    parameter integer s_axis_input_BDIM = 16,
    parameter integer s_axis_input_SDIM = 256,
    
    // Output interface parameters  
    parameter integer m_axis_output_BDIM = 16,
    
    // Data width
    parameter integer DATA_WIDTH = 32
) (
    // Global control interface (required)
    input wire ap_clk,
    input wire ap_rst_n,
    
    // AXI-Stream input interface
    input wire [DATA_WIDTH-1:0] s_axis_input_tdata,
    input wire s_axis_input_tvalid,
    output wire s_axis_input_tready,
    
    // AXI-Stream output interface
    output wire [DATA_WIDTH-1:0] m_axis_output_tdata,
    output wire m_axis_output_tvalid,
    input wire m_axis_output_tready
);

    // Minimal implementation - direct passthrough
    assign m_axis_output_tdata = s_axis_input_tdata;
    assign m_axis_output_tvalid = s_axis_input_tvalid;
    assign s_axis_input_tready = m_axis_output_tready;

endmodule