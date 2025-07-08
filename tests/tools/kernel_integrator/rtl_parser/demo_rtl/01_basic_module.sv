////////////////////////////////////////////////////////////////////////////
// Demo 01: Basic Module
// 
// This example shows the minimal RTL module structure that the parser
// can handle. It demonstrates:
// - Basic AXI Stream interfaces
// - Module parameters
// - No pragmas (relying on defaults)
////////////////////////////////////////////////////////////////////////////

module basic_accelerator #(
    parameter int unsigned DATA_WIDTH = 16,
    parameter int unsigned BATCH_SIZE = 32
) (
    // Global signals
    input  logic                    ap_clk,
    input  logic                    ap_rst_n,
    
    // Input stream
    input  logic [DATA_WIDTH-1:0]   s_axis_input_tdata,
    input  logic                    s_axis_input_tvalid,
    output logic                    s_axis_input_tready,
    
    // Output stream
    output logic [DATA_WIDTH-1:0]   m_axis_output_tdata,
    output logic                    m_axis_output_tvalid,
    input  logic                    m_axis_output_tready
);

    // Simple pass-through implementation
    assign m_axis_output_tdata = s_axis_input_tdata;
    assign m_axis_output_tvalid = s_axis_input_tvalid;
    assign s_axis_input_tready = m_axis_output_tready;

endmodule : basic_accelerator

// Expected parser behavior:
// - Detects INPUT interface: s_axis_input
// - Detects OUTPUT interface: m_axis_output
// - Detects CONTROL interface: ap (clock/reset)
// - Exposes parameters: DATA_WIDTH, BATCH_SIZE
// - No auto-linking occurs (parameter names don't match conventions)