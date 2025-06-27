////////////////////////////////////////////////////////////////////////////
// Minimal AXI-Stream interfaces (only required signals)
//
// This fixture demonstrates the minimum required AXI-Stream signals:
// - TDATA (payload)
// - TVALID (data valid indicator)
// - TREADY (backpressure)
//
// Multiple interfaces show different naming patterns
////////////////////////////////////////////////////////////////////////////

module axi_stream_minimal #(
    // Interface parameters - different patterns
    parameter integer in0_BDIM = 16,
    parameter integer in0_SDIM = 224,
    parameter integer in1_BDIM = 8,
    parameter integer in1_SDIM = 224,
    parameter integer out0_BDIM = 32,
    
    // Shared parameters
    parameter integer DATA_WIDTH = 32,
    parameter integer NARROW_WIDTH = 16
) (
    // Global control
    input wire ap_clk,
    input wire ap_rst_n,
    
    // First input - standard naming
    input wire [DATA_WIDTH-1:0] s_axis_in0_tdata,
    input wire s_axis_in0_tvalid,
    output wire s_axis_in0_tready,
    
    // Second input - different width
    input wire [NARROW_WIDTH-1:0] s_axis_in1_tdata,
    input wire s_axis_in1_tvalid,
    output wire s_axis_in1_tready,
    
    // Output - standard naming
    output wire [DATA_WIDTH-1:0] m_axis_out0_tdata,
    output wire m_axis_out0_tvalid,
    input wire m_axis_out0_tready
);

    // Simple arbitration - in0 has priority
    wire sel_in1 = !s_axis_in0_tvalid;
    
    // Mux inputs based on priority
    assign m_axis_out0_tdata = sel_in1 ? 
        {{(DATA_WIDTH-NARROW_WIDTH){1'b0}}, s_axis_in1_tdata} : 
        s_axis_in0_tdata;
    
    assign m_axis_out0_tvalid = s_axis_in0_tvalid || s_axis_in1_tvalid;
    
    // Ready signals
    assign s_axis_in0_tready = m_axis_out0_tready && !sel_in1;
    assign s_axis_in1_tready = m_axis_out0_tready && sel_in1;

endmodule