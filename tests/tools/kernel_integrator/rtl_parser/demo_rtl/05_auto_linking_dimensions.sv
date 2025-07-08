////////////////////////////////////////////////////////////////////////////
// Demo 05: Auto-linking Dimensions
// 
// This example demonstrates automatic BDIM/SDIM parameter detection
// for interface dimensions using naming conventions.
////////////////////////////////////////////////////////////////////////////

// Mark weight interfaces
// @brainsmith WEIGHT s_axis_weights
// @brainsmith WEIGHT bias_values

module auto_linking_dimensions #(
    // Single BDIM/SDIM parameters (auto-linked)
    parameter int unsigned s_axis_input_BDIM = 64,      // Block dimension for input
    parameter int unsigned s_axis_input_SDIM = 1024,    // Stream dimension for input
    
    // Weight dimensions (auto-linked, WEIGHT pragma above)
    parameter int unsigned s_axis_weights_BDIM = 32,
    parameter int unsigned s_axis_weights_SDIM = 512,
    
    // Bias dimensions (auto-linked, WEIGHT pragma above) 
    parameter int unsigned bias_values_BDIM = 32,
    parameter int unsigned bias_values_SDIM = 32,
    
    // Output dimensions (auto-linked, SDIM ignored for outputs!)
    parameter int unsigned m_axis_output_BDIM = 16,
    parameter int unsigned m_axis_output_SDIM = 256,    // This will be ignored!
    
    // Config interface dimensions (both ignored!)
    parameter int unsigned s_axilite_config_BDIM = 4,   // Ignored - CONFIG doesn't support BDIM
    parameter int unsigned s_axilite_config_SDIM = 8,   // Ignored - CONFIG doesn't support SDIM
    
    // Control interface dimensions (both ignored!)
    parameter int unsigned ap_BDIM = 1,                  // Ignored - CONTROL has no dimensions
    parameter int unsigned ap_SDIM = 1,                  // Ignored - CONTROL has no dimensions
    
    // Datatype parameters (also auto-linked)
    parameter int unsigned s_axis_input_WIDTH = 16,
    parameter bit s_axis_input_SIGNED = 1,
    parameter int unsigned s_axis_weights_WIDTH = 8,
    parameter bit s_axis_weights_SIGNED = 1,
    parameter int unsigned bias_values_WIDTH = 32,
    parameter bit bias_values_SIGNED = 1,
    parameter int unsigned m_axis_output_WIDTH = 32,
    parameter bit m_axis_output_SIGNED = 1,
    
    // Other parameters
    parameter int unsigned PARALLELISM = 8
) (
    // Clock and reset
    input  logic                                ap_clk,
    input  logic                                ap_rst_n,
    
    // Input stream (supports BDIM + SDIM)
    input  logic [s_axis_input_WIDTH-1:0]      s_axis_input_tdata,
    input  logic                                s_axis_input_tvalid,
    output logic                                s_axis_input_tready,
    
    // Weight stream (supports BDIM + SDIM due to WEIGHT pragma)
    input  logic [s_axis_weights_WIDTH-1:0]    s_axis_weights_tdata,
    input  logic                                s_axis_weights_tvalid,
    output logic                                s_axis_weights_tready,
    
    // Bias stream (supports BDIM + SDIM due to WEIGHT pragma)
    input  logic [bias_values_WIDTH-1:0]       bias_values_tdata,
    input  logic                                bias_values_tvalid,
    output logic                                bias_values_tready,
    
    // Output stream (supports BDIM only, not SDIM)
    output logic [m_axis_output_WIDTH-1:0]     m_axis_output_tdata,
    output logic                                m_axis_output_tvalid,
    input  logic                                m_axis_output_tready,
    
    // Config interface (no dimension support)
    input  logic                                s_axilite_config_awvalid,
    output logic                                s_axilite_config_awready,
    input  logic [11:0]                         s_axilite_config_awaddr,
    input  logic                                s_axilite_config_wvalid,
    output logic                                s_axilite_config_wready,
    input  logic [31:0]                         s_axilite_config_wdata,
    output logic                                s_axilite_config_bvalid,
    input  logic                                s_axilite_config_bready,
    output logic [1:0]                          s_axilite_config_bresp
);

    // Implementation...
    
endmodule : auto_linking_dimensions

// Expected parser behavior:
// - s_axis_input: BDIM and SDIM auto-linked (INPUT interface)
// - s_axis_weights: BDIM and SDIM auto-linked (WEIGHT interface)
// - bias_values: BDIM and SDIM auto-linked (WEIGHT interface)
// - m_axis_output: Only BDIM auto-linked, SDIM ignored (OUTPUT interface)
// - s_axilite_config: Neither BDIM nor SDIM linked (CONFIG interface)
// - ap: Neither BDIM nor SDIM linked (CONTROL interface)
// - All datatype parameters are also auto-linked
// - Only PARALLELISM remains exposed