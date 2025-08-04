////////////////////////////////////////////////////////////////////////////
// Demo 04: Auto-linking Datatypes
// 
// This example demonstrates automatic datatype parameter detection
// based on naming conventions. No pragmas needed!
////////////////////////////////////////////////////////////////////////////

module auto_linking_datatypes #(
    // Following naming convention: <interface>_<PROPERTY>
    // These will be auto-linked to interfaces
    
    // Input interface parameters (auto-linked)
    parameter int unsigned s_axis_input_WIDTH = 16,
    parameter bit s_axis_input_SIGNED = 1,
    parameter int unsigned s_axis_input_FRACTIONAL_WIDTH = 8,
    
    // Weight interface parameters (auto-linked)
    parameter int unsigned s_axis_weights_WIDTH = 8,
    parameter bit s_axis_weights_SIGNED = 1,
    parameter int unsigned s_axis_weights_BIAS = 0,
    
    // Output interface parameters (auto-linked)
    parameter int unsigned m_axis_output_WIDTH = 32,
    parameter bit m_axis_output_SIGNED = 0,
    
    // Config interface parameters (auto-linked)
    parameter int unsigned s_axilite_control_WIDTH = 32,
    parameter bit s_axilite_control_SIGNED = 0,
    
    // Internal datatype parameters (auto-linked as internal)
    parameter int unsigned ACCUMULATOR_WIDTH = 48,
    parameter bit ACCUMULATOR_SIGNED = 1,
    parameter int unsigned THRESHOLD_WIDTH = 16,
    parameter int unsigned THRESHOLD_BIAS = 128,
    
    // Parameters that won't be auto-linked (don't follow pattern)
    parameter int unsigned PE = 16,
    parameter int unsigned SIMD = 4,
    parameter int unsigned BUFFER_SIZE = 1024
) (
    // Clock and reset
    input  logic                                    ap_clk,
    input  logic                                    ap_rst_n,
    
    // Input stream
    input  logic [s_axis_input_WIDTH-1:0]          s_axis_input_tdata,
    input  logic                                    s_axis_input_tvalid,
    output logic                                    s_axis_input_tready,
    
    // Weight stream (needs WEIGHT pragma without it won't be detected)
    input  logic [s_axis_weights_WIDTH-1:0]        s_axis_weights_tdata,
    input  logic                                    s_axis_weights_tvalid,
    output logic                                    s_axis_weights_tready,
    
    // Output stream
    output logic [m_axis_output_WIDTH-1:0]         m_axis_output_tdata,
    output logic                                    m_axis_output_tvalid,
    input  logic                                    m_axis_output_tready,
    
    // Configuration interface
    input  logic                                    s_axilite_control_awvalid,
    output logic                                    s_axilite_control_awready,
    input  logic [11:0]                             s_axilite_control_awaddr,
    input  logic                                    s_axilite_control_wvalid,
    output logic                                    s_axilite_control_wready,
    input  logic [s_axilite_control_WIDTH-1:0]     s_axilite_control_wdata,
    input  logic [3:0]                              s_axilite_control_wstrb,
    output logic                                    s_axilite_control_bvalid,
    input  logic                                    s_axilite_control_bready,
    output logic [1:0]                              s_axilite_control_bresp
);

    // Internal signals using auto-linked datatypes
    logic signed [ACCUMULATOR_WIDTH-1:0] accumulator;
    logic [THRESHOLD_WIDTH-1:0] threshold_value;
    
    // Implementation...
    
endmodule : auto_linking_datatypes

// Expected parser behavior:
// - Interface parameters auto-linked based on naming pattern
// - Internal parameters (ACCUMULATOR_*, THRESHOLD_*) grouped by prefix
// - PE, SIMD, BUFFER_SIZE remain exposed (don't match patterns)
// - No pragmas needed for basic datatype linking!

// Note: To make s_axis_weights a WEIGHT interface, you'd still need:
// @brainsmith WEIGHT s_axis_weights