////////////////////////////////////////////////////////////////////////////
// Demo 11: Interface Relationships
// 
// This example demonstrates complex interface dimension relationships
// using the shape specification features with RINDEX.
////////////////////////////////////////////////////////////////////////////

// Define dimension relationships between interfaces
// @brainsmith BDIM s_axis_input [BATCH, H, W, C_IN]
// @brainsmith BDIM s_axis_weights [C_OUT, C_IN, KH, KW]
// @brainsmith BDIM m_axis_output [BATCH, OH, OW, C_OUT]

// Mark weight interface
// @brainsmith WEIGHT s_axis_weights

// Datatype specifications
// @brainsmith DATATYPE s_axis_input FIXED 8 16
// @brainsmith DATATYPE s_axis_weights FIXED 8 8
// @brainsmith DATATYPE m_axis_output FIXED 16 32

module interface_relationships #(
    // Shared dimension parameters
    parameter int unsigned BATCH = 1,        // Batch size
    parameter int unsigned C_IN = 32,        // Input channels (shared)
    parameter int unsigned C_OUT = 64,       // Output channels (shared)
    
    // Input dimensions
    parameter int unsigned H = 32,           // Input height
    parameter int unsigned W = 32,           // Input width
    
    // Kernel dimensions
    parameter int unsigned KH = 3,           // Kernel height
    parameter int unsigned KW = 3,           // Kernel width
    
    // Output dimensions (computed based on convolution)
    parameter int unsigned OH = 30,          // Output height (H - KH + 1)
    parameter int unsigned OW = 30,          // Output width (W - KW + 1)
    
    // Shape parameters no longer used with new pragma syntax
    // Dimensions are directly specified in BDIM pragmas above
    
    // Stream dimensions
    parameter int unsigned s_axis_input_SDIM = 1024,
    parameter int unsigned s_axis_weights_SDIM = 2048,
    
    // Datatype parameters (auto-linked)
    parameter int unsigned s_axis_input_WIDTH = 8,
    parameter bit s_axis_input_SIGNED = 1,
    parameter int unsigned s_axis_weights_WIDTH = 8,
    parameter bit s_axis_weights_SIGNED = 1,
    parameter int unsigned m_axis_output_WIDTH = 16,
    parameter bit m_axis_output_SIGNED = 1,
    
    // Processing parameters
    parameter int unsigned PE = 16,          // Processing elements
    parameter int unsigned SIMD = 8          // SIMD lanes
) (
    // Clock and reset
    input  logic                                ap_clk,
    input  logic                                ap_rst_n,
    
    // Input tensor stream [BATCH, H, W, C_IN]
    input  logic [s_axis_input_WIDTH-1:0]      s_axis_input_tdata,
    input  logic                                s_axis_input_tvalid,
    output logic                                s_axis_input_tready,
    input  logic                                s_axis_input_tlast,
    
    // Weight tensor stream [C_OUT, C_IN, KH, KW]
    input  logic [s_axis_weights_WIDTH-1:0]    s_axis_weights_tdata,
    input  logic                                s_axis_weights_tvalid,
    output logic                                s_axis_weights_tready,
    
    // Output tensor stream [BATCH, OH, OW, C_OUT]
    output logic [m_axis_output_WIDTH-1:0]     m_axis_output_tdata,
    output logic                                m_axis_output_tvalid,
    input  logic                                m_axis_output_tready,
    output logic                                m_axis_output_tlast
);

    // The dimensions establish relationships:
    // - Input: [BATCH, H, W, C_IN]
    // - Weights: [C_OUT, C_IN, KH, KW] - shares C_IN with input
    // - Output: [BATCH, OH, OW, C_OUT] - shares BATCH with input, C_OUT with weights
    
    // Internal buffers sized according to relationships
    logic [C_IN-1:0][s_axis_input_WIDTH-1:0] input_buffer;
    logic [C_OUT-1:0][C_IN-1:0][KH-1:0][KW-1:0][s_axis_weights_WIDTH-1:0] weight_buffer;
    logic [C_OUT-1:0][m_axis_output_WIDTH-1:0] output_accumulator;
    
    // Convolution engine implementation
    // Uses the dimension relationships to properly index data
    
endmodule : interface_relationships

// Expected parser behavior:
// - BDIM shapes specified directly:
//   - s_axis_input: [BATCH, H, W, C_IN]
//   - s_axis_weights: [C_OUT, C_IN, KH, KW]
//   - m_axis_output: [BATCH, OH, OW, C_OUT]
// - Shared parameters (BATCH, C_IN, C_OUT) link interfaces
// - No need for SHAPE or RINDEX with new syntax
// - PE and SIMD remain exposed
// - All datatype and SDIM parameters are auto-linked