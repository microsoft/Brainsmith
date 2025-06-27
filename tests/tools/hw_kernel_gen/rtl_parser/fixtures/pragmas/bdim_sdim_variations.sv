////////////////////////////////////////////////////////////////////////////
// Different BDIM/SDIM parameter patterns and pragma variations
//
// This fixture demonstrates:
// - Single BDIM/SDIM parameters
// - Multi-dimensional BDIM/SDIM with lists
// - Indexed parameter naming (e.g., IN0_BDIM0, IN0_BDIM1)
// - Mixed singleton dimensions (using '1')
// - Default parameter inference patterns
////////////////////////////////////////////////////////////////////////////

module bdim_sdim_variations #(
    // Pattern 1: Single BDIM/SDIM parameters
    parameter integer s_axis_simple_BDIM = 32,
    parameter integer s_axis_simple_SDIM = 512,
    
    // Pattern 2: Indexed parameters (auto-linked)
    parameter integer in0_BDIM0 = 16,
    parameter integer in0_BDIM1 = 16,
    parameter integer in0_BDIM2 = 3,
    parameter integer in0_SDIM0 = 224,
    parameter integer in0_SDIM1 = 224,
    
    // Pattern 3: Custom named parameters via pragma
    parameter integer TILE_H = 8,
    parameter integer TILE_W = 8,
    parameter integer TILE_C = 64,
    parameter integer STREAM_SIZE = 1024,
    
    // Pattern 4: Output only needs BDIM
    parameter integer OUTPUT_DIM = 128,
    
    // Pattern 5: Weights with mixed patterns
    parameter integer W0_BLOCK = 64,
    parameter integer W0_STREAM = 512,
    parameter integer weights1_BDIM = 32,  // Auto-linked
    parameter integer WEIGHT1_STREAM_DIM = 256
) (
    // Global control
    input wire ap_clk,
    input wire ap_rst_n,
    
    // Pattern 1: Simple single dimension (auto-linked)
    input wire [31:0] s_axis_simple_tdata,
    input wire s_axis_simple_tvalid,
    output wire s_axis_simple_tready,
    
    // Pattern 2: Multi-dimensional with indexed params (auto-linked)
    input wire [15:0] s_axis_in0_tdata,
    input wire s_axis_in0_tvalid,
    output wire s_axis_in0_tready,
    
    // Pattern 3: Custom names via pragma
    // @brainsmith BDIM s_axis_tiled [TILE_H, TILE_W, TILE_C]
    // @brainsmith SDIM s_axis_tiled STREAM_SIZE
    input wire [7:0] s_axis_tiled_tdata,
    input wire s_axis_tiled_tvalid,
    output wire s_axis_tiled_tready,
    
    // Pattern 4: Mixed singleton dimensions
    // @brainsmith BDIM s_axis_mixed [1, 32, 1]
    // @brainsmith SDIM s_axis_mixed [256, 1]
    input wire [31:0] s_axis_mixed_tdata,
    input wire s_axis_mixed_tvalid,
    output wire s_axis_mixed_tready,
    
    // Pattern 5: Weight interfaces
    // @brainsmith WEIGHT s_axis_w0
    // @brainsmith BDIM s_axis_w0 W0_BLOCK
    // @brainsmith SDIM s_axis_w0 W0_STREAM
    input wire [7:0] s_axis_w0_tdata,
    input wire s_axis_w0_tvalid,
    output wire s_axis_w0_tready,
    
    // @brainsmith WEIGHT s_axis_weights1
    // @brainsmith SDIM s_axis_weights1 WEIGHT1_STREAM_DIM
    // BDIM will be auto-linked to weights1_BDIM
    input wire [15:0] s_axis_weights1_tdata,
    input wire s_axis_weights1_tvalid,
    output wire s_axis_weights1_tready,
    
    // Output interfaces
    // @brainsmith BDIM m_axis_output OUTPUT_DIM
    output wire [31:0] m_axis_output_tdata,
    output wire m_axis_output_tvalid,
    input wire m_axis_output_tready,
    
    // Multi-dimensional output
    // @brainsmith BDIM m_axis_tiled_out [TILE_H, TILE_W, 1, TILE_C]
    output wire [7:0] m_axis_tiled_out_tdata,
    output wire m_axis_tiled_out_tvalid,
    input wire m_axis_tiled_out_tready
);

    // Simple datapath - OR all inputs
    wire [31:0] combined = 
        s_axis_simple_tdata |
        {16'h0, s_axis_in0_tdata} |
        {24'h0, s_axis_tiled_tdata} |
        s_axis_mixed_tdata |
        {24'h0, s_axis_w0_tdata} |
        {16'h0, s_axis_weights1_tdata};
    
    // All inputs must be valid
    wire all_valid = 
        s_axis_simple_tvalid &
        s_axis_in0_tvalid &
        s_axis_tiled_tvalid &
        s_axis_mixed_tvalid &
        s_axis_w0_tvalid &
        s_axis_weights1_tvalid;
    
    // Broadcast ready
    wire all_ready = m_axis_output_tready & m_axis_tiled_out_tready;
    
    assign s_axis_simple_tready = all_ready;
    assign s_axis_in0_tready = all_ready;
    assign s_axis_tiled_tready = all_ready;
    assign s_axis_mixed_tready = all_ready;
    assign s_axis_w0_tready = all_ready;
    assign s_axis_weights1_tready = all_ready;
    
    // Output assignments
    assign m_axis_output_tdata = combined;
    assign m_axis_output_tvalid = all_valid;
    assign m_axis_tiled_out_tdata = combined[7:0];
    assign m_axis_tiled_out_tvalid = all_valid;

endmodule