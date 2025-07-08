////////////////////////////////////////////////////////////////////////////
// Demo 06: Indexed Dimensions
// 
// This example demonstrates multi-dimensional BDIM/SDIM parameters
// using indexed naming convention (e.g., _BDIM0, _BDIM1, _BDIM2).
// Also shows gap handling where missing indices become singletons.
////////////////////////////////////////////////////////////////////////////

// Mark weight interface
// @brainsmith WEIGHT s_axis_kernel

module indexed_dimensions #(
    // Example 1: Contiguous 3D input tensor (H x W x C)
    parameter int unsigned s_axis_image_BDIM0 = 224,    // Height
    parameter int unsigned s_axis_image_BDIM1 = 224,    // Width
    parameter int unsigned s_axis_image_BDIM2 = 3,      // Channels (RGB)
    
    // Corresponding multi-dimensional streaming
    parameter int unsigned s_axis_image_SDIM0 = 1024,
    parameter int unsigned s_axis_image_SDIM1 = 512,
    parameter int unsigned s_axis_image_SDIM2 = 256,
    
    // Example 2: Non-contiguous weight dimensions (with gaps)
    parameter int unsigned s_axis_kernel_BDIM0 = 64,    // Output channels
    parameter int unsigned s_axis_kernel_BDIM2 = 3,     // Kernel height (BDIM1 missing!)
    parameter int unsigned s_axis_kernel_BDIM3 = 3,     // Kernel width
    // Missing BDIM1 will be treated as singleton "1"
    // Result: [64, 1, 3, 3]
    
    // Single SDIM for kernel (mixing styles is allowed)
    parameter int unsigned s_axis_kernel_SDIM = 2048,
    
    // Example 3: Output with mixed case (lowercase also works)
    parameter int unsigned m_axis_features_bdim0 = 16,  // Lowercase 'bdim'
    parameter int unsigned m_axis_features_bdim1 = 16,
    parameter int unsigned m_axis_features_bdim2 = 64,
    // Note: SDIM parameters would be ignored for outputs
    
    // Example 4: Mixing single and indexed (single takes precedence)
    parameter int unsigned s_axis_bias_BDIM = 64,       // This wins
    parameter int unsigned s_axis_bias_BDIM0 = 32,      // Ignored
    parameter int unsigned s_axis_bias_BDIM1 = 2,       // Ignored
    
    // Datatype parameters
    parameter int unsigned s_axis_image_WIDTH = 8,
    parameter bit s_axis_image_SIGNED = 0,
    parameter int unsigned s_axis_kernel_WIDTH = 8,
    parameter bit s_axis_kernel_SIGNED = 1,
    parameter int unsigned m_axis_features_WIDTH = 16,
    parameter bit m_axis_features_SIGNED = 1,
    
    // Regular parameters
    parameter int unsigned TILE_SIZE = 8,
    parameter int unsigned NUM_ENGINES = 4
) (
    // Clock and reset
    input  logic                                    ap_clk,
    input  logic                                    ap_rst_n,
    
    // Image input stream (3D tensor)
    input  logic [s_axis_image_WIDTH-1:0]          s_axis_image_tdata,
    input  logic                                    s_axis_image_tvalid,
    output logic                                    s_axis_image_tready,
    input  logic                                    s_axis_image_tlast,
    
    // Kernel weight stream (4D with gap)
    input  logic [s_axis_kernel_WIDTH-1:0]         s_axis_kernel_tdata,
    input  logic                                    s_axis_kernel_tvalid,
    output logic                                    s_axis_kernel_tready,
    
    // Bias input (single BDIM wins over indexed)
    input  logic [31:0]                             s_axis_bias_tdata,
    input  logic                                    s_axis_bias_tvalid,
    output logic                                    s_axis_bias_tready,
    
    // Feature output stream
    output logic [m_axis_features_WIDTH-1:0]       m_axis_features_tdata,
    output logic                                    m_axis_features_tvalid,
    input  logic                                    m_axis_features_tready,
    output logic                                    m_axis_features_tlast
);

    // Implementation...
    
endmodule : indexed_dimensions

// Expected parser behavior:
// - s_axis_image: 
//   - BDIM = [s_axis_image_BDIM0, s_axis_image_BDIM1, s_axis_image_BDIM2]
//   - SDIM = [s_axis_image_SDIM0, s_axis_image_SDIM1, s_axis_image_SDIM2]
// - s_axis_kernel:
//   - BDIM = [s_axis_kernel_BDIM0, "1", s_axis_kernel_BDIM2, s_axis_kernel_BDIM3]
//   - SDIM = s_axis_kernel_SDIM (single parameter)
// - m_axis_features:
//   - BDIM = [m_axis_features_bdim0, m_axis_features_bdim1, m_axis_features_bdim2]
//   - No SDIM (output interface)
// - s_axis_bias:
//   - BDIM = s_axis_bias_BDIM (single wins over indexed)
//   - Indexed parameters ignored
// - TILE_SIZE and NUM_ENGINES remain exposed