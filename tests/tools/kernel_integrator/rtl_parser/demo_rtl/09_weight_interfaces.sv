////////////////////////////////////////////////////////////////////////////
// Demo 09: Weight Interfaces
// 
// This example demonstrates various weight interface patterns and the
// WEIGHT pragma. Weight interfaces support both BDIM and SDIM dimensions.
////////////////////////////////////////////////////////////////////////////

// Mark weight interfaces - required for non-standard naming
// @brainsmith WEIGHT filter_weights
// @brainsmith WEIGHT bias_stream
// @brainsmith WEIGHT normalization_params

// Demonstrate BDIM with multi-dimensional specification
// @brainsmith BDIM filter_weights [K, C, FH, FW]
// @brainsmith BDIM bias_stream BIAS_DIM

// Interface datatypes
// @brainsmith DATATYPE s_axis_activations FIXED 8 16
// @brainsmith DATATYPE filter_weights FIXED 8 8
// @brainsmith DATATYPE bias_stream FIXED 16 16
// @brainsmith DATATYPE normalization_params FIXED 16 16

module weight_interfaces_demo #(
    // Convolution parameters
    parameter int unsigned K = 64,      // Output channels
    parameter int unsigned C = 32,      // Input channels  
    parameter int unsigned FH = 3,      // Filter height
    parameter int unsigned FW = 3,      // Filter width
    
    // Shape parameter for weights (overrides individual params due to pragma)
    parameter int unsigned CONV_WEIGHT_SHAPE = 64,  // Will be ignored
    parameter int unsigned BIAS_DIM = 64,
    
    // Activation interface (auto-linked)
    parameter int unsigned s_axis_activations_WIDTH = 8,
    parameter bit s_axis_activations_SIGNED = 1,
    parameter int unsigned s_axis_activations_BDIM = 32,
    parameter int unsigned s_axis_activations_SDIM = 1024,
    
    // Filter weights (needs WEIGHT pragma)
    parameter int unsigned filter_weights_WIDTH = 8,
    parameter bit filter_weights_SIGNED = 1,
    // BDIM handled by pragma with shape
    parameter int unsigned filter_weights_SDIM = 4096,
    
    // Bias weights (needs WEIGHT pragma)
    parameter int unsigned bias_stream_WIDTH = 16,
    parameter bit bias_stream_SIGNED = 1,
    // BDIM handled by pragma
    parameter int unsigned bias_stream_SDIM = 64,
    
    // Normalization weights (indexed dimensions)
    parameter int unsigned normalization_params_WIDTH = 16,
    parameter bit normalization_params_SIGNED = 1,
    parameter int unsigned normalization_params_BDIM0 = 64,   // Channels
    parameter int unsigned normalization_params_BDIM1 = 4,    // Params per channel
    parameter int unsigned normalization_params_SDIM = 256,
    
    // Standard weight interface (detected by name pattern)
    parameter int unsigned s_axis_weights_WIDTH = 8,
    parameter bit s_axis_weights_SIGNED = 1,
    parameter int unsigned s_axis_weights_BDIM = 128,
    parameter int unsigned s_axis_weights_SDIM = 2048,
    
    // Output parameters
    parameter int unsigned m_axis_output_WIDTH = 32,
    parameter bit m_axis_output_SIGNED = 1,
    parameter int unsigned m_axis_output_BDIM = 64
) (
    // Clock and reset
    input  logic                                        ap_clk,
    input  logic                                        ap_rst_n,
    
    // Activation input
    input  logic [s_axis_activations_WIDTH-1:0]        s_axis_activations_tdata,
    input  logic                                        s_axis_activations_tvalid,
    output logic                                        s_axis_activations_tready,
    
    // Convolutional filter weights (custom interface name)
    input  logic [filter_weights_WIDTH-1:0]            filter_weights_tdata,
    input  logic                                        filter_weights_tvalid,
    output logic                                        filter_weights_tready,
    
    // Bias values
    input  logic [bias_stream_WIDTH-1:0]               bias_stream_tdata,
    input  logic                                        bias_stream_tvalid,
    output logic                                        bias_stream_tready,
    
    // Batch normalization parameters
    input  logic [normalization_params_WIDTH-1:0]      normalization_params_tdata,
    input  logic                                        normalization_params_tvalid,
    output logic                                        normalization_params_tready,
    
    // Standard weight interface (auto-detected by s_axis + 'weights')
    input  logic [s_axis_weights_WIDTH-1:0]            s_axis_weights_tdata,
    input  logic                                        s_axis_weights_tvalid,
    output logic                                        s_axis_weights_tready,
    
    // Output
    output logic [m_axis_output_WIDTH-1:0]             m_axis_output_tdata,
    output logic                                        m_axis_output_tvalid,
    input  logic                                        m_axis_output_tready
);

    // Weight buffer dimensions based on shapes
    logic [K-1:0][C-1:0][FH-1:0][FW-1:0][filter_weights_WIDTH-1:0] conv_weights;
    logic [BIAS_DIM-1:0][bias_stream_WIDTH-1:0] bias_values;
    logic [63:0][3:0][normalization_params_WIDTH-1:0] norm_params;
    
    // Implementation...
    
endmodule : weight_interfaces_demo

// Expected parser behavior:
// - filter_weights: WEIGHT interface due to pragma
//   - BDIM uses shape [K,C,FH,FW] with reference index 0
//   - Supports SDIM (weight interface capability)
// - bias_stream: WEIGHT interface due to pragma
//   - BDIM = BIAS_DIM parameter
//   - Supports SDIM
// - normalization_params: WEIGHT interface due to pragma
//   - BDIM = [normalization_params_BDIM0, normalization_params_BDIM1]
//   - SDIM = normalization_params_SDIM
// - s_axis_weights: Auto-detected as WEIGHT (name pattern)
//   - Standard BDIM/SDIM support
// - All weight interfaces support full datatype specification