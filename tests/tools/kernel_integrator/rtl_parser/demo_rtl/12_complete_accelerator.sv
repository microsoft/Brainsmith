////////////////////////////////////////////////////////////////////////////
// Demo 12: Complete Accelerator Example
// 
// This example combines all pragma types and features into a realistic
// accelerator design, demonstrating best practices and advanced usage.
////////////////////////////////////////////////////////////////////////////

// Module selection (if multiple modules in file)
// @brainsmith TOP_MODULE cnn_accelerator

// Interface type specifications
// @brainsmith WEIGHT s_axis_weights
// @brainsmith WEIGHT s_axis_bias

// Datatype constraints
// @brainsmith DATATYPE s_axis_activations FIXED 8 16
// @brainsmith DATATYPE s_axis_weights FIXED 8 8
// @brainsmith DATATYPE s_axis_bias FIXED 16 16
// @brainsmith DATATYPE m_axis_features FIXED 16 32
// @brainsmith DATATYPE s_axilite_control UINT 32 32

// Dimension specifications with shapes
// @brainsmith BDIM s_axis_activations [N, IH, IW, IC]
// @brainsmith BDIM s_axis_weights [OC, IC, KH, KW]
// @brainsmith BDIM s_axis_bias [OC]
// @brainsmith BDIM m_axis_features [N, OH, OW, OC]

// Stream dimensions
// @brainsmith SDIM s_axis_activations ACT_STREAM_DIM
// @brainsmith SDIM s_axis_weights WEIGHT_STREAM_DIM
// @brainsmith SDIM s_axis_bias BIAS_STREAM_DIM

// User-friendly parameter aliases
// @brainsmith ALIAS PE parallelism
// @brainsmith ALIAS SIMD simd_lanes
// @brainsmith ALIAS N batch_size
// @brainsmith ALIAS IC input_channels
// @brainsmith ALIAS OC output_channels
// @brainsmith ALIAS PRECISION activation_precision

// Derived parameters
// @brainsmith DERIVED_PARAMETER TOTAL_OPS self.get_nodeattr("batch_size") * self.get_nodeattr("output_height") * self.get_nodeattr("output_width") * self.get_nodeattr("output_channels") * self.get_nodeattr("kernel_height") * self.get_nodeattr("kernel_width") * self.get_nodeattr("input_channels")
// @brainsmith DERIVED_PARAMETER LATENCY (self.get_nodeattr("output_channels") // self.get_nodeattr("parallelism")) * (self.get_nodeattr("kernel_height") * self.get_nodeattr("kernel_width")) + 10
// @brainsmith DERIVED_PARAMETER WEIGHT_BUFFER_DEPTH self.calc_wmem()

// Additional aliases for clarity
// @brainsmith ALIAS IH input_height
// @brainsmith ALIAS IW input_width
// @brainsmith ALIAS OH output_height
// @brainsmith ALIAS OW output_width
// @brainsmith ALIAS KH kernel_height
// @brainsmith ALIAS KW kernel_width

module cnn_accelerator #(
    // Architecture parameters
    parameter int unsigned PE = 16,              // Processing elements
    parameter int unsigned SIMD = 8,             // SIMD width
    parameter int unsigned PRECISION = 8,        // Activation bit width
    
    // Tensor dimensions
    parameter int unsigned N = 1,                // Batch size
    parameter int unsigned IC = 64,              // Input channels
    parameter int unsigned OC = 128,             // Output channels
    parameter int unsigned IH = 32,              // Input height
    parameter int unsigned IW = 32,              // Input width
    parameter int unsigned OH = 30,              // Output height (IH-KH+1)
    parameter int unsigned OW = 30,              // Output width (IW-KW+1)
    parameter int unsigned KH = 3,               // Kernel height
    parameter int unsigned KW = 3,               // Kernel width
    
    // Shape parameters no longer used with new pragma syntax
    // Dimensions are directly specified in BDIM pragmas above
    
    // Stream dimensions
    parameter int unsigned ACT_STREAM_DIM = 2048,
    parameter int unsigned WEIGHT_STREAM_DIM = 4096,
    parameter int unsigned BIAS_STREAM_DIM = 128,
    
    // Datatype parameters (auto-linked)
    parameter int unsigned s_axis_activations_WIDTH = 8,
    parameter bit s_axis_activations_SIGNED = 1,
    parameter int unsigned s_axis_activations_FRACTIONAL_WIDTH = 4,
    
    parameter int unsigned s_axis_weights_WIDTH = 8,
    parameter bit s_axis_weights_SIGNED = 1,
    parameter int unsigned s_axis_weights_FRACTIONAL_WIDTH = 7,
    
    parameter int unsigned s_axis_bias_WIDTH = 16,
    parameter bit s_axis_bias_SIGNED = 1,
    parameter int unsigned s_axis_bias_FRACTIONAL_WIDTH = 8,
    
    parameter int unsigned m_axis_features_WIDTH = 16,
    parameter bit m_axis_features_SIGNED = 1,
    
    parameter int unsigned s_axilite_control_WIDTH = 32,
    
    // Internal datatype parameters (auto-linked)
    parameter int unsigned ACCUMULATOR_WIDTH = 32,
    parameter bit ACCUMULATOR_SIGNED = 1,
    parameter int unsigned POST_SCALE_WIDTH = 16,
    parameter bit POST_SCALE_SIGNED = 1,
    
    // Derived parameters (computed in Python)
    parameter int unsigned TOTAL_OPS = 58982400,        // Will be overridden
    parameter int unsigned LATENCY = 82,                 // Will be overridden  
    parameter int unsigned WEIGHT_BUFFER_DEPTH = 1152,   // Will be overridden
    
    // Configuration parameters
    parameter int unsigned FIFO_DEPTH = 512,
    parameter int unsigned BURST_LENGTH = 16,
    parameter bit ENABLE_RELU = 1,
    parameter bit ENABLE_POOLING = 0
) (
    // Global control signals
    input  logic                                    ap_clk,
    input  logic                                    ap_rst_n,
    
    // Activation input stream
    input  logic [s_axis_activations_WIDTH-1:0]    s_axis_activations_tdata,
    input  logic                                    s_axis_activations_tvalid,
    output logic                                    s_axis_activations_tready,
    input  logic                                    s_axis_activations_tlast,
    
    // Weight input stream
    input  logic [s_axis_weights_WIDTH-1:0]        s_axis_weights_tdata,
    input  logic                                    s_axis_weights_tvalid,
    output logic                                    s_axis_weights_tready,
    
    // Bias input stream
    input  logic [s_axis_bias_WIDTH-1:0]           s_axis_bias_tdata,
    input  logic                                    s_axis_bias_tvalid,
    output logic                                    s_axis_bias_tready,
    
    // Feature output stream
    output logic [m_axis_features_WIDTH-1:0]       m_axis_features_tdata,
    output logic                                    m_axis_features_tvalid,
    input  logic                                    m_axis_features_tready,
    output logic                                    m_axis_features_tlast,
    
    // Control/Status interface
    input  logic                                    s_axilite_control_awvalid,
    output logic                                    s_axilite_control_awready,
    input  logic [11:0]                             s_axilite_control_awaddr,
    input  logic                                    s_axilite_control_wvalid,
    output logic                                    s_axilite_control_wready,
    input  logic [s_axilite_control_WIDTH-1:0]     s_axilite_control_wdata,
    input  logic [3:0]                              s_axilite_control_wstrb,
    output logic                                    s_axilite_control_bvalid,
    input  logic                                    s_axilite_control_bready,
    output logic [1:0]                              s_axilite_control_bresp,
    input  logic                                    s_axilite_control_arvalid,
    output logic                                    s_axilite_control_arready,
    input  logic [11:0]                             s_axilite_control_araddr,
    output logic                                    s_axilite_control_rvalid,
    input  logic                                    s_axilite_control_rready,
    output logic [s_axilite_control_WIDTH-1:0]     s_axilite_control_rdata,
    output logic [1:0]                              s_axilite_control_rresp
);

    // Internal architecture implementation
    // This represents a complete CNN accelerator with:
    // - Configurable parallelism (PE)
    // - SIMD processing (SIMD)
    // - Weight buffering
    // - Accumulation and post-processing
    // - Optional ReLU and pooling
    
    // Weight buffer
    logic [WEIGHT_BUFFER_DEPTH-1:0][s_axis_weights_WIDTH-1:0] weight_buffer;
    
    // Processing array
    logic signed [PE-1:0][ACCUMULATOR_WIDTH-1:0] accumulators;
    
    // Post-processing pipeline
    logic signed [POST_SCALE_WIDTH-1:0] scaled_result;
    
    // Control FSM would go here...
    
endmodule : cnn_accelerator

// Expected parser behavior:
// - Exposed parameters (with aliases):
//   - parallelism (PE)
//   - simd_lanes (SIMD)
//   - batch_size (N)
//   - input_channels (IC)
//   - output_channels (OC)
//   - activation_precision (PRECISION)
//   - input_height (IH), input_width (IW)
//   - output_height (OH), output_width (OW)
//   - kernel_height (KH), kernel_width (KW)
//   - FIFO_DEPTH, BURST_LENGTH, ENABLE_RELU, ENABLE_POOLING
// - Hidden parameters:
//   - All interface datatype parameters (auto-linked)
//   - Internal datatype parameters (auto-linked)
//   - Shape parameters (no longer used with new syntax)
//   - Derived parameters (computed in Python)
// - Interface configuration:
//   - Activations: INPUT with dimensions [N, IH, IW, IC]
//   - Weights: WEIGHT with dimensions [OC, IC, KH, KW]
//   - Bias: WEIGHT with dimensions [OC]
//   - Features: OUTPUT with dimensions [N, OH, OW, OC]
//   - Control: CONFIG with 32-bit datatype