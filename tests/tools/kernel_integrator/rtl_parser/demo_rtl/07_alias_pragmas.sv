////////////////////////////////////////////////////////////////////////////
// Demo 07: ALIAS Pragmas
// 
// This example demonstrates using ALIAS pragmas to expose RTL parameters
// with user-friendly names in the Python API. This allows hardware-
// specific naming in RTL while providing clean software interfaces.
////////////////////////////////////////////////////////////////////////////

// Create user-friendly aliases for hardware parameters
// @brainsmith ALIAS PE parallelism_factor
// @brainsmith ALIAS SIMD input_vector_width  
// @brainsmith ALIAS C num_channels
// @brainsmith ALIAS K num_kernels
// @brainsmith ALIAS FOLD folding_factor
// @brainsmith ALIAS BUF_DEPTH fifo_depth
// @brainsmith ALIAS LATENCY pipeline_cycles

// Interface pragmas
// @brainsmith DATATYPE s_axis_activations FIXED 8 16
// @brainsmith DATATYPE s_axis_weights FIXED 8 8
// @brainsmith WEIGHT s_axis_weights

module alias_demo #(
    // Hardware-oriented parameter names
    parameter int unsigned PE = 16,          // Processing elements
    parameter int unsigned SIMD = 8,         // Single instruction multiple data
    parameter int unsigned C = 64,           // Input channels
    parameter int unsigned K = 128,          // Output channels/kernels
    parameter int unsigned FOLD = 4,         // Folding factor for resource sharing
    parameter int unsigned BUF_DEPTH = 512,  // Internal buffer depth
    parameter int unsigned LATENCY = 23,     // Total pipeline latency
    
    // Interface parameters (auto-linked)
    parameter int unsigned s_axis_activations_WIDTH = 8,
    parameter bit s_axis_activations_SIGNED = 1,
    parameter int unsigned s_axis_activations_BDIM = 64,
    parameter int unsigned s_axis_activations_SDIM = 1024,
    
    parameter int unsigned s_axis_weights_WIDTH = 8,
    parameter bit s_axis_weights_SIGNED = 1,
    parameter int unsigned s_axis_weights_BDIM = 32,
    parameter int unsigned s_axis_weights_SDIM = 512,
    
    parameter int unsigned m_axis_results_WIDTH = 32,
    parameter bit m_axis_results_SIGNED = 1,
    parameter int unsigned m_axis_results_BDIM = 128,
    
    // Non-aliased parameters (exposed with original names)
    parameter int unsigned PRECISION_CONFIG = 0,
    parameter int unsigned DEBUG_LEVEL = 0
) (
    // Clock and reset
    input  logic                                    ap_clk,
    input  logic                                    ap_rst_n,
    
    // Activation input stream
    input  logic [s_axis_activations_WIDTH-1:0]    s_axis_activations_tdata,
    input  logic                                    s_axis_activations_tvalid,
    output logic                                    s_axis_activations_tready,
    
    // Weight input stream
    input  logic [s_axis_weights_WIDTH-1:0]        s_axis_weights_tdata,
    input  logic                                    s_axis_weights_tvalid,
    output logic                                    s_axis_weights_tready,
    
    // Result output stream
    output logic [m_axis_results_WIDTH-1:0]        m_axis_results_tdata,
    output logic                                    m_axis_results_tvalid,
    input  logic                                    m_axis_results_tready
);

    // Implementation using hardware names internally
    localparam int unsigned COMPUTE_CYCLES = LATENCY - 3;
    localparam int unsigned PARALLEL_MULTS = PE * SIMD;
    localparam int unsigned WEIGHT_BUFFER_SIZE = K * C / FOLD;
    
    // Processing logic...
    
endmodule : alias_demo

// Expected parser behavior:
// - Aliased parameters exposed with friendly names:
//   - parallelism_factor (PE)
//   - input_vector_width (SIMD)
//   - num_channels (C)
//   - num_kernels (K)
//   - folding_factor (FOLD)
//   - fifo_depth (BUF_DEPTH)
//   - pipeline_cycles (LATENCY)
// - Non-aliased parameters exposed with original names:
//   - PRECISION_CONFIG
//   - DEBUG_LEVEL
// - Interface parameters are auto-linked (not exposed)

// Generated Python code will use:
// self.get_nodeattr("parallelism_factor")  # Instead of PE
// self.get_nodeattr("num_channels")        # Instead of C