////////////////////////////////////////////////////////////////////////////
// Demo 08: Derived Parameters
// 
// This example demonstrates DERIVED_PARAMETER pragmas for computing
// parameter values in Python rather than exposing them as node attributes.
// This is useful for parameters that depend on other settings.
////////////////////////////////////////////////////////////////////////////

// Compute parameters from Python expressions
// @brainsmith DERIVED_PARAMETER SIMD self.get_input_datatype().bitwidth() // self.get_nodeattr("activation_width")
// @brainsmith DERIVED_PARAMETER TOTAL_WEIGHTS self.get_nodeattr("num_filters") * self.get_nodeattr("filter_size") * self.get_nodeattr("filter_size") * self.get_nodeattr("input_channels")
// @brainsmith DERIVED_PARAMETER MEM_DEPTH self.calc_wmem()
// @brainsmith DERIVED_PARAMETER OUTPUT_WIDTH self.get_nodeattr("accumulator_width")
// @brainsmith DERIVED_PARAMETER LATENCY_CYCLES self.get_nodeattr("num_filters") // self.get_nodeattr("parallelism") + 5

// Create aliases for exposed parameters
// @brainsmith ALIAS FILTERS num_filters
// @brainsmith ALIAS FSIZE filter_size
// @brainsmith ALIAS CH_IN input_channels
// @brainsmith ALIAS PE parallelism
// @brainsmith ALIAS ACT_WIDTH activation_width
// @brainsmith ALIAS ACC_WIDTH accumulator_width

// Interface datatypes
// @brainsmith DATATYPE s_axis_input FIXED 8 16
// @brainsmith DATATYPE s_axis_weights FIXED 8 8
// @brainsmith WEIGHT s_axis_weights

module derived_params_demo #(
    // User-configurable parameters (exposed via aliases)
    parameter int unsigned FILTERS = 64,        // Number of convolutional filters
    parameter int unsigned FSIZE = 3,           // Filter size (3x3)
    parameter int unsigned CH_IN = 3,           // Input channels
    parameter int unsigned PE = 16,             // Parallelism factor
    parameter int unsigned ACT_WIDTH = 8,       // Activation width
    parameter int unsigned ACC_WIDTH = 32,      // Accumulator width
    
    // Derived parameters (computed in Python, not exposed)
    parameter int unsigned SIMD = 8,            // Will be overridden
    parameter int unsigned TOTAL_WEIGHTS = 1728,// Will be overridden (64*3*3*3=1728)
    parameter int unsigned MEM_DEPTH = 256,     // Will be overridden
    parameter int unsigned OUTPUT_WIDTH = 32,   // Will be overridden
    parameter int unsigned LATENCY_CYCLES = 9,  // Will be overridden
    
    // Interface parameters (auto-linked)
    parameter int unsigned s_axis_input_WIDTH = 8,
    parameter bit s_axis_input_SIGNED = 1,
    parameter int unsigned s_axis_weights_WIDTH = 8,
    parameter bit s_axis_weights_SIGNED = 1,
    parameter int unsigned m_axis_output_WIDTH = 32,
    parameter bit m_axis_output_SIGNED = 1
) (
    // Clock and reset
    input  logic                            ap_clk,
    input  logic                            ap_rst_n,
    
    // Input stream
    input  logic [s_axis_input_WIDTH-1:0]  s_axis_input_tdata,
    input  logic                            s_axis_input_tvalid,
    output logic                            s_axis_input_tready,
    
    // Weight stream
    input  logic [s_axis_weights_WIDTH-1:0] s_axis_weights_tdata,
    input  logic                            s_axis_weights_tvalid,
    output logic                            s_axis_weights_tready,
    
    // Output stream
    output logic [OUTPUT_WIDTH-1:0]         m_axis_output_tdata,
    output logic                            m_axis_output_tvalid,
    input  logic                            m_axis_output_tready
);

    // Internal logic using derived parameters
    logic [SIMD-1:0][ACT_WIDTH-1:0] simd_buffer;
    logic [MEM_DEPTH-1:0][s_axis_weights_WIDTH-1:0] weight_memory;
    logic [$clog2(LATENCY_CYCLES)-1:0] cycle_counter;
    
    // Implementation...
    
endmodule : derived_params_demo

// Expected parser behavior:
// - Exposed parameters (with aliases):
//   - num_filters (FILTERS)
//   - filter_size (FSIZE)
//   - input_channels (CH_IN)
//   - parallelism (PE)
//   - activation_width (ACT_WIDTH)
//   - accumulator_width (ACC_WIDTH)
// - Derived parameters NOT exposed:
//   - SIMD (computed from input datatype and activation_width)
//   - TOTAL_WEIGHTS (computed from filters * filter_size^2 * channels)
//   - MEM_DEPTH (computed by calc_wmem() method)
//   - OUTPUT_WIDTH (copied from accumulator_width)
//   - LATENCY_CYCLES (computed from filters/parallelism + 5)

// In generated Python:
// code_gen_dict["$SIMD$"] = [str(self.get_input_datatype().bitwidth() // self.get_nodeattr("activation_width"))]
// code_gen_dict["$TOTAL_WEIGHTS$"] = [str(self.get_nodeattr("num_filters") * self.get_nodeattr("filter_size") * self.get_nodeattr("filter_size") * self.get_nodeattr("input_channels"))]