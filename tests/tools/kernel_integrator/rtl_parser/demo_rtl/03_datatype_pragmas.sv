////////////////////////////////////////////////////////////////////////////
// Demo 03: Datatype Pragmas
// 
// This example demonstrates DATATYPE and DATATYPE_PARAM pragma usage
// for specifying interface data types and linking to RTL parameters.
////////////////////////////////////////////////////////////////////////////

// Datatype constraints for interfaces
// @brainsmith DATATYPE s_axis_input FIXED 8 16
// @brainsmith DATATYPE weights_stream FIXED 8 8
// @brainsmith DATATYPE m_axis_output UINT 16 32
// @brainsmith DATATYPE s_axilite_config UINT 32 64

// Link specific properties to parameters
// @brainsmith DATATYPE_PARAM s_axis_input width INPUT_PRECISION
// @brainsmith DATATYPE_PARAM s_axis_input signed INPUT_IS_SIGNED
// @brainsmith DATATYPE_PARAM s_axis_input fractional_width INPUT_FRAC_BITS

// @brainsmith DATATYPE_PARAM weights_stream width WEIGHT_BITS
// @brainsmith DATATYPE_PARAM weights_stream signed WEIGHT_SIGNED

// @brainsmith DATATYPE_PARAM m_axis_output width OUTPUT_WIDTH
// @brainsmith DATATYPE_PARAM m_axis_output signed OUTPUT_SIGNED

// CONFIG interfaces support datatypes (but not dimensions)
// @brainsmith DATATYPE_PARAM s_axilite_config width CONFIG_BUS_WIDTH

// Mark weight interface
// @brainsmith WEIGHT weights_stream

// Add dimension specifications
// @brainsmith BDIM s_axis_input INPUT_BATCH_SIZE
// @brainsmith SDIM s_axis_input INPUT_STREAM_SIZE
// @brainsmith BDIM weights_stream WEIGHT_BATCH_SIZE
// @brainsmith SDIM weights_stream WEIGHT_STREAM_SIZE
// @brainsmith BDIM m_axis_output OUTPUT_BATCH_SIZE

module datatype_demo #(
    // Input parameters (linked by pragma)
    parameter int unsigned INPUT_PRECISION = 16,
    parameter bit INPUT_IS_SIGNED = 1,
    parameter int unsigned INPUT_FRAC_BITS = 8,
    
    // Weight parameters (linked by pragma)
    parameter int unsigned WEIGHT_BITS = 8,
    parameter bit WEIGHT_SIGNED = 1,
    
    // Output parameters (linked by pragma)
    parameter int unsigned OUTPUT_WIDTH = 32,
    parameter bit OUTPUT_SIGNED = 0,
    
    // Config parameters (linked by pragma)
    parameter int unsigned CONFIG_BUS_WIDTH = 32,
    
    // Dimension parameters for interfaces
    parameter int unsigned INPUT_BATCH_SIZE = 32,
    parameter int unsigned INPUT_STREAM_SIZE = 1024,
    parameter int unsigned WEIGHT_BATCH_SIZE = 64,
    parameter int unsigned WEIGHT_STREAM_SIZE = 2048,
    parameter int unsigned OUTPUT_BATCH_SIZE = 32,
    
    // Other parameters (will be exposed)
    parameter int unsigned COMPUTE_UNITS = 4,
    parameter int unsigned BUFFER_DEPTH = 1024
) (
    // Standard interfaces
    input  logic                            ap_clk,
    input  logic                            ap_rst_n,
    
    // Input stream with parameterized width
    input  logic [INPUT_PRECISION-1:0]      s_axis_input_tdata,
    input  logic                            s_axis_input_tvalid,
    output logic                            s_axis_input_tready,
    
    // Weight stream
    input  logic [WEIGHT_BITS-1:0]          weights_stream_tdata,
    input  logic                            weights_stream_tvalid,
    output logic                            weights_stream_tready,
    
    // Output stream
    output logic [OUTPUT_WIDTH-1:0]         m_axis_output_tdata,
    output logic                            m_axis_output_tvalid,
    input  logic                            m_axis_output_tready,
    
    // Configuration interface
    input  logic                            s_axilite_config_awvalid,
    output logic                            s_axilite_config_awready,
    input  logic [11:0]                     s_axilite_config_awaddr,
    input  logic                            s_axilite_config_wvalid,
    output logic                            s_axilite_config_wready,
    input  logic [CONFIG_BUS_WIDTH-1:0]     s_axilite_config_wdata,
    input  logic [3:0]                      s_axilite_config_wstrb,
    output logic                            s_axilite_config_bvalid,
    input  logic                            s_axilite_config_bready,
    output logic [1:0]                      s_axilite_config_bresp
);

    // Implementation...
    
endmodule : datatype_demo

// Expected parser behavior:
// - DATATYPE pragmas set base type constraints for each interface
// - DATATYPE_PARAM pragmas link specific parameters to interface properties
// - Linked parameters are NOT exposed as node attributes
// - COMPUTE_UNITS and BUFFER_DEPTH remain exposed
// - weights_stream recognized as WEIGHT interface due to pragma