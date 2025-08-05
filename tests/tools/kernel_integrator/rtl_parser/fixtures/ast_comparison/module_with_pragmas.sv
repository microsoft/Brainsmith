////////////////////////////////////////////////////////////////////////////
// Module with Pragmas for AST Testing
// 
// This module includes pragma annotations for testing pragma parsing
////////////////////////////////////////////////////////////////////////////

module matrix_multiply #(
    parameter int unsigned SIMD = 8,
    parameter int unsigned PE = 4,
    parameter int unsigned INPUT_WIDTH = 16,
    parameter int unsigned OUTPUT_WIDTH = 32
) (
    // Global signals
    input  logic                            ap_clk,
    input  logic                            ap_rst_n,
    
    // brainsmith:pragma:INTERFACE input
    // brainsmith:pragma:DATATYPE ap_fixed<16,6>
    // brainsmith:pragma:BDIM [*, 768]
    // brainsmith:pragma:SDIM [1, SIMD]
    input  logic [INPUT_WIDTH-1:0]          s_axis_input_tdata,
    input  logic                            s_axis_input_tvalid,
    output logic                            s_axis_input_tready,
    
    // brainsmith:pragma:INTERFACE weight
    // brainsmith:pragma:DATATYPE ap_fixed<16,6>
    // brainsmith:pragma:BDIM [768, 512]
    // brainsmith:pragma:SDIM [SIMD, PE]
    input  logic [INPUT_WIDTH-1:0]          s_axis_weight_tdata,
    input  logic                            s_axis_weight_tvalid,
    output logic                            s_axis_weight_tready,
    
    // brainsmith:pragma:INTERFACE output
    // brainsmith:pragma:DATATYPE ap_fixed<32,16>
    // brainsmith:pragma:BDIM [*, 512]
    // brainsmith:pragma:SDIM [1, PE]
    output logic [OUTPUT_WIDTH-1:0]         m_axis_output_tdata,
    output logic                            m_axis_output_tvalid,
    input  logic                            m_axis_output_tready
);

    // Implementation would go here
    
endmodule