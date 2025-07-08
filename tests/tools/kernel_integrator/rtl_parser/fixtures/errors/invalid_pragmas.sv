////////////////////////////////////////////////////////////////////////////
// Module with various invalid pragma syntax
//
// This fixture demonstrates pragma errors that should generate
// warnings but not break parsing entirely. The parser should
// be resilient to pragma errors.
////////////////////////////////////////////////////////////////////////////

module invalid_pragmas #(
    parameter integer VALID_PARAM = 32,
    parameter integer WIDTH = 16,
    parameter integer DEPTH = 8,
    parameter integer s_axis_input_BDIM = 32,
    parameter integer s_axis_input_SDIM = 512,
    parameter integer m_axis_output_BDIM = 32
) (
    // Global control
    input wire ap_clk,
    input wire ap_rst_n,
    
    // Valid interface to ensure module can still work
    input wire [WIDTH-1:0] s_axis_input_tdata,
    input wire s_axis_input_tvalid,
    output wire s_axis_input_tready,
    
    output wire [WIDTH-1:0] m_axis_output_tdata,
    output wire m_axis_output_tvalid,
    input wire m_axis_output_tready
);

    // Invalid pragma: Unknown pragma type
    // @brainsmith UNKNOWN_PRAGMA some arguments
    
    // Invalid pragma: BDIM with no arguments
    // @brainsmith BDIM
    
    // Invalid pragma: BDIM with only interface name
    // @brainsmith BDIM s_axis_input
    
    // Invalid pragma: DATATYPE with wrong number of arguments
    // @brainsmith DATATYPE s_axis_input UINT
    // @brainsmith DATATYPE s_axis_input UINT 8
    // @brainsmith DATATYPE s_axis_input UINT 8 4 extra_arg
    
    // Invalid pragma: DATATYPE with invalid base type
    // @brainsmith DATATYPE s_axis_input INVALID_TYPE 8 32
    
    // Invalid pragma: DATATYPE with invalid bit widths
    // @brainsmith DATATYPE s_axis_input UINT -8 32
    // @brainsmith DATATYPE s_axis_input UINT 0 32
    // @brainsmith DATATYPE s_axis_input UINT 64 32
    
    // Invalid pragma: ALIAS without target name
    // @brainsmith ALIAS VALID_PARAM
    
    // Invalid pragma: DERIVED_PARAMETER without expression
    // @brainsmith DERIVED_PARAMETER RESULT
    
    // Invalid pragma: WEIGHT with no interface name
    // @brainsmith WEIGHT
    
    // Invalid pragma: RELATIONSHIP with insufficient arguments
    // @brainsmith RELATIONSHIP s_axis_input
    // @brainsmith RELATIONSHIP s_axis_input m_axis_output
    
    // Invalid pragma: TOP_MODULE with multiple modules (though only one exists)
    // @brainsmith TOP_MODULE nonexistent_module
    
    // Invalid pragma: BDIM with invalid list syntax
    // @brainsmith BDIM s_axis_input [32, 32
    // @brainsmith BDIM s_axis_input 32, 32]
    // @brainsmith BDIM s_axis_input [32 32]
    
    // Invalid pragma: Named argument with invalid syntax
    // @brainsmith BDIM s_axis_input WIDTH SHAPE=[32,32
    // @brainsmith BDIM s_axis_input WIDTH SHAPE=
    
    // Invalid pragma: DATATYPE_PARAM with invalid property
    // @brainsmith DATATYPE_PARAM s_axis_input invalid_property SOME_PARAM
    
    // Invalid pragma: Applying to non-existent interface
    // @brainsmith BDIM nonexistent_interface SOME_PARAM
    // @brainsmith DATATYPE nonexistent_interface UINT 8 32
    
    // Simple implementation
    assign m_axis_output_tdata = s_axis_input_tdata;
    assign m_axis_output_tvalid = s_axis_input_tvalid;
    assign s_axis_input_tready = m_axis_output_tready;

endmodule