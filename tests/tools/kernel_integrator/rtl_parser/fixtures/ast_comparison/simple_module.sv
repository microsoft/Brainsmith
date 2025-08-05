////////////////////////////////////////////////////////////////////////////
// Simple Module for AST Testing
// 
// This module provides a basic structure for testing AST serialization
////////////////////////////////////////////////////////////////////////////

module simple_module (
    input  logic        clk,
    input  logic        rst_n,
    output logic [7:0]  data_out
);

    // Simple assignment
    assign data_out = 8'hAB;

endmodule