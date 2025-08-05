////////////////////////////////////////////////////////////////////////////
// Malformed Module for AST Error Testing
// 
// This module contains syntax errors for testing error detection
////////////////////////////////////////////////////////////////////////////

module malformed_module (
    input  logic        clk,
    input  logic        rst_n  // Missing comma
    output logic [7:0]  data_out
);

    // Missing semicolon
    logic [7:0] temp_data
    
    // Invalid syntax
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            temp_data <= 8'h00;
        end else begin
            temp_data <= temp_data + 1
        // Missing end
    end
    
    assign data_out = temp_data;

// Missing endmodule