////////////////////////////////////////////////////////////////////////////
// Parameterized Module for AST Testing
// 
// This module includes parameters and more complex port declarations
////////////////////////////////////////////////////////////////////////////

module parameterized_module #(
    parameter int unsigned WIDTH = 32,
    parameter int unsigned DEPTH = 16,
    parameter bit ENABLE_PARITY = 1'b0
) (
    input  logic                    clk,
    input  logic                    rst_n,
    
    // Data interface
    input  logic [WIDTH-1:0]        data_in,
    input  logic                    valid_in,
    output logic                    ready_out,
    
    // Output interface  
    output logic [WIDTH-1:0]        data_out,
    output logic                    valid_out,
    input  logic                    ready_in
);

    // Internal signals
    logic [WIDTH-1:0] data_reg;
    logic valid_reg;
    
    // Sequential logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_reg <= '0;
            valid_reg <= 1'b0;
        end else if (valid_in && ready_out) begin
            data_reg <= data_in;
            valid_reg <= 1'b1;
        end else if (valid_reg && ready_in) begin
            valid_reg <= 1'b0;
        end
    end
    
    // Output assignments
    assign data_out = data_reg;
    assign valid_out = valid_reg;
    assign ready_out = !valid_reg || ready_in;

endmodule