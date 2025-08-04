////////////////////////////////////////////////////////////////////////////
// Minimal SystemVerilog module for non-strict validation tests
//
// This fixture is used for testing basic parsing functionality
// without the full requirements of strict mode.
// It lacks:
// - Proper global control interface
// - AXI-Stream interfaces
// - BDIM/SDIM parameters
////////////////////////////////////////////////////////////////////////////

module non_strict_minimal #(
    parameter integer WIDTH = 32,
    parameter integer DEPTH = 16
) (
    input wire clk,
    input wire rst,
    input wire [WIDTH-1:0] data_in,
    output wire [WIDTH-1:0] data_out,
    output wire valid
);

    // Simple registered passthrough
    reg [WIDTH-1:0] data_reg;
    reg valid_reg;
    
    always @(posedge clk) begin
        if (rst) begin
            data_reg <= '0;
            valid_reg <= 1'b0;
        end else begin
            data_reg <= data_in;
            valid_reg <= 1'b1;
        end
    end
    
    assign data_out = data_reg;
    assign valid = valid_reg;

endmodule