////////////////////////////////////////////////////////////////////////////
// Module with no AXI-Stream interfaces
//
// This fixture demonstrates a module that has parameters and ports
// but no valid AXI-Stream interfaces. Should fail strict validation
// requiring at least one input and one output interface.
////////////////////////////////////////////////////////////////////////////

module no_interfaces #(
    parameter integer WIDTH = 32,
    parameter integer DEPTH = 16,
    parameter integer ADDR_WIDTH = 4
) (
    // Has clock/reset but not standard names
    input wire clk,
    input wire rst,
    
    // Has ports but not AXI-Stream protocol
    input wire [WIDTH-1:0] data_in,
    input wire valid_in,
    input wire [ADDR_WIDTH-1:0] addr,
    input wire write_enable,
    
    output reg [WIDTH-1:0] data_out,
    output reg valid_out
);

    // Memory array
    reg [WIDTH-1:0] mem [0:DEPTH-1];
    
    // Simple memory with registered output
    always @(posedge clk) begin
        if (rst) begin
            data_out <= '0;
            valid_out <= 1'b0;
        end else begin
            if (write_enable && valid_in) begin
                mem[addr] <= data_in;
            end
            
            data_out <= mem[addr];
            valid_out <= 1'b1;
        end
    end

endmodule