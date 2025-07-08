////////////////////////////////////////////////////////////////////////////
// Standard global control interface patterns
//
// This fixture demonstrates different control signal naming conventions
// that should all be recognized as valid global control interfaces
////////////////////////////////////////////////////////////////////////////

module global_control #(
    parameter integer DATA_WIDTH = 32
) (
    // Standard Xilinx/FINN naming
    input wire ap_clk,
    input wire ap_rst_n,
    
    // Alternative valid patterns that could be recognized:
    // input wire aclk,          // AXI standard clock
    // input wire aresetn,       // AXI standard active-low reset
    // input wire clk,           // Simple clock
    // input wire rst_n,         // Simple active-low reset
    // input wire clock,         // Verbose clock
    // input wire reset_n,       // Verbose active-low reset
    
    // Simple data path to make module valid
    input wire [DATA_WIDTH-1:0] data_in,
    output reg [DATA_WIDTH-1:0] data_out
);

    // Register data on clock
    always @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            data_out <= '0;
        end else begin
            data_out <= data_in;
        end
    end

endmodule