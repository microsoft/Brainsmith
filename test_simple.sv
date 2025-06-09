module test_simple (
    input wire clk,
    input wire rst_n,
    input wire [7:0] data_in,
    output wire [7:0] data_out
);

assign data_out = data_in;

endmodule