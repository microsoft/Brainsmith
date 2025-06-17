// Test file with INVALID BDIM pragma on CONTROL interface

// @brainsmith BDIM ap AP_BDIM

module test_invalid_control_bdim #(
    parameter AP_BDIM = 1
)(
    input wire ap_clk,
    input wire ap_rst_n
);

// No logic needed for this test

endmodule