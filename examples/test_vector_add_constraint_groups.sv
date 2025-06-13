// Test SystemVerilog module with DATATYPE pragmas for constraint groups
// @brainsmith TOP_MODULE vector_add
// @brainsmith DATATYPE in0 UINT 8 16
// @brainsmith DATATYPE in1 UINT 8 16  
// @brainsmith DATATYPE out0 UINT 8 16
// @brainsmith BDIM in0 [PE]
// @brainsmith BDIM in1 [PE]
// @brainsmith BDIM out0 [PE]

module vector_add #(
    parameter PE = 8
)(
    input wire clk,
    input wire rst_n,
    
    // AXI-Stream input 0
    input wire [63:0] in0_V_data_V_TDATA,
    input wire in0_V_data_V_TVALID,
    output wire in0_V_data_V_TREADY,
    
    // AXI-Stream input 1  
    input wire [63:0] in1_V_data_V_TDATA,
    input wire in1_V_data_V_TVALID,
    output wire in1_V_data_V_TREADY,
    
    // AXI-Stream output
    output wire [63:0] out0_V_data_V_TDATA,
    output wire out0_V_data_V_TVALID,
    input wire out0_V_data_V_TREADY
);

// Vector addition logic
assign out0_V_data_V_TDATA = in0_V_data_V_TDATA + in1_V_data_V_TDATA;
assign out0_V_data_V_TVALID = in0_V_data_V_TVALID & in1_V_data_V_TVALID;
assign in0_V_data_V_TREADY = out0_V_data_V_TREADY;
assign in1_V_data_V_TREADY = out0_V_data_V_TREADY;

endmodule