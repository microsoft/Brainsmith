// Test module for AXI-Lite suffix matching

module test_axilite_suffix #(
    parameter INPUT_WIDTH = 8,
    parameter OUTPUT_WIDTH = 8,
    // These should be detected as AXI-Lite params by suffix
    parameter CONFIG_AXILITE = 1,
    parameter DEBUG_EN = 0,
    parameter MONITOR_EN = 1,
    // This should remain general
    parameter NORMAL_PARAM = 42
) (
    input wire ap_clk,
    input wire ap_rst_n,
    
    // Input interface
    input wire [INPUT_WIDTH-1:0] s_axis_input_TDATA,
    input wire s_axis_input_TVALID,
    output wire s_axis_input_TREADY,
    
    // Output interface  
    output wire [OUTPUT_WIDTH-1:0] m_axis_output_TDATA,
    output wire m_axis_output_TVALID,
    input wire m_axis_output_TREADY
);

// Dummy implementation
assign s_axis_input_TREADY = 1'b1;
assign m_axis_output_TDATA = s_axis_input_TDATA;
assign m_axis_output_TVALID = s_axis_input_TVALID;

endmodule