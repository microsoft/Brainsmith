////////////////////////////////////////////////////////////////////////////
// Input interface missing required SDIM parameter
//
// This fixture demonstrates the error when an INPUT interface
// lacks the required SDIM parameter. INPUT and WEIGHT interfaces
// must have both BDIM and SDIM, while OUTPUT only needs BDIM.
////////////////////////////////////////////////////////////////////////////

module missing_sdim_input #(
    // Input has BDIM but missing SDIM
    parameter integer s_axis_input_BDIM = 64,
    // Missing: parameter integer s_axis_input_SDIM = ...,
    
    // Weights missing SDIM (weights also require SDIM)
    parameter integer s_axis_weights_BDIM = 32,
    // Missing: parameter integer s_axis_weights_SDIM = ...,
    
    // Output correctly has only BDIM (SDIM not required for outputs)
    parameter integer m_axis_output_BDIM = 64,
    
    parameter integer DATA_WIDTH = 16,
    parameter integer WEIGHT_WIDTH = 8
) (
    // Global control (present)
    input wire ap_clk,
    input wire ap_rst_n,
    
    // Input interface - missing SDIM
    input wire [DATA_WIDTH-1:0] s_axis_input_tdata,
    input wire s_axis_input_tvalid,
    output wire s_axis_input_tready,
    
    // Weight interface - also missing SDIM
    // @brainsmith WEIGHT s_axis_weights
    input wire [WEIGHT_WIDTH-1:0] s_axis_weights_tdata,
    input wire s_axis_weights_tvalid,
    output wire s_axis_weights_tready,
    
    // Output interface - correctly has only BDIM
    output wire [DATA_WIDTH-1:0] m_axis_output_tdata,
    output wire m_axis_output_tvalid,
    input wire m_axis_output_tready
);

    // Simple multiply-accumulate
    reg [DATA_WIDTH+WEIGHT_WIDTH-1:0] acc;
    
    always @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            acc <= '0;
        end else if (s_axis_input_tvalid && s_axis_weights_tvalid) begin
            acc <= acc + (s_axis_input_tdata * s_axis_weights_tdata);
        end
    end
    
    assign m_axis_output_tdata = acc[DATA_WIDTH-1:0];
    assign m_axis_output_tvalid = (acc != 0);
    assign s_axis_input_tready = m_axis_output_tready;
    assign s_axis_weights_tready = m_axis_output_tready;

endmodule