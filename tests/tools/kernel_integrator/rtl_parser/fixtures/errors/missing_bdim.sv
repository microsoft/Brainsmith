////////////////////////////////////////////////////////////////////////////
// AXI-Stream interface missing required BDIM parameter
//
// This fixture demonstrates the error when an AXI-Stream interface
// lacks the required BDIM parameter and no pragma is provided.
// Should fail strict validation.
////////////////////////////////////////////////////////////////////////////

module missing_bdim #(
    // Has SDIM but missing BDIM
    parameter integer s_axis_data_SDIM = 1024,
    parameter integer m_axis_result_SDIM = 1024,
    
    // Output has BDIM but input doesn't
    parameter integer m_axis_result_BDIM = 32,
    
    // Other parameters
    parameter integer DATA_WIDTH = 32
) (
    // Global control (present)
    input wire ap_clk,
    input wire ap_rst_n,
    
    // Input missing BDIM parameter (neither s_axis_data_BDIM nor pragma)
    input wire [DATA_WIDTH-1:0] s_axis_data_tdata,
    input wire s_axis_data_tvalid,
    output wire s_axis_data_tready,
    
    // Output has BDIM (valid)
    output wire [DATA_WIDTH-1:0] m_axis_result_tdata,
    output wire m_axis_result_tvalid,
    input wire m_axis_result_tready
);

    // Implementation
    reg [DATA_WIDTH-1:0] data_reg;
    reg valid_reg;
    
    always @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            data_reg <= '0;
            valid_reg <= 1'b0;
        end else if (s_axis_data_tvalid && s_axis_data_tready) begin
            data_reg <= s_axis_data_tdata;
            valid_reg <= 1'b1;
        end else if (m_axis_result_tready) begin
            valid_reg <= 1'b0;
        end
    end
    
    assign m_axis_result_tdata = data_reg;
    assign m_axis_result_tvalid = valid_reg;
    assign s_axis_data_tready = !valid_reg || m_axis_result_tready;

endmodule