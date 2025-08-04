////////////////////////////////////////////////////////////////////////////
// Complete AXI-Stream interfaces with all optional signals
//
// This fixture demonstrates full AXI-Stream protocol with:
// - TDATA, TVALID, TREADY (required)
// - TLAST (packet boundaries)
// - TKEEP (byte enables)
// - TUSER (sideband data)
// - BDIM/SDIM parameters for dataflow
////////////////////////////////////////////////////////////////////////////

module axi_stream_complete #(
    // Interface parameters
    parameter integer S_AXIS_INPUT_BDIM = 32,
    parameter integer S_AXIS_INPUT_SDIM = 512,
    parameter integer M_AXIS_OUTPUT_BDIM = 32,
    
    // Data widths
    parameter integer DATA_WIDTH = 64,
    parameter integer USER_WIDTH = 8,
    
    // Derived parameters
    parameter integer KEEP_WIDTH = DATA_WIDTH / 8
) (
    // Global control
    input wire ap_clk,
    input wire ap_rst_n,
    
    // Complete AXI-Stream slave interface
    input wire [DATA_WIDTH-1:0] s_axis_input_tdata,
    input wire s_axis_input_tvalid,
    output wire s_axis_input_tready,
    input wire s_axis_input_tlast,
    input wire [KEEP_WIDTH-1:0] s_axis_input_tkeep,
    input wire [USER_WIDTH-1:0] s_axis_input_tuser,
    
    // Complete AXI-Stream master interface
    output wire [DATA_WIDTH-1:0] m_axis_output_tdata,
    output wire m_axis_output_tvalid,
    input wire m_axis_output_tready,
    output wire m_axis_output_tlast,
    output wire [KEEP_WIDTH-1:0] m_axis_output_tkeep,
    output wire [USER_WIDTH-1:0] m_axis_output_tuser
);

    // Register all signals for timing
    reg [DATA_WIDTH-1:0] tdata_reg;
    reg tvalid_reg;
    reg tlast_reg;
    reg [KEEP_WIDTH-1:0] tkeep_reg;
    reg [USER_WIDTH-1:0] tuser_reg;
    
    // Input ready when output is ready or no valid data
    assign s_axis_input_tready = m_axis_output_tready || !tvalid_reg;
    
    always @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            tdata_reg <= '0;
            tvalid_reg <= 1'b0;
            tlast_reg <= 1'b0;
            tkeep_reg <= '0;
            tuser_reg <= '0;
        end else begin
            if (s_axis_input_tready) begin
                tdata_reg <= s_axis_input_tdata;
                tvalid_reg <= s_axis_input_tvalid;
                tlast_reg <= s_axis_input_tlast;
                tkeep_reg <= s_axis_input_tkeep;
                tuser_reg <= s_axis_input_tuser;
            end
        end
    end
    
    // Connect registered outputs
    assign m_axis_output_tdata = tdata_reg;
    assign m_axis_output_tvalid = tvalid_reg;
    assign m_axis_output_tlast = tlast_reg;
    assign m_axis_output_tkeep = tkeep_reg;
    assign m_axis_output_tuser = tuser_reg;

endmodule