////////////////////////////////////////////////////////////////////////////
// Module demonstrating all pragma types
//
// This fixture shows every pragma type in action with valid syntax
// and appropriate context for each pragma.
////////////////////////////////////////////////////////////////////////////

// @brainsmith TOP_MODULE all_pragmas
module all_pragmas #(
    // Interface dimension parameters
    parameter integer IN0_BDIM0 = 16,
    parameter integer IN0_BDIM1 = 16,
    parameter integer IN0_BDIM2 = 3,
    parameter integer IN0_SDIM0 = 224,
    parameter integer IN0_SDIM1 = 224,
    parameter integer WEIGHTS_BDIM = 64,
    parameter integer WEIGHTS_SDIM = 512,
    parameter integer OUT0_BDIM = 32,
    
    // Datatype parameters
    parameter integer INPUT_WIDTH = 16,
    parameter integer INPUT_SIGNED = 0,
    parameter integer WEIGHT_WIDTH = 8,
    parameter integer WEIGHT_SIGNED = 1,
    parameter integer OUTPUT_WIDTH = 32,
    parameter integer ACC_WIDTH = 48,
    parameter integer ACC_SIGNED = 1,
    parameter integer THRESH_WIDTH = 32,
    
    // Configuration parameters
    parameter integer PE = 8,           // @brainsmith ALIAS PE ParallelismFactor
    parameter integer BATCH_SIZE = 16,  // @brainsmith AXILITE_PARAM BATCH_SIZE s_axi_config
    parameter integer MEM_DEPTH = 1024,
    parameter integer MEM_SIZE = 32768  // @brainsmith DERIVED_PARAMETER MEM_SIZE MEM_DEPTH * 32
) (
    // Global control
    input wire ap_clk,
    input wire ap_rst_n,
    
    // @brainsmith DATATYPE s_axis_in0 UINT 8 32
    // @brainsmith BDIM s_axis_in0 [IN0_BDIM0, IN0_BDIM1, IN0_BDIM2]
    // @brainsmith SDIM s_axis_in0 [IN0_SDIM0, IN0_SDIM1]
    // @brainsmith DATATYPE_PARAM s_axis_in0 width INPUT_WIDTH
    // @brainsmith DATATYPE_PARAM s_axis_in0 signed INPUT_SIGNED
    input wire [INPUT_WIDTH-1:0] s_axis_in0_tdata,
    input wire s_axis_in0_tvalid,
    output wire s_axis_in0_tready,
    
    // @brainsmith WEIGHT s_axis_weights
    // @brainsmith DATATYPE s_axis_weights INT 8 8
    // @brainsmith BDIM s_axis_weights WEIGHTS_BDIM
    // @brainsmith SDIM s_axis_weights WEIGHTS_SDIM
    // @brainsmith DATATYPE_PARAM s_axis_weights width WEIGHT_WIDTH
    // @brainsmith DATATYPE_PARAM s_axis_weights signed WEIGHT_SIGNED
    input wire [WEIGHT_WIDTH-1:0] s_axis_weights_tdata,
    input wire s_axis_weights_tvalid,
    output wire s_axis_weights_tready,
    
    // @brainsmith DATATYPE m_axis_out0 UINT 16 64
    // @brainsmith BDIM m_axis_out0 OUT0_BDIM
    // @brainsmith DATATYPE_PARAM m_axis_out0 width OUTPUT_WIDTH
    // @brainsmith RELATIONSHIP s_axis_in0 m_axis_out0 EQUAL
    output wire [OUTPUT_WIDTH-1:0] m_axis_out0_tdata,
    output wire m_axis_out0_tvalid,
    input wire m_axis_out0_tready,
    
    // AXI-Lite configuration interface
    input wire [5:0] s_axi_config_awaddr,
    input wire s_axi_config_awvalid,
    output wire s_axi_config_awready,
    input wire [31:0] s_axi_config_wdata,
    input wire [3:0] s_axi_config_wstrb,
    input wire s_axi_config_wvalid,
    output wire s_axi_config_wready,
    output wire [1:0] s_axi_config_bresp,
    output wire s_axi_config_bvalid,
    input wire s_axi_config_bready,
    input wire [5:0] s_axi_config_araddr,
    input wire s_axi_config_arvalid,
    output wire s_axi_config_arready,
    output wire [31:0] s_axi_config_rdata,
    output wire [1:0] s_axi_config_rresp,
    output wire s_axi_config_rvalid,
    input wire s_axi_config_rready
);

    // @brainsmith DATATYPE_PARAM accumulator width ACC_WIDTH
    // @brainsmith DATATYPE_PARAM accumulator signed ACC_SIGNED
    // @brainsmith DATATYPE_PARAM threshold width THRESH_WIDTH
    
    // Internal accumulator using pragma-defined width
    reg [ACC_WIDTH-1:0] accumulator;
    reg [THRESH_WIDTH-1:0] threshold;
    
    // Simple processing pipeline
    always @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            accumulator <= '0;
            threshold <= '0;
        end else begin
            if (s_axis_in0_tvalid && s_axis_weights_tvalid) begin
                // Multiply-accumulate with sign extension
                if (INPUT_SIGNED && WEIGHT_SIGNED) begin
                    accumulator <= accumulator + 
                        $signed(s_axis_in0_tdata) * $signed(s_axis_weights_tdata);
                end else begin
                    accumulator <= accumulator + 
                        s_axis_in0_tdata * s_axis_weights_tdata;
                end
            end
        end
    end
    
    // Output logic
    assign m_axis_out0_tdata = accumulator[OUTPUT_WIDTH-1:0];
    assign m_axis_out0_tvalid = (accumulator > threshold);
    assign s_axis_in0_tready = m_axis_out0_tready;
    assign s_axis_weights_tready = m_axis_out0_tready;
    
    // Stub AXI-Lite interface
    assign s_axi_config_awready = 1'b1;
    assign s_axi_config_wready = 1'b1;
    assign s_axi_config_bresp = 2'b00;
    assign s_axi_config_bvalid = 1'b0;
    assign s_axi_config_arready = 1'b1;
    assign s_axi_config_rdata = 32'h0;
    assign s_axi_config_rresp = 2'b00;
    assign s_axi_config_rvalid = 1'b0;

endmodule

// This module should be ignored due to TOP_MODULE pragma
module ignored_module (
    input wire dummy
);
endmodule