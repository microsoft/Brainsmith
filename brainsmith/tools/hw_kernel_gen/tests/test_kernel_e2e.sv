////////////////////////////////////////////////////////////////////////////
// Test kernel for end-to-end HKG verification
// This module exercises all pragma types and features
////////////////////////////////////////////////////////////////////////////

// @brainsmith top_module test_kernel_e2e
// @brainsmith datatype s_axis_input UINT 8 32
// @brainsmith datatype m_axis_output UINT 8 32
// @brainsmith datatype s_axis_weights FIXED 8 16
// @brainsmith weight s_axis_weights
// @brainsmith bdim s_axis_input INPUT_BDIM SHAPE=[:]
// @brainsmith sdim s_axis_input INPUT_SDIM
// @brainsmith bdim s_axis_weights WEIGHT_BDIM SHAPE=[:,:] RINDEX=1
// @brainsmith alias PE num_engines
// @brainsmith derived_parameter MEM_DEPTH self.calc_memory_depth()
// @brainsmith datatype_param s_axis_input width INPUT_WIDTH
// @brainsmith datatype_param s_axis_weights width WEIGHT_WIDTH
// @brainsmith datatype_param s_axis_weights signed WEIGHT_SIGNED
// @brainsmith datatype_param accumulator width ACC_WIDTH
// @brainsmith datatype_param accumulator signed ACC_SIGNED
// @brainsmith datatype_param threshold width THRESH_WIDTH

module test_kernel_e2e #(
    parameter int INPUT_WIDTH = 8,
    parameter int WEIGHT_WIDTH = 8,
    parameter int WEIGHT_SIGNED = 1,
    parameter int OUTPUT_WIDTH = 8,
    parameter int ACC_WIDTH = 32,
    parameter int ACC_SIGNED = 1,
    parameter int THRESH_WIDTH = 16,
    parameter int INPUT_BDIM = 1,
    parameter int INPUT_SDIM = 1,
    parameter int WEIGHT_BDIM = 1,
    parameter int MEM_DEPTH = 1024,
    parameter int ACTIVATION_TYPE = 0  // 0=ReLU, 1=None
) (
    // Global control
    input  logic ap_clk,
    input  logic ap_rst_n,
    
    // Input stream
    input  logic [INPUT_WIDTH-1:0] s_axis_input_tdata,
    input  logic s_axis_input_tvalid,
    output logic s_axis_input_tready,
    
    // Weight stream
    input  logic [WEIGHT_WIDTH-1:0] s_axis_weights_tdata,
    input  logic s_axis_weights_tvalid,
    output logic s_axis_weights_tready,
    
    // Output stream
    output logic [OUTPUT_WIDTH-1:0] m_axis_output_tdata,
    output logic m_axis_output_tvalid,
    input  logic m_axis_output_tready,
    
    // AXI-Lite control interface
    input  logic [15:0] s_axilite_config_awaddr,
    input  logic s_axilite_config_awvalid,
    output logic s_axilite_config_awready,
    input  logic [31:0] s_axilite_config_wdata,
    input  logic [3:0] s_axilite_config_wstrb,
    input  logic s_axilite_config_wvalid,
    output logic s_axilite_config_wready,
    output logic [1:0] s_axilite_config_bresp,
    output logic s_axilite_config_bvalid,
    input  logic s_axilite_config_bready,
    input  logic [15:0] s_axilite_config_araddr,
    input  logic s_axilite_config_arvalid,
    output logic s_axilite_config_arready,
    output logic [31:0] s_axilite_config_rdata,
    output logic [1:0] s_axilite_config_rresp,
    output logic s_axilite_config_rvalid,
    input  logic s_axilite_config_rready
);

    // Internal signals
    logic [ACC_WIDTH-1:0] accumulator;
    logic [THRESH_WIDTH-1:0] threshold_reg;
    logic processing_done;
    
    // Simple processing logic for testing
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            accumulator <= '0;
            processing_done <= 1'b0;
            m_axis_output_tdata <= '0;
            m_axis_output_tvalid <= 1'b0;
            s_axis_input_tready <= 1'b1;
            s_axis_weights_tready <= 1'b1;
        end else begin
            // Simple MAC operation
            if (s_axis_input_tvalid && s_axis_input_tready && 
                s_axis_weights_tvalid && s_axis_weights_tready) begin
                if (WEIGHT_SIGNED) begin
                    accumulator <= accumulator + $signed(s_axis_input_tdata) * $signed(s_axis_weights_tdata);
                end else begin
                    accumulator <= accumulator + s_axis_input_tdata * s_axis_weights_tdata;
                end
            end
            
            // Output logic with threshold
            if (processing_done && m_axis_output_tready) begin
                if (ACTIVATION_TYPE == 0) begin
                    // ReLU activation
                    m_axis_output_tdata <= (accumulator > threshold_reg) ? accumulator[OUTPUT_WIDTH-1:0] : '0;
                end else begin
                    // No activation
                    m_axis_output_tdata <= accumulator[OUTPUT_WIDTH-1:0];
                end
                m_axis_output_tvalid <= 1'b1;
            end else if (m_axis_output_tvalid && m_axis_output_tready) begin
                m_axis_output_tvalid <= 1'b0;
            end
        end
    end
    
    // AXI-Lite interface (simplified)
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            s_axilite_config_awready <= 1'b1;
            s_axilite_config_wready <= 1'b1;
            s_axilite_config_bresp <= 2'b00;
            s_axilite_config_bvalid <= 1'b0;
            s_axilite_config_arready <= 1'b1;
            s_axilite_config_rdata <= '0;
            s_axilite_config_rresp <= 2'b00;
            s_axilite_config_rvalid <= 1'b0;
            threshold_reg <= '0;
        end else begin
            // Write handling
            if (s_axilite_config_awvalid && s_axilite_config_awready &&
                s_axilite_config_wvalid && s_axilite_config_wready) begin
                if (s_axilite_config_awaddr == 16'h0010) begin
                    threshold_reg <= s_axilite_config_wdata[THRESH_WIDTH-1:0];
                end
                s_axilite_config_bvalid <= 1'b1;
            end else if (s_axilite_config_bvalid && s_axilite_config_bready) begin
                s_axilite_config_bvalid <= 1'b0;
            end
            
            // Read handling
            if (s_axilite_config_arvalid && s_axilite_config_arready) begin
                case (s_axilite_config_araddr)
                    16'h0010: s_axilite_config_rdata <= {{(32-THRESH_WIDTH){1'b0}}, threshold_reg};
                    16'h0020: s_axilite_config_rdata <= 32'hDEADBEEF; // ID register
                    default: s_axilite_config_rdata <= 32'h0;
                endcase
                s_axilite_config_rvalid <= 1'b1;
            end else if (s_axilite_config_rvalid && s_axilite_config_rready) begin
                s_axilite_config_rvalid <= 1'b0;
            end
        end
    end

endmodule

// Decoy module to test TOP_MODULE pragma
module decoy_module (
    input logic clk,
    input logic rst
);
    // Empty module
endmodule