////////////////////////////////////////////////////////////////////////////
// Complete AXI-Lite slave interface
//
// This fixture demonstrates a full AXI-Lite configuration interface with:
// - All five channels (AW, W, B, AR, R)
// - Proper handshaking signals
// - Address decoding for multiple registers
// - No BDIM/SDIM (control interfaces don't use them)
////////////////////////////////////////////////////////////////////////////

module axi_lite_complete #(
    // AXI-Lite parameters
    parameter integer C_S_AXI_ADDR_WIDTH = 6,  // 64 bytes = 16 registers
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    
    // Data interface parameters
    parameter integer INPUT_BDIM = 64,
    parameter integer INPUT_SDIM = 1024,
    parameter integer OUTPUT_BDIM = 64
) (
    // Global control
    input wire ap_clk,
    input wire ap_rst_n,
    
    // AXI-Lite slave interface
    // Write address channel
    input wire [C_S_AXI_ADDR_WIDTH-1:0] s_axi_control_awaddr,
    input wire s_axi_control_awvalid,
    output wire s_axi_control_awready,
    
    // Write data channel
    input wire [C_S_AXI_DATA_WIDTH-1:0] s_axi_control_wdata,
    input wire [(C_S_AXI_DATA_WIDTH/8)-1:0] s_axi_control_wstrb,
    input wire s_axi_control_wvalid,
    output wire s_axi_control_wready,
    
    // Write response channel
    output wire [1:0] s_axi_control_bresp,
    output wire s_axi_control_bvalid,
    input wire s_axi_control_bready,
    
    // Read address channel
    input wire [C_S_AXI_ADDR_WIDTH-1:0] s_axi_control_araddr,
    input wire s_axi_control_arvalid,
    output wire s_axi_control_arready,
    
    // Read data channel
    output wire [C_S_AXI_DATA_WIDTH-1:0] s_axi_control_rdata,
    output wire [1:0] s_axi_control_rresp,
    output wire s_axi_control_rvalid,
    input wire s_axi_control_rready,
    
    // Data interfaces (to show complete module)
    input wire [31:0] s_axis_input_tdata,
    input wire s_axis_input_tvalid,
    output wire s_axis_input_tready,
    
    output wire [31:0] m_axis_output_tdata,
    output wire m_axis_output_tvalid,
    input wire m_axis_output_tready
);

    // Configuration registers
    reg [31:0] ctrl_reg;      // 0x00: Control register
    reg [31:0] status_reg;    // 0x04: Status register
    reg [31:0] config_reg;    // 0x08: Configuration
    reg [31:0] threshold_reg; // 0x0C: Threshold value
    
    // AXI-Lite write FSM
    reg aw_ready_reg;
    reg w_ready_reg;
    reg [1:0] b_resp_reg;
    reg b_valid_reg;
    reg [C_S_AXI_ADDR_WIDTH-1:0] aw_addr_reg;
    
    // AXI-Lite read FSM
    reg ar_ready_reg;
    reg [31:0] r_data_reg;
    reg [1:0] r_resp_reg;
    reg r_valid_reg;
    
    // Write handling
    wire aw_handshake = s_axi_control_awvalid && aw_ready_reg;
    wire w_handshake = s_axi_control_wvalid && w_ready_reg;
    wire write_enable = aw_handshake && w_handshake;
    
    always @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            ctrl_reg <= 32'h0;
            config_reg <= 32'h0;
            threshold_reg <= 32'h100;
            aw_ready_reg <= 1'b1;
            w_ready_reg <= 1'b1;
            b_valid_reg <= 1'b0;
            b_resp_reg <= 2'b00;
            aw_addr_reg <= '0;
        end else begin
            // Address write channel
            if (aw_handshake) begin
                aw_addr_reg <= s_axi_control_awaddr;
                aw_ready_reg <= 1'b0;
            end else if (write_enable) begin
                aw_ready_reg <= 1'b1;
            end
            
            // Data write channel
            if (w_handshake) begin
                w_ready_reg <= 1'b0;
            end else if (write_enable) begin
                w_ready_reg <= 1'b1;
            end
            
            // Write response channel
            if (write_enable) begin
                b_valid_reg <= 1'b1;
                b_resp_reg <= 2'b00; // OKAY
                
                // Decode address and write registers
                case (aw_addr_reg[5:2])
                    4'h0: ctrl_reg <= s_axi_control_wdata;
                    4'h2: config_reg <= s_axi_control_wdata;
                    4'h3: threshold_reg <= s_axi_control_wdata;
                    default: b_resp_reg <= 2'b10; // SLVERR
                endcase
            end else if (b_valid_reg && s_axi_control_bready) begin
                b_valid_reg <= 1'b0;
            end
        end
    end
    
    // Read handling
    always @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            ar_ready_reg <= 1'b1;
            r_valid_reg <= 1'b0;
            r_data_reg <= 32'h0;
            r_resp_reg <= 2'b00;
        end else begin
            // Read address channel
            if (s_axi_control_arvalid && ar_ready_reg) begin
                ar_ready_reg <= 1'b0;
                r_valid_reg <= 1'b1;
                r_resp_reg <= 2'b00; // OKAY
                
                // Decode address and read registers
                case (s_axi_control_araddr[5:2])
                    4'h0: r_data_reg <= ctrl_reg;
                    4'h1: r_data_reg <= status_reg;
                    4'h2: r_data_reg <= config_reg;
                    4'h3: r_data_reg <= threshold_reg;
                    default: begin
                        r_data_reg <= 32'hDEADBEEF;
                        r_resp_reg <= 2'b10; // SLVERR
                    end
                endcase
            end else if (r_valid_reg && s_axi_control_rready) begin
                r_valid_reg <= 1'b0;
                ar_ready_reg <= 1'b1;
            end
        end
    end
    
    // Update status register
    always @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            status_reg <= 32'h0;
        end else begin
            status_reg[0] <= s_axis_input_tvalid;
            status_reg[1] <= m_axis_output_tvalid;
            status_reg[2] <= ctrl_reg[0]; // Echo enable bit
        end
    end
    
    // Connect AXI-Lite outputs
    assign s_axi_control_awready = aw_ready_reg;
    assign s_axi_control_wready = w_ready_reg;
    assign s_axi_control_bresp = b_resp_reg;
    assign s_axi_control_bvalid = b_valid_reg;
    assign s_axi_control_arready = ar_ready_reg;
    assign s_axi_control_rdata = r_data_reg;
    assign s_axi_control_rresp = r_resp_reg;
    assign s_axi_control_rvalid = r_valid_reg;
    
    // Simple data path controlled by config
    assign m_axis_output_tdata = ctrl_reg[0] ? 
        (s_axis_input_tdata + threshold_reg) : 
        s_axis_input_tdata;
    assign m_axis_output_tvalid = s_axis_input_tvalid && ctrl_reg[0];
    assign s_axis_input_tready = m_axis_output_tready || !ctrl_reg[0];

endmodule