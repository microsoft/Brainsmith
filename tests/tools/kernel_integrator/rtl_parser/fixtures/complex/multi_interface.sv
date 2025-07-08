////////////////////////////////////////////////////////////////////////////
// Complex module with multiple inputs, outputs, weights, and control
//
// This fixture demonstrates:
// - Multiple input streams with different properties
// - Multiple output streams
// - Multiple weight interfaces
// - AXI-Lite control interface
// - Mix of pragmas and auto-linking
// - Complex parameter relationships
////////////////////////////////////////////////////////////////////////////

module multi_interface #(
    // Input 0: RGB image tensor
    parameter integer IMG_BDIM0 = 8,
    parameter integer IMG_BDIM1 = 8,
    parameter integer IMG_BDIM2 = 3,
    parameter integer IMG_SDIM0 = 224,
    parameter integer IMG_SDIM1 = 224,
    parameter integer IMG_WIDTH = 8,
    
    // Input 1: Feature maps
    parameter integer FEAT_BDIM0 = 16,
    parameter integer FEAT_BDIM1 = 16,
    parameter integer FEAT_BDIM2 = 64,
    parameter integer FEAT_SDIM0 = 56,
    parameter integer FEAT_SDIM1 = 56,
    parameter integer FEAT_WIDTH = 16,
    
    // Weight parameters
    parameter integer CONV_WEIGHTS_BDIM = 576,  // 3x3x64
    parameter integer CONV_WEIGHTS_SDIM = 128,
    parameter integer FC_WEIGHTS_BDIM = 1024,
    parameter integer FC_WEIGHTS_SDIM = 1000,
    parameter integer BIAS_BDIM = 128,
    
    // Output parameters
    parameter integer CONV_OUT_BDIM0 = 16,
    parameter integer CONV_OUT_BDIM1 = 16,
    parameter integer CONV_OUT_BDIM2 = 128,
    parameter integer CONV_OUT_WIDTH = 32,
    parameter integer FC_OUT_BDIM = 1000,
    parameter integer FC_OUT_WIDTH = 16,
    
    // Control parameters
    parameter integer C_S_AXI_ADDR_WIDTH = 7,
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    
    // Processing parameters
    parameter integer PE = 16,  // @brainsmith ALIAS PE ProcessingElements
    parameter integer BATCH_SIZE = 8,  // @brainsmith AXILITE_PARAM BATCH_SIZE s_axi_control
    parameter integer ACC_WIDTH = 48,
    parameter integer ACTIVATION_TYPE = 0  // 0=ReLU, 1=Sigmoid, 2=Tanh
) (
    // Global control
    input wire ap_clk,
    input wire ap_rst_n,
    
    // Image input stream
    // @brainsmith DATATYPE s_axis_img UINT 8 8
    // @brainsmith BDIM s_axis_img [IMG_BDIM0, IMG_BDIM1, IMG_BDIM2]
    // @brainsmith SDIM s_axis_img [IMG_SDIM0, IMG_SDIM1]
    // @brainsmith DATATYPE_PARAM s_axis_img width IMG_WIDTH
    input wire [IMG_WIDTH-1:0] s_axis_img_tdata,
    input wire s_axis_img_tvalid,
    output wire s_axis_img_tready,
    input wire s_axis_img_tlast,
    
    // Feature map input stream
    // @brainsmith DATATYPE s_axis_feat UINT 8 32
    // @brainsmith BDIM s_axis_feat [FEAT_BDIM0, FEAT_BDIM1, FEAT_BDIM2]
    // @brainsmith SDIM s_axis_feat [FEAT_SDIM0, FEAT_SDIM1]
    // @brainsmith DATATYPE_PARAM s_axis_feat width FEAT_WIDTH
    input wire [FEAT_WIDTH-1:0] s_axis_feat_tdata,
    input wire s_axis_feat_tvalid,
    output wire s_axis_feat_tready,
    
    // Convolution weights
    // @brainsmith WEIGHT s_axis_conv_w
    // @brainsmith DATATYPE s_axis_conv_w INT 8 8
    // @brainsmith BDIM s_axis_conv_w CONV_WEIGHTS_BDIM
    // @brainsmith SDIM s_axis_conv_w CONV_WEIGHTS_SDIM
    input wire [7:0] s_axis_conv_w_tdata,
    input wire s_axis_conv_w_tvalid,
    output wire s_axis_conv_w_tready,
    
    // Fully connected weights
    // @brainsmith WEIGHT s_axis_fc_w
    // @brainsmith DATATYPE s_axis_fc_w INT 8 16
    // @brainsmith BDIM s_axis_fc_w FC_WEIGHTS_BDIM
    // @brainsmith SDIM s_axis_fc_w FC_WEIGHTS_SDIM
    input wire [15:0] s_axis_fc_w_tdata,
    input wire s_axis_fc_w_tvalid,
    output wire s_axis_fc_w_tready,
    
    // Bias values
    // @brainsmith WEIGHT s_axis_bias
    // @brainsmith DATATYPE s_axis_bias INT 16 32
    // @brainsmith BDIM s_axis_bias BIAS_BDIM
    // @brainsmith SDIM s_axis_bias 1
    input wire [31:0] s_axis_bias_tdata,
    input wire s_axis_bias_tvalid,
    output wire s_axis_bias_tready,
    
    // Convolution output
    // @brainsmith DATATYPE m_axis_conv UINT 16 64
    // @brainsmith BDIM m_axis_conv [CONV_OUT_BDIM0, CONV_OUT_BDIM1, CONV_OUT_BDIM2]
    // @brainsmith DATATYPE_PARAM m_axis_conv width CONV_OUT_WIDTH
    // @brainsmith RELATIONSHIP s_axis_feat m_axis_conv DEPENDENT 2 2 scaled 2
    output wire [CONV_OUT_WIDTH-1:0] m_axis_conv_tdata,
    output wire m_axis_conv_tvalid,
    input wire m_axis_conv_tready,
    output wire m_axis_conv_tlast,
    
    // FC output
    // @brainsmith DATATYPE m_axis_fc UINT 8 32
    // @brainsmith BDIM m_axis_fc FC_OUT_BDIM
    // @brainsmith DATATYPE_PARAM m_axis_fc width FC_OUT_WIDTH
    output wire [FC_OUT_WIDTH-1:0] m_axis_fc_tdata,
    output wire m_axis_fc_tvalid,
    input wire m_axis_fc_tready,
    
    // AXI-Lite control interface
    // @brainsmith AXILITE_PARAM ACTIVATION_TYPE s_axi_control
    input wire [C_S_AXI_ADDR_WIDTH-1:0] s_axi_control_awaddr,
    input wire s_axi_control_awvalid,
    output wire s_axi_control_awready,
    input wire [C_S_AXI_DATA_WIDTH-1:0] s_axi_control_wdata,
    input wire [(C_S_AXI_DATA_WIDTH/8)-1:0] s_axi_control_wstrb,
    input wire s_axi_control_wvalid,
    output wire s_axi_control_wready,
    output wire [1:0] s_axi_control_bresp,
    output wire s_axi_control_bvalid,
    input wire s_axi_control_bready,
    input wire [C_S_AXI_ADDR_WIDTH-1:0] s_axi_control_araddr,
    input wire s_axi_control_arvalid,
    output wire s_axi_control_arready,
    output wire [C_S_AXI_DATA_WIDTH-1:0] s_axi_control_rdata,
    output wire [1:0] s_axi_control_rresp,
    output wire s_axi_control_rvalid,
    input wire s_axi_control_rready
);

    // @brainsmith DATATYPE_PARAM accumulator width ACC_WIDTH
    // @brainsmith DATATYPE_PARAM accumulator signed 1
    
    // Internal processing state
    reg [ACC_WIDTH-1:0] conv_accumulator;
    reg [ACC_WIDTH-1:0] fc_accumulator;
    reg [31:0] config_reg;
    reg processing_enable;
    
    // Control signals
    wire img_processing = s_axis_img_tvalid && s_axis_conv_w_tvalid;
    wire feat_processing = s_axis_feat_tvalid && s_axis_conv_w_tvalid;
    wire fc_processing = (conv_accumulator != 0) && s_axis_fc_w_tvalid;
    
    // Processing pipeline
    always @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            conv_accumulator <= '0;
            fc_accumulator <= '0;
            processing_enable <= 1'b1;
        end else begin
            // Convolution processing
            if (img_processing || feat_processing) begin
                if (img_processing) begin
                    // Process image data
                    conv_accumulator <= conv_accumulator + 
                        (s_axis_img_tdata * $signed(s_axis_conv_w_tdata));
                end else begin
                    // Process feature data
                    conv_accumulator <= conv_accumulator + 
                        (s_axis_feat_tdata * $signed(s_axis_conv_w_tdata));
                end
                
                // Add bias
                if (s_axis_bias_tvalid) begin
                    conv_accumulator <= conv_accumulator + $signed(s_axis_bias_tdata);
                end
            end
            
            // FC processing
            if (fc_processing) begin
                fc_accumulator <= fc_accumulator + 
                    (conv_accumulator[31:0] * $signed(s_axis_fc_w_tdata));
            end
        end
    end
    
    // Activation function (simplified)
    wire [CONV_OUT_WIDTH-1:0] activated_conv = 
        (config_reg[1:0] == 2'b00) ? // ReLU
            (conv_accumulator[ACC_WIDTH-1] ? '0 : conv_accumulator[CONV_OUT_WIDTH-1:0]) :
        conv_accumulator[CONV_OUT_WIDTH-1:0]; // Pass-through for others
    
    // Output generation
    assign m_axis_conv_tdata = activated_conv;
    assign m_axis_conv_tvalid = processing_enable && (conv_accumulator != 0);
    assign m_axis_conv_tlast = s_axis_img_tlast;
    
    assign m_axis_fc_tdata = fc_accumulator[FC_OUT_WIDTH-1:0];
    assign m_axis_fc_tvalid = processing_enable && (fc_accumulator != 0);
    
    // Input ready signals
    assign s_axis_img_tready = m_axis_conv_tready && processing_enable;
    assign s_axis_feat_tready = m_axis_conv_tready && processing_enable;
    assign s_axis_conv_w_tready = m_axis_conv_tready && processing_enable;
    assign s_axis_fc_w_tready = m_axis_fc_tready && processing_enable;
    assign s_axis_bias_tready = m_axis_conv_tready && processing_enable;
    
    // Simplified AXI-Lite (just store config_reg)
    always @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            config_reg <= 32'h00000001; // Enable + ReLU
        end else if (s_axi_control_wvalid && s_axi_control_awvalid) begin
            if (s_axi_control_awaddr[6:2] == 5'h00) begin
                config_reg <= s_axi_control_wdata;
            end
        end
    end
    
    // AXI-Lite outputs (simplified)
    assign s_axi_control_awready = 1'b1;
    assign s_axi_control_wready = 1'b1;
    assign s_axi_control_bresp = 2'b00;
    assign s_axi_control_bvalid = s_axi_control_wvalid;
    assign s_axi_control_arready = 1'b1;
    assign s_axi_control_rdata = config_reg;
    assign s_axi_control_rresp = 2'b00;
    assign s_axi_control_rvalid = s_axi_control_arvalid;

endmodule