////////////////////////////////////////////////////////////////////////////
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// @author       Thomas Keller <thomaskeller@microsoft.com>
////////////////////////////////////////////////////////////////////////////
//
// Demo: Indexed Dimension Parameters
// 
// This module demonstrates the new auto-linking feature for multi-dimensional
// BDIM/SDIM parameters using indexed naming convention.
//
// Key features demonstrated:
// 1. Contiguous indexed BDIM parameters (3D tensor)
// 2. Non-contiguous indexed parameters (with gaps)
// 3. Mixed single and indexed parameters
// 4. Interface type restrictions (SDIM only on INPUT/WEIGHT)
// 5. Pragma override behavior
//

// Pragmas for datatypes and special interfaces
// @brainsmith DATATYPE s_axis_input FIXED 16 16
// @brainsmith DATATYPE s_axis_output FIXED 32 32
// @brainsmith DATATYPE m_axis_result UINT 8 32
// @brainsmith WEIGHT s_axis_weights
// @brainsmith BDIM m_axis_result [OUT_H, OUT_W]  // Pragma overrides any indexed params

module tensor_processor_indexed #(
    // Core parameters
    parameter int unsigned COMPUTE_UNITS = 4,
    parameter int unsigned PRECISION = 16,
    
    // Example 1: Contiguous indexed BDIM for 3D input tensor (H x W x C)
    parameter int unsigned s_axis_input_BDIM0 = 224,   // Height (e.g., image height)
    parameter int unsigned s_axis_input_BDIM1 = 224,   // Width (e.g., image width)
    parameter int unsigned s_axis_input_BDIM2 = 3,     // Channels (e.g., RGB)
    
    // Example 2: Indexed SDIM for multi-dimensional streaming
    parameter int unsigned s_axis_input_SDIM0 = 1024,  // Stream dimension 0
    parameter int unsigned s_axis_input_SDIM1 = 512,   // Stream dimension 1
    parameter int unsigned s_axis_input_SDIM2 = 256,   // Stream dimension 2
    
    // Example 3: Non-contiguous BDIM (missing index 1 - will be singleton)
    parameter int unsigned s_axis_weights_BDIM0 = 64,   // Output channels
    parameter int unsigned s_axis_weights_BDIM2 = 3,    // Kernel size
    parameter int unsigned s_axis_weights_BDIM3 = 3,    // Kernel size
    // Note: s_axis_weights_BDIM1 is missing - will be treated as "1"
    
    // Example 4: Single parameter style (traditional)
    parameter int unsigned s_axis_weights_SDIM = 4096,
    
    // Example 5: Output uses BDIM only (SDIM not allowed on OUTPUT interfaces)
    parameter int unsigned s_axis_output_BDIM = 128,
    // Note: s_axis_output_SDIM* parameters would be ignored since
    // SDIM only applies to INPUT and WEIGHT interfaces
    
    // Example 6: Result interface has pragma that overrides indexed params
    parameter int unsigned m_axis_result_BDIM0 = 16,    // These will be ignored
    parameter int unsigned m_axis_result_BDIM1 = 16,    // due to pragma
    parameter int unsigned OUT_H = 8,                   // Pragma uses these
    parameter int unsigned OUT_W = 8,
    
    // Example 7: Config interfaces support datatype parameters but NOT dimensions
    // Datatype parameters can be auto-linked or set via pragma
    parameter int unsigned s_axilite_config_WIDTH = 32,    // Will be auto-linked
    parameter int unsigned s_axilite_config_SIGNED = 0,    // Will be auto-linked
    
    // Example 8: Control interfaces (clk/rst) don't support any parameterization
    // Global control signals have fixed behavior and no configurable datatypes
    // Parameters like ap_clk_WIDTH would be ignored and serve no purpose
    
    // Other parameters
    parameter int unsigned FIFO_DEPTH = 512,
    parameter int unsigned BURST_LENGTH = 16,
    
    // Internal precision parameters (will be auto-linked as internal datatypes)
    parameter int unsigned ACC_WIDTH = 32,
    parameter int unsigned ACC_SIGNED = 1,
    parameter int unsigned THRESH_WIDTH = 16,
    parameter int unsigned THRESH_BIAS = 127
) (
    // Global signals
    input  logic                          ap_clk,
    input  logic                          ap_rst_n,
    
    // Input stream (3D tensor data)
    input  logic                          s_axis_input_tvalid,
    output logic                          s_axis_input_tready,
    input  logic [PRECISION-1:0]          s_axis_input_tdata,
    input  logic                          s_axis_input_tlast,
    
    // Weight stream
    input  logic                          s_axis_weights_tvalid,
    output logic                          s_axis_weights_tready,
    input  logic [PRECISION-1:0]          s_axis_weights_tdata,
    
    // Output stream
    output logic                          s_axis_output_tvalid,
    input  logic                          s_axis_output_tready,
    output logic [31:0]                   s_axis_output_tdata,
    
    // Result stream (with pragma-defined BDIM)
    output logic                          m_axis_result_tvalid,
    input  logic                          m_axis_result_tready,
    output logic [31:0]                   m_axis_result_tdata,
    
    // Configuration interface (AXI-Lite)
    input  logic                          s_axilite_config_awvalid,
    output logic                          s_axilite_config_awready,
    input  logic [11:0]                   s_axilite_config_awaddr,
    
    input  logic                          s_axilite_config_wvalid,
    output logic                          s_axilite_config_wready,
    input  logic [31:0]                   s_axilite_config_wdata,
    input  logic [3:0]                    s_axilite_config_wstrb,
    
    output logic                          s_axilite_config_bvalid,
    input  logic                          s_axilite_config_bready,
    output logic [1:0]                    s_axilite_config_bresp,
    
    input  logic                          s_axilite_config_arvalid,
    output logic                          s_axilite_config_arready,
    input  logic [11:0]                   s_axilite_config_araddr,
    
    output logic                          s_axilite_config_rvalid,
    input  logic                          s_axilite_config_rready,
    output logic [31:0]                   s_axilite_config_rdata,
    output logic [1:0]                    s_axilite_config_rresp
);

    // Implementation details would go here
    // For demo purposes, just wire up some basic connections
    
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            s_axis_input_tready <= 1'b0;
            s_axis_weights_tready <= 1'b0;
            s_axis_output_tvalid <= 1'b0;
            s_axis_output_tdata <= '0;
            m_axis_result_tvalid <= 1'b0;
            m_axis_result_tdata <= '0;
        end else begin
            // Simple pass-through for demo
            s_axis_input_tready <= s_axis_output_tready;
            s_axis_weights_tready <= 1'b1;
            s_axis_output_tvalid <= s_axis_input_tvalid;
            s_axis_output_tdata <= {16'b0, s_axis_input_tdata};
            m_axis_result_tvalid <= s_axis_input_tvalid && s_axis_input_tlast;
            m_axis_result_tdata <= {16'b0, s_axis_input_tdata};
        end
    end
    
    // AXI-Lite dummy responses
    assign s_axilite_config_awready = 1'b1;
    assign s_axilite_config_wready = 1'b1;
    assign s_axilite_config_bvalid = 1'b0;
    assign s_axilite_config_bresp = 2'b00;
    assign s_axilite_config_arready = 1'b1;
    assign s_axilite_config_rvalid = 1'b0;
    assign s_axilite_config_rdata = '0;
    assign s_axilite_config_rresp = 2'b00;

endmodule : tensor_processor_indexed