////////////////////////////////////////////////////////////////////////////
// Demo 02: All Interface Types
// 
// This example demonstrates all five interface types that the RTL parser
// recognizes and their characteristics:
// - INPUT: Data input streams
// - OUTPUT: Data output streams  
// - WEIGHT: Weight/parameter streams
// - CONFIG: AXI-Lite configuration
// - CONTROL: Clock/reset signals
////////////////////////////////////////////////////////////////////////////

module all_interfaces #(
    parameter int unsigned INPUT_WIDTH = 16,
    parameter int unsigned OUTPUT_WIDTH = 32,
    parameter int unsigned WEIGHT_WIDTH = 8,
    parameter int unsigned CONFIG_WIDTH = 32
) (
    // CONTROL interface (ap prefix = clock/reset)
    // - No parameterization allowed
    // - Fixed behavior
    input  logic                        ap_clk,
    input  logic                        ap_rst_n,
    
    // INPUT interface (s_axis prefix)
    // - Supports: datatypes, BDIM, SDIM
    input  logic [INPUT_WIDTH-1:0]      s_axis_data_tdata,
    input  logic                        s_axis_data_tvalid,
    output logic                        s_axis_data_tready,
    input  logic                        s_axis_data_tlast,
    
    // WEIGHT interface (detected by suffix or pragma)
    // - Supports: datatypes, BDIM, SDIM
    input  logic [WEIGHT_WIDTH-1:0]     weights_V_dout,
    input  logic                        weights_V_empty_n,
    output logic                        weights_V_read,
    
    // OUTPUT interface (m_axis prefix)
    // - Supports: datatypes, BDIM
    // - Does NOT support: SDIM
    output logic [OUTPUT_WIDTH-1:0]     m_axis_result_tdata,
    output logic                        m_axis_result_tvalid,
    input  logic                        m_axis_result_tready,
    output logic                        m_axis_result_tlast,
    
    // CONFIG interface (s_axilite prefix)
    // - Supports: datatypes only
    // - Does NOT support: BDIM, SDIM
    input  logic                        s_axilite_ctrl_awvalid,
    output logic                        s_axilite_ctrl_awready,
    input  logic [11:0]                 s_axilite_ctrl_awaddr,
    input  logic                        s_axilite_ctrl_wvalid,
    output logic                        s_axilite_ctrl_wready,
    input  logic [CONFIG_WIDTH-1:0]     s_axilite_ctrl_wdata,
    input  logic [3:0]                  s_axilite_ctrl_wstrb,
    output logic                        s_axilite_ctrl_bvalid,
    input  logic                        s_axilite_ctrl_bready,
    output logic [1:0]                  s_axilite_ctrl_bresp,
    input  logic                        s_axilite_ctrl_arvalid,
    output logic                        s_axilite_ctrl_arready,
    input  logic [11:0]                 s_axilite_ctrl_araddr,
    output logic                        s_axilite_ctrl_rvalid,
    input  logic                        s_axilite_ctrl_rready,
    output logic [CONFIG_WIDTH-1:0]     s_axilite_ctrl_rdata,
    output logic [1:0]                  s_axilite_ctrl_rresp
);

    // Implementation would go here...
    
endmodule : all_interfaces

// Expected parser behavior:
// - Detects all 5 interface types correctly
// - weights_V needs WEIGHT pragma to be recognized as weight interface
// - Parameters are exposed (no auto-linking due to naming)