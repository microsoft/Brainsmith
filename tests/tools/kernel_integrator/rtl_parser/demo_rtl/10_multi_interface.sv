////////////////////////////////////////////////////////////////////////////
// Demo 10: Multi-Interface Modules
// 
// This example demonstrates modules with multiple interfaces of the same
// type, showing how to handle naming and parameterization for each.
////////////////////////////////////////////////////////////////////////////

// Datatype specifications for each interface
// @brainsmith DATATYPE s_axis_input0 FIXED 16 16
// @brainsmith DATATYPE s_axis_input1 FIXED 16 16
// @brainsmith DATATYPE s_axis_input2 UINT 8 8
// @brainsmith DATATYPE m_axis_output0 FIXED 32 32
// @brainsmith DATATYPE m_axis_output1 UINT 16 16

// Dimension specifications with indexed parameters
// @brainsmith BDIM s_axis_input0 [H0, W0, C0]
// @brainsmith BDIM s_axis_input1 [H1, W1, C1]

module multi_interface_demo #(
    // Input 0 parameters (3D tensor)
    parameter int unsigned s_axis_input0_WIDTH = 16,
    parameter bit s_axis_input0_SIGNED = 1,
    parameter int unsigned s_axis_input0_BDIM0 = 32,    // Height
    parameter int unsigned s_axis_input0_BDIM1 = 32,    // Width
    parameter int unsigned s_axis_input0_BDIM2 = 16,    // Channels
    parameter int unsigned s_axis_input0_SDIM = 1024,
    
    // Pragma overrides for input0 BDIM
    parameter int unsigned H0 = 64,
    parameter int unsigned W0 = 64,
    parameter int unsigned C0 = 32,
    
    // Input 1 parameters (3D tensor)
    parameter int unsigned s_axis_input1_WIDTH = 16,
    parameter bit s_axis_input1_SIGNED = 1,
    parameter int unsigned s_axis_input1_BDIM0 = 16,
    parameter int unsigned s_axis_input1_BDIM1 = 16,
    parameter int unsigned s_axis_input1_BDIM2 = 32,
    parameter int unsigned s_axis_input1_SDIM = 512,
    
    // Pragma overrides for input1 BDIM
    parameter int unsigned H1 = 32,
    parameter int unsigned W1 = 32, 
    parameter int unsigned C1 = 64,
    
    // Input 2 parameters (1D vector)
    parameter int unsigned s_axis_input2_WIDTH = 8,
    parameter bit s_axis_input2_SIGNED = 0,
    parameter int unsigned s_axis_input2_BDIM = 256,
    parameter int unsigned s_axis_input2_SDIM = 256,
    
    // Output 0 parameters
    parameter int unsigned m_axis_output0_WIDTH = 32,
    parameter bit m_axis_output0_SIGNED = 1,
    parameter int unsigned m_axis_output0_BDIM0 = 16,
    parameter int unsigned m_axis_output0_BDIM1 = 16,
    parameter int unsigned m_axis_output0_BDIM2 = 64,
    
    // Output 1 parameters
    parameter int unsigned m_axis_output1_WIDTH = 16,
    parameter bit m_axis_output1_SIGNED = 0,
    parameter int unsigned m_axis_output1_BDIM = 128,
    
    // Shared parameters
    parameter int unsigned OPERATION_MODE = 0,  // 0=add, 1=multiply, 2=concat
    parameter int unsigned PIPELINE_STAGES = 4
) (
    // Clock and reset
    input  logic                                ap_clk,
    input  logic                                ap_rst_n,
    
    // First input stream
    input  logic [s_axis_input0_WIDTH-1:0]     s_axis_input0_tdata,
    input  logic                                s_axis_input0_tvalid,
    output logic                                s_axis_input0_tready,
    input  logic                                s_axis_input0_tlast,
    
    // Second input stream
    input  logic [s_axis_input1_WIDTH-1:0]     s_axis_input1_tdata,
    input  logic                                s_axis_input1_tvalid,
    output logic                                s_axis_input1_tready,
    input  logic                                s_axis_input1_tlast,
    
    // Third input stream
    input  logic [s_axis_input2_WIDTH-1:0]     s_axis_input2_tdata,
    input  logic                                s_axis_input2_tvalid,
    output logic                                s_axis_input2_tready,
    
    // First output stream
    output logic [m_axis_output0_WIDTH-1:0]    m_axis_output0_tdata,
    output logic                                m_axis_output0_tvalid,
    input  logic                                m_axis_output0_tready,
    output logic                                m_axis_output0_tlast,
    
    // Second output stream
    output logic [m_axis_output1_WIDTH-1:0]    m_axis_output1_tdata,
    output logic                                m_axis_output1_tvalid,
    input  logic                                m_axis_output1_tready
);

    // Operation-specific logic
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            m_axis_output0_tvalid <= 1'b0;
            m_axis_output1_tvalid <= 1'b0;
        end else begin
            case (OPERATION_MODE)
                0: begin // Addition mode
                    // Add corresponding elements
                end
                1: begin // Multiplication mode
                    // Multiply corresponding elements
                end
                2: begin // Concatenation mode
                    // Concatenate inputs
                end
                default: begin
                    // Default behavior
                end
            endcase
        end
    end
    
    // Ready signal management
    assign s_axis_input0_tready = m_axis_output0_tready;
    assign s_axis_input1_tready = m_axis_output0_tready;
    assign s_axis_input2_tready = m_axis_output1_tready;
    
endmodule : multi_interface_demo

// Expected parser behavior:
// - Three input interfaces detected: s_axis_input0, s_axis_input1, s_axis_input2
// - Two output interfaces detected: m_axis_output0, m_axis_output1
// - Each interface has independent parameterization
// - Pragma overrides work per-interface:
//   - s_axis_input0 BDIM uses [H0, W0, C0] instead of indexed params
//   - s_axis_input1 BDIM uses [H1, W1, C1] instead of indexed params
// - All datatype parameters are auto-linked
// - Dimension parameters are auto-linked (except where overridden)
// - Only OPERATION_MODE and PIPELINE_STAGES remain exposed