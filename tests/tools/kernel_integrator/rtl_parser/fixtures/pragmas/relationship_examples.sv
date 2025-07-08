////////////////////////////////////////////////////////////////////////////
// RELATIONSHIP pragma examples
//
// This fixture demonstrates various interface relationship types:
// - EQUAL: All dimensions must match
// - DEPENDENT: Specific dimension dependencies
// - MULTIPLE/DIVISIBLE: Dimension constraints
// - Complex multi-interface relationships
////////////////////////////////////////////////////////////////////////////

module relationship_examples #(
    // Interface A parameters (3D tensor)
    parameter integer A_BDIM0 = 16,
    parameter integer A_BDIM1 = 16,
    parameter integer A_BDIM2 = 3,
    parameter integer A_SDIM0 = 224,
    parameter integer A_SDIM1 = 224,
    
    // Interface B parameters (must equal A)
    parameter integer B_BDIM0 = 16,
    parameter integer B_BDIM1 = 16,
    parameter integer B_BDIM2 = 3,
    parameter integer B_SDIM0 = 224,
    parameter integer B_SDIM1 = 224,
    
    // Interface C parameters (dependent on A)
    parameter integer C_BDIM0 = 32,    // 2x A_BDIM0
    parameter integer C_BDIM1 = 8,     // A_BDIM1 / 2
    parameter integer C_BDIM2 = 3,     // Same as A_BDIM2
    parameter integer C_SDIM = 512,
    
    // Interface D parameters (divisible constraint)
    parameter integer D_BDIM = 64,      // Must be divisible by A_BDIM0
    parameter integer D_SDIM = 1024,
    
    // Output parameters
    parameter integer OUT_BDIM = 32,
    parameter integer OUT_SDIM = 256,
    
    // Pooling parameters (dimension reduction)
    parameter integer POOL_BDIM0 = 8,   // A_BDIM0 / 2
    parameter integer POOL_BDIM1 = 8,   // A_BDIM1 / 2
    parameter integer POOL_BDIM2 = 3    // Same channels
) (
    // Global control
    input wire ap_clk,
    input wire ap_rst_n,
    
    // Interface A - base tensor
    // @brainsmith BDIM s_axis_a [A_BDIM0, A_BDIM1, A_BDIM2]
    // @brainsmith SDIM s_axis_a [A_SDIM0, A_SDIM1]
    input wire [31:0] s_axis_a_tdata,
    input wire s_axis_a_tvalid,
    output wire s_axis_a_tready,
    
    // Interface B - must equal A
    // @brainsmith BDIM s_axis_b [B_BDIM0, B_BDIM1, B_BDIM2]
    // @brainsmith SDIM s_axis_b [B_SDIM0, B_SDIM1]
    // @brainsmith RELATIONSHIP s_axis_a s_axis_b EQUAL
    input wire [31:0] s_axis_b_tdata,
    input wire s_axis_b_tvalid,
    output wire s_axis_b_tready,
    
    // Interface C - dependent dimensions
    // @brainsmith BDIM s_axis_c [C_BDIM0, C_BDIM1, C_BDIM2]
    // @brainsmith SDIM s_axis_c C_SDIM
    // @brainsmith RELATIONSHIP s_axis_a s_axis_c DEPENDENT 0 0 scaled 2
    // @brainsmith RELATIONSHIP s_axis_a s_axis_c DEPENDENT 1 1 scaled 0.5
    // @brainsmith RELATIONSHIP s_axis_a s_axis_c DEPENDENT 2 2 copy
    input wire [31:0] s_axis_c_tdata,
    input wire s_axis_c_tvalid,
    output wire s_axis_c_tready,
    
    // Interface D - divisibility constraint
    // @brainsmith BDIM s_axis_d D_BDIM
    // @brainsmith SDIM s_axis_d D_SDIM
    // @brainsmith RELATIONSHIP s_axis_a s_axis_d DIVISIBLE 0
    input wire [31:0] s_axis_d_tdata,
    input wire s_axis_d_tvalid,
    output wire s_axis_d_tready,
    
    // Weights - different relationship
    // @brainsmith WEIGHT s_axis_weights
    // @brainsmith BDIM s_axis_weights [3, 3, A_BDIM2, 32]
    // @brainsmith SDIM s_axis_weights 1
    // @brainsmith RELATIONSHIP s_axis_a s_axis_weights DEPENDENT 2 2 copy
    input wire [7:0] s_axis_weights_tdata,
    input wire s_axis_weights_tvalid,
    output wire s_axis_weights_tready,
    
    // Output - independent dimensions
    // @brainsmith BDIM m_axis_output OUT_BDIM
    // @brainsmith SDIM m_axis_output OUT_SDIM
    output wire [31:0] m_axis_output_tdata,
    output wire m_axis_output_tvalid,
    input wire m_axis_output_tready,
    
    // Pooled output - dimension reduction from A
    // @brainsmith BDIM m_axis_pooled [POOL_BDIM0, POOL_BDIM1, POOL_BDIM2]
    // @brainsmith RELATIONSHIP s_axis_a m_axis_pooled DEPENDENT 0 0 scaled 0.5
    // @brainsmith RELATIONSHIP s_axis_a m_axis_pooled DEPENDENT 1 1 scaled 0.5
    // @brainsmith RELATIONSHIP s_axis_a m_axis_pooled DEPENDENT 2 2 copy
    output wire [31:0] m_axis_pooled_tdata,
    output wire m_axis_pooled_tvalid,
    input wire m_axis_pooled_tready
);

    // Complex multi-input processing
    reg [31:0] result_reg;
    reg [31:0] pooled_reg;
    reg valid_reg;
    
    wire all_inputs_valid = s_axis_a_tvalid & s_axis_b_tvalid & 
                           s_axis_c_tvalid & s_axis_d_tvalid & 
                           s_axis_weights_tvalid;
    
    wire all_outputs_ready = m_axis_output_tready & m_axis_pooled_tready;
    
    always @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            result_reg <= 32'h0;
            pooled_reg <= 32'h0;
            valid_reg <= 1'b0;
        end else if (all_inputs_valid && all_outputs_ready) begin
            // Complex operation respecting relationships
            result_reg <= (s_axis_a_tdata + s_axis_b_tdata) +  // Equal dims
                         (s_axis_c_tdata >> 1) +               // Scaled relationship
                         (s_axis_d_tdata >> 2) +               // Divisible relationship
                         {24'h0, s_axis_weights_tdata};       // Weights
            
            // Simulated pooling (2x2 reduction)
            pooled_reg <= (s_axis_a_tdata >> 2) + (s_axis_b_tdata >> 2);
            
            valid_reg <= 1'b1;
        end else if (all_outputs_ready) begin
            valid_reg <= 1'b0;
        end
    end
    
    // Output assignments
    assign m_axis_output_tdata = result_reg;
    assign m_axis_output_tvalid = valid_reg;
    assign m_axis_pooled_tdata = pooled_reg;
    assign m_axis_pooled_tvalid = valid_reg;
    
    // Input ready signals
    assign s_axis_a_tready = all_outputs_ready;
    assign s_axis_b_tready = all_outputs_ready;
    assign s_axis_c_tready = all_outputs_ready;
    assign s_axis_d_tready = all_outputs_ready;
    assign s_axis_weights_tready = all_outputs_ready;

endmodule