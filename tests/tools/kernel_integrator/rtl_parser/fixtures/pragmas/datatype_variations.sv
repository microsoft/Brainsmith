////////////////////////////////////////////////////////////////////////////
// DATATYPE and DATATYPE_PARAM pragma variations
//
// This fixture demonstrates:
// - Basic DATATYPE constraints (base type, min/max bits)
// - DATATYPE_PARAM mappings (width, signed, format, etc.)
// - Internal datatype definitions (accumulator, threshold)
// - Multiple constraints on single interface
////////////////////////////////////////////////////////////////////////////

module datatype_variations #(
    // Width parameters
    parameter integer NARROW_WIDTH = 8,
    parameter integer STANDARD_WIDTH = 16,
    parameter integer WIDE_WIDTH = 32,
    parameter integer ULTRA_WIDE = 64,
    
    // Signed parameters
    parameter integer INPUT_SIGNED = 0,
    parameter integer WEIGHT_SIGNED = 1,
    parameter integer OUTPUT_SIGNED = 0,
    
    // Internal datatype parameters
    parameter integer ACC_WIDTH = 48,
    parameter integer ACC_SIGNED = 1,
    parameter integer ACC_FRACTIONAL = 16,
    parameter integer THRESH_WIDTH = 32,
    parameter integer BIAS_WIDTH = 16,
    parameter integer BIAS_VALUE = 128,
    
    // Interface dimensions (required for strict)
    parameter integer INPUT_BDIM = 32,
    parameter integer INPUT_SDIM = 512,
    parameter integer WEIGHT_BDIM = 64,
    parameter integer WEIGHT_SDIM = 512,
    parameter integer OUTPUT_BDIM = 32
) (
    // Global control
    input wire ap_clk,
    input wire ap_rst_n,
    
    // Narrow unsigned input with tight constraints
    // @brainsmith DATATYPE s_axis_narrow UINT 8 8
    // @brainsmith DATATYPE_PARAM s_axis_narrow width NARROW_WIDTH
    // @brainsmith BDIM s_axis_narrow INPUT_BDIM
    // @brainsmith SDIM s_axis_narrow INPUT_SDIM
    input wire [NARROW_WIDTH-1:0] s_axis_narrow_tdata,
    input wire s_axis_narrow_tvalid,
    output wire s_axis_narrow_tready,
    
    // Standard input with flexible constraints
    // @brainsmith DATATYPE s_axis_standard UINT 8 32
    // @brainsmith DATATYPE_PARAM s_axis_standard width STANDARD_WIDTH
    // @brainsmith DATATYPE_PARAM s_axis_standard signed INPUT_SIGNED
    // @brainsmith BDIM s_axis_standard INPUT_BDIM
    // @brainsmith SDIM s_axis_standard INPUT_SDIM
    input wire [STANDARD_WIDTH-1:0] s_axis_standard_tdata,
    input wire s_axis_standard_tvalid,
    output wire s_axis_standard_tready,
    
    // Signed weight input
    // @brainsmith WEIGHT s_axis_weights
    // @brainsmith DATATYPE s_axis_weights INT 8 16
    // @brainsmith DATATYPE_PARAM s_axis_weights width WEIGHT_WIDTH
    // @brainsmith DATATYPE_PARAM s_axis_weights signed WEIGHT_SIGNED
    // @brainsmith BDIM s_axis_weights WEIGHT_BDIM
    // @brainsmith SDIM s_axis_weights WEIGHT_SDIM
    input wire [STANDARD_WIDTH-1:0] s_axis_weights_tdata,
    input wire s_axis_weights_tvalid,
    output wire s_axis_weights_tready,
    
    // Wide output with multiple datatype constraints
    // @brainsmith DATATYPE m_axis_wide UINT 16 64
    // @brainsmith DATATYPE m_axis_wide FIXED 16 32
    // @brainsmith DATATYPE_PARAM m_axis_wide width WIDE_WIDTH
    // @brainsmith DATATYPE_PARAM m_axis_wide signed OUTPUT_SIGNED
    // @brainsmith BDIM m_axis_wide OUTPUT_BDIM
    output wire [WIDE_WIDTH-1:0] m_axis_wide_tdata,
    output wire m_axis_wide_tvalid,
    input wire m_axis_wide_tready,
    
    // Ultra-wide output
    // @brainsmith DATATYPE m_axis_ultra UINT 32 64
    // @brainsmith DATATYPE_PARAM m_axis_ultra width ULTRA_WIDE
    // @brainsmith BDIM m_axis_ultra OUTPUT_BDIM
    output wire [ULTRA_WIDE-1:0] m_axis_ultra_tdata,
    output wire m_axis_ultra_tvalid,
    input wire m_axis_ultra_tready
);

    // Internal datatype definitions via pragmas
    // @brainsmith DATATYPE_PARAM accumulator width ACC_WIDTH
    // @brainsmith DATATYPE_PARAM accumulator signed ACC_SIGNED
    // @brainsmith DATATYPE_PARAM accumulator fractional_width ACC_FRACTIONAL
    
    // @brainsmith DATATYPE_PARAM threshold width THRESH_WIDTH
    
    // @brainsmith DATATYPE_PARAM bias width BIAS_WIDTH
    // @brainsmith DATATYPE_PARAM bias bias BIAS_VALUE
    
    // Internal signals using pragma-defined widths
    reg [ACC_WIDTH-1:0] accumulator;
    reg [THRESH_WIDTH-1:0] threshold;
    reg [BIAS_WIDTH-1:0] bias;
    
    // Processing logic
    wire all_inputs_valid = s_axis_narrow_tvalid & 
                           s_axis_standard_tvalid & 
                           s_axis_weights_tvalid;
    
    wire all_outputs_ready = m_axis_wide_tready & m_axis_ultra_tready;
    
    always @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            accumulator <= '0;
            threshold <= 32'h00001000;
            bias <= BIAS_VALUE;
        end else if (all_inputs_valid && all_outputs_ready) begin
            // Complex calculation respecting signed parameters
            if (WEIGHT_SIGNED) begin
                accumulator <= accumulator + 
                    ($signed(s_axis_weights_tdata) * $signed({8'b0, s_axis_narrow_tdata})) +
                    {{(ACC_WIDTH-STANDARD_WIDTH){1'b0}}, s_axis_standard_tdata} +
                    {{(ACC_WIDTH-BIAS_WIDTH){1'b0}}, bias};
            end else begin
                accumulator <= accumulator + 
                    (s_axis_weights_tdata * s_axis_narrow_tdata) +
                    {{(ACC_WIDTH-STANDARD_WIDTH){1'b0}}, s_axis_standard_tdata} +
                    {{(ACC_WIDTH-BIAS_WIDTH){1'b0}}, bias};
            end
        end
    end
    
    // Output generation
    wire [WIDE_WIDTH-1:0] clamped_result = 
        (accumulator > {1'b0, {WIDE_WIDTH{1'b1}}}) ? {WIDE_WIDTH{1'b1}} :
        accumulator[WIDE_WIDTH-1:0];
    
    assign m_axis_wide_tdata = clamped_result;
    assign m_axis_wide_tvalid = all_inputs_valid;
    
    assign m_axis_ultra_tdata = {{(ULTRA_WIDE-ACC_WIDTH){accumulator[ACC_WIDTH-1]}}, accumulator};
    assign m_axis_ultra_tvalid = all_inputs_valid;
    
    // Input ready signals
    assign s_axis_narrow_tready = all_outputs_ready;
    assign s_axis_standard_tready = all_outputs_ready;
    assign s_axis_weights_tready = all_outputs_ready;

endmodule