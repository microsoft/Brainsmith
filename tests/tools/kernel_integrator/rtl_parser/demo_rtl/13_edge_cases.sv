////////////////////////////////////////////////////////////////////////////
// Demo 13: Edge Cases and Common Pitfalls
// 
// This example demonstrates edge cases, common mistakes, and how the
// parser handles various corner cases.
////////////////////////////////////////////////////////////////////////////

// Attempting to use pragmas on wrong interface types
// @brainsmith SDIM m_axis_output 1024              // ERROR: SDIM not allowed on OUTPUT
// @brainsmith BDIM s_axilite_config 32             // ERROR: BDIM not allowed on CONFIG
// @brainsmith DATATYPE ap UINT 1 1                 // ERROR: DATATYPE not allowed on CONTROL
// @brainsmith WEIGHT m_axis_result                 // ERROR: WEIGHT only for INPUT interfaces

// Valid pragmas
// @brainsmith WEIGHT weird_weight_name
// @brainsmith DATATYPE s_axis_input FIXED 8 16
// @brainsmith ALIAS WEIRD_PARAM sensible_name

module edge_cases_demo #(
    // Case 1: Parameter name conflicts
    parameter int unsigned s_axis_input_WIDTH = 16,     // Auto-linked
    parameter int unsigned s_axis_input_width = 8,      // Different case - separate param!
    parameter int unsigned S_AXIS_INPUT_WIDTH = 32,     // Another case variant
    
    // Case 2: Partial naming matches (won't auto-link)
    parameter int unsigned s_axis_WIDTH = 8,            // Missing interface suffix
    parameter int unsigned input_WIDTH = 16,            // Missing s_axis prefix
    parameter int unsigned WIDTH_s_axis_input = 24,     // Wrong order
    
    // Case 3: Mixed single and indexed (single wins)
    parameter int unsigned s_axis_data_BDIM = 64,       // This is used
    parameter int unsigned s_axis_data_BDIM0 = 32,      // Ignored
    parameter int unsigned s_axis_data_BDIM1 = 32,      // Ignored
    parameter int unsigned s_axis_data_BDIM2 = 16,      // Ignored
    
    // Case 4: Non-contiguous indexed with large gaps
    parameter int unsigned weights_BDIM0 = 8,           // Index 0
    parameter int unsigned weights_BDIM5 = 16,          // Index 5 (gaps: 1,2,3,4)
    parameter int unsigned weights_BDIM99 = 32,         // Index 99 (many gaps!)
    // Results in 100-element list with many "1" singletons
    
    // Case 5: Invalid dimension parameters (ignored)
    parameter int unsigned m_axis_output_SDIM = 512,    // Ignored - OUTPUT no SDIM
    parameter int unsigned s_axilite_config_BDIM = 4,   // Ignored - CONFIG no BDIM
    parameter int unsigned s_axilite_config_SDIM = 8,   // Ignored - CONFIG no SDIM
    parameter int unsigned ap_BDIM = 1,                 // Ignored - CONTROL no BDIM
    parameter int unsigned ap_WIDTH = 1,                // Ignored - CONTROL no WIDTH
    
    // Case 6: Weird but valid parameter names
    parameter int unsigned WEIRD_PARAM = 42,            // Exposed with alias
    parameter int unsigned ______WIDTH = 8,             // Valid identifier!
    parameter int unsigned ACC0_WIDTH = 32,             // Auto-linked internal
    parameter int unsigned ACC0_SIGNED = 1,             // Auto-linked internal
    
    // Case 7: Parameters that look like they should link but don't
    parameter int unsigned s_axis_input_BDIM_0 = 16,    // Extra underscore
    parameter int unsigned s_axis_input_bdim00 = 32,    // Double zero
    parameter int unsigned s_axis_inputBDIM0 = 64,      // Missing underscore
    
    // Case 8: Unicode and special characters (if supported)
    // parameter int unsigned søme_pāram = 16;           // May not compile
    
    // Case 9: Very long parameter names
    parameter int unsigned this_is_a_very_long_parameter_name_that_might_cause_issues_somewhere_WIDTH = 8,
    parameter bit this_is_a_very_long_parameter_name_that_might_cause_issues_somewhere_SIGNED = 1
) (
    // Standard interfaces
    input  logic                                    ap_clk,
    input  logic                                    ap_rst_n,
    
    // Input with case-sensitive params
    input  logic [s_axis_input_WIDTH-1:0]          s_axis_input_tdata,
    input  logic                                    s_axis_input_tvalid,
    output logic                                    s_axis_input_tready,
    
    // Data interface with mixed dimension styles
    input  logic [15:0]                             s_axis_data_tdata,
    input  logic                                    s_axis_data_tvalid,
    output logic                                    s_axis_data_tready,
    
    // Non-standard weight interface name
    input  logic [7:0]                              weird_weight_name_tdata,
    input  logic                                    weird_weight_name_tvalid,
    output logic                                    weird_weight_name_tready,
    
    // Output (SDIM parameter ignored)
    output logic [31:0]                             m_axis_output_tdata,
    output logic                                    m_axis_output_tvalid,
    input  logic                                    m_axis_output_tready,
    
    // Config (dimension parameters ignored)
    input  logic                                    s_axilite_config_awvalid,
    output logic                                    s_axilite_config_awready,
    input  logic [11:0]                             s_axilite_config_awaddr,
    input  logic                                    s_axilite_config_wvalid,
    output logic                                    s_axilite_config_wready,
    input  logic [31:0]                             s_axilite_config_wdata,
    output logic                                    s_axilite_config_bvalid,
    input  logic                                    s_axilite_config_bready,
    output logic [1:0]                              s_axilite_config_bresp
);

    // Implementation demonstrating parameter usage
    logic [ACC0_WIDTH-1:0] accumulator;
    
endmodule : edge_cases_demo

// Common parser warnings/errors you might see:
// - "SDIM pragma cannot be applied to OUTPUT interface"
// - "BDIM pragma cannot be applied to CONFIG interface"
// - "DATATYPE pragma cannot be applied to CONTROL interface"
// - "WEIGHT pragma can only be applied to INPUT interfaces"
// - "Parameter 's_axis_data_BDIM0' ignored - single parameter 's_axis_data_BDIM' takes precedence"
// - "Non-contiguous index detected: weights_BDIM5 (missing indices will be filled with '1')"

// Expected parser behavior:
// - Case sensitivity matters: WIDTH, width, Width are different
// - Single parameters override indexed ones
// - Large gaps in indices create many singleton dimensions
// - Invalid auto-linking attempts are silently ignored
// - Pragma errors are reported
// - Very long parameter names work fine
// - Internal datatypes (ACC0_*) are auto-linked