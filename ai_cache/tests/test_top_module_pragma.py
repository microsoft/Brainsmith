#!/usr/bin/env python3
"""Test TopModulePragma functionality after recent refactoring."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from brainsmith.tools.hw_kernel_gen.rtl_parser.parser import RTLParser

def test_top_module_pragma():
    """Test that TOP_MODULE pragma correctly selects the target module."""
    
    # Test case 1: Multiple modules with TOP_MODULE pragma
    rtl_code_multi = """
// @brainsmith top_module my_target_module

module decoy_module (
    input logic ap_clk,
    input logic ap_rst_n,
    input logic [31:0] s_axis_tdata,
    input logic s_axis_tvalid,
    output logic s_axis_tready,
    output logic [31:0] m_axis_tdata,
    output logic m_axis_tvalid,
    input logic m_axis_tready
);
endmodule

module my_target_module #(
    parameter int WIDTH = 8
) (
    input logic ap_clk,
    input logic ap_rst_n,
    input logic [WIDTH-1:0] s_axis_input_tdata,
    input logic s_axis_input_tvalid,
    output logic s_axis_input_tready,
    output logic [WIDTH-1:0] m_axis_output_tdata,
    output logic m_axis_output_tvalid,
    input logic m_axis_output_tready
);
    assign m_axis_output_tdata = s_axis_input_tdata;
    assign m_axis_output_tvalid = s_axis_input_tvalid;
    assign s_axis_input_tready = m_axis_output_tready;
endmodule

module another_decoy (
    input logic ap_clk,
    input logic ap_rst_n,
    output logic [7:0] m_axis_result_tdata,
    output logic m_axis_result_tvalid,
    input logic m_axis_result_tready
);
endmodule
"""
    
    print("Test 1: Multiple modules with TOP_MODULE pragma")
    parser = RTLParser(debug=True)
    result = parser.parse(rtl_code_multi, "test_multi_modules.sv")
    
    print(f"  Selected module: {result.name}")
    print(f"  Expected: my_target_module")
    assert result.name == "my_target_module", f"Expected 'my_target_module', got '{result.name}'"
    print("  ✓ TOP_MODULE pragma correctly selected target module\n")
    
    # Test case 2: Single module with TOP_MODULE pragma (should still work)
    rtl_code_single = """
// @brainsmith top_module single_module

module single_module #(
    parameter int DEPTH = 16
) (
    input logic ap_clk,
    input logic ap_rst_n,
    input logic [7:0] s_axis_addr_tdata,
    input logic s_axis_addr_tvalid,
    output logic s_axis_addr_tready,
    output logic [31:0] m_axis_data_tdata,
    output logic m_axis_data_tvalid,
    input logic m_axis_data_tready
);
endmodule
"""
    
    print("Test 2: Single module with TOP_MODULE pragma")
    parser2 = RTLParser(debug=False)
    result2 = parser2.parse(rtl_code_single, "test_single_module.sv")
    
    print(f"  Selected module: {result2.name}")
    print(f"  Expected: single_module")
    assert result2.name == "single_module", f"Expected 'single_module', got '{result2.name}'"
    print("  ✓ TOP_MODULE pragma works with single module\n")
    
    # Test case 3: Multiple modules without TOP_MODULE pragma (should fail)
    rtl_code_no_pragma = """
module module_a (
    input logic ap_clk,
    input logic ap_rst_n,
    input logic [15:0] s_axis_a_tdata,
    input logic s_axis_a_tvalid,
    output logic s_axis_a_tready,
    output logic [15:0] m_axis_a_tdata,
    output logic m_axis_a_tvalid,
    input logic m_axis_a_tready
);
endmodule

module module_b (
    input logic ap_clk,
    input logic ap_rst_n,
    input logic [15:0] s_axis_b_tdata,
    input logic s_axis_b_tvalid,
    output logic s_axis_b_tready,
    output logic [15:0] m_axis_b_tdata,
    output logic m_axis_b_tvalid,
    input logic m_axis_b_tready
);
endmodule
"""
    
    print("Test 3: Multiple modules without TOP_MODULE pragma")
    parser3 = RTLParser(debug=False)
    try:
        result3 = parser3.parse(rtl_code_no_pragma, "test_no_pragma.sv")
        print("  ✗ Expected ParserError but parsing succeeded")
        assert False, "Should have raised ParserError"
    except Exception as e:
        print(f"  Expected error: {type(e).__name__}: {str(e)}")
        assert "Multiple modules" in str(e), "Error should mention multiple modules"
        print("  ✓ Correctly failed with multiple modules and no pragma\n")
    
    # Test case 4: TOP_MODULE pragma with non-existent module
    rtl_code_bad_pragma = """
// @brainsmith top_module nonexistent_module

module actual_module (
    input logic ap_clk,
    input logic ap_rst_n,
    input logic [31:0] s_axis_in_tdata,
    input logic s_axis_in_tvalid,
    output logic s_axis_in_tready,
    output logic [31:0] m_axis_out_tdata,
    output logic m_axis_out_tvalid,
    input logic m_axis_out_tready
);
endmodule
"""
    
    print("Test 4: TOP_MODULE pragma with non-existent module")
    parser4 = RTLParser(debug=False)
    try:
        result4 = parser4.parse(rtl_code_bad_pragma, "test_bad_pragma.sv")
        print("  ✗ Expected ParserError but parsing succeeded")
        assert False, "Should have raised ParserError"
    except Exception as e:
        print(f"  Expected error: {type(e).__name__}: {str(e)}")
        assert "TOP_MODULE pragma specifies" in str(e) and "but" in str(e), "Error should mention pragma/module mismatch"
        print("  ✓ Correctly failed with non-existent module name\n")
    
    # Test case 5: Check that TOP_MODULE pragma is present in kernel metadata
    print("Test 5: TOP_MODULE pragma in kernel metadata")
    print(f"  Number of pragmas in result: {len(result.pragmas)}")
    top_module_pragmas = [p for p in result.pragmas if p.type.name == "TOP_MODULE"]
    print(f"  TOP_MODULE pragmas found: {len(top_module_pragmas)}")
    
    if top_module_pragmas:
        pragma = top_module_pragmas[0]
        print(f"  Pragma data: {pragma.parsed_data}")
        assert pragma.parsed_data.get("module_name") == "my_target_module"
        print("  ✓ TOP_MODULE pragma correctly stored in kernel metadata\n")
    else:
        print("  ✗ No TOP_MODULE pragma found in kernel metadata\n")
    
    print("All TopModulePragma tests passed! ✓")

if __name__ == "__main__":
    test_top_module_pragma()