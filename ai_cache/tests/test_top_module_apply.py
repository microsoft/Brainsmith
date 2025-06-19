#!/usr/bin/env python3
"""Test that TopModulePragma's apply_to_kernel is called during parsing."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from brainsmith.tools.hw_kernel_gen.rtl_parser.parser import RTLParser
from brainsmith.tools.hw_kernel_gen.rtl_parser.data import TopModulePragma

# Monkey patch to track if apply_to_kernel is called
original_apply = TopModulePragma.apply_to_kernel
apply_called = False

def tracked_apply(self, kernel):
    global apply_called
    apply_called = True
    print(f"  apply_to_kernel called for TopModulePragma with data: {self.parsed_data}")
    return original_apply(self, kernel)

TopModulePragma.apply_to_kernel = tracked_apply

def test_apply_to_kernel():
    """Test that TopModulePragma.apply_to_kernel is called during parsing."""
    
    rtl_code = """
// @brainsmith top_module test_module

module test_module (
    input logic ap_clk,
    input logic ap_rst_n,
    input logic [31:0] s_axis_in_tdata,
    input logic s_axis_in_tvalid,
    output logic s_axis_in_tready,
    output logic [31:0] m_axis_out_tdata,
    output logic m_axis_out_tvalid,
    input logic m_axis_out_tready
);
    assign m_axis_out_tdata = s_axis_in_tdata;
    assign m_axis_out_tvalid = s_axis_in_tvalid;
    assign s_axis_in_tready = m_axis_out_tready;
endmodule
"""
    
    print("Testing TopModulePragma.apply_to_kernel call:")
    
    global apply_called
    apply_called = False
    
    parser = RTLParser(debug=False)
    result = parser.parse(rtl_code, "test_apply.sv")
    
    print(f"  Module parsed: {result.name}")
    print(f"  apply_to_kernel was called: {apply_called}")
    
    if apply_called:
        print("  ✓ TopModulePragma.apply_to_kernel is being called correctly")
    else:
        print("  ✗ TopModulePragma.apply_to_kernel was NOT called")
    
    # Restore original method
    TopModulePragma.apply_to_kernel = original_apply
    
    return apply_called

if __name__ == "__main__":
    success = test_apply_to_kernel()
    if not success:
        sys.exit(1)