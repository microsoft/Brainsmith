#!/usr/bin/env python3
"""Test the refactored parser with early KernelMetadata creation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from brainsmith.tools.hw_kernel_gen.rtl_parser.parser import RTLParser

def test_refactored_parser():
    """Test that the refactored parser works correctly."""
    
    rtl_code = '''
module test_refactor #(
    parameter s_axis_input_WIDTH = 8,
    parameter s_axis_input_SIGNED = 0,
    parameter m_axis_output_WIDTH = 16,
    parameter m_axis_output_SIGNED = 1,
    parameter THRESH_WIDTH = 8,
    parameter THRESH_SIGNED = 0,
    parameter T_WIDTH = 4,
    parameter ITERATIONS = 10
) (
    // Global Control
    input logic ap_clk,
    input logic ap_rst_n,
    
    // AXI-Stream Input
    input logic [7:0] s_axis_input_tdata,
    input logic s_axis_input_tvalid,
    output logic s_axis_input_tready,
    
    // AXI-Stream Output  
    output logic [15:0] m_axis_output_tdata,
    output logic m_axis_output_tvalid,
    input logic m_axis_output_tready
);

    // @brainsmith DATATYPE_PARAM threshold width THRESH_WIDTH
    // @brainsmith DATATYPE_PARAM threshold signed THRESH_SIGNED
    // @brainsmith ALIAS ITERATIONS num_iterations

endmodule
'''
    
    print("=== Testing Refactored Parser ===")
    
    parser = RTLParser(auto_link_parameters=True)
    kernel_metadata = parser.parse(rtl_code, "test_refactor.sv")
    
    print(f"\nModule: {kernel_metadata.name}")
    print(f"Parameters: {len(kernel_metadata.parameters)}")
    print(f"Exposed parameters: {kernel_metadata.exposed_parameters}")
    print(f"Interfaces: {len(kernel_metadata.interfaces)}")
    print(f"Internal datatypes: {len(kernel_metadata.internal_datatypes)}")
    
    print(f"\nParameter pragma data: {kernel_metadata.parameter_pragma_data}")
    
    print("\nInterface details:")
    for iface in kernel_metadata.interfaces:
        dt = iface.datatype_metadata
        dt_info = f"datatype={dt.name}, width={dt.width}, signed={dt.signed}" if dt else "no datatype"
        print(f"  - {iface.name} ({iface.interface_type.value}): {dt_info}")
    
    print("\nInternal datatype details:")
    for dt in kernel_metadata.internal_datatypes:
        print(f"  - {dt.name}: width={dt.width}, signed={dt.signed}")
    
    print("\n✓ Refactored parser test completed successfully!")

if __name__ == "__main__":
    test_refactored_parser()