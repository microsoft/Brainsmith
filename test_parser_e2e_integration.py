#!/usr/bin/env python3
"""Test end-to-end parser integration with new InterfaceMetadata flow."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from brainsmith.tools.hw_kernel_gen.rtl_parser.parser import RTLParser
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata
from brainsmith.dataflow.core.interface_types import InterfaceType


def test_simple_module_parsing():
    """Test parsing a simple module with pragmas."""
    print("Testing end-to-end parser integration...")
    
    # Simple SystemVerilog module with pragmas
    systemverilog_code = """
// @brainsmith datatype weights INT 8 8
// @brainsmith weight weights
module simple_test(
    input ap_clk,
    input ap_rst_n,
    
    input [7:0] weights_tdata,
    input weights_tvalid,
    output weights_tready,
    
    output [7:0] out_tdata,
    output out_tvalid,
    input out_tready
);
    // Module implementation would go here
endmodule
"""
    
    parser = RTLParser(debug=True)
    
    try:
        # Parse the SystemVerilog code
        kernel_metadata = parser.parse_string(systemverilog_code, source_name="test_module")
        
        print(f"\n--- Parser Results ---")
        print(f"Module name: {kernel_metadata.name}")
        print(f"Number of interfaces: {len(kernel_metadata.interfaces)}")
        print(f"Number of pragmas: {len(kernel_metadata.pragmas)}")
        
        # Verify we got InterfaceMetadata objects
        for interface in kernel_metadata.interfaces:
            if not isinstance(interface, InterfaceMetadata):
                print(f"❌ Expected InterfaceMetadata, got {type(interface)}")
                return False
            print(f"  - {interface.name}: {interface.interface_type.value}")
            print(f"    Allowed datatypes: {len(interface.allowed_datatypes)}")
            for dt in interface.allowed_datatypes:
                print(f"      * {dt.finn_type} ({dt.bit_width} bits, signed={dt.signed})")
        
        # Verify expected interfaces
        interface_names = {iface.name for iface in kernel_metadata.interfaces}
        interface_types = {iface.interface_type for iface in kernel_metadata.interfaces}
        
        expected_names = {"ap", "weights", "out"}  # Control, input with weight pragma, output
        expected_types = {InterfaceType.CONTROL, InterfaceType.WEIGHT, InterfaceType.OUTPUT}
        
        print(f"\n--- Verification ---")
        print(f"Interface names: {interface_names}")
        print(f"Interface types: {interface_types}")
        
        # Check interface names
        if not expected_names.issubset(interface_names):
            missing = expected_names - interface_names
            print(f"❌ Missing interface names: {missing}")
            return False
        
        # Check interface types
        if not expected_types.issubset(interface_types):
            missing = expected_types - interface_types
            print(f"❌ Missing interface types: {missing}")
            return False
        
        # Verify weight pragma was applied
        weights_interface = next((iface for iface in kernel_metadata.interfaces if iface.name == "weights"), None)
        if not weights_interface:
            print("❌ Weights interface not found")
            return False
        
        if weights_interface.interface_type != InterfaceType.WEIGHT:
            print(f"❌ Weight pragma not applied - interface type is {weights_interface.interface_type}")
            return False
        
        # Verify datatype pragma was applied
        has_int8 = any(dt.finn_type == "INT8" and dt.signed for dt in weights_interface.allowed_datatypes)
        if not has_int8:
            print("❌ DATATYPE pragma not applied - no INT8 datatype found")
            return False
        
        print("✅ All end-to-end integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Parser failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_thresholding_style_module():
    """Test with a more realistic thresholding-style module."""
    print("\nTesting with thresholding-style module...")
    
    systemverilog_code = """
module thresholding_test #(
    parameter THRESHOLD = 8
)(
    input ap_clk,
    input ap_rst_n,
    
    input [31:0] s_axis_V_data_V_TDATA,
    input s_axis_V_data_V_TVALID,
    output s_axis_V_data_V_TREADY,
    
    output [31:0] m_axis_V_data_V_TDATA,
    output m_axis_V_data_V_TVALID,
    input m_axis_V_data_V_TREADY
);
    // Module implementation would go here
endmodule
"""
    
    parser = RTLParser(debug=False)  # Less verbose for this test
    
    try:
        kernel_metadata = parser.parse_string(systemverilog_code, source_name="thresholding_test")
        
        print(f"Module: {kernel_metadata.name}")
        print(f"Parameters: {len(kernel_metadata.parameters)}")
        print(f"Interfaces: {len(kernel_metadata.interfaces)}")
        
        # Verify basic structure
        interface_types = {iface.interface_type for iface in kernel_metadata.interfaces}
        expected_types = {InterfaceType.CONTROL, InterfaceType.INPUT, InterfaceType.OUTPUT}
        
        if expected_types.issubset(interface_types):
            print("✅ Thresholding-style module parsing successful!")
            return True
        else:
            missing = expected_types - interface_types
            print(f"❌ Missing interface types: {missing}")
            return False
            
    except Exception as e:
        print(f"❌ Thresholding parser failed: {e}")
        return False


if __name__ == "__main__":
    try:
        tests = [
            test_simple_module_parsing,
            test_thresholding_style_module,
        ]
        
        all_passed = True
        for test in tests:
            try:
                if not test():
                    all_passed = False
            except Exception as e:
                print(f"❌ {test.__name__} failed with exception: {e}")
                import traceback
                traceback.print_exc()
                all_passed = False
        
        if all_passed:
            print("\n🎉 All end-to-end integration tests passed!")
        else:
            print("\n❌ Some end-to-end integration tests failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test suite failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)