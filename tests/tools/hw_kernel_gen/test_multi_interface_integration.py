#!/usr/bin/env python3
"""
Integration tests for multi-interface datatype parameter mapping.

Tests the complete RTL parser -> pragma processing -> template generation pipeline
with real RTL modules to ensure end-to-end functionality works correctly.
"""

import pytest
import tempfile
import os
from pathlib import Path
from typing import List, Dict

from brainsmith.tools.hw_kernel_gen.rtl_parser.parser import RTLParser
from brainsmith.tools.hw_kernel_gen.rtl_parser.data import DatatypeParamPragma, PragmaType
from brainsmith.tools.hw_kernel_gen.templates.context_generator import TemplateContextGenerator
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata
from brainsmith.dataflow.core.interface_types import InterfaceType


class TestMultiInterfaceIntegration:
    """Integration tests for multi-interface datatype parameter mapping."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_rtl_dir = Path(tempfile.mkdtemp())
        self.maxDiff = None  # Allow full diff output for debugging
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_rtl_dir, ignore_errors=True)
    
    def test_elementwise_add_scenario(self):
        """Test elementwise add with indexed inputs and pragmas."""
        # Create test RTL module with pragmas
        rtl_content = '''
// Elementwise addition with two inputs
// @brainsmith DATATYPE_PARAM s_axis_input0 width INPUT0_WIDTH
// @brainsmith DATATYPE_PARAM s_axis_input0 signed SIGNED_INPUT0
// @brainsmith DATATYPE_PARAM s_axis_input1 width INPUT1_WIDTH
// @brainsmith DATATYPE_PARAM s_axis_input1 signed SIGNED_INPUT1

module elementwise_add #(
    parameter INPUT0_WIDTH = 8,
    parameter SIGNED_INPUT0 = 0,
    parameter INPUT1_WIDTH = 8,
    parameter SIGNED_INPUT1 = 0,
    parameter OUTPUT_WIDTH = 8
) (
    input clk,
    input rst_n,
    
    // First input stream
    input [INPUT0_WIDTH-1:0] s_axis_input0_tdata,
    input s_axis_input0_tvalid,
    output s_axis_input0_tready,
    
    // Second input stream
    input [INPUT1_WIDTH-1:0] s_axis_input1_tdata,
    input s_axis_input1_tvalid,
    output s_axis_input1_tready,
    
    // Output stream
    output [OUTPUT_WIDTH-1:0] m_axis_output0_tdata,
    output m_axis_output0_tvalid,
    input m_axis_output0_tready
);

// Simple addition logic
assign m_axis_output0_tdata = s_axis_input0_tdata + s_axis_input1_tdata;
assign m_axis_output0_tvalid = s_axis_input0_tvalid & s_axis_input1_tvalid;
assign s_axis_input0_tready = m_axis_output0_tready;
assign s_axis_input1_tready = m_axis_output0_tready;

endmodule
'''
        
        rtl_file = self.test_rtl_dir / "elementwise_add.sv"
        rtl_file.write_text(rtl_content)
        
        # Parse RTL and test pragma processing
        parser = RTLParser(debug=True)
        kernel_metadata = parser.parse_file(str(rtl_file))
        
        # Verify parsing succeeded
        assert kernel_metadata is not None
        assert kernel_metadata.name == "elementwise_add"
        
        # Verify interfaces have datatype_params applied
        interfaces = kernel_metadata.interfaces
        input_interfaces = [iface for iface in interfaces if iface.interface_type == InterfaceType.INPUT]
        assert len(input_interfaces) == 2
        
        # Find input0 and input1 interfaces
        input0 = next((iface for iface in input_interfaces if 'input0' in iface.name), None)
        input1 = next((iface for iface in input_interfaces if 'input1' in iface.name), None)
        
        assert input0 is not None
        assert input1 is not None
        
        # Verify datatype_params were applied correctly
        assert input0.datatype_params is not None
        assert input0.datatype_params['width'] == 'INPUT0_WIDTH'
        assert input0.datatype_params['signed'] == 'SIGNED_INPUT0'
        
        assert input1.datatype_params is not None
        assert input1.datatype_params['width'] == 'INPUT1_WIDTH'
        assert input1.datatype_params['signed'] == 'SIGNED_INPUT1'
        
        print("✅ Elementwise add scenario test passed")
    
    def test_multihead_attention_scenario(self):
        """Test multihead attention with named parameters and pragmas."""
        rtl_content = '''
// Multi-head attention with query, key, value inputs
// @brainsmith DATATYPE_PARAM s_axis_query width QUERY_WIDTH
// @brainsmith DATATYPE_PARAM s_axis_query signed QUERY_SIGNED
// @brainsmith DATATYPE_PARAM s_axis_key width KEY_WIDTH
// @brainsmith DATATYPE_PARAM s_axis_value width VALUE_WIDTH

module multihead_attention #(
    parameter QUERY_WIDTH = 8,
    parameter QUERY_SIGNED = 0,
    parameter KEY_WIDTH = 8,
    parameter VALUE_WIDTH = 8,
    parameter OUTPUT_WIDTH = 8,
    parameter NUM_HEADS = 8
) (
    input clk,
    input rst_n,
    
    // Query input
    input [QUERY_WIDTH-1:0] s_axis_query_tdata,
    input s_axis_query_tvalid,
    output s_axis_query_tready,
    
    // Key input
    input [KEY_WIDTH-1:0] s_axis_key_tdata,
    input s_axis_key_tvalid,
    output s_axis_key_tready,
    
    // Value input
    input [VALUE_WIDTH-1:0] s_axis_value_tdata,
    input s_axis_value_tvalid,
    output s_axis_value_tready,
    
    // Output
    output [OUTPUT_WIDTH-1:0] m_axis_output_tdata,
    output m_axis_output_tvalid,
    input m_axis_output_tready
);

// Simplified attention logic (placeholder)
assign m_axis_output_tdata = s_axis_query_tdata ^ s_axis_key_tdata ^ s_axis_value_tdata;
assign m_axis_output_tvalid = s_axis_query_tvalid & s_axis_key_tvalid & s_axis_value_tvalid;
assign s_axis_query_tready = m_axis_output_tready;
assign s_axis_key_tready = m_axis_output_tready;
assign s_axis_value_tready = m_axis_output_tready;

endmodule
'''
        
        rtl_file = self.test_rtl_dir / "multihead_attention.sv"
        rtl_file.write_text(rtl_content)
        
        # Parse RTL
        parser = RTLParser(debug=True)
        kernel_metadata = parser.parse_file(str(rtl_file))
        
        # Verify parsing succeeded
        assert kernel_metadata is not None
        assert kernel_metadata.name == "multihead_attention"
        
        # Verify custom datatype parameter mapping
        interfaces = kernel_metadata.interfaces
        input_interfaces = [iface for iface in interfaces if iface.interface_type == InterfaceType.INPUT]
        
        # Find specific interfaces
        query_iface = next((iface for iface in input_interfaces if 'query' in iface.name), None)
        key_iface = next((iface for iface in input_interfaces if 'key' in iface.name), None)
        value_iface = next((iface for iface in input_interfaces if 'value' in iface.name), None)
        
        assert query_iface is not None
        assert key_iface is not None
        assert value_iface is not None
        
        # Verify custom parameter mapping
        assert query_iface.datatype_params is not None
        assert query_iface.datatype_params['width'] == 'QUERY_WIDTH'
        assert query_iface.datatype_params['signed'] == 'QUERY_SIGNED'
        
        assert key_iface.datatype_params is not None
        assert key_iface.datatype_params['width'] == 'KEY_WIDTH'
        
        # Value interface has no pragmas, should use defaults
        assert value_iface.datatype_params is not None
        assert value_iface.datatype_params['width'] == 'VALUE_WIDTH'
        
        print("✅ Multihead attention scenario test passed")
    
    def test_default_parameter_generation(self):
        """Test automatic parameter naming without pragmas."""
        rtl_content = '''
// Simple dual input module without pragmas
module simple_dual_input #(
    parameter WIDTH = 8
) (
    input clk,
    input rst_n,
    
    input [WIDTH-1:0] s_axis_input0_tdata,
    input s_axis_input0_tvalid,
    output s_axis_input0_tready,
    
    input [WIDTH-1:0] s_axis_input1_tdata,
    input s_axis_input1_tvalid,
    output s_axis_input1_tready,
    
    output [WIDTH-1:0] m_axis_output0_tdata,
    output m_axis_output0_tvalid,
    input m_axis_output0_tready
);

assign m_axis_output0_tdata = s_axis_input0_tdata | s_axis_input1_tdata;
assign m_axis_output0_tvalid = s_axis_input0_tvalid & s_axis_input1_tvalid;
assign s_axis_input0_tready = m_axis_output0_tready;
assign s_axis_input1_tready = m_axis_output0_tready;

endmodule
'''
        
        rtl_file = self.test_rtl_dir / "simple_dual_input.sv"
        rtl_file.write_text(rtl_content)
        
        # Parse RTL
        parser = RTLParser(debug=True)
        kernel_metadata = parser.parse_file(str(rtl_file))
        
        # Verify parsing succeeded
        assert kernel_metadata is not None
        assert kernel_metadata.name == "simple_dual_input"
        
        # Verify default parameter naming through template context
        context_generator = TemplateContextGenerator()
        context = context_generator.generate_context(kernel_metadata)
        
        # Check dataflow interfaces in context
        dataflow_interfaces = context['dataflow_interfaces']
        assert len(dataflow_interfaces) >= 3  # 2 inputs + 1 output
        
        # Find interfaces and verify default parameter names
        input0 = next((iface for iface in dataflow_interfaces if 'input0' in iface['name']), None)
        input1 = next((iface for iface in dataflow_interfaces if 'input1' in iface['name']), None)
        output0 = next((iface for iface in dataflow_interfaces if 'output0' in iface['name']), None)
        
        assert input0 is not None
        assert input1 is not None
        assert output0 is not None
        
        # Verify default parameter naming
        assert input0['width_param'] == 'INPUT0_WIDTH'
        assert input0['signed_param'] == 'SIGNED_INPUT0'
        assert input1['width_param'] == 'INPUT1_WIDTH'
        assert input1['signed_param'] == 'SIGNED_INPUT1'
        assert output0['width_param'] == 'OUTPUT0_WIDTH'
        assert output0['signed_param'] == 'SIGNED_OUTPUT0'
        
        print("✅ Default parameter generation test passed")
    
    def test_template_context_integration(self):
        """Test template context generation with multi-interface scenarios."""
        rtl_content = '''
// Mixed scenario with some pragmas and some defaults
// @brainsmith DATATYPE_PARAM s_axis_input0 width CUSTOM_INPUT_WIDTH
// @brainsmith DATATYPE_PARAM weights signed WEIGHTS_SIGNED
// @brainsmith WEIGHT weights

module mixed_scenario #(
    parameter CUSTOM_INPUT_WIDTH = 8,
    parameter WEIGHTS_SIGNED = 1,
    parameter DEFAULT_WIDTH = 16
) (
    input clk,
    input rst_n,
    
    input [CUSTOM_INPUT_WIDTH-1:0] s_axis_input0_tdata,
    input s_axis_input0_tvalid,
    output s_axis_input0_tready,
    
    input [DEFAULT_WIDTH-1:0] s_axis_input1_tdata,
    input s_axis_input1_tvalid,
    output s_axis_input1_tready,
    
    input [DEFAULT_WIDTH-1:0] weights_tdata,
    input weights_tvalid,
    output weights_tready,
    
    output [DEFAULT_WIDTH-1:0] m_axis_output_tdata,
    output m_axis_output_tvalid,
    input m_axis_output_tready
);

// Simple processing
assign m_axis_output_tdata = s_axis_input0_tdata + s_axis_input1_tdata + weights_tdata;

endmodule
'''
        
        rtl_file = self.test_rtl_dir / "mixed_scenario.sv"
        rtl_file.write_text(rtl_content)
        
        # Parse RTL
        parser = RTLParser(debug=True)
        kernel_metadata = parser.parse_file(str(rtl_file))
        
        # Generate template context
        context_generator = TemplateContextGenerator()
        context = context_generator.generate_context(kernel_metadata)
        
        # Verify mixed parameter naming
        input_interfaces = context['input_interfaces']
        weight_interfaces = context['weight_interfaces']
        output_interfaces = context['output_interfaces']
        
        # Find specific interfaces
        input0 = next((iface for iface in input_interfaces if 'input0' in iface['name']), None)
        input1 = next((iface for iface in input_interfaces if 'input1' in iface['name']), None)
        weights = next((iface for iface in weight_interfaces if 'weights' in iface['name']), None)
        
        assert input0 is not None
        assert input1 is not None
        assert weights is not None
        
        # Debug print
        print(f"input0: {input0}")
        print(f"input1: {input1}")
        print(f"weights: {weights}")
        
        # Verify mixed parameter mapping
        assert input0['width_param'] == 'CUSTOM_INPUT_WIDTH'  # Custom from pragma
        assert input0['signed_param'] == 'SIGNED_INPUT0'      # Default (no pragma)
        assert input1['width_param'] == 'INPUT1_WIDTH'       # Default (no pragma)
        assert input1['signed_param'] == 'SIGNED_INPUT1'     # Default (no pragma)
        assert weights['width_param'] == 'WEIGHTS_WIDTH'     # Default (no pragma)
        assert weights['signed_param'] == 'WEIGHTS_SIGNED'   # Custom from pragma
        
        print("✅ Template context integration test passed")


def run_integration_tests():
    """Run all integration tests."""
    print("🧪 Starting multi-interface integration tests...\n")
    
    test_suite = TestMultiInterfaceIntegration()
    
    # Run each test
    tests = [
        test_suite.test_elementwise_add_scenario,
        test_suite.test_multihead_attention_scenario,
        test_suite.test_default_parameter_generation,
        test_suite.test_template_context_integration,
    ]
    
    for test_func in tests:
        test_suite.setup_method()
        try:
            test_func()
        finally:
            test_suite.teardown_method()
        print()
    
    print("🎉 All multi-interface integration tests passed!")


if __name__ == "__main__":
    run_integration_tests()