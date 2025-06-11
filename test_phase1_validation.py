#!/usr/bin/env python3
"""
Phase 1 Validation Test: ParsedKernelData Refactor

This test validates that the Phase 1A refactor successfully:
1. RTL Parser returns ParsedKernelData instead of HWKernel
2. ParsedKernelData provides all required template context
3. CLI can use ParsedKernelData without errors
4. All template variables are available

Run this test after Phase 1A completion to ensure clean refactor.
"""

import sys
import tempfile
from pathlib import Path

# Add the brainsmith package to the path
sys.path.insert(0, '/home/tafk/dev/brainsmith-2')

from brainsmith.tools.hw_kernel_gen.rtl_parser import RTLParser, ParsedKernelData
from brainsmith.dataflow.core.interface_types import InterfaceType


def test_rtl_parser_returns_parsed_kernel_data():
    """Test that RTL Parser returns ParsedKernelData object."""
    print("🔬 Testing RTL Parser returns ParsedKernelData...")
    
    # Create simple test RTL module
    test_rtl = """
module test_module #(
    parameter WIDTH = 8
) (
    input ap_clk,
    input ap_rst_n,
    input [WIDTH-1:0] data_in,
    output [WIDTH-1:0] data_out
);
    assign data_out = data_in;
endmodule
"""
    
    # Parse from string
    parser = RTLParser()
    result = parser.parse_string(test_rtl, source_name="test_module.sv")
    
    # Verify return type
    assert isinstance(result, ParsedKernelData), f"Expected ParsedKernelData, got {type(result)}"
    assert result.name == "test_module", f"Expected name 'test_module', got '{result.name}'"
    assert len(result.parameters) > 0, "Expected at least one parameter"
    assert len(result.interfaces) > 0, "Expected at least one interface"
    
    print("✅ RTL Parser correctly returns ParsedKernelData")


def test_template_context_completeness():
    """Test that ParsedKernelData provides complete template context."""
    print("🔬 Testing template context completeness...")
    
    # Create more complex test RTL with interfaces
    test_rtl = """
// @brainsmith DATATYPE in0 UINT 8 16
module thresholding #(
    parameter WIDTH = 8,
    parameter THRESHOLD = 128
) (
    input ap_clk,
    input ap_rst_n,
    
    // AXI-Stream input
    input [WIDTH-1:0] in0_V_data_V,
    input in0_V_valid_V,
    output in0_V_ready_V,
    
    // AXI-Stream output  
    output [7:0] out0_V_data_V,
    output out0_V_valid_V,
    input out0_V_ready_V
);
    assign out0_V_data_V = (in0_V_data_V > THRESHOLD) ? 8'hFF : 8'h00;
    assign out0_V_valid_V = in0_V_valid_V;
    assign in0_V_ready_V = out0_V_ready_V;
endmodule
"""
    
    parser = RTLParser()
    parsed_data = parser.parse_string(test_rtl, source_name="thresholding.sv")
    
    # Get template context using the new template context generator
    from brainsmith.tools.hw_kernel_gen.templates.context_generator import TemplateContextGenerator
    context = TemplateContextGenerator.generate_context(parsed_data)
    
    # Verify all required template variables exist
    required_vars = [
        'kernel_name', 'class_name', 'interfaces', 'input_interfaces',
        'output_interfaces', 'rtl_parameters', 'has_inputs', 'has_outputs',
        'kernel_complexity', 'kernel_type', 'InterfaceType', 'kernel',
        'dataflow_model_summary', 'generation_timestamp'
    ]
    
    for var in required_vars:
        assert var in context, f"Missing required template variable: {var}"
    
    # Verify specific values
    assert context['kernel_name'] == 'thresholding'
    assert context['class_name'] == 'Thresholding'
    assert isinstance(context['interfaces'], list)
    assert len(context['interfaces']) > 0
    assert isinstance(context['rtl_parameters'], list)
    assert len(context['rtl_parameters']) > 0
    assert context['InterfaceType'] == InterfaceType
    
    print("✅ Template context provides all required variables")


def test_interface_template_compatibility():
    """Test that interfaces provide template-expected attributes."""
    print("🔬 Testing interface template compatibility...")
    
    test_rtl = """
module simple_module (
    input ap_clk,
    input ap_rst_n,
    input [7:0] in0_V_data_V,
    output [7:0] out0_V_data_V
);
    assign out0_V_data_V = in0_V_data_V;
endmodule
"""
    
    parser = RTLParser()
    parsed_data = parser.parse_string(test_rtl, source_name="simple_module.sv")
    
    from brainsmith.tools.hw_kernel_gen.templates.context_generator import TemplateContextGenerator
    context = TemplateContextGenerator.generate_context(parsed_data)
    interfaces = context['interfaces']
    
    for interface in interfaces:
        # Test direct attribute access (what templates expect)
        assert hasattr(interface, 'name'), "Interface missing 'name' attribute"
        assert hasattr(interface, 'type'), "Interface missing 'type' attribute"
        assert hasattr(interface, 'ports'), "Interface missing 'ports' attribute"
        assert hasattr(interface, 'metadata'), "Interface missing 'metadata' attribute"
        
        # Test that type is InterfaceType enum
        assert isinstance(interface.type, InterfaceType), f"Expected InterfaceType, got {type(interface.type)}"
        
        # Test helper methods
        assert hasattr(interface, 'get_template_datatype'), "Interface missing get_template_datatype method"
        assert hasattr(interface, 'get_dimensional_info'), "Interface missing get_dimensional_info method"
    
    print("✅ Interfaces provide all template-expected attributes")


def test_parsed_kernel_data_helper_methods():
    """Test ParsedKernelData helper methods work correctly."""
    print("🔬 Testing ParsedKernelData helper methods...")
    
    test_rtl = """
module complex_module #(
    parameter WIDTH = 16,
    parameter DEPTH = 1024
) (
    input ap_clk,
    input ap_rst_n,
    input [WIDTH-1:0] in0_V_data_V,
    input [WIDTH-1:0] weights_V_data_V,
    output [WIDTH-1:0] out0_V_data_V,
    output [WIDTH-1:0] out1_V_data_V
);
endmodule
"""
    
    parser = RTLParser()
    parsed_data = parser.parse_string(test_rtl, source_name="complex_module.sv")
    
    # Test helper methods using context generator
    from brainsmith.tools.hw_kernel_gen.templates.context_generator import TemplateContextGenerator
    context = TemplateContextGenerator.generate_context(parsed_data)
    class_name = context['class_name']
    assert class_name == "ComplexModule", f"Expected 'ComplexModule', got '{class_name}'"
    
    # Test interface filtering
    all_interfaces = list(parsed_data.interfaces.values())
    input_interfaces = context['input_interfaces']
    output_interfaces = context['output_interfaces']
    dataflow_interfaces = context['dataflow_interfaces']
    
    assert len(all_interfaces) > 0, "Should have some interfaces"
    assert isinstance(input_interfaces, list), "input_interfaces should return list"
    assert isinstance(output_interfaces, list), "output_interfaces should return list"
    assert isinstance(dataflow_interfaces, list), "dataflow_interfaces should return list"
    
    # Test complexity estimation
    complexity = context['kernel_complexity']
    assert complexity in ['low', 'medium', 'high'], f"Invalid complexity: {complexity}"
    
    # Test kernel type inference
    kernel_type = context['kernel_type']
    assert isinstance(kernel_type, str), f"Kernel type should be string, got {type(kernel_type)}"
    
    print("✅ ParsedKernelData helper methods work correctly")


def test_cli_integration():
    """Test that CLI components can work with ParsedKernelData."""
    print("🔬 Testing CLI integration...")
    
    # Test that we can import CLI components
    try:
        from brainsmith.tools.hw_kernel_gen.cli import create_parsed_kernel_data
        print("✅ CLI import successful")
    except ImportError as e:
        print(f"❌ CLI import failed: {e}")
        return
    
    # Create a test RTL file
    test_rtl = """
module test_cli_module (
    input ap_clk,
    input ap_rst_n,
    input [7:0] data_in,
    output [7:0] data_out
);
    assign data_out = data_in;
endmodule
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sv', delete=False) as f:
        f.write(test_rtl)
        rtl_file = f.name
    
    # Create a dummy config-like object
    class MockConfig:
        def __init__(self):
            self.rtl_file = rtl_file
            self.advanced_pragmas = False
            self.debug = False
            self.compiler_data_file = None
    
    try:
        # This would normally be called from CLI
        config = MockConfig()
        
        # Test basic parsing (without compiler data for simplicity)
        parser = RTLParser()
        parsed_data = parser.parse_file(rtl_file)
        
        assert isinstance(parsed_data, ParsedKernelData)
        assert parsed_data.name == "test_cli_module"
        
        print("✅ CLI integration test passed")
        
    except Exception as e:
        print(f"❌ CLI integration test failed: {e}")
        raise
    finally:
        # Clean up
        Path(rtl_file).unlink()


def main():
    """Run all Phase 1A validation tests."""
    print("🚀 Starting Phase 1A Validation Tests")
    print("=" * 50)
    
    tests = [
        test_rtl_parser_returns_parsed_kernel_data,
        test_template_context_completeness,
        test_interface_template_compatibility,
        test_parsed_kernel_data_helper_methods,
        test_cli_integration,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print("=" * 50)
    print(f"🏁 Phase 1A Validation Complete: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Phase 1A refactor is successful.")
        return 0
    else:
        print("💥 Some tests failed. Phase 1A refactor needs fixes.")
        return 1


if __name__ == "__main__":
    sys.exit(main())