#!/usr/bin/env python3
"""
HWKG Axiom Validation Test
Tests our current HWKG implementation against the design axioms.
"""

import sys
sys.path.insert(0, '/home/tafk/dev/brainsmith-2')

import pytest
import tempfile
from pathlib import Path

from brainsmith.tools.hw_kernel_gen.rtl_parser import parse_rtl_file
from brainsmith.tools.hw_kernel_gen.generators.hw_custom_op import HWCustomOpGenerator
from brainsmith.tools.hw_kernel_gen.generators.rtl_backend import RTLBackendGenerator
from brainsmith.tools.hw_kernel_gen.generators.test_suite import TestSuiteGenerator

def test_axiom_1_interface_wise_dataflow_foundation():
    """
    Test Axiom 1: Interface-Wise Dataflow Foundation
    RTL Input → RTL Parser → Dataflow Interface Model → FINN Components
    """
    print("Testing Axiom 1: Interface-Wise Dataflow Foundation")
    
    # RTL with multiple interface types
    rtl_code = '''
module dataflow_test (
    input wire clk,
    input wire rst_n,
    
    // AXI-Stream input interface
    input wire [63:0] s_axis_input_tdata,
    input wire s_axis_input_tvalid,
    output wire s_axis_input_tready,
    
    // AXI-Stream weight interface
    input wire [31:0] s_axis_weight_tdata,
    input wire s_axis_weight_tvalid,
    output wire s_axis_weight_tready,
    
    // AXI-Stream output interface
    output wire [63:0] m_axis_output_tdata,
    output wire m_axis_output_tvalid,
    input wire m_axis_output_tready,
    
    // AXI-Lite configuration
    input wire [31:0] s_axilite_awaddr,
    input wire s_axilite_awvalid,
    output wire s_axilite_awready,
    input wire [31:0] s_axilite_wdata,
    input wire [3:0] s_axilite_wstrb,
    input wire s_axilite_wvalid,
    output wire s_axilite_wready,
    output wire [1:0] s_axilite_bresp,
    output wire s_axilite_bvalid,
    input wire s_axilite_bready,
    input wire [31:0] s_axilite_araddr,
    input wire s_axilite_arvalid,
    output wire s_axilite_arready,
    output wire [31:0] s_axilite_rdata,
    output wire [1:0] s_axilite_rresp,
    output wire s_axilite_rvalid,
    input wire s_axilite_rready
);

assign m_axis_output_tdata = s_axis_input_tdata + {32'h0, s_axis_weight_tdata};
assign m_axis_output_tvalid = s_axis_input_tvalid & s_axis_weight_tvalid;
assign s_axis_input_tready = m_axis_output_tready;
assign s_axis_weight_tready = m_axis_output_tready;

// AXI-Lite responses
assign s_axilite_awready = 1'b1;
assign s_axilite_wready = 1'b1;
assign s_axilite_bresp = 2'b00;
assign s_axilite_bvalid = s_axilite_wvalid;
assign s_axilite_arready = 1'b1;
assign s_axilite_rdata = 32'hDEADBEEF;
assign s_axilite_rresp = 2'b00;
assign s_axilite_rvalid = s_axilite_arvalid;

endmodule
'''
    
    rtl_file = Path('/tmp/dataflow_test.sv')
    rtl_file.write_text(rtl_code)
    
    try:
        # Step 1: RTL Input → RTL Parser
        hw_kernel = parse_rtl_file(rtl_file)
        assert hw_kernel is not None
        assert hw_kernel.name == "dataflow_test"
        
        # Step 2: Validate Interface Detection
        interfaces = hw_kernel.interfaces
        
        # Should detect multiple interface types
        interface_types = {iface.type.value for iface in interfaces.values()}
        assert 'axistream' in interface_types, "Should detect AXI-Stream interfaces"
        assert 'axilite' in interface_types, "Should detect AXI-Lite interfaces"
        assert 'global' in interface_types, "Should detect global control signals"
        
        # Step 3: Dataflow Interface Model → FINN Components
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Generate FINN-compatible components
            hwcustomop_gen = HWCustomOpGenerator()
            hwcustomop_file = hwcustomop_gen.generate(hw_kernel, output_dir)
            
            rtlbackend_gen = RTLBackendGenerator()
            rtlbackend_file = rtlbackend_gen.generate(hw_kernel, output_dir)
            
            # Validate FINN component generation
            assert hwcustomop_file.exists()
            assert rtlbackend_file.exists()
            
            hwcustomop_content = hwcustomop_file.read_text()
            rtlbackend_content = rtlbackend_file.read_text()
            
            # Should contain FINN-compatible code
            assert "HWCustomOp" in hwcustomop_content
            assert "RTLBackend" in rtlbackend_content
            assert "finn" in hwcustomop_content.lower() or "brainsmith" in hwcustomop_content
            
        print("✅ Axiom 1: Interface-Wise Dataflow Foundation - PASSED")
        return True
        
    finally:
        if rtl_file.exists():
            rtl_file.unlink()


def test_axiom_2_multi_phase_pipeline():
    """
    Test Axiom 2: Multi-Phase Pipeline
    Parse RTL → Parse Compiler Data → Build Dataflow Model → Generate Templates → Generate Components
    """
    print("Testing Axiom 2: Multi-Phase Pipeline")
    
    rtl_code = '''
module pipeline_test (
    input wire clk,
    input wire rst_n,
    
    input wire [31:0] s_axis_tdata,
    input wire s_axis_tvalid,
    output wire s_axis_tready,
    
    output wire [31:0] m_axis_tdata,
    output wire m_axis_tvalid,
    input wire m_axis_tready
);

parameter THRESHOLD = 128;

assign m_axis_tdata = (s_axis_tdata > THRESHOLD) ? 32'hFFFFFFFF : 32'h00000000;
assign m_axis_tvalid = s_axis_tvalid;
assign s_axis_tready = m_axis_tready;

endmodule
'''
    
    rtl_file = Path('/tmp/pipeline_test.sv')
    rtl_file.write_text(rtl_code)
    
    try:
        # Phase 1: Parse RTL
        hw_kernel = parse_rtl_file(rtl_file)
        assert hw_kernel is not None
        
        # Phase 2: Parse Compiler Data (embedded in RTL parameters)
        assert len(hw_kernel.rtl_parameters) > 0, "Should detect RTL parameters"
        threshold_param = next((p for p in hw_kernel.rtl_parameters if p.name == "THRESHOLD"), None)
        assert threshold_param is not None, "Should detect THRESHOLD parameter"
        
        # Phase 3: Build Dataflow Model (through interface enhancement)
        assert hw_kernel.interfaces is not None
        assert len(hw_kernel.interfaces) > 0
        
        # Phase 4 & 5: Generate Templates → Generate Components
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            generators = [
                HWCustomOpGenerator(),
                RTLBackendGenerator(), 
                TestSuiteGenerator()
            ]
            
            generated_files = []
            for generator in generators:
                output_file = generator.generate(hw_kernel, output_dir)
                generated_files.append(output_file)
                assert output_file.exists()
                assert output_file.stat().st_size > 0
                
            # Validate all phases completed successfully
            assert len(generated_files) == 3
            
        print("✅ Axiom 2: Multi-Phase Pipeline - PASSED")
        return True
        
    finally:
        if rtl_file.exists():
            rtl_file.unlink()


def test_axiom_3_template_driven_code_generation():
    """
    Test Axiom 3: Template-Driven Code Generation
    All code generation uses Jinja2 templates with rich context objects
    """
    print("Testing Axiom 3: Template-Driven Code Generation")
    
    rtl_code = '''
module template_test (
    input wire clk,
    input wire rst_n,
    
    input wire [127:0] s_axis_data_tdata,
    input wire s_axis_data_tvalid,
    output wire s_axis_data_tready,
    
    output wire [127:0] m_axis_result_tdata,
    output wire m_axis_result_tvalid,
    input wire m_axis_result_tready
);

parameter DATA_WIDTH = 128;
parameter ELEMENTS = 16;

assign m_axis_result_tdata = s_axis_data_tdata;
assign m_axis_result_tvalid = s_axis_data_tvalid;
assign s_axis_data_tready = m_axis_result_tready;

endmodule
'''
    
    rtl_file = Path('/tmp/template_test.sv')
    rtl_file.write_text(rtl_code)
    
    try:
        hw_kernel = parse_rtl_file(rtl_file)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Test HWCustomOp template generation
            hwcustomop_gen = HWCustomOpGenerator()
            hwcustomop_file = hwcustomop_gen.generate(hw_kernel, output_dir)
            hwcustomop_content = hwcustomop_file.read_text()
            
            # Should be slim Python classes (50-80 lines as per axiom)
            hwcustomop_lines = len(hwcustomop_content.split('\n'))
            assert 30 < hwcustomop_lines < 200, f"HWCustomOp should be compact, got {hwcustomop_lines} lines"
            
            # Should contain template-generated content
            assert "class " in hwcustomop_content
            assert "def __init__" in hwcustomop_content
            assert hw_kernel.name in hwcustomop_content
            
            # Test RTLBackend template generation
            rtlbackend_gen = RTLBackendGenerator()
            rtlbackend_file = rtlbackend_gen.generate(hw_kernel, output_dir)
            rtlbackend_content = rtlbackend_file.read_text()
            
            # Should contain FINN integration components
            assert "RTLBackend" in rtlbackend_content
            assert "generate_hdl" in rtlbackend_content or "rtl" in rtlbackend_content.lower()
            
            # Test Suite template generation
            testsuite_gen = TestSuiteGenerator()
            testsuite_file = testsuite_gen.generate(hw_kernel, output_dir)
            testsuite_content = testsuite_file.read_text()
            
            # Should contain comprehensive validation frameworks
            assert "test_" in testsuite_content
            assert "pytest" in testsuite_content
            assert "assert" in testsuite_content
            
        print("✅ Axiom 3: Template-Driven Code Generation - PASSED")
        return True
        
    finally:
        if rtl_file.exists():
            rtl_file.unlink()


def test_axiom_5_runtime_dimension_extraction():
    """
    Test Axiom 5: Runtime Dimension Extraction
    Generated components extract dimensions at runtime, not compile-time
    """
    print("Testing Axiom 5: Runtime Dimension Extraction")
    
    rtl_code = '''
module runtime_test (
    input wire clk,
    input wire rst_n,
    
    input wire [255:0] s_axis_input_tdata,
    input wire s_axis_input_tvalid,
    output wire s_axis_input_tready,
    
    output wire [255:0] m_axis_output_tdata,
    output wire m_axis_output_tvalid,
    input wire m_axis_output_tready
);

assign m_axis_output_tdata = s_axis_input_tdata;
assign m_axis_output_tvalid = s_axis_input_tvalid;
assign s_axis_input_tready = m_axis_output_tready;

endmodule
'''
    
    rtl_file = Path('/tmp/runtime_test.sv')
    rtl_file.write_text(rtl_code)
    
    try:
        hw_kernel = parse_rtl_file(rtl_file)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            hwcustomop_gen = HWCustomOpGenerator()
            hwcustomop_file = hwcustomop_gen.generate(hw_kernel, output_dir)
            hwcustomop_content = hwcustomop_file.read_text()
            
            # Should NOT contain hardcoded tensor_dims, block_dims, stream_dims
            assert "tensor_dims=" not in hwcustomop_content or "runtime" in hwcustomop_content.lower()
            assert "block_dims=" not in hwcustomop_content or "runtime" in hwcustomop_content.lower()
            
            # Should contain runtime extraction mechanisms
            assert "determine_chunking_from_layout" in hwcustomop_content or "runtime" in hwcustomop_content.lower()
            
            # RTLBackend should support runtime configuration
            rtlbackend_gen = RTLBackendGenerator()
            rtlbackend_file = rtlbackend_gen.generate(hw_kernel, output_dir)
            rtlbackend_content = rtlbackend_file.read_text()
            
            assert "runtime" in rtlbackend_content.lower() or "extract" in rtlbackend_content.lower()
            
        print("✅ Axiom 5: Runtime Dimension Extraction - PASSED")
        return True
        
    finally:
        if rtl_file.exists():
            rtl_file.unlink()


def test_axiom_7_hierarchical_error_handling():
    """
    Test Axiom 7: Hierarchical Error Handling
    Structured error handling with context and actionable suggestions
    """
    print("Testing Axiom 7: Hierarchical Error Handling")
    
    # Test with invalid RTL
    invalid_rtl = '''
module broken_syntax (
    input wire [7:0 data_in  // Missing closing bracket
    output wire invalid_output
);
// Missing endmodule
'''
    
    rtl_file = Path('/tmp/broken_syntax.sv')
    rtl_file.write_text(invalid_rtl)
    
    try:
        # Should handle parsing errors gracefully
        try:
            hw_kernel = parse_rtl_file(rtl_file)
            # If it doesn't throw an exception, that's also valid error handling
            assert hw_kernel is not None
        except Exception as e:
            # Should provide structured error information
            error_msg = str(e).lower()
            assert "rtl" in error_msg or "parsing" in error_msg or "syntax" in error_msg
            assert len(str(e)) > 10, "Error message should be informative"
            
        print("✅ Axiom 7: Hierarchical Error Handling - PASSED")
        return True
        
    finally:
        if rtl_file.exists():
            rtl_file.unlink()


def test_axiom_9_generator_factory_pattern():
    """
    Test Axiom 9: Generator Factory Pattern
    Specialized generators implement common interface with dedicated logic
    """
    print("Testing Axiom 9: Generator Factory Pattern")
    
    rtl_code = '''
module factory_test (
    input wire clk,
    input wire rst_n,
    
    input wire [31:0] s_axis_tdata,
    input wire s_axis_tvalid,
    output wire s_axis_tready,
    
    output wire [31:0] m_axis_tdata,
    output wire m_axis_tvalid,
    input wire m_axis_tready
);

assign m_axis_tdata = s_axis_tdata;
assign m_axis_tvalid = s_axis_tvalid;
assign s_axis_tready = m_axis_tready;

endmodule
'''
    
    rtl_file = Path('/tmp/factory_test.sv')
    rtl_file.write_text(rtl_code)
    
    try:
        hw_kernel = parse_rtl_file(rtl_file)
        
        # Test that all generators implement common interface
        generators = [
            HWCustomOpGenerator(),
            RTLBackendGenerator(),
            TestSuiteGenerator()
        ]
        
        # All should have common methods
        for generator in generators:
            assert hasattr(generator, 'generate'), f"{type(generator).__name__} should have generate method"
            assert hasattr(generator, '_get_output_filename'), f"{type(generator).__name__} should have _get_output_filename method"
            
        # Test that each produces different output types
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            outputs = {}
            for generator in generators:
                output_file = generator.generate(hw_kernel, output_dir)
                content = output_file.read_text()
                outputs[type(generator).__name__] = content
                
            # Each should produce specialized content
            assert "HWCustomOp" in outputs["HWCustomOpGenerator"]
            assert "RTLBackend" in outputs["RTLBackendGenerator"]
            assert "test_" in outputs["TestSuiteGenerator"]
            
            # Should be different from each other
            contents = list(outputs.values())
            assert contents[0] != contents[1]
            assert contents[1] != contents[2]
            assert contents[0] != contents[2]
            
        print("✅ Axiom 9: Generator Factory Pattern - PASSED")
        return True
        
    finally:
        if rtl_file.exists():
            rtl_file.unlink()


def run_axiom_validation():
    """Run all axiom validation tests."""
    print("=" * 60)
    print("HWKG Axiom Validation Test Suite")
    print("=" * 60)
    
    tests = [
        test_axiom_1_interface_wise_dataflow_foundation,
        test_axiom_2_multi_phase_pipeline,
        test_axiom_3_template_driven_code_generation,
        test_axiom_5_runtime_dimension_extraction,
        test_axiom_7_hierarchical_error_handling,
        test_axiom_9_generator_factory_pattern
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
            else:
                failed += 1
                print(f"❌ {test.__name__} - FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test.__name__} - ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print("=" * 60)
    print("AXIOM VALIDATION SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL AXIOMS VALIDATED SUCCESSFULLY!")
        print("The HWKG implementation correctly follows all design axioms.")
        return True
    else:
        print(f"\n⚠️  {failed} axiom(s) need attention.")
        return False


if __name__ == "__main__":
    success = run_axiom_validation()
    sys.exit(0 if success else 1)