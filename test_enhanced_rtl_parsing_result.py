#!/usr/bin/env python3
"""
Unit tests for EnhancedRTLParsingResult functionality.

This validates that the enhanced RTL parsing result provides all template
variables without DataflowModel conversion overhead.
"""

import pytest
from pathlib import Path
from brainsmith.tools.hw_kernel_gen.rtl_parser import (
    parse_rtl_file, 
    create_enhanced_rtl_parsing_result,
    EnhancedRTLParsingResult
)

def test_enhanced_rtl_parsing_result_creation():
    """Test creating EnhancedRTLParsingResult from RTLParsingResult."""
    
    # Parse RTL file to get baseline RTLParsingResult
    rtl_file = Path("examples/thresholding/thresholding_axi.sv")
    rtl_result = parse_rtl_file(rtl_file)
    
    # Convert to enhanced version
    enhanced_result = create_enhanced_rtl_parsing_result(rtl_result)
    
    # Verify core data preservation
    assert enhanced_result.name == rtl_result.name
    assert enhanced_result.interfaces == rtl_result.interfaces
    assert enhanced_result.pragmas == rtl_result.pragmas
    assert enhanced_result.parameters == rtl_result.parameters
    assert enhanced_result.source_file == rtl_result.source_file
    
    print("✅ Enhanced RTL parsing result creation successful")

def test_template_context_generation():
    """Test complete template context generation."""
    
    # Parse and enhance RTL file
    rtl_file = Path("examples/thresholding/thresholding_axi.sv")
    rtl_result = parse_rtl_file(rtl_file)
    enhanced_result = create_enhanced_rtl_parsing_result(rtl_result)
    
    # Generate template context
    context = enhanced_result.get_template_context()
    
    # Verify required template variables
    required_vars = [
        "kernel_name", "class_name", "source_file", "generation_timestamp",
        "interfaces", "input_interfaces", "output_interfaces", "config_interfaces",
        "dataflow_interfaces", "rtl_parameters", "interface_metadata",
        "dataflow_model_summary"
    ]
    
    for var in required_vars:
        assert var in context, f"Missing template variable: {var}"
    
    # Verify basic metadata
    assert context["kernel_name"] == "thresholding_axi"
    assert context["class_name"] == "ThresholdingAxi"
    assert "thresholding_axi.sv" in context["source_file"]
    
    print("✅ Template context generation successful")
    print(f"   Generated {len(context)} template variables")

def test_interface_categorization():
    """Test interface categorization logic."""
    
    # Parse and enhance RTL file
    rtl_file = Path("examples/thresholding/thresholding_axi.sv")
    rtl_result = parse_rtl_file(rtl_file)
    enhanced_result = create_enhanced_rtl_parsing_result(rtl_result)
    
    context = enhanced_result.get_template_context()
    
    # Verify interface categorization
    input_interfaces = context["input_interfaces"]
    output_interfaces = context["output_interfaces"]
    config_interfaces = context["config_interfaces"]
    
    # Expected categorization for thresholding_axi.sv
    expected_input = ["s_axis"]  # AXI-Stream input
    expected_output = ["m_axis"]  # AXI-Stream output
    expected_config = ["ap", "s_axilite"]  # Global control and AXI-Lite
    
    actual_input = [iface["name"] for iface in input_interfaces]
    actual_output = [iface["name"] for iface in output_interfaces]
    actual_config = [iface["name"] for iface in config_interfaces]
    
    assert set(actual_input) == set(expected_input), f"Input mismatch: {actual_input} vs {expected_input}"
    assert set(actual_output) == set(expected_output), f"Output mismatch: {actual_output} vs {expected_output}"
    assert set(actual_config) == set(expected_config), f"Config mismatch: {actual_config} vs {expected_config}"
    
    print("✅ Interface categorization successful")
    print(f"   Input: {actual_input}")
    print(f"   Output: {actual_output}")
    print(f"   Config: {actual_config}")

def test_datatype_constraint_extraction():
    """Test datatype constraint extraction from RTL ports."""
    
    # Parse and enhance RTL file
    rtl_file = Path("examples/thresholding/thresholding_axi.sv")
    rtl_result = parse_rtl_file(rtl_file)
    enhanced_result = create_enhanced_rtl_parsing_result(rtl_result)
    
    context = enhanced_result.get_template_context()
    
    # Check datatype constraints for dataflow interfaces
    for interface in context["dataflow_interfaces"]:
        dtype_constraints = interface["datatype_constraints"]
        assert len(dtype_constraints) > 0, f"No datatype constraints for {interface['name']}"
        
        constraint = dtype_constraints[0]
        assert "finn_type" in constraint
        assert "base_type" in constraint
        assert "bitwidth" in constraint
        assert constraint["bitwidth"] > 0
        
        print(f"   {interface['name']}: {constraint['finn_type']} ({constraint['bitwidth']} bits)")
    
    print("✅ Datatype constraint extraction successful")

def test_dimensional_metadata():
    """Test dimensional metadata extraction and defaults."""
    
    # Parse and enhance RTL file
    rtl_file = Path("examples/thresholding/thresholding_axi.sv")
    rtl_result = parse_rtl_file(rtl_file)
    enhanced_result = create_enhanced_rtl_parsing_result(rtl_result)
    
    context = enhanced_result.get_template_context()
    
    # Check dimensional metadata for all interfaces
    for interface in context["dataflow_interfaces"]:
        assert "tensor_dims" in interface
        assert "block_dims" in interface
        assert "stream_dims" in interface
        
        assert isinstance(interface["tensor_dims"], list)
        assert isinstance(interface["block_dims"], list)
        assert isinstance(interface["stream_dims"], list)
        
        assert len(interface["tensor_dims"]) > 0
        assert len(interface["block_dims"]) > 0
        assert len(interface["stream_dims"]) > 0
        
        print(f"   {interface['name']}: tensor={interface['tensor_dims']}, block={interface['block_dims']}, stream={interface['stream_dims']}")
    
    print("✅ Dimensional metadata successful")

def test_template_context_caching():
    """Test that template context is cached for performance."""
    
    # Parse and enhance RTL file
    rtl_file = Path("examples/thresholding/thresholding_axi.sv")
    rtl_result = parse_rtl_file(rtl_file)
    enhanced_result = create_enhanced_rtl_parsing_result(rtl_result)
    
    # Generate context twice
    context1 = enhanced_result.get_template_context()
    context2 = enhanced_result.get_template_context()
    
    # Should be the same object (cached)
    assert context1 is context2, "Template context should be cached"
    
    print("✅ Template context caching successful")

def test_rtl_parameters_formatting():
    """Test RTL parameter formatting for templates."""
    
    # Parse and enhance RTL file
    rtl_file = Path("examples/thresholding/thresholding_axi.sv")
    rtl_result = parse_rtl_file(rtl_file)
    enhanced_result = create_enhanced_rtl_parsing_result(rtl_result)
    
    context = enhanced_result.get_template_context()
    rtl_parameters = context["rtl_parameters"]
    
    # Verify parameter formatting
    assert len(rtl_parameters) > 0, "Should have RTL parameters"
    
    for param in rtl_parameters:
        assert "name" in param
        assert "param_type" in param
        assert "default_value" in param
        assert "template_param_name" in param
        
        # Template parameter should be uppercase with $
        expected_template_name = f"${param['name'].upper()}$"
        assert param["template_param_name"] == expected_template_name
        
        print(f"   {param['name']}: {param['param_type']} = {param['default_value']} → {param['template_param_name']}")
    
    print("✅ RTL parameter formatting successful")

def test_dataflow_model_summary():
    """Test dataflow model summary generation."""
    
    # Parse and enhance RTL file
    rtl_file = Path("examples/thresholding/thresholding_axi.sv")
    rtl_result = parse_rtl_file(rtl_file)
    enhanced_result = create_enhanced_rtl_parsing_result(rtl_result)
    
    context = enhanced_result.get_template_context()
    summary = context["dataflow_model_summary"]
    
    # Verify summary statistics
    assert "num_interfaces" in summary
    assert "input_count" in summary
    assert "output_count" in summary
    assert "weight_count" in summary
    
    # Expected counts for thresholding_axi.sv
    assert summary["num_interfaces"] == 4  # ap, s_axis, m_axis, s_axilite
    assert summary["input_count"] == 1     # s_axis
    assert summary["output_count"] == 1    # m_axis
    assert summary["weight_count"] == 0    # no weight interfaces
    
    print("✅ Dataflow model summary successful")
    print(f"   Total interfaces: {summary['num_interfaces']}")
    print(f"   Input: {summary['input_count']}, Output: {summary['output_count']}, Weight: {summary['weight_count']}")

def test_kernel_analysis():
    """Test kernel complexity and type analysis."""
    
    # Parse and enhance RTL file
    rtl_file = Path("examples/thresholding/thresholding_axi.sv")
    rtl_result = parse_rtl_file(rtl_file)
    enhanced_result = create_enhanced_rtl_parsing_result(rtl_result)
    
    context = enhanced_result.get_template_context()
    
    # Verify kernel analysis
    assert "kernel_complexity" in context
    assert "kernel_type" in context
    assert "resource_estimation_required" in context
    assert "verification_required" in context
    
    # Expected analysis for thresholding_axi.sv
    assert context["kernel_type"] == "threshold"  # Should detect from name
    assert context["kernel_complexity"] in ["low", "medium", "high"]
    
    print("✅ Kernel analysis successful")
    print(f"   Type: {context['kernel_type']}")
    print(f"   Complexity: {context['kernel_complexity']}")
    print(f"   Resource estimation: {context['resource_estimation_required']}")

def run_all_tests():
    """Run all enhanced RTL parsing result tests."""
    
    print("🧪 ENHANCED RTL PARSING RESULT VALIDATION")
    print("=" * 55)
    
    try:
        test_enhanced_rtl_parsing_result_creation()
        test_template_context_generation()
        test_interface_categorization()
        test_datatype_constraint_extraction()
        test_dimensional_metadata()
        test_template_context_caching()
        test_rtl_parameters_formatting()
        test_dataflow_model_summary()
        test_kernel_analysis()
        
        print()
        print("🎉 ALL TESTS PASSED")
        print("✅ EnhancedRTLParsingResult ready for template generation")
        print("✅ No DataflowModel conversion required")
        print("✅ All template variables available")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)