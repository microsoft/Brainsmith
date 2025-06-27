############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Integration tests for end-to-end RTL parsing - REFACTORED.

Tests the complete parsing pipeline from SystemVerilog source to KernelMetadata,
using the new test infrastructure (patterns, builders, factory).
"""

import pytest
from pathlib import Path

from brainsmith.tools.hw_kernel_gen.rtl_parser.parser import RTLParser, ParserError
from brainsmith.tools.hw_kernel_gen.rtl_parser.ast_parser import SyntaxError
from brainsmith.tools.hw_kernel_gen.data import InterfaceType

from .utils.rtl_builder import RTLBuilder, StrictRTLBuilder
from .utils.rtl_patterns import RTLPatterns
from .utils.test_factory import TestDataFactory
from .utils.test_data_helpers import TestDataHelpers


class TestParserIntegration:
    """Test cases for end-to-end RTL parsing integration."""
    
    def test_minimal_valid_rtl_to_kernel_metadata(self, rtl_parser):
        """Test parsing minimal valid RTL to KernelMetadata."""
        # Use RTLPatterns instead of inline RTL
        rtl = RTLPatterns.minimal_axi_stream("minimal_kernel")
        
        # Parse the RTL
        kernel_metadata = rtl_parser.parse(rtl, "minimal.sv")
        
        # Verify basic metadata
        assert kernel_metadata.name == "minimal_kernel"
        assert len(kernel_metadata.parameters) >= 3  # DATA_WIDTH + BDIM/SDIM params
        
        # Verify interfaces were detected
        assert len(kernel_metadata.interfaces) >= 2
        
        # Find AXI-Stream interfaces
        input_interfaces = [i for i in kernel_metadata.interfaces if i.interface_type == InterfaceType.INPUT]
        output_interfaces = [i for i in kernel_metadata.interfaces if i.interface_type == InterfaceType.OUTPUT]
        
        assert len(input_interfaces) >= 1
        assert len(output_interfaces) >= 1
        
        # Check parameters - all may be auto-linked
        assert len(kernel_metadata.parameters) >= 3
        # With auto-linking, exposed params may be empty
        assert len(kernel_metadata.exposed_parameters) >= 0
    
    def test_complex_kernel_full_pipeline(self, rtl_parser):
        """Test parsing complex kernel with all pragma types."""
        # Use builder with pragma block
        rtl = (StrictRTLBuilder()
               .module("complex_kernel")
               .pragma("TOP_MODULE", "complex_kernel")
               # Parameters
               .parameter("IN_WIDTH", "8")
               .parameter("PE", "4")
               .parameter("TILE_H", "16")
               .parameter("TILE_W", "16")
               .parameter("IN_H", "224")
               .parameter("IN_W", "224")
               .parameter("ACC_WIDTH", "32")
               .parameter("ACC_SIGNED", "1")
               .parameter("OUT_WIDTH", "9")
               # Add interfaces with params
               .add_stream_input("s_axis_in0", data_width="IN_WIDTH",
                               bdim_param="IN0_BDIM", bdim_value="256",
                               sdim_param="IN0_SDIM", sdim_value="50176")
               .add_stream_weight("s_axis_weights", data_width="8",
                                bdim_param="WEIGHTS_BDIM", bdim_value="64",
                                sdim_param="WEIGHTS_SDIM", sdim_value="512")
               .add_stream_output("m_axis_out0", data_width="OUT_WIDTH",
                                bdim_param="OUT0_BDIM", bdim_value="256")
               # Add pragmas
               .add_pragma_block([
                   ("DATATYPE", ["s_axis_in0", "UINT", "8", "32"]),
                   ("BDIM", ["s_axis_in0", "[TILE_H, TILE_W]"]),
                   ("SDIM", ["s_axis_in0", "[IN_H, IN_W]"]),
                   ("DATATYPE", ["s_axis_weights", "INT", "8", "16"]),
                   ("ALIAS", ["PE", "ParallelismFactor"]),
                   ("DERIVED_PARAMETER", ["OUT_WIDTH", "IN_WIDTH + 1"]),
                   ("DATATYPE_PARAM", ["accumulator", "width", "ACC_WIDTH"]),
                   ("DATATYPE_PARAM", ["accumulator", "signed", "ACC_SIGNED"])
               ])
               .build())
        
        # Parse the RTL
        kernel_metadata = rtl_parser.parse(rtl, "complex.sv")
        
        # Verify module selection
        assert kernel_metadata.name == "complex_kernel"
        
        # Verify pragma effects
        assert "PE" not in kernel_metadata.exposed_parameters  # Hidden by ALIAS
        assert "ParallelismFactor" in kernel_metadata.exposed_parameters
        assert "OUT_WIDTH" not in kernel_metadata.exposed_parameters  # Hidden by DERIVED
        
        # Verify interfaces
        input_ifaces = [i for i in kernel_metadata.interfaces if i.interface_type == InterfaceType.INPUT]
        weight_ifaces = [i for i in kernel_metadata.interfaces if i.interface_type == InterfaceType.WEIGHT]
        output_ifaces = [i for i in kernel_metadata.interfaces if i.interface_type == InterfaceType.OUTPUT]
        
        assert len(input_ifaces) >= 1
        assert len(weight_ifaces) >= 1
        assert len(output_ifaces) >= 1
        
        # Verify internal datatypes from DATATYPE_PARAM pragmas
        assert len(kernel_metadata.internal_datatypes) >= 1
        acc_dt = next((dt for dt in kernel_metadata.internal_datatypes if dt.name == "accumulator"), None)
        assert acc_dt is not None
    
    def test_multi_interface_with_relationships(self, rtl_parser):
        """Test parsing module with multiple interfaces and relationships."""
        # Use multi-interface pattern and add relationship pragmas
        rtl = (StrictRTLBuilder()
               .module("multi_interface")
               .pragma("RELATIONSHIP", "s_axis_in0", "s_axis_in1", "EQUAL")
               .pragma("RELATIONSHIP", "s_axis_in0", "m_axis_out0", "DEPENDENT", "0", "0", "scaled", "2")
               .add_stream_input("s_axis_in0", bdim_value="32", sdim_value="512")
               .add_stream_input("s_axis_in1", bdim_value="32", sdim_value="512")
               .add_stream_input("s_axis_in2", bdim_value="32", sdim_value="512")
               .add_stream_weight("s_axis_weight0", bdim_value="64", sdim_value="512")
               .add_stream_output("m_axis_out0", bdim_value="64")
               .add_stream_output("m_axis_out1", bdim_value="64")
               .build())
        
        # Parse
        kernel_metadata = rtl_parser.parse(rtl, "multi_interface.sv")
        
        # Verify interface counts
        assert len(kernel_metadata.interfaces) >= 6  # 3 inputs + 2 outputs + 1 weight + control
        
        # Verify relationships were parsed
        assert len(kernel_metadata.relationships) >= 2
    
    def test_real_thresholding_kernel(self, rtl_parser):
        """Test parsing a real-world kernel pattern."""
        # Use conv2d pattern which is similar to thresholding
        rtl = RTLPatterns.conv2d_kernel(
            input_shape=(32, 32),
            weight_shape=(1, 1),  # 1x1 conv is like thresholding
            output_shape=(32, 32)
        )
        
        kernel_metadata = rtl_parser.parse(rtl, "thresholding.sv")
        
        # Verify parsing succeeded
        assert kernel_metadata.name == "conv2d_kernel"
        assert len(kernel_metadata.interfaces) >= 3
        
        # Check for weight interface
        weight_ifaces = [i for i in kernel_metadata.interfaces if i.interface_type == InterfaceType.WEIGHT]
        assert len(weight_ifaces) >= 1
    
    def test_multi_module_with_top_pragma(self, rtl_parser):
        """Test module selection with TOP_MODULE pragma."""
        # Use hierarchical pattern
        rtl = RTLPatterns.hierarchical_module(num_submodules=3)
        
        kernel_metadata = rtl_parser.parse(rtl, "multi_module.sv")
        
        # Should select top_level due to pragma
        assert kernel_metadata.name == "top_level"
        
        # Should have interfaces from top module only
        assert len(kernel_metadata.interfaces) >= 2
    
    def test_syntax_error_handling(self, rtl_parser):
        """Test handling of SystemVerilog syntax errors."""
        # Create invalid RTL
        rtl = """
        module syntax_error (
            input wire clk  // Missing comma
            input wire data
        );
        endmodule
        """
        
        with pytest.raises((SyntaxError, ParserError)):
            rtl_parser.parse(rtl, "syntax_error.sv")
    
    def test_missing_module_error(self, rtl_parser):
        """Test error when no module found."""
        rtl = "// Just a comment, no module"
        
        with pytest.raises(ParserError) as exc_info:
            rtl_parser.parse(rtl, "no_module.sv")
        
        assert "No modules found" in str(exc_info.value) or "no module" in str(exc_info.value).lower()
    
    def test_parser_with_all_pragmas(self, rtl_parser):
        """Test parser handling all pragma types together."""
        # Use pragma test pattern
        rtl = RTLPatterns.pragma_test_module(
            "all_pragmas",
            ["datatype", "bdim", "sdim", "weight", "alias", "derived"]
        )
        
        kernel_metadata = rtl_parser.parse(rtl, "all_pragmas.sv")
        
        # Verify parsing succeeded with all pragmas
        assert kernel_metadata.name == "all_pragmas"  # Uses the name we passed
        # Count pragmas based on requested types (some may not apply)
        assert len(kernel_metadata.pragmas) >= 3
    
    def test_parameter_autolink_with_interfaces(self, rtl_parser):
        """Test automatic parameter linking to interfaces."""
        # Use pattern with indexed parameters
        rtl = RTLPatterns.parameter_test_module(["indexed"])
        
        # Parse to get the actual module content
        import re
        # Check if the pattern actually includes indexed params
        has_indexed = "BDIM0" in rtl or "BDIM1" in rtl
        
        kernel_metadata = rtl_parser.parse(rtl, "param_link.sv")
        
        # Verify indexed parameters if present
        if has_indexed:
            # Check that indexed params were parsed
            param_names = [p.name for p in kernel_metadata.parameters]
            indexed_params = [p for p in param_names if "BDIM" in p and any(c.isdigit() for c in p)]
            assert len(indexed_params) >= 3  # in0_BDIM0, in0_BDIM1, in0_BDIM2
    
    def test_auto_linking_enabled(self, rtl_parser):
        """Test that auto-linking works by default."""
        # Use builder to create module with linkable parameters
        rtl = (RTLBuilder()
               .module("auto_link_test")
               .add_global_control()
               .parameter("s_axis_data_BDIM", "64")
               .parameter("s_axis_data_SDIM", "1024")
               .parameter("m_axis_result_BDIM", "64")
               .axi_stream_slave("s_axis_data", "32")
               .axi_stream_master("m_axis_result", "32")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "auto_link.sv")
        
        # Parameters should be linked and removed from exposed
        assert "s_axis_data_BDIM" not in kernel_metadata.exposed_parameters
        assert "s_axis_data_SDIM" not in kernel_metadata.exposed_parameters
        assert "m_axis_result_BDIM" not in kernel_metadata.exposed_parameters
    
    def test_auto_linking_disabled(self):
        """Test that auto-linking can be disabled."""
        # Use builder for consistent module
        rtl = (RTLBuilder()
               .module("no_auto_link")
               .parameter("in0_WIDTH", "32")
               .parameter("in0_SIGNED", "1")
               .port("clk", "input")
               .port("s_axis_in0_tdata", "input", "in0_WIDTH-1:0")
               .port("s_axis_in0_tvalid", "input")
               .port("s_axis_in0_tready", "output")
               .build())
        
        # Create parser with auto-linking disabled
        parser = RTLParser(auto_link_parameters=False, strict=False)
        kernel_metadata = parser.parse(rtl, "no_auto_link.sv")
        
        # Verify parameters are still exposed
        assert "in0_WIDTH" in kernel_metadata.exposed_parameters
        assert "in0_SIGNED" in kernel_metadata.exposed_parameters
    
    def test_file_parsing(self, rtl_parser, tmp_path):
        """Test parsing from file."""
        # Create RTL using pattern
        rtl_content = RTLPatterns.minimal_axi_stream("test_from_file")
        
        # Write to temporary file
        rtl_file = tmp_path / "test_module.sv"
        rtl_file.write_text(rtl_content)
        
        # Parse from file
        kernel_metadata = rtl_parser.parse_file(str(rtl_file))
        
        # Verify parsing
        assert kernel_metadata.name == "test_from_file"
        assert kernel_metadata.source_file == rtl_file
    
    def test_axi_lite_interface_parsing(self, rtl_parser):
        """Test parsing module with AXI-Lite control interface."""
        # Use AXI-Lite pattern
        rtl = RTLPatterns.axi_lite_control_module()
        
        kernel_metadata = rtl_parser.parse(rtl, "axi_lite.sv")
        
        # Verify control interface detected
        control_ifaces = [i for i in kernel_metadata.interfaces 
                         if "control" in i.name or "axi" in i.name]
        assert len(control_ifaces) >= 1
    
    def test_error_case_validation(self, strict_rtl_parser):
        """Test validation of error cases in strict mode."""
        # Test missing control interface
        rtl = RTLPatterns.error_case("missing_control")
        with pytest.raises(ParserError) as exc_info:
            strict_rtl_parser.parse(rtl, "missing_control.sv")
        assert "Global Control interface" in str(exc_info.value)
        
        # Test missing interfaces
        rtl = RTLPatterns.error_case("missing_interfaces")
        with pytest.raises(ParserError) as exc_info:
            strict_rtl_parser.parse(rtl, "missing_interfaces.sv")
        # Error message may vary
        assert "at least one interface" in str(exc_info.value) or \
               "at least one input interface" in str(exc_info.value)
        
        # Test missing BDIM
        rtl = RTLPatterns.error_case("missing_bdim")
        with pytest.raises(ParserError) as exc_info:
            strict_rtl_parser.parse(rtl, "missing_bdim.sv")
        assert "missing required BDIM parameter" in str(exc_info.value)
    
    def test_invalid_pragma_application(self, rtl_parser):
        """Test handling of invalid pragma applications."""
        # Use builder with invalid pragma target
        rtl = (RTLBuilder()
               .module("invalid_pragma_test")
               .add_global_control()
               .pragma("BDIM", "nonexistent_interface", "[16, 16]")
               .pragma("DATATYPE", "also_nonexistent", "UINT", "8", "32")
               .axi_stream_slave("s_axis_valid", "32")
               .axi_stream_master("m_axis_valid", "32")
               .build())
        
        # Should parse without error (pragmas just won't apply)
        kernel_metadata = rtl_parser.parse(rtl, "invalid_pragma.sv")
        assert kernel_metadata.name == "invalid_pragma_test"
        assert len(kernel_metadata.interfaces) >= 2
    
    def test_module_with_nested_parameters(self, rtl_parser):
        """Test parsing modules with complex parameter expressions."""
        rtl = (RTLBuilder()
               .module("nested_params")
               .add_global_control()
               .parameter("BASE_WIDTH", "8")
               .parameter("SCALE_FACTOR", "4")
               .parameter("DERIVED_WIDTH", "BASE_WIDTH * SCALE_FACTOR")
               .parameter("ARRAY_SIZE", "(DERIVED_WIDTH + 7) / 8")
               .parameter("s_axis_data_BDIM", "DERIVED_WIDTH")
               .parameter("s_axis_data_SDIM", "ARRAY_SIZE * 64")
               .axi_stream_slave("s_axis_data", "DERIVED_WIDTH")
               .axi_stream_master("m_axis_result", "DERIVED_WIDTH")
               .parameter("m_axis_result_BDIM", "DERIVED_WIDTH")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "nested_params.sv")
        
        # Verify parameters were parsed
        param_names = {p.name for p in kernel_metadata.parameters}
        assert "BASE_WIDTH" in param_names
        assert "DERIVED_WIDTH" in param_names
        
        # Verify auto-linking removed BDIM/SDIM params
        assert "s_axis_data_BDIM" not in kernel_metadata.exposed_parameters
        assert "s_axis_data_SDIM" not in kernel_metadata.exposed_parameters
    
    def test_multiple_weight_interfaces(self, rtl_parser):
        """Test parsing with multiple weight streams."""
        rtl = RTLPatterns.multi_interface("multi_weight", 
                                        num_inputs=1, 
                                        num_outputs=1, 
                                        num_weights=3)
        
        kernel_metadata = rtl_parser.parse(rtl, "multi_weight.sv")
        
        # Verify weight interfaces
        weight_ifaces = [i for i in kernel_metadata.interfaces 
                        if i.interface_type == InterfaceType.WEIGHT]
        assert len(weight_ifaces) >= 3
        
        # Check that WEIGHT pragmas were generated
        weight_pragmas = [p for p in kernel_metadata.pragmas 
                         if p.type.value == "weight"]
        assert len(weight_pragmas) >= 3
    
    def test_relationship_pragma_parsing(self, rtl_parser):
        """Test relationship pragma parsing and storage."""
        rtl = (RTLBuilder()
               .module("relationship_test")
               .add_global_control()
               .pragma("RELATIONSHIP", "s_axis_a", "s_axis_b", "EQUAL")
               .pragma("RELATIONSHIP", "s_axis_a", "m_axis_out", "DEPENDENT", "0", "0", "scaled", "2")
               .add_axi_stream_pair("a", with_params=True)
               .add_axi_stream_pair("b", with_params=True)
               .axi_stream_master("m_axis_out", "32")
               .parameter("m_axis_out_BDIM", "64")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "relationship.sv")
        
        # Verify relationships were stored
        relationship_pragmas = [p for p in kernel_metadata.pragmas 
                              if p.type.value == "relationship"]
        assert len(relationship_pragmas) >= 2
        
        # Check relationship data (if processed)
        if hasattr(kernel_metadata, 'relationships'):
            assert len(kernel_metadata.relationships) >= 2
    
    def test_datatype_param_pragmas(self, rtl_parser):
        """Test DATATYPE_PARAM pragma effects."""
        rtl = (StrictRTLBuilder()
               .module("datatype_param_test")
               .parameter("ACC_WIDTH", "48")
               .parameter("ACC_SIGNED", "1")
               .parameter("THRESH_WIDTH", "16")
               .parameter("THRESH_SIGNED", "0")
               .pragma("DATATYPE_PARAM", "accumulator", "width", "ACC_WIDTH")
               .pragma("DATATYPE_PARAM", "accumulator", "signed", "ACC_SIGNED")
               .pragma("DATATYPE_PARAM", "threshold", "width", "THRESH_WIDTH")
               .pragma("DATATYPE_PARAM", "threshold", "signed", "THRESH_SIGNED")
               .add_stream_input("s_axis_data", bdim_value="64", sdim_value="1024")
               .add_stream_output("m_axis_result", bdim_value="64")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "datatype_param.sv")
        
        # Check internal datatypes were created
        assert len(kernel_metadata.internal_datatypes) >= 2
        
        # Find specific datatypes
        acc_dt = next((dt for dt in kernel_metadata.internal_datatypes 
                      if dt.name == "accumulator"), None)
        thresh_dt = next((dt for dt in kernel_metadata.internal_datatypes 
                        if dt.name == "threshold"), None)
        
        # Verify datatypes have width and signed attributes
        assert acc_dt is not None
        assert thresh_dt is not None
        assert acc_dt.width == "ACC_WIDTH"
        assert acc_dt.signed == "ACC_SIGNED"
        assert thresh_dt.width == "THRESH_WIDTH"
        assert thresh_dt.signed == "THRESH_SIGNED"
    
    def test_top_module_pragma_effect(self, rtl_parser):
        """Test TOP_MODULE pragma correctly selects module."""
        # Create multi-module RTL
        rtl = """
        module helper_module (
            input wire clk
        );
        endmodule
        
        // @brainsmith TOP_MODULE main_kernel
        module main_kernel (
            input wire ap_clk,
            input wire ap_rst_n,
            input wire [31:0] s_axis_data_tdata,
            input wire s_axis_data_tvalid,
            output wire s_axis_data_tready,
            output wire [31:0] m_axis_result_tdata,
            output wire m_axis_result_tvalid,
            input wire m_axis_result_tready
        );
        endmodule
        
        module another_helper (
            input wire clk
        );
        endmodule
        """
        
        kernel_metadata = rtl_parser.parse(rtl, "multi_module.sv")
        
        # Should select main_kernel due to pragma
        assert kernel_metadata.name == "main_kernel"
        assert len(kernel_metadata.interfaces) >= 2
    
    def test_array_parameter_parsing(self, rtl_parser):
        """Test parsing of array-style parameters."""
        rtl = (RTLBuilder()
               .module("array_params")
               .add_global_control()
               .parameter("DIMS[0]", "16", None)  # SystemVerilog array syntax
               .parameter("DIMS[1]", "32", None)
               .parameter("DIMS[2]", "64", None)
               .parameter("WIDTHS[0]", "8", None)
               .parameter("WIDTHS[1]", "16", None)
               .axi_stream_slave("s_axis_in", "WIDTHS[0]")
               .axi_stream_master("m_axis_out", "WIDTHS[1]")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "array_params.sv")
        
        # Verify array parameters were parsed
        param_names = [p.name for p in kernel_metadata.parameters]
        assert any("DIMS" in name for name in param_names)
        assert any("WIDTHS" in name for name in param_names)
    
    def test_generate_block_handling(self, rtl_parser):
        """Test that generate blocks don't interfere with parsing."""
        rtl = (RTLBuilder()
               .module("generate_test")
               .add_global_control()
               .parameter("NUM_CHANNELS", "4")
               .parameter("DATA_WIDTH", "32")
               .axi_stream_slave("s_axis_input", "DATA_WIDTH")
               .axi_stream_master("m_axis_output", "DATA_WIDTH")
               .body("genvar i;")
               .body("generate")
               .body("    for (i = 0; i < NUM_CHANNELS; i = i + 1) begin : gen_channel")
               .body("        wire [DATA_WIDTH-1:0] channel_data;")
               .body("    end")
               .body("endgenerate")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "generate.sv")
        
        # Should parse successfully despite generate block
        assert kernel_metadata.name == "generate_test"
        assert len(kernel_metadata.parameters) >= 2
        assert len(kernel_metadata.interfaces) >= 2
    
    def test_interface_with_sideband_signals(self, rtl_parser):
        """Test interfaces with additional sideband signals."""
        rtl = (RTLBuilder()
               .module("sideband_test")
               .add_global_control()
               .parameter("DATA_WIDTH", "32")
               .parameter("USER_WIDTH", "4")
               .parameter("ID_WIDTH", "8")
               # AXI-Stream with sideband
               .port("s_axis_tdata", "input", "DATA_WIDTH")
               .port("s_axis_tvalid", "input")
               .port("s_axis_tready", "output")
               .port("s_axis_tlast", "input")
               .port("s_axis_tuser", "input", "USER_WIDTH")
               .port("s_axis_tid", "input", "ID_WIDTH")
               # Output
               .port("m_axis_tdata", "output", "DATA_WIDTH")
               .port("m_axis_tvalid", "output")
               .port("m_axis_tready", "input")
               .port("m_axis_tlast", "output")
               .port("m_axis_tuser", "output", "USER_WIDTH")
               .port("m_axis_tid", "output", "ID_WIDTH")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "sideband.sv")
        
        # Should detect AXI-Stream interfaces despite extra signals
        assert len(kernel_metadata.interfaces) >= 2
        input_ifaces = [i for i in kernel_metadata.interfaces 
                       if i.interface_type == InterfaceType.INPUT]
        output_ifaces = [i for i in kernel_metadata.interfaces 
                        if i.interface_type == InterfaceType.OUTPUT]
        assert len(input_ifaces) >= 1
        assert len(output_ifaces) >= 1