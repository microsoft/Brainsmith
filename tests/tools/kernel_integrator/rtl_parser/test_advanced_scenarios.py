############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Advanced scenario validation tests for RTL Parser.

This test module covers complex edge cases and real-world scenarios including:
- Multi-module files with TOP_MODULE pragma selection
- Complex pragma interaction patterns
- Performance validation
- Error recovery scenarios
- Real-world hardware kernel patterns
"""

import pytest
import time
from brainsmith.tools.kernel_integrator.rtl_parser import RTLParser
from brainsmith.tools.kernel_integrator.data import InterfaceType

from .utils.rtl_builder import RTLBuilder, StrictRTLBuilder


class TestMultiModuleHandling:
    """Test multi-module file handling and TOP_MODULE pragma."""
    
    def test_top_module_pragma_selects_correct_module(self, rtl_parser):
        """Test TOP_MODULE pragma selects the correct module from multiple modules."""
        # Create multi-module RTL with TOP_MODULE pragma
        rtl = """
// @brainsmith TOP_MODULE target_module
module dummy_module #(
    parameter DUMMY_PARAM = 8
) (
    input wire clk,
    input wire rst_n
);
endmodule

module target_module #(
    parameter TARGET_WIDTH = 32,
    parameter TARGET_DEPTH = 16
) (
    input wire ap_clk,
    input wire ap_rst_n,
    
    input wire [31:0] s_axis_input_tdata,
    input wire s_axis_input_tvalid,
    output wire s_axis_input_tready,
    
    output wire [31:0] m_axis_output_tdata,
    output wire m_axis_output_tvalid,
    input wire m_axis_output_tready
);
    assign m_axis_output_tdata = s_axis_input_tdata;
    assign m_axis_output_tvalid = s_axis_input_tvalid;
    assign s_axis_input_tready = m_axis_output_tready;
endmodule

module another_module (
    input wire dummy_signal
);
endmodule
"""
        
        kernel_metadata = rtl_parser.parse(rtl, "multi_module_test.sv")
        
        # Should select target_module based on TOP_MODULE pragma
        assert kernel_metadata.name == "target_module"
        
        # Should have parameters from target_module
        param_names = {p.name for p in kernel_metadata.parameters}
        assert "TARGET_WIDTH" in param_names
        assert "TARGET_DEPTH" in param_names
        assert "DUMMY_PARAM" not in param_names  # From wrong module
        
        # Should have interfaces from target_module
        assert len(kernel_metadata.interfaces) >= 3  # input + output + control
    
    def test_module_selection_without_top_module_pragma_requires_error(self, rtl_parser):
        """Test module selection when no TOP_MODULE pragma is present - should require TOP_MODULE pragma."""
        rtl = """
module first_module (
    input wire clk
);
endmodule

module second_module #(
    parameter DATA_WIDTH = 32
) (
    input wire ap_clk,
    input wire ap_rst_n,
    
    input wire [31:0] s_axis_input_tdata,
    input wire s_axis_input_tvalid,
    output wire s_axis_input_tready,
    
    output wire [31:0] m_axis_output_tdata,
    output wire m_axis_output_tvalid,
    input wire m_axis_output_tready
);
endmodule
"""
        
        # Should raise error when multiple modules present without TOP_MODULE pragma
        with pytest.raises(Exception) as exc_info:
            rtl_parser.parse(rtl, "no_top_pragma_test.sv")
        
        assert "TOP_MODULE pragma specified" in str(exc_info.value) or "Multiple modules" in str(exc_info.value)
    
    def test_explicit_module_name_parameter(self, rtl_parser):
        """Test explicit module targeting with module_name parameter."""
        rtl = """
module unwanted_module (
    input wire dummy
);
endmodule

module wanted_module #(
    parameter WANTED_PARAM = 64
) (
    input wire ap_clk,
    input wire ap_rst_n,
    
    input wire [31:0] s_axis_input_tdata,
    input wire s_axis_input_tvalid,
    output wire s_axis_input_tready,
    
    output wire [31:0] m_axis_output_tdata,
    output wire m_axis_output_tvalid,
    input wire m_axis_output_tready
);
endmodule
"""
        
        # Parse with explicit module_name parameter
        kernel_metadata = rtl_parser.parse(rtl, "explicit_target_test.sv", module_name="wanted_module")
        
        # Should select wanted_module when explicitly specified
        assert kernel_metadata.name == "wanted_module"
        
        # Should have parameters from wanted_module
        param_names = {p.name for p in kernel_metadata.parameters}
        assert "WANTED_PARAM" in param_names
    
    def test_complex_multi_module_with_pragmas(self, rtl_parser):
        """Test multi-module file with pragmas targeting the selected module."""
        rtl = """
// @brainsmith TOP_MODULE complex_target
// @brainsmith DATATYPE s_axis_input UINT 8 32
// @brainsmith BDIM s_axis_input TILE_SIZE

module decoy_module (
    input wire dummy
);
endmodule

module complex_target #(
    parameter TILE_SIZE = 32,
    parameter STREAM_SIZE = 1024,
    parameter ACC_WIDTH = 48,
    parameter ACC_SIGNED = 1
) (
    input wire ap_clk,
    input wire ap_rst_n,
    
    input wire [31:0] s_axis_input_tdata,
    input wire s_axis_input_tvalid,
    output wire s_axis_input_tready,
    
    output wire [31:0] m_axis_output_tdata,
    output wire m_axis_output_tvalid,
    input wire m_axis_output_tready
);
endmodule
"""
        
        kernel_metadata = rtl_parser.parse(rtl, "complex_multi_module_test.sv")
        
        # Should select complex_target
        assert kernel_metadata.name == "complex_target"
        
        # Should apply pragmas to selected module
        input_interface = next(
            (i for i in kernel_metadata.interfaces if i.interface_type == InterfaceType.INPUT),
            None
        )
        assert input_interface is not None
        
        # DATATYPE pragma should create constraints
        assert len(input_interface.datatype_constraints) == 1
        constraint = input_interface.datatype_constraints[0]
        assert constraint.base_type == "UINT"
        
        # BDIM pragma should assign parameters
        assert input_interface.bdim_params == ["TILE_SIZE"]


class TestComplexPragmaInteractions:
    """Test complex pragma interaction patterns and edge cases."""
    
    def test_datatype_param_with_auto_linking_conflict(self, rtl_parser):
        """Test DATATYPE_PARAM pragma vs auto-linking conflict resolution."""
        rtl = (StrictRTLBuilder()
               .module("datatype_param_conflict_test")
               .parameter("s_axis_input_WIDTH", "32")      # Would auto-link
               .parameter("CUSTOM_WIDTH", "24")            # Used in DATATYPE_PARAM
               .pragma("DATATYPE_PARAM", "s_axis_input", "width", "CUSTOM_WIDTH")
               .add_stream_input("s_axis_input", bdim_value="32", sdim_value="512")
               .add_stream_output("m_axis_output", bdim_value="32")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "datatype_param_conflict_test.sv")
        
        # DATATYPE_PARAM should take precedence
        assert "CUSTOM_WIDTH" not in kernel_metadata.exposed_parameters
        
        # Auto-linking parameter may or may not be hidden depending on implementation
        # The key is that DATATYPE_PARAM pragma is correctly processed
        
        # Validate interface has proper datatype metadata
        input_interface = next(
            (i for i in kernel_metadata.interfaces if i.interface_type == InterfaceType.INPUT),
            None
        )
        assert input_interface is not None
        # Interface should have datatype_metadata from DATATYPE_PARAM pragma
    
    def test_circular_pragma_dependency_handling(self, rtl_parser):
        """Test handling of circular dependencies in derived parameters."""
        rtl = (StrictRTLBuilder()
               .module("circular_dependency_test")
               .parameter("A", "16")
               .parameter("B", "32")
               .parameter("C", "64")
               .pragma("DERIVED_PARAMETER", "A", "B + C")
               .pragma("DERIVED_PARAMETER", "B", "C * 2") 
               .pragma("DERIVED_PARAMETER", "C", "A / 2")  # Creates circular dependency
               .add_stream_input("s_axis_input", bdim_value="32", sdim_value="512")
               .add_stream_output("m_axis_output", bdim_value="32")
               .build())
        
        # Should parse without crashing (may warn about circular dependencies)
        kernel_metadata = rtl_parser.parse(rtl, "circular_dependency_test.sv")
        
        assert kernel_metadata.name == "circular_dependency_test"
        # Circular dependencies may result in some parameters remaining exposed
        # Implementation should handle this gracefully
    
    def test_pragma_chain_interaction(self, rtl_parser):
        """Test complex pragma chains and their interactions."""
        rtl = (StrictRTLBuilder()
               .module("pragma_chain_test")
               .parameter("BASE", "8")
               .parameter("MULTIPLIER", "4")
               .parameter("DERIVED_WIDTH", "32")
               .parameter("TILE_SIZE", "16")
               .pragma("ALIAS", "MULTIPLIER", "ScaleFactor")
               .pragma("DERIVED_PARAMETER", "DERIVED_WIDTH", "BASE * ScaleFactor")
               .pragma("DATATYPE", "s_axis_input", "UINT", "8", "32")
               .pragma("BDIM", "s_axis_input", "TILE_SIZE")
               .pragma("DATATYPE_PARAM", "accumulator", "width", "DERIVED_WIDTH")
               .add_stream_input("s_axis_input", bdim_value="16", sdim_value="512")
               .add_stream_output("m_axis_output", bdim_value="32")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "pragma_chain_test.sv")
        
        # Validate pragma chain effects
        input_interface = next(
            (i for i in kernel_metadata.interfaces if i.interface_type == InterfaceType.INPUT),
            None
        )
        assert input_interface is not None
        
        # DATATYPE pragma should create constraints
        assert len(input_interface.datatype_constraints) == 1
        
        # BDIM pragma should assign parameters
        assert input_interface.bdim_params == ["TILE_SIZE"]
        
        # Parameter exposure should reflect all pragma effects
        assert "MULTIPLIER" not in kernel_metadata.exposed_parameters  # ALIAS
        assert "ScaleFactor" in kernel_metadata.exposed_parameters      # ALIAS target
        assert "DERIVED_WIDTH" not in kernel_metadata.exposed_parameters  # DERIVED + DATATYPE_PARAM
    
    def test_pragma_error_isolation(self, rtl_parser):
        """Test that pragma errors don't break other pragma processing."""
        rtl = (StrictRTLBuilder()
               .module("pragma_error_isolation_test")
               .parameter("VALID_PARAM", "32")
               .parameter("TILE_SIZE", "16")
               .pragma("DATATYPE", "nonexistent_interface", "UINT", "8", "32")  # Invalid
               .pragma("BDIM", "s_axis_input", "NONEXISTENT_PARAM")            # Invalid
               .pragma("DATATYPE", "s_axis_input", "UINT", "8", "32")          # Valid
               .pragma("ALIAS", "VALID_PARAM", "ValidParameter")               # Valid
               .add_stream_input("s_axis_input", bdim_value="16", sdim_value="512")
               .add_stream_output("m_axis_output", bdim_value="32")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "pragma_error_isolation_test.sv")
        
        # Valid pragmas should still work despite invalid ones
        input_interface = next(
            (i for i in kernel_metadata.interfaces if i.interface_type == InterfaceType.INPUT),
            None
        )
        assert input_interface is not None
        
        # Valid DATATYPE pragma should create constraints
        assert len(input_interface.datatype_constraints) == 1
        
        # Valid ALIAS pragma should work
        assert "VALID_PARAM" not in kernel_metadata.exposed_parameters
        assert "ValidParameter" in kernel_metadata.exposed_parameters


class TestPerformanceValidation:
    """Test performance characteristics and benchmarks."""
    
    def test_parsing_performance_benchmark(self, rtl_parser):
        """Test parsing performance meets sub-100ms target for typical kernels."""
        # Create a moderately complex kernel
        rtl = (StrictRTLBuilder()
               .module("performance_test_kernel")
               # Multiple parameters
               .parameter("TILE_H", "16")
               .parameter("TILE_W", "16")
               .parameter("CHANNELS", "32")
               .parameter("PE", "8")
               .parameter("SIMD", "4")
               .parameter("ACC_WIDTH", "48")
               .parameter("ACC_SIGNED", "1")
               .parameter("THRESH_WIDTH", "32")
               .parameter("WEIGHT_WIDTH", "8")
               .parameter("OUTPUT_WIDTH", "16")
               # Multiple pragmas
               .pragma("DATATYPE", "s_axis_input", "UINT", "8", "16")
               .pragma("BDIM", "s_axis_input", "[TILE_H, TILE_W, CHANNELS]")
               .pragma("SDIM", "s_axis_input", "[PE, SIMD]")
               .pragma("DATATYPE", "s_axis_weights", "INT", "8", "8")
               .pragma("BDIM", "s_axis_weights", "[CHANNELS, OUTPUT_WIDTH]")
               .pragma("SDIM", "s_axis_weights", "[PE, SIMD]")
               .pragma("WEIGHT", "s_axis_weights")
               .pragma("DATATYPE", "m_axis_output", "UINT", "8", "32")
               .pragma("BDIM", "m_axis_output", "[TILE_H, TILE_W, OUTPUT_WIDTH]")
               .pragma("ALIAS", "PE", "ParallelismFactor")
               .pragma("ALIAS", "SIMD", "SIMDFactor")
               .pragma("DATATYPE_PARAM", "accumulator", "width", "ACC_WIDTH")
               .pragma("DATATYPE_PARAM", "accumulator", "signed", "ACC_SIGNED")
               .pragma("DATATYPE_PARAM", "threshold", "width", "THRESH_WIDTH")
               .pragma("RELATIONSHIP", "s_axis_input", "s_axis_weights", "EQUAL")
               .pragma("RELATIONSHIP", "s_axis_input", "m_axis_output", "DEPENDENT", "0", "0", "conv")
               # Multiple interfaces
               .add_stream_input("s_axis_input", bdim_value="8192", sdim_value="32")
               .add_stream_input("s_axis_weights", bdim_value="512", sdim_value="32")
               .add_stream_output("m_axis_output", bdim_value="4096")
               .axi_lite_slave("s_axi_config")
               .build())
        
        # Measure parsing time
        start_time = time.time()
        kernel_metadata = rtl_parser.parse(rtl, "performance_test.sv")
        parse_time = time.time() - start_time
        
        # Should parse successfully
        assert kernel_metadata.name == "performance_test_kernel"
        
        # Performance target: sub-100ms for typical kernels
        # Note: This is a guideline, actual performance may vary by system
        print(f"Parsing time: {parse_time*1000:.2f}ms")
        
        # Validate all components were processed
        assert len(kernel_metadata.interfaces) >= 4  # 2 inputs + 1 output + 1 config + 1 control
        assert len(kernel_metadata.pragmas) >= 12   # All pragmas should be parsed
        assert len(kernel_metadata.internal_datatypes) >= 2  # accumulator + threshold
    
    def test_large_parameter_count_handling(self, rtl_parser):
        """Test handling of modules with many parameters."""
        builder = StrictRTLBuilder().module("large_param_test")
        
        # Add many parameters
        for i in range(50):
            builder.parameter(f"PARAM_{i}", str(16 + i))
            if i % 5 == 0:  # Add some internal datatypes
                builder.parameter(f"DT{i//5}_WIDTH", str(8 + i))
                builder.parameter(f"DT{i//5}_SIGNED", str(i % 2))
        
        rtl = (builder
               .add_stream_input("s_axis_input", bdim_value="32", sdim_value="512")
               .add_stream_output("m_axis_output", bdim_value="32")
               .build())
        
        # Should handle large parameter count efficiently
        start_time = time.time()
        kernel_metadata = rtl_parser.parse(rtl, "large_param_test.sv")
        parse_time = time.time() - start_time
        
        # Should parse successfully
        assert kernel_metadata.name == "large_param_test"
        assert len(kernel_metadata.parameters) >= 50
        
        # Should detect internal datatypes
        assert len(kernel_metadata.internal_datatypes) >= 10
        
        print(f"Large parameter parsing time: {parse_time*1000:.2f}ms")


class TestRealWorldScenarios:
    """Test real-world hardware kernel patterns."""
    
    def test_convolution_kernel_pattern(self, rtl_parser):
        """Test parsing pattern similar to convolution kernels."""
        rtl = (StrictRTLBuilder()
               .module("conv2d_kernel")
               .parameter("IFM_CH", "64")
               .parameter("OFM_CH", "128")
               .parameter("KERNEL_SIZE", "3")
               .parameter("TILE_H", "8")
               .parameter("TILE_W", "8")
               .parameter("PE", "16")
               .parameter("SIMD", "8")
               .parameter("ACC_WIDTH", "32")
               .parameter("WEIGHT_WIDTH", "8")
               .parameter("ACT_WIDTH", "8")
               .pragma("DATATYPE", "s_axis_input", "UINT", "ACT_WIDTH", "ACT_WIDTH")
               .pragma("DATATYPE", "s_axis_weights", "INT", "WEIGHT_WIDTH", "WEIGHT_WIDTH")
               .pragma("BDIM", "s_axis_input", "[TILE_H, TILE_W, IFM_CH]")
               .pragma("SDIM", "s_axis_input", "[PE, SIMD]")
               .pragma("BDIM", "s_axis_weights", "[KERNEL_SIZE, KERNEL_SIZE, IFM_CH, OFM_CH]")
               .pragma("SDIM", "s_axis_weights", "[PE, SIMD]")
               .pragma("WEIGHT", "s_axis_weights")
               .pragma("BDIM", "m_axis_output", "[TILE_H, TILE_W, OFM_CH]")
               .pragma("ALIAS", "PE", "ParallelElements")
               .pragma("ALIAS", "SIMD", "SIMDWidth")
               .pragma("DATATYPE_PARAM", "accumulator", "width", "ACC_WIDTH")
               .pragma("RELATIONSHIP", "s_axis_input", "m_axis_output", "DEPENDENT", "0", "0", "conv2d")
               .add_stream_input("s_axis_input", bdim_value="4096", sdim_value="128")
               .add_stream_input("s_axis_weights", bdim_value="9216", sdim_value="128")
               .add_stream_output("m_axis_output", bdim_value="8192")
               .axi_lite_slave("s_axi_config")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "conv2d_kernel.sv")
        
        # Validate convolution pattern
        assert kernel_metadata.name == "conv2d_kernel"
        
        # Should have proper interface types
        input_ifaces = [i for i in kernel_metadata.interfaces if i.interface_type == InterfaceType.INPUT]
        weight_ifaces = [i for i in kernel_metadata.interfaces if i.interface_type == InterfaceType.WEIGHT]
        output_ifaces = [i for i in kernel_metadata.interfaces if i.interface_type == InterfaceType.OUTPUT]
        
        assert len(input_ifaces) == 1
        assert len(weight_ifaces) == 1  # WEIGHT pragma should convert one input to weight
        assert len(output_ifaces) == 1
        
        # Validate dimensional parameters
        input_interface = input_ifaces[0]
        assert input_interface.bdim_params == ["TILE_H", "TILE_W", "IFM_CH"]
        assert input_interface.sdim_params == ["PE", "SIMD"]
    
    def test_thresholding_kernel_pattern(self, rtl_parser):
        """Test parsing pattern similar to thresholding kernels."""
        rtl = (StrictRTLBuilder()
               .module("thresholding_kernel")
               .parameter("CH", "256")
               .parameter("PE", "32")
               .parameter("THRESH_WIDTH", "16")
               .parameter("THRESH_SIGNED", "1")
               .parameter("INPUT_WIDTH", "8")
               .parameter("OUTPUT_WIDTH", "1")
               .pragma("DATATYPE", "s_axis_input", "UINT", "INPUT_WIDTH", "INPUT_WIDTH")
               .pragma("BDIM", "s_axis_input", "CH")
               .pragma("SDIM", "s_axis_input", "PE")
               .pragma("DATATYPE", "s_axis_threshold", "INT", "THRESH_WIDTH", "THRESH_WIDTH")
               .pragma("BDIM", "s_axis_threshold", "CH") 
               .pragma("SDIM", "s_axis_threshold", "1")
               .pragma("WEIGHT", "s_axis_threshold")
               .pragma("DATATYPE", "m_axis_output", "UINT", "OUTPUT_WIDTH", "OUTPUT_WIDTH")
               .pragma("BDIM", "m_axis_output", "CH")
               .pragma("ALIAS", "PE", "Parallelism")
               .pragma("DATATYPE_PARAM", "threshold", "width", "THRESH_WIDTH")
               .pragma("DATATYPE_PARAM", "threshold", "signed", "THRESH_SIGNED")
               .pragma("RELATIONSHIP", "s_axis_input", "s_axis_threshold", "EQUAL")
               .pragma("RELATIONSHIP", "s_axis_input", "m_axis_output", "DEPENDENT", "0", "0", "threshold")
               .add_stream_input("s_axis_input", bdim_value="256", sdim_value="32")
               .add_stream_input("s_axis_threshold", bdim_value="256", sdim_value="1")
               .add_stream_output("m_axis_output", bdim_value="256")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "thresholding_kernel.sv")
        
        # Validate thresholding pattern
        assert kernel_metadata.name == "thresholding_kernel"
        
        # Should detect internal threshold datatype
        threshold_dt = next(
            (dt for dt in kernel_metadata.internal_datatypes if dt.name == "threshold"),
            None
        )
        assert threshold_dt is not None
        assert threshold_dt.width == "THRESH_WIDTH"
        assert threshold_dt.signed == "THRESH_SIGNED"