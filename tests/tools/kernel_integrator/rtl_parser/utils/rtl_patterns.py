############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""RTL pattern library for consistent test module generation.

This module provides pre-defined RTL patterns commonly used in tests,
reducing duplication and ensuring consistency across the test suite.
"""

from typing import List, Dict, Optional, Tuple
from .rtl_builder import RTLBuilder, StrictRTLBuilder


class RTLPatterns:
    """Library of common RTL patterns for testing."""
    
    @staticmethod
    def minimal_axi_stream(name: str = "minimal_axi") -> str:
        """Create minimal AXI-Stream module with single input/output."""
        return (StrictRTLBuilder()
                .module(name)
                .add_stream_input("s_axis", bdim_value="32", sdim_value="512")
                .add_stream_output("m_axis", bdim_value="32")
                .parameter("DATA_WIDTH", "32")
                .assign("m_axis_tdata", "s_axis_tdata")
                .assign("m_axis_tvalid", "s_axis_tvalid")
                .assign("s_axis_tready", "m_axis_tready")
                .build())
    
    @staticmethod
    def multi_interface(name: str = "multi_interface",
                       num_inputs: int = 2,
                       num_outputs: int = 1,
                       num_weights: int = 1) -> str:
        """Create module with multiple interfaces of each type."""
        builder = StrictRTLBuilder().module(name)
        
        # Add inputs
        for i in range(num_inputs):
            iname = f"s_axis_in{i}"
            builder.add_stream_input(iname, 
                                   bdim_value=str(16 * (i + 1)),
                                   sdim_value=str(256 * (i + 1)))
        
        # Add weights
        for i in range(num_weights):
            wname = f"s_axis_weight{i}"
            builder.add_stream_weight(wname,
                                    bdim_value=str(32 * (i + 1)),
                                    sdim_value="512")
        
        # Add outputs
        for i in range(num_outputs):
            oname = f"m_axis_out{i}"
            builder.add_stream_output(oname,
                                    bdim_value=str(64 * (i + 1)))
        
        # Simple passthrough logic
        if num_inputs > 0 and num_outputs > 0:
            builder.assign("m_axis_out0_tdata", "s_axis_in0_tdata")
            builder.assign("m_axis_out0_tvalid", "s_axis_in0_tvalid")
            builder.assign("s_axis_in0_tready", "m_axis_out0_tready")
        
        return builder.build()
    
    @staticmethod
    def pragma_test_module(name: str = "pragma_test",
                          pragma_types: List[str] = None) -> str:
        """Create module with specific pragma combinations."""
        if pragma_types is None:
            pragma_types = ["datatype", "bdim", "alias"]
        
        builder = StrictRTLBuilder().module(name)
        
        # Add parameters based on pragma types
        if "alias" in pragma_types:
            builder.parameter("PE", "8")
            builder.pragma("ALIAS", "PE", "ProcessingElements")
        
        if "derived" in pragma_types:
            builder.parameter("WIDTH", "32")
            builder.parameter("DEPTH", "16")
            builder.pragma("DERIVED_PARAMETER", "TOTAL_SIZE", "WIDTH * DEPTH")
        
        # Add interface
        builder.add_stream_input("s_axis_input", bdim_value="32", sdim_value="512")
        builder.add_stream_output("m_axis_output", bdim_value="32")
        
        # Add interface pragmas
        if "datatype" in pragma_types:
            builder.pragma("DATATYPE", "s_axis_input", "UINT", "8", "32")
        
        if "bdim" in pragma_types:
            builder.pragma("BDIM", "s_axis_input", "[16, 16, 3]")
        
        if "weight" in pragma_types:
            builder.add_stream_weight("s_axis_weights", bdim_value="64")
        
        # Simple logic
        builder.assign("m_axis_output_tdata", "s_axis_input_tdata")
        builder.assign("m_axis_output_tvalid", "s_axis_input_tvalid")
        builder.assign("s_axis_input_tready", "m_axis_output_tready")
        
        return builder.build()
    
    @staticmethod
    def conv2d_kernel(input_shape: Tuple[int, int] = (28, 28),
                     weight_shape: Tuple[int, int] = (3, 3),
                     output_shape: Tuple[int, int] = (26, 26)) -> str:
        """Create 2D convolution kernel with proper parameters."""
        builder = StrictRTLBuilder().module("conv2d_kernel")
        
        # Shape parameters
        builder.parameter("INPUT_HEIGHT", str(input_shape[0]))
        builder.parameter("INPUT_WIDTH", str(input_shape[1]))
        builder.parameter("WEIGHT_HEIGHT", str(weight_shape[0]))
        builder.parameter("WEIGHT_WIDTH", str(weight_shape[1]))
        builder.parameter("OUTPUT_HEIGHT", str(output_shape[0]))
        builder.parameter("OUTPUT_WIDTH", str(output_shape[1]))
        
        # Data parameters
        builder.parameter("INPUT_WIDTH_BITS", "16")
        builder.parameter("WEIGHT_WIDTH_BITS", "8")
        builder.parameter("OUTPUT_WIDTH_BITS", "32")
        
        # Add interfaces with shape-based BDIM/SDIM
        builder.add_stream_input("s_axis_input",
                               data_width="INPUT_WIDTH_BITS",
                               bdim_param="INPUT_BDIM",
                               bdim_value=f"{input_shape[0] * input_shape[1]}",
                               sdim_param="INPUT_SDIM",
                               sdim_value="1")
        
        builder.add_stream_weight("s_axis_weights",
                                data_width="WEIGHT_WIDTH_BITS",
                                bdim_param="WEIGHT_BDIM",
                                bdim_value=f"{weight_shape[0] * weight_shape[1]}",
                                sdim_param="WEIGHT_SDIM",
                                sdim_value="32")
        
        builder.add_stream_output("m_axis_output",
                                data_width="OUTPUT_WIDTH_BITS",
                                bdim_param="OUTPUT_BDIM",
                                bdim_value=f"{output_shape[0] * output_shape[1]}")
        
        # Add pragmas
        builder.pragma("DATATYPE", "s_axis_input", "UINT", "8", "32")
        builder.pragma("DATATYPE", "s_axis_weights", "INT", "8", "8")
        builder.pragma("DATATYPE", "m_axis_output", "UINT", "16", "64")
        
        # Simple convolution logic placeholder
        builder.body("// Convolution logic here")
        builder.assign("m_axis_output_tdata", "32'h0")
        builder.assign("m_axis_output_tvalid", "1'b0")
        builder.assign("s_axis_input_tready", "1'b1")
        builder.assign("s_axis_weights_tready", "1'b1")
        
        return builder.build()
    
    @staticmethod
    def hierarchical_module(num_submodules: int = 3) -> str:
        """Create module with hierarchy and TOP_MODULE pragma."""
        lines = []
        
        # Add TOP_MODULE pragma
        lines.append("// @brainsmith TOP_MODULE top_level")
        lines.append("")
        
        # Top module
        builder = StrictRTLBuilder()
        lines.append(builder.module("top_level")
                    .add_stream_input("s_axis_in", bdim_value="32", sdim_value="512")
                    .add_stream_output("m_axis_out", bdim_value="32")
                    .body("// Instantiate submodules")
                    .assign("m_axis_out_tdata", "s_axis_in_tdata")
                    .assign("m_axis_out_tvalid", "s_axis_in_tvalid")
                    .assign("s_axis_in_tready", "m_axis_out_tready")
                    .build())
        
        # Add submodules
        for i in range(num_submodules):
            lines.append("")
            lines.append(f"module submodule_{i} (")
            lines.append("    input wire clk,")
            lines.append("    input wire [31:0] data")
            lines.append(");")
            lines.append("endmodule")
        
        return "\n".join(lines)
    
    @staticmethod
    def error_case(error_type: str = "missing_control") -> str:
        """Create modules with specific errors for validation testing."""
        if error_type == "missing_control":
            # Missing ap_clk/ap_rst_n
            return (RTLBuilder()
                    .module("missing_control")
                    .port("clk", "input")  # Wrong name
                    .port("rst", "input")  # Wrong name
                    .axi_stream_slave("s_axis_data", "32")
                    .axi_stream_master("m_axis_data", "32")
                    .build())
        
        elif error_type == "missing_interfaces":
            # No AXI interfaces
            return (RTLBuilder()
                    .module("no_interfaces")
                    .port("ap_clk", "input")
                    .port("ap_rst_n", "input")
                    .port("data_in", "input", "31:0")
                    .port("data_out", "output", "31:0")
                    .build())
        
        elif error_type == "missing_bdim":
            # AXI interface without BDIM parameter
            return (RTLBuilder()
                    .module("missing_bdim")
                    .add_global_control()
                    .parameter("s_axis_input_SDIM", "512")  # Has SDIM but no BDIM
                    .axi_stream_slave("s_axis_input", "32")
                    .axi_stream_master("m_axis_output", "32")
                    .parameter("m_axis_output_BDIM", "32")  # Output has BDIM
                    .build())
        
        else:
            raise ValueError(f"Unknown error type: {error_type}")
    
    @staticmethod
    def parameter_test_module(param_patterns: List[str] = None) -> str:
        """Create module with various parameter patterns."""
        if param_patterns is None:
            param_patterns = ["standard", "indexed", "derived"]
        
        builder = StrictRTLBuilder().module("param_test")
        
        if "standard" in param_patterns:
            builder.parameter("WIDTH", "32")
            builder.parameter("DEPTH", "16")
        
        if "indexed" in param_patterns:
            builder.parameter("in0_BDIM0", "16")
            builder.parameter("in0_BDIM1", "16")
            builder.parameter("in0_BDIM2", "3")
        
        if "derived" in param_patterns:
            builder.parameter("BASE_WIDTH", "8")
            builder.pragma("DERIVED_PARAMETER", "DOUBLE_WIDTH", "BASE_WIDTH * 2")
            builder.pragma("DERIVED_PARAMETER", "QUAD_WIDTH", "BASE_WIDTH * 4")
        
        if "alias" in param_patterns:
            builder.parameter("PE", "4")
            builder.pragma("ALIAS", "PE", "ProcessingEngines")
        
        # Add basic interfaces
        builder.add_stream_input("s_axis_in0", bdim_value="32", sdim_value="512")
        builder.add_stream_output("m_axis_out0", bdim_value="32")
        
        return builder.build()
    
    @staticmethod
    def axi_lite_control_module() -> str:
        """Create module with AXI-Lite control interface."""
        return (RTLBuilder()
                .module("axi_lite_control")
                .add_global_control()
                .parameter("C_S_AXI_ADDR_WIDTH", "6")
                .parameter("C_S_AXI_DATA_WIDTH", "32")
                .parameter("DATA_WIDTH", "32")
                .parameter("s_axis_data_BDIM", "64")
                .parameter("s_axis_data_SDIM", "1024")
                .parameter("m_axis_result_BDIM", "64")
                .axi_lite_slave("s_axi_control", "C_S_AXI_ADDR_WIDTH", "C_S_AXI_DATA_WIDTH")
                .axi_stream_slave("s_axis_data", "DATA_WIDTH")
                .axi_stream_master("m_axis_result", "DATA_WIDTH")
                .body("// Control logic")
                .body("reg [31:0] control_reg;")
                .body("always @(posedge ap_clk) begin")
                .body("    if (!ap_rst_n) control_reg <= 32'h0;")
                .body("end")
                .assign("m_axis_result_tdata", "s_axis_data_tdata")
                .assign("m_axis_result_tvalid", "s_axis_data_tvalid & control_reg[0]")
                .assign("s_axis_data_tready", "m_axis_result_tready")
                .build())