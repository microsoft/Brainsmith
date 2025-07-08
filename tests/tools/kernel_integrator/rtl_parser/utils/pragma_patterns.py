############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""RTL patterns for pragma-specific testing.

This module provides pre-defined RTL patterns focused on pragma testing,
including single-pragma isolation, pragma combinations, conflicts, and
edge cases.
"""

from typing import List, Dict, Optional, Tuple
from .rtl_builder import RTLBuilder, StrictRTLBuilder


class PragmaPatterns:
    """RTL patterns for pragma-specific testing."""
    
    @staticmethod
    def single_pragma_module(pragma_type: str, *args, 
                           module_name: str = "pragma_test") -> str:
        """Create module with single pragma for isolated testing.
        
        Args:
            pragma_type: Type of pragma (e.g., "BDIM", "DATATYPE")
            args: Pragma arguments
            module_name: Module name
            
        Returns:
            RTL string with single pragma
        """
        return (StrictRTLBuilder()
                .module(module_name)
                .pragma(pragma_type, *args)
                .parameter("DATA_WIDTH", "32")
                .add_stream_input("s_axis_input", bdim_value="32", sdim_value="512")
                .add_stream_output("m_axis_output", bdim_value="32")
                .assign("m_axis_output_tdata", "s_axis_input_tdata")
                .assign("m_axis_output_tvalid", "s_axis_input_tvalid")
                .assign("s_axis_input_tready", "m_axis_output_tready")
                .build())
    
    @staticmethod
    def multi_pragma_cascade(interface_name: str = "s_axis_data") -> str:
        """Create module with multiple pragmas on same interface.
        
        Tests pragma application order and interaction.
        """
        return (StrictRTLBuilder()
                .module("pragma_cascade")
                .parameter("TILE_H", "16")
                .parameter("TILE_W", "16")
                .parameter("CHANNELS", "3")
                .parameter("IMG_H", "224")
                .parameter("IMG_W", "224")
                .pragma("DATATYPE", interface_name, "UINT", "8", "32")
                .pragma("BDIM", interface_name, "[TILE_H, TILE_W, CHANNELS]")
                .pragma("SDIM", interface_name, "[IMG_H, IMG_W, CHANNELS]")
                .add_stream_input(interface_name, bdim_value="768", sdim_value="150528")
                .add_stream_output("m_axis_result", bdim_value="1000")
                .build())
    
    @staticmethod
    def pragma_conflict_test() -> str:
        """Create module with conflicting pragmas.
        
        Tests pragma validation and conflict resolution.
        """
        return (StrictRTLBuilder()
                .module("pragma_conflict")
                # Conflicting datatype pragmas
                .pragma("DATATYPE", "s_axis_input", "UINT", "8", "32")
                .pragma("DATATYPE", "s_axis_input", "INT", "16", "16")
                # Conflicting BDIM
                .pragma("BDIM", "s_axis_input", "PARAM_A")
                .pragma("BDIM", "s_axis_input", "PARAM_B")
                .parameter("PARAM_A", "32")
                .parameter("PARAM_B", "64")
                .add_stream_input("s_axis_input", bdim_value="32", sdim_value="512")
                .add_stream_output("m_axis_output", bdim_value="32")
                .build())
    
    @staticmethod
    def pragma_with_invalid_target() -> str:
        """Create module with pragma targeting non-existent interface."""
        return (StrictRTLBuilder()
                .module("invalid_target")
                # Pragmas for non-existent interfaces
                .pragma("BDIM", "nonexistent_interface", "SOME_PARAM")
                .pragma("DATATYPE", "also_missing", "UINT", "8", "8")
                .pragma("WEIGHT", "phantom_weights")
                # Real interfaces
                .add_stream_input("s_axis_real", bdim_value="32", sdim_value="512")
                .add_stream_output("m_axis_real", bdim_value="32")
                .build())
    
    @staticmethod
    def pragma_ordering_test() -> str:
        """Test pragma application order effects."""
        return (StrictRTLBuilder()
                .module("pragma_order")
                # Parameters that pragmas will reference
                .parameter("BASE_WIDTH", "8")
                .parameter("SCALE", "4")
                # Derived parameter should come after base params
                .pragma("DERIVED_PARAMETER", "TOTAL_WIDTH", "BASE_WIDTH * SCALE")
                # Alias should come after parameter definition
                .pragma("ALIAS", "SCALE", "ScalingFactor")
                # Interface pragmas
                .pragma("DATATYPE", "s_axis_input", "UINT", "BASE_WIDTH", "TOTAL_WIDTH")
                .add_stream_input("s_axis_input", bdim_value="32", sdim_value="512")
                .add_stream_output("m_axis_output", bdim_value="32")
                .build())
    
    @staticmethod
    def pragma_with_expressions() -> str:
        """Module with complex pragma arguments including expressions."""
        return (StrictRTLBuilder()
                .module("pragma_expressions")
                .parameter("TILE_SIZE", "16")
                .parameter("CHANNELS", "3")
                .parameter("BATCH", "8")
                .parameter("STREAM_H", "14")
                .parameter("STREAM_W", "14")
                # Complex BDIM expression
                .pragma("BDIM", "s_axis_input", "[BATCH, TILE_SIZE, TILE_SIZE, CHANNELS]")
                # Complex SDIM expression
                .pragma("SDIM", "s_axis_input", "[1, STREAM_H, STREAM_W, 1]")
                # Derived with expression
                .pragma("DERIVED_PARAMETER", "TOTAL_ELEMENTS", 
                       "BATCH * TILE_SIZE * TILE_SIZE * CHANNELS")
                .add_stream_input("s_axis_input", 
                                bdim_value="6144",  # 8*16*16*3
                                sdim_value="196")   # 14*14
                .add_stream_output("m_axis_output", bdim_value="1024")
                .build())
    
    @staticmethod
    def invalid_pragma_syntax() -> str:
        """Module with malformed pragma syntax for error testing."""
        # Use base RTLBuilder to inject malformed pragmas
        return (RTLBuilder()
                .module("invalid_syntax")
                .add_global_control()
                .comment("@brainsmith BDIM s_axis_input [32 32]  // Missing comma")
                .comment("@brainsmith DATATYPE s_axis_input UINT  // Missing args")
                .comment("@brainsmith WEIGHT  // Missing interface name")
                .comment("@brainsmith UNKNOWN_PRAGMA foo bar  // Unknown pragma")
                .axi_stream_slave("s_axis_input", "32")
                .axi_stream_master("m_axis_output", "32")
                .build())
    
    @staticmethod
    def datatype_param_cascade() -> str:
        """Test DATATYPE_PARAM pragma combinations."""
        return (StrictRTLBuilder()
                .module("datatype_params")
                .parameter("ACC_WIDTH", "48")
                .parameter("ACC_SIGNED", "1")
                .parameter("WEIGHT_WIDTH", "8")
                .parameter("WEIGHT_SIGNED", "1")
                # Multiple params for same datatype
                .pragma("DATATYPE_PARAM", "accumulator", "width", "ACC_WIDTH")
                .pragma("DATATYPE_PARAM", "accumulator", "signed", "ACC_SIGNED")
                .pragma("DATATYPE_PARAM", "accumulator", "round_mode", "truncate")
                # Different internal datatype
                .pragma("DATATYPE_PARAM", "weight_buffer", "width", "WEIGHT_WIDTH")
                .pragma("DATATYPE_PARAM", "weight_buffer", "signed", "WEIGHT_SIGNED")
                .add_stream_input("s_axis_input", bdim_value="32", sdim_value="512")
                .add_stream_output("m_axis_output", bdim_value="32")
                .build())
    
    @staticmethod
    def relationship_pragma_test() -> str:
        """Test RELATIONSHIP pragma variations."""
        return (StrictRTLBuilder()
                .module("relationships")
                # Various relationship types
                .pragma("RELATIONSHIP", "s_axis_a", "s_axis_b", "EQUAL")
                .pragma("RELATIONSHIP", "s_axis_a", "m_axis_out", "DEPENDENT", 
                       "0", "0", "scaled", "2")
                .pragma("RELATIONSHIP", "s_axis_b", "m_axis_out", "DEPENDENT",
                       "1", "0", "copy")
                .add_stream_input("s_axis_a", bdim_value="32", sdim_value="512")
                .add_stream_input("s_axis_b", bdim_value="32", sdim_value="512")
                .add_stream_output("m_axis_out", bdim_value="64")
                .build())
    
    @staticmethod
    def weight_pragma_variations() -> str:
        """Test WEIGHT pragma with different scenarios."""
        return (StrictRTLBuilder()
                .module("weight_pragmas")
                # Weight pragma before interface
                .pragma("WEIGHT", "s_axis_weights")
                .add_stream_input("s_axis_data", bdim_value="32", sdim_value="512")
                .add_stream_input("s_axis_weights", bdim_value="64", sdim_value="1024")
                # Weight pragma after interface  
                .pragma("WEIGHT", "s_axis_bias")
                .add_stream_input("s_axis_bias", bdim_value="32", sdim_value="1")
                .add_stream_output("m_axis_output", bdim_value="32")
                .build())
    
    @staticmethod
    def alias_and_derived_chain() -> str:
        """Test ALIAS and DERIVED_PARAMETER interaction."""
        return (StrictRTLBuilder()
                .module("alias_derived")
                .parameter("A", "8")
                .parameter("B", "4")
                .parameter("C", "2")
                # Create alias chain
                .pragma("ALIAS", "A", "BaseWidth")
                .pragma("ALIAS", "B", "ScaleFactor")
                # Derived using original names
                .pragma("DERIVED_PARAMETER", "D", "A * B")
                # Derived using aliases (may not work)
                .pragma("DERIVED_PARAMETER", "E", "BaseWidth * ScaleFactor")
                # Derived using other derived
                .pragma("DERIVED_PARAMETER", "F", "D * C")
                .add_stream_input("s_axis_input", bdim_value="32", sdim_value="512")
                .add_stream_output("m_axis_output", bdim_value="64")
                .build())
    
    @staticmethod
    def pragma_with_special_chars() -> str:
        """Test pragma parsing with special characters."""
        return (RTLBuilder()
                .module("special_chars")
                .add_global_control()
                # Pragmas with special characters
                .comment("@brainsmith BDIM s_axis_input [A+B, C*D, E/F]")
                .comment("@brainsmith ALIAS my_param \"User Friendly Name\"")
                .comment("@brainsmith DATATYPE s_axis_input UINT 8:0 31:0")
                .parameter("A", "4")
                .parameter("B", "4") 
                .parameter("C", "8")
                .parameter("D", "2")
                .parameter("E", "16")
                .parameter("F", "2")
                .parameter("my_param", "42")
                .axi_stream_slave("s_axis_input", "32")
                .axi_stream_master("m_axis_output", "32")
                .build())
    
    @staticmethod
    def all_pragma_types() -> str:
        """Module demonstrating all pragma types."""
        return (StrictRTLBuilder()
                .module("all_pragmas")
                .pragma("TOP_MODULE", "all_pragmas")
                # Parameters
                .parameter("WIDTH", "32")
                .parameter("DEPTH", "16")
                .parameter("PE", "4")
                # Parameter pragmas
                .pragma("ALIAS", "PE", "ProcessingElements")
                .pragma("DERIVED_PARAMETER", "TOTAL_SIZE", "WIDTH * DEPTH")
                # Interfaces
                .add_stream_input("s_axis_input", bdim_value="512", sdim_value="1024")
                .add_stream_weight("s_axis_weights", bdim_value="64", sdim_value="512")
                .add_stream_output("m_axis_output", bdim_value="512")
                # Interface pragmas
                .pragma("DATATYPE", "s_axis_input", "UINT", "8", "32")
                .pragma("DATATYPE", "s_axis_weights", "INT", "8", "8")
                .pragma("BDIM", "s_axis_input", "[16, 32]")
                .pragma("SDIM", "s_axis_input", "[32, 32]")
                # Internal datatypes
                .pragma("DATATYPE_PARAM", "accumulator", "width", "48")
                .pragma("DATATYPE_PARAM", "accumulator", "signed", "1")
                # Relationships
                .pragma("RELATIONSHIP", "s_axis_input", "m_axis_output", "EQUAL")
                .build())
    
    @staticmethod
    def pragma_error_cases(error_type: str) -> str:
        """Generate specific pragma error cases for testing.
        
        Args:
            error_type: Type of error to generate
                - "missing_args": Pragma missing required arguments
                - "invalid_interface": Pragma targeting wrong interface
                - "type_mismatch": Type conflicts in pragmas
                - "circular_dep": Circular dependency in derived params
        """
        if error_type == "missing_args":
            return (RTLBuilder()
                    .module("missing_args")
                    .add_global_control()
                    .comment("@brainsmith DATATYPE s_axis_input")  # Missing type and widths
                    .comment("@brainsmith BDIM")  # Missing interface and param
                    .comment("@brainsmith RELATIONSHIP s_axis_input")  # Missing target
                    .axi_stream_slave("s_axis_input", "32")
                    .axi_stream_master("m_axis_output", "32")
                    .build())
        
        elif error_type == "invalid_interface":
            return (StrictRTLBuilder()
                    .module("invalid_interface")
                    .pragma("DATATYPE", "m_axis_input", "UINT", "8", "32")  # Wrong prefix
                    .pragma("BDIM", "axis_data", "32")  # Missing s_/m_ prefix
                    .pragma("WEIGHT", "s_axis_output")  # Output can't be weight
                    .add_stream_input("s_axis_input", bdim_value="32", sdim_value="512")
                    .add_stream_output("m_axis_output", bdim_value="32")
                    .build())
        
        elif error_type == "type_mismatch":
            return (StrictRTLBuilder()
                    .module("type_mismatch")
                    .parameter("WIDTH", "32")
                    .pragma("DATATYPE", "s_axis_input", "UINT", "WIDTH", "16")  # Max < min
                    .pragma("BDIM", "s_axis_input", "not_a_number")  # Non-numeric
                    .add_stream_input("s_axis_input", bdim_value="32", sdim_value="512")
                    .add_stream_output("m_axis_output", bdim_value="32")
                    .build())
        
        elif error_type == "circular_dep":
            return (StrictRTLBuilder()
                    .module("circular_dep")
                    .parameter("A", "B + 1")
                    .parameter("B", "C * 2")
                    .parameter("C", "A - 1")
                    .pragma("DERIVED_PARAMETER", "D", "A + B + C")
                    .add_stream_input("s_axis_input", bdim_value="32", sdim_value="512")
                    .add_stream_output("m_axis_output", bdim_value="32")
                    .build())
        
        else:
            raise ValueError(f"Unknown error type: {error_type}")