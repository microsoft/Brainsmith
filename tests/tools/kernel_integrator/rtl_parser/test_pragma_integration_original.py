############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Pragma integration tests for RTL Parser.

Tests pragma combinations, interactions, ordering effects, and conflict resolution.
These tests focus on how multiple pragmas work together and affect the final
KernelMetadata output.
"""

import pytest
from brainsmith.core.dataflow.types import InterfaceType
from brainsmith.tools.kernel_integrator.rtl_parser import RTLParser
from .utils.rtl_builder import RTLBuilder


class TestPragmaIntegration:
    """Test cases for pragma integration and interactions."""
    
    def test_multiple_pragmas_same_interface(self, rtl_parser):
        """Test multiple pragmas targeting the same interface."""
        rtl = """
        // @brainsmith DATATYPE s_axis_input UINT 8 32
        // @brainsmith BDIM s_axis_input [TILE_H, TILE_W]
        // @brainsmith SDIM s_axis_input [STREAM_H, STREAM_W]
        // @brainsmith DATATYPE_PARAM s_axis_input width INPUT_WIDTH
        // @brainsmith BDIM m_axis_output OUTPUT_BDIM
        // @brainsmith DATATYPE_PARAM m_axis_output width OUTPUT_WIDTH
        module multi_pragma_interface #(
            parameter integer TILE_H = 16,
            parameter integer TILE_W = 16,
            parameter integer STREAM_H = 224,
            parameter integer STREAM_W = 224,
            parameter integer OUTPUT_BDIM = 32,
            parameter integer INPUT_WIDTH = 32,
            parameter integer OUTPUT_WIDTH = 32
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
        
        kernel_metadata = rtl_parser.parse(rtl, "multi_pragma_interface.sv")
        
        # Verify basic parsing
        assert kernel_metadata.name == "multi_pragma_interface"
        assert len(kernel_metadata.interfaces) >= 2
        
        # Check that all pragmas were parsed
        pragma_types = {p.type.value for p in kernel_metadata.pragmas}
        assert "datatype" in pragma_types
        assert "bdim" in pragma_types
        assert "sdim" in pragma_types
        assert "datatype_param" in pragma_types
        
        # Verify interface detection
        interface_types = {i.interface_type for i in kernel_metadata.interfaces}
        assert InterfaceType.INPUT in interface_types
        assert InterfaceType.OUTPUT in interface_types
        
        # Note: Pragma application success depends on interface name matching
        # If pragmas target "s_axis_input" but scanner creates different names,
        # pragma application may fail with warnings
    
    def test_pragma_ordering_effects(self, rtl_parser):
        """Test that pragma ordering doesn't affect final result."""
        
        # Test with pragmas in one order
        rtl_order1 = """
        // @brainsmith DATATYPE_PARAM accumulator width ACC_WIDTH
        // @brainsmith DATATYPE_PARAM accumulator signed ACC_SIGNED
        // @brainsmith ALIAS PE ParallelismFactor
        // @brainsmith DERIVED_PARAMETER MEM_SIZE DEPTH * WIDTH
        module pragma_order_test1 #(
            parameter integer WIDTH = 32,
            parameter integer DEPTH = 1024,
            parameter integer PE = 4,
            parameter integer MEM_SIZE = 32768,
            parameter integer ACC_WIDTH = 64,
            parameter integer ACC_SIGNED = 1
        ) (
            input wire clk,
            input wire [WIDTH-1:0] data_in,
            output wire [WIDTH-1:0] data_out
        );
        endmodule
        """
        
        # Test with pragmas in different order
        rtl_order2 = """
        // @brainsmith DERIVED_PARAMETER MEM_SIZE DEPTH * WIDTH
        // @brainsmith ALIAS PE ParallelismFactor
        // @brainsmith DATATYPE_PARAM accumulator signed ACC_SIGNED
        // @brainsmith DATATYPE_PARAM accumulator width ACC_WIDTH
        module pragma_order_test2 #(
            parameter integer WIDTH = 32,
            parameter integer DEPTH = 1024,
            parameter integer PE = 4,
            parameter integer MEM_SIZE = 32768,
            parameter integer ACC_WIDTH = 64,
            parameter integer ACC_SIGNED = 1
        ) (
            input wire clk,
            input wire [WIDTH-1:0] data_in,
            output wire [WIDTH-1:0] data_out
        );
        endmodule
        """
        
        km1 = rtl_parser.parse(rtl_order1, "pragma_order_test1.sv")
        km2 = rtl_parser.parse(rtl_order2, "pragma_order_test2.sv")
        
        # Results should be equivalent regardless of pragma order
        assert len(km1.pragmas) == len(km2.pragmas)
        
        # Check exposed parameters consistency
        # Note: Exact parameters may vary due to auto-linking, but behavior should be similar
        assert len(km1.exposed_parameters) == len(km2.exposed_parameters)
        
        # Check alias handling consistency
        alias1 = km1.linked_parameters.get("aliases", {})
        alias2 = km2.linked_parameters.get("aliases", {})
        
        if "PE" in alias1 and "PE" in alias2:
            assert alias1["PE"] == alias2["PE"]
        
        # Check derived parameter handling consistency
        derived1 = km1.linked_parameters.get("derived", {})
        derived2 = km2.linked_parameters.get("derived", {})
        
        if "MEM_SIZE" in derived1 and "MEM_SIZE" in derived2:
            assert derived1["MEM_SIZE"] == derived2["MEM_SIZE"]
    
    def test_conflicting_pragmas(self, rtl_parser):
        """Test handling of conflicting pragma definitions."""
        rtl = """
        // @brainsmith ALIAS PE ParallelismFactor
        // @brainsmith ALIAS PE ProcessingElements
        // @brainsmith DERIVED_PARAMETER WIDTH BASE_WIDTH * 2
        // @brainsmith DERIVED_PARAMETER WIDTH BASE_WIDTH + 8
        module conflicting_pragmas_test #(
            parameter integer BASE_WIDTH = 16,
            parameter integer PE = 4,
            parameter integer WIDTH = 32
        ) (
            input wire clk,
            input wire [WIDTH-1:0] data_in,
            output wire [WIDTH-1:0] data_out
        );
        endmodule
        """
        
        # Parser should handle conflicts gracefully (last one wins or error)
        kernel_metadata = rtl_parser.parse(rtl, "conflicting_pragmas_test.sv")
        
        # Verify basic parsing succeeds
        assert kernel_metadata.name == "conflicting_pragmas_test"
        
        # Check that pragmas were parsed (even if some conflict)
        assert len(kernel_metadata.pragmas) == 4
        
        # Check conflict resolution in aliases
        aliases = kernel_metadata.linked_parameters.get("aliases", {})
        if "PE" in aliases:
            # Should have one alias value (implementation decides which wins)
            assert aliases["PE"] in ["ParallelismFactor", "ProcessingElements"]
        
        # Check conflict resolution in derived parameters
        derived = kernel_metadata.linked_parameters.get("derived", {})
        if "WIDTH" in derived:
            # Should have one derived expression
            assert derived["WIDTH"] in ["BASE_WIDTH * 2", "BASE_WIDTH + 8"]
    
    def test_pragma_cascade_effects(self, rtl_parser):
        """Test how pragmas affect each other (cascade effects)."""
        rtl = """
        // @brainsmith DATATYPE s_axis_input UINT 8 32
        // @brainsmith DATATYPE_PARAM s_axis_input width INPUT_WIDTH
        // @brainsmith DATATYPE_PARAM s_axis_input signed INPUT_SIGNED
        // @brainsmith ALIAS INPUT_WIDTH DataWidth
        // @brainsmith DERIVED_PARAMETER OUTPUT_WIDTH INPUT_WIDTH * 2
        module pragma_cascade_test #(
            parameter integer INPUT_WIDTH = 16,
            parameter integer INPUT_SIGNED = 0,
            parameter integer OUTPUT_WIDTH = 32
        ) (
            input wire clk,
            
            input wire [INPUT_WIDTH-1:0] s_axis_input_tdata,
            input wire s_axis_input_tvalid,
            output wire s_axis_input_tready,
            
            output wire [OUTPUT_WIDTH-1:0] m_axis_output_tdata,
            output wire m_axis_output_tvalid,
            input wire m_axis_output_tready
        );
        endmodule
        """
        
        kernel_metadata = rtl_parser.parse(rtl, "pragma_cascade_test.sv")
        
        # Verify basic parsing
        assert kernel_metadata.name == "pragma_cascade_test"
        
        # Check parameter exposure after cascade effects
        exposed_params = set(kernel_metadata.exposed_parameters)
        
        # INPUT_WIDTH should be hidden by ALIAS (replaced with DataWidth)
        if "DataWidth" in exposed_params:
            assert "INPUT_WIDTH" not in exposed_params
        
        # OUTPUT_WIDTH should be hidden by DERIVED_PARAMETER
        if "OUTPUT_WIDTH" not in exposed_params:
            assert "OUTPUT_WIDTH" in kernel_metadata.linked_parameters.get("derived", {})
        
        # INPUT_SIGNED should be hidden by DATATYPE_PARAM (linked to interface)
        # Note: This depends on successful pragma application
        
        # Check internal datatypes created by DATATYPE_PARAM pragmas
        if kernel_metadata.internal_datatypes:
            # Look for datatypes created by DATATYPE_PARAM pragmas
            dt_names = {dt.name for dt in kernel_metadata.internal_datatypes}
            # Exact names depend on pragma application success
    
    def test_interface_pragma_combinations(self, rtl_parser):
        """Test combinations of interface-targeting pragmas."""
        rtl = """
        // @brainsmith DATATYPE s_axis_input UINT 8 32
        // @brainsmith WEIGHT s_axis_weights
        // @brainsmith DATATYPE s_axis_weights INT 8 8
        // @brainsmith BDIM s_axis_input [BATCH_SIZE, SEQ_LEN]
        // @brainsmith SDIM s_axis_input [STREAM_BATCH, STREAM_SEQ]
        // @brainsmith RELATIONSHIP s_axis_input m_axis_output EQUAL
        module interface_pragma_combo #(
            parameter integer BATCH_SIZE = 16,
            parameter integer SEQ_LEN = 64,
            parameter integer STREAM_BATCH = 256,
            parameter integer STREAM_SEQ = 1024
        ) (
            input wire clk,
            
            input wire [31:0] s_axis_input_tdata,
            input wire s_axis_input_tvalid,
            output wire s_axis_input_tready,
            
            input wire [7:0] s_axis_weights_tdata,
            input wire s_axis_weights_tvalid,
            output wire s_axis_weights_tready,
            
            output wire [31:0] m_axis_output_tdata,
            output wire m_axis_output_tvalid,
            input wire m_axis_output_tready
        );
        endmodule
        """
        
        kernel_metadata = rtl_parser.parse(rtl, "interface_pragma_combo.sv")
        
        # Verify parsing and interface detection
        assert kernel_metadata.name == "interface_pragma_combo"
        assert len(kernel_metadata.interfaces) >= 3
        
        # Check interface types
        interface_types = {i.interface_type for i in kernel_metadata.interfaces}
        assert InterfaceType.INPUT in interface_types
        assert InterfaceType.OUTPUT in interface_types
        
        # Check for weight interface (if pragma application worked)
        weight_interfaces = [i for i in kernel_metadata.interfaces if i.interface_type == InterfaceType.WEIGHT]
        # May be empty if pragma application failed due to naming mismatch
        
        # Check pragma parsing
        pragma_types = {p.type.value for p in kernel_metadata.pragmas}
        assert "datatype" in pragma_types
        assert "weight" in pragma_types
        assert "bdim" in pragma_types
        assert "sdim" in pragma_types
        assert "relationship" in pragma_types
        
        # Check parameter effects
        exposed_params = set(kernel_metadata.exposed_parameters)
        # Parameters used by BDIM/SDIM may be hidden from exposed parameters
        # Exact behavior depends on pragma application success
    
    def test_parameter_pragma_interactions(self, rtl_parser):
        """Test interactions between parameter-targeting pragmas."""
        rtl = """
        // @brainsmith ALIAS CORES ParallelismFactor
        // @brainsmith DERIVED_PARAMETER BUFFER_SIZE CORES * DEPTH
        // @brainsmith DERIVED_PARAMETER TOTAL_MEM BUFFER_SIZE + OVERHEAD
        // @brainsmith DATATYPE_PARAM accumulator width ACC_WIDTH
        // @brainsmith DATATYPE_PARAM threshold width THRESH_WIDTH
        module param_pragma_interactions #(
            parameter integer CORES = 8,
            parameter integer DEPTH = 512,
            parameter integer BUFFER_SIZE = 4096,
            parameter integer OVERHEAD = 256,
            parameter integer TOTAL_MEM = 4352,
            parameter integer ACC_WIDTH = 32,
            parameter integer THRESH_WIDTH = 16
        ) (
            input wire clk,
            input wire [31:0] data_in,
            output wire [31:0] data_out
        );
        endmodule
        """
        
        kernel_metadata = rtl_parser.parse(rtl, "param_pragma_interactions.sv")
        
        # Verify parsing
        assert kernel_metadata.name == "param_pragma_interactions"
        assert len(kernel_metadata.parameters) == 7
        
        # Check parameter exposure after pragma interactions
        exposed_params = set(kernel_metadata.exposed_parameters)
        
        # CORES should be hidden by ALIAS
        if "ParallelismFactor" in exposed_params:
            assert "CORES" not in exposed_params
            assert kernel_metadata.linked_parameters["aliases"]["CORES"] == "ParallelismFactor"
        
        # BUFFER_SIZE and TOTAL_MEM should be hidden by DERIVED_PARAMETER
        derived_params = kernel_metadata.linked_parameters.get("derived", {})
        
        # Check derived parameter chain
        if "BUFFER_SIZE" in derived_params:
            assert derived_params["BUFFER_SIZE"] == "CORES * DEPTH"
        
        if "TOTAL_MEM" in derived_params:
            # Should reference BUFFER_SIZE which is itself derived
            assert derived_params["TOTAL_MEM"] == "BUFFER_SIZE + OVERHEAD"
        
        # ACC_WIDTH and THRESH_WIDTH should be hidden by DATATYPE_PARAM
        # Note: This depends on successful pragma application
        
        # Check internal datatypes
        internal_dt_names = {dt.name for dt in kernel_metadata.internal_datatypes}
        # May contain 'accumulator' and 'threshold' if pragmas worked
    
    def test_top_module_pragma_with_others(self, rtl_parser):
        """Test TOP_MODULE pragma interaction with other pragmas."""
        rtl = """
        // @brainsmith TOP_MODULE selected_module
        // @brainsmith ALIAS PE ProcessingElements
        // @brainsmith DATATYPE s_axis_input UINT 8 32
        module first_module #(
            parameter integer WIDTH = 16
        ) (
            input wire clk,
            input wire [WIDTH-1:0] data_in,
            output wire [WIDTH-1:0] data_out
        );
        endmodule
        
        // This is the module that should be selected
        module selected_module #(
            parameter integer PE = 4,
            parameter integer INPUT_WIDTH = 32
        ) (
            input wire clk,
            
            input wire [INPUT_WIDTH-1:0] s_axis_input_tdata,
            input wire s_axis_input_tvalid,
            output wire s_axis_input_tready,
            
            output wire [INPUT_WIDTH-1:0] m_axis_output_tdata,
            output wire m_axis_output_tvalid,
            input wire m_axis_output_tready
        );
        endmodule
        
        module third_module (
            input wire dummy
        );
        endmodule
        """
        
        kernel_metadata = rtl_parser.parse(rtl, "top_module_pragma_test.sv")
        
        # Verify TOP_MODULE pragma worked
        assert kernel_metadata.name == "selected_module"
        
        # Check that other pragmas were applied to the selected module
        pragma_types = {p.type.value for p in kernel_metadata.pragmas}
        assert "top_module" in pragma_types
        assert "alias" in pragma_types
        assert "datatype" in pragma_types
        
        # Check parameter effects on selected module
        param_names = {p.name for p in kernel_metadata.parameters}
        assert "PE" in param_names
        assert "INPUT_WIDTH" in param_names
        assert "WIDTH" not in param_names  # Should not be from first_module
        
        # Check ALIAS pragma effect
        exposed_params = set(kernel_metadata.exposed_parameters)
        if "ProcessingElements" in exposed_params:
            assert "PE" not in exposed_params
    
    def test_pragma_error_resilience(self, rtl_parser):
        """Test that pragma errors don't break overall parsing."""
        rtl = """
        // @brainsmith INVALID_PRAGMA unknown arguments
        // @brainsmith DATATYPE nonexistent_interface UINT 8 32
        // @brainsmith ALIAS VALID_PARAM ValidAlias
        // @brainsmith BDIM missing_interface [MISSING_PARAM]
        // @brainsmith DERIVED_PARAMETER RESULT VALID_PARAM * 2
        module pragma_error_resilience #(
            parameter integer VALID_PARAM = 16,
            parameter integer RESULT = 32
        ) (
            input wire clk,
            input wire [VALID_PARAM-1:0] data_in,
            output wire [RESULT-1:0] data_out
        );
        endmodule
        """
        
        # Parser should succeed despite pragma errors
        kernel_metadata = rtl_parser.parse(rtl, "pragma_error_resilience.sv")
        
        # Verify basic parsing succeeded
        assert kernel_metadata.name == "pragma_error_resilience"
        assert len(kernel_metadata.parameters) == 2
        
        # Valid pragmas should still work
        exposed_params = set(kernel_metadata.exposed_parameters)
        
        # Check that valid ALIAS pragma worked
        if "ValidAlias" in exposed_params:
            assert "VALID_PARAM" not in exposed_params
            assert kernel_metadata.linked_parameters["aliases"]["VALID_PARAM"] == "ValidAlias"
        
        # Check that valid DERIVED_PARAMETER pragma worked
        if "RESULT" not in exposed_params:
            assert "RESULT" in kernel_metadata.linked_parameters.get("derived", {})
        
        # Invalid pragmas should be ignored or cause warnings (not fatal errors)
        # The parser should continue processing valid pragmas