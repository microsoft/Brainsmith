############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Unit tests for the Pragma Handler component.

Tests pragma extraction and handling including:
- Pragma extraction from AST
- All pragma types (TOP_MODULE, DATATYPE, WEIGHT, BDIM, SDIM, etc.)
- Pragma validation and error handling
- Complex pragma input parsing
- Integration with kernel metadata
"""

import pytest
from pathlib import Path

from brainsmith.tools.hw_kernel_gen.rtl_parser.pragma import PragmaHandler
from brainsmith.tools.hw_kernel_gen.rtl_parser.rtl_data import PragmaType
from brainsmith.tools.hw_kernel_gen.rtl_parser.pragmas import (
    TopModulePragma, DatatypePragma, WeightPragma, BDimPragma, SDimPragma,
    AliasPragma, DerivedParameterPragma, AxiLiteParamPragma, DatatypeParamPragma,
    RelationshipPragma
)

from .utils.rtl_builder import RTLBuilder
from .utils.pragma_patterns import PragmaPatterns


class TestPragmaHandler:
    """Test cases for Pragma Handler functionality."""
    
    def test_extract_simple_pragmas(self, pragma_handler, ast_parser):
        """Test extracting simple pragmas from RTL."""
        rtl = (RTLBuilder()
               .pragma("TOP_MODULE", "my_module")
               .pragma("ALIAS", "WIDTH", "DataWidth")
               .module("my_module")
               .parameter("WIDTH", "32")
               .port("clk", "input")
               .build())
        
        tree = ast_parser.parse_source(rtl)
        pragmas = pragma_handler.extract_pragmas(tree.root_node)
        
        assert len(pragmas) == 2
        
        # Check TOP_MODULE pragma
        top_pragma = next(p for p in pragmas if p.type == PragmaType.TOP_MODULE)
        assert isinstance(top_pragma, TopModulePragma)
        assert top_pragma.parsed_data["module_name"] == "my_module"
        
        # Check ALIAS pragma
        alias_pragma = next(p for p in pragmas if p.type == PragmaType.ALIAS)
        assert isinstance(alias_pragma, AliasPragma)
        assert alias_pragma.parsed_data["rtl_param"] == "WIDTH"
        assert alias_pragma.parsed_data["nodeattr_name"] == "DataWidth"
    
    def test_extract_interface_pragmas(self, pragma_handler, ast_parser):
        """Test extracting interface-related pragmas."""
        # Build RTL with interface pragmas
        rtl = (RTLBuilder()
               .pragma("DATATYPE", "in0", "UINT", "8", "32")
               .pragma("BDIM", "in0", "[WIDTH]")
               .pragma("SDIM", "in0", "[DEPTH]")
               .pragma("WEIGHT", "weights")
               .module("test_module")
               .parameter("WIDTH", "32")
               .parameter("DEPTH", "16")
               .port("s_axis_in0_tdata", "input", "WIDTH")
               .port("s_axis_in0_tvalid", "input")
               .port("s_axis_in0_tready", "output")
               .port("s_axis_weights_tdata", "input", "8")
               .port("s_axis_weights_tvalid", "input")
               .port("s_axis_weights_tready", "output")
               .build())
        
        tree = ast_parser.parse_source(rtl)
        pragmas = pragma_handler.extract_pragmas(tree.root_node)
        
        assert len(pragmas) == 4
        
        # Check DATATYPE pragma
        datatype_pragma = next(p for p in pragmas if p.type == PragmaType.DATATYPE)
        assert isinstance(datatype_pragma, DatatypePragma)
        assert datatype_pragma.parsed_data["interface_name"] == "in0"
        assert datatype_pragma.parsed_data["base_types"] == ["UINT"]  # Now always a list
        assert datatype_pragma.parsed_data["min_width"] == 8
        assert datatype_pragma.parsed_data["max_width"] == 32
        
        # Check BDIM pragma
        bdim_pragma = next(p for p in pragmas if p.type == PragmaType.BDIM)
        assert isinstance(bdim_pragma, BDimPragma)
        assert bdim_pragma.parsed_data["interface_name"] == "in0"
        assert bdim_pragma.parsed_data["bdim_params"] == ["WIDTH"]
        
        # Check SDIM pragma
        sdim_pragma = next(p for p in pragmas if p.type == PragmaType.SDIM)
        assert isinstance(sdim_pragma, SDimPragma)
        assert sdim_pragma.parsed_data["interface_name"] == "in0"
        assert sdim_pragma.parsed_data["sdim_params"] == ["DEPTH"]
        
        # Check WEIGHT pragma
        weight_pragma = next(p for p in pragmas if p.type == PragmaType.WEIGHT)
        assert isinstance(weight_pragma, WeightPragma)
        assert weight_pragma.parsed_data["interface_names"] == ["weights"]
    
    def test_extract_parameter_pragmas(self, pragma_handler, ast_parser):
        """Test extracting parameter-related pragmas."""
        # Build RTL with parameter pragmas
        rtl = (RTLBuilder()
               .pragma("DERIVED_PARAMETER", "OUT_WIDTH", "WIDTH + 1")
               .pragma("AXILITE_PARAM", "s_axi_config", "threshold")
               .pragma("DATATYPE_PARAM", "internal", "signed", "SIGNED_WIDTH")
               .module("test")
               .parameter("WIDTH", "32")
               .parameter("SIGNED_WIDTH", "1")
               .parameter("threshold", "127")
               .port("clk", "input")
               .build())
        
        tree = ast_parser.parse_source(rtl)
        pragmas = pragma_handler.extract_pragmas(tree.root_node)
        
        assert len(pragmas) == 3
        
        # Check DERIVED_PARAMETER pragma
        derived_pragma = next(p for p in pragmas if p.type == PragmaType.DERIVED_PARAMETER)
        assert isinstance(derived_pragma, DerivedParameterPragma)
        assert derived_pragma.parsed_data["param_name"] == "OUT_WIDTH"
        assert derived_pragma.parsed_data["python_expression"] == "WIDTH + 1"
        
        # Check AXILITE_PARAM pragma
        axilite_pragma = next(p for p in pragmas if p.type == PragmaType.AXILITE_PARAM)
        assert isinstance(axilite_pragma, AxiLiteParamPragma)
        assert axilite_pragma.parsed_data["param_name"] == "threshold"
        assert axilite_pragma.parsed_data["interface_name"] == "s_axi_config"
        
        # Check DATATYPE_PARAM pragma
        datatype_param_pragma = next(p for p in pragmas if p.type == PragmaType.DATATYPE_PARAM)
        assert isinstance(datatype_param_pragma, DatatypeParamPragma)
        assert datatype_param_pragma.parsed_data["interface_name"] == "internal"
        assert datatype_param_pragma.parsed_data["property_type"] == "signed"
        assert datatype_param_pragma.parsed_data["parameter_name"] == "SIGNED_WIDTH"
    
    def test_extract_complex_pragma_inputs(self, pragma_handler, ast_parser):
        """Test extracting pragmas with complex inputs like lists."""
        # Use PragmaPatterns for complex pragma test
        rtl = (RTLBuilder()
               .pragma("BDIM", "output", "[B,H,W,C]")
               .pragma("DATATYPE", "output", "INT", "8", "32")
               .pragma("RELATIONSHIP", "input0", "output0", "MULTIPLE", "0", "0", "factor=8")
               .module("test")
               .port("s_axis_output_tdata", "input", "32")
               .build())
        
        tree = ast_parser.parse_source(rtl)
        pragmas = pragma_handler.extract_pragmas(tree.root_node)
        
        assert len(pragmas) == 3
        
        # Check BDIM
        bdim_pragma = next(p for p in pragmas if p.type == PragmaType.BDIM)
        assert bdim_pragma.parsed_data["bdim_params"] == ["B", "H", "W", "C"]
        
        # Check DATATYPE with multiple types
        datatype_pragma = next(p for p in pragmas if p.type == PragmaType.DATATYPE)
        assert datatype_pragma.parsed_data["base_types"] == ["INT"]  # Now always a list
        
        # Check RELATIONSHIP pragma
        relationship_pragma = next(p for p in pragmas if p.type == PragmaType.RELATIONSHIP)
        assert isinstance(relationship_pragma, RelationshipPragma)
        assert relationship_pragma.parsed_data["source_interface"] == "input0"
        assert relationship_pragma.parsed_data["target_interface"] == "output0"
        assert relationship_pragma.parsed_data["relationship_type"] == "MULTIPLE"
        assert relationship_pragma.parsed_data["source_dim"] == 0
        assert relationship_pragma.parsed_data["target_dim"] == 0
        assert relationship_pragma.parsed_data["scale_factor"] == 8
    
    def test_invalid_pragma_handling(self, pragma_handler, ast_parser):
        """Test handling of invalid pragmas."""
        # Use PragmaPatterns for invalid pragma cases
        rtl = PragmaPatterns.pragma_error_cases("missing_args")
        
        tree = ast_parser.parse_source(rtl)
        pragmas = pragma_handler.extract_pragmas(tree.root_node)
        
        # Only valid pragmas should be returned
        assert len(pragmas) == 0  # All pragmas in this test are invalid
    
    def test_get_pragmas_by_type(self, pragma_handler, ast_parser):
        """Test filtering pragmas by type."""
        # Build RTL with multiple pragma types
        rtl = (RTLBuilder()
               .pragma("TOP_MODULE", "main")
               .pragma("WEIGHT", "w1")
               .pragma("WEIGHT", "w2")
               .pragma("ALIAS", "PARAM1", "param_one")
               .pragma("ALIAS", "PARAM2", "param_two")
               .pragma("ALIAS", "PARAM3", "param_three")
               .module("main")
               .port("clk", "input")
               .build())
        
        tree = ast_parser.parse_source(rtl)
        pragmas = pragma_handler.extract_pragmas(tree.root_node)
        
        assert len(pragmas) == 6
        
        # Get specific pragma types
        weight_pragmas = pragma_handler.get_pragmas_by_type(PragmaType.WEIGHT)
        assert len(weight_pragmas) == 2
        assert all(isinstance(p, WeightPragma) for p in weight_pragmas)
        
        alias_pragmas = pragma_handler.get_pragmas_by_type(PragmaType.ALIAS)
        assert len(alias_pragmas) == 3
        assert all(isinstance(p, AliasPragma) for p in alias_pragmas)
        
        top_pragmas = pragma_handler.get_pragmas_by_type(PragmaType.TOP_MODULE)
        assert len(top_pragmas) == 1
    
    def test_pragma_line_numbers(self, pragma_handler, ast_parser):
        """Test that pragmas track correct line numbers."""
        # Build RTL with pragmas on specific lines
        rtl = (RTLBuilder()
               .pragma("TOP_MODULE", "test")
               .body("")  # Empty line
               .comment("Some other comment")
               .body("")  # Empty line
               .pragma("WEIGHT", "weights")
               .module("test")
               .port("clk", "input")
               .pragma("ALIAS", "WIDTH", "data_width", location="after_ports")
               .build())
        
        tree = ast_parser.parse_source(rtl)
        pragmas = pragma_handler.extract_pragmas(tree.root_node)
        
        assert len(pragmas) == 3
        
        # Sort by line number
        pragmas_sorted = sorted(pragmas, key=lambda p: p.line_number)
        
        assert pragmas_sorted[0].type == PragmaType.TOP_MODULE
        # Line numbers may vary with RTLBuilder generation
        assert pragmas_sorted[0].line_number >= 1
        
        assert pragmas_sorted[1].type == PragmaType.WEIGHT
        assert pragmas_sorted[1].line_number > pragmas_sorted[0].line_number
        
        assert pragmas_sorted[2].type == PragmaType.ALIAS
        assert pragmas_sorted[2].line_number > pragmas_sorted[1].line_number
    
    def test_collect_internal_datatype_pragmas(self, pragma_handler, ast_parser):
        """Test collecting internal datatype pragmas."""
        # Build RTL with valid DATATYPE_PARAM pragmas
        rtl = (RTLBuilder()
               .module("datatype_params")
               .parameter("ACC_WIDTH", "48")
               .parameter("ACC_SIGNED", "1")
               .parameter("THRESH_WIDTH", "32")
               .parameter("THRESH_SIGNED", "0")
               # Multiple params for same datatype
               .pragma("DATATYPE_PARAM", "accumulator", "width", "ACC_WIDTH")
               .pragma("DATATYPE_PARAM", "accumulator", "signed", "ACC_SIGNED")
               # Different internal datatype
               .pragma("DATATYPE_PARAM", "threshold", "width", "THRESH_WIDTH")
               .pragma("DATATYPE_PARAM", "threshold", "signed", "THRESH_SIGNED")
               .port("clk", "input")
               .port("s_axis_input_tdata", "input", "32")
               .port("s_axis_input_tvalid", "input")
               .port("s_axis_input_tready", "output")
               .port("m_axis_output_tdata", "output", "32")
               .port("m_axis_output_tvalid", "output")
               .port("m_axis_output_tready", "input")
               .build())
        
        tree = ast_parser.parse_source(rtl)
        pragmas = pragma_handler.extract_pragmas(tree.root_node)
        
        # Test collecting internal datatypes (not matching any interface)
        interface_names = ["s_axis_input", "m_axis_output"]  # Real interfaces to exclude
        internal_datatypes = pragma_handler.collect_internal_datatype_pragmas(interface_names)
        
        # Should get 2 internal datatypes (accumulator and threshold)
        assert len(internal_datatypes) == 2
        assert any(dt.name == "accumulator" for dt in internal_datatypes)
        assert any(dt.name == "threshold" for dt in internal_datatypes)
        
        # Verify the datatypes have the correct properties
        accumulator = next(dt for dt in internal_datatypes if dt.name == "accumulator")
        assert accumulator.width == "ACC_WIDTH"
        assert accumulator.signed == "ACC_SIGNED"
        
        threshold = next(dt for dt in internal_datatypes if dt.name == "threshold")
        assert threshold.width == "THRESH_WIDTH"
        assert threshold.signed == "THRESH_SIGNED"
    
    def test_pragma_with_brackets(self, pragma_handler, ast_parser):
        """Test pragmas containing bracket expressions."""
        # Build RTL with bracket expressions in pragmas
        rtl = (RTLBuilder()
               .pragma("BDIM", "output", "[B,H,W,C]")
               .pragma("RELATIONSHIP", "in0", "in1", "EQUAL")
               .module("test")
               .port("clk", "input")
               .build())
        
        tree = ast_parser.parse_source(rtl)
        pragmas = pragma_handler.extract_pragmas(tree.root_node)
        
        assert len(pragmas) == 2
        
        # Check BDIM
        bdim_pragma = next(p for p in pragmas if p.type == PragmaType.BDIM)
        assert bdim_pragma.parsed_data["bdim_params"] == ["B", "H", "W", "C"]
        
        # Check RELATIONSHIP
        rel_pragma = next(p for p in pragmas if p.type == PragmaType.RELATIONSHIP)
        assert rel_pragma.parsed_data["relationship_type"] == "EQUAL"
    
    def test_empty_pragma_extraction(self, pragma_handler, ast_parser):
        """Test extracting pragmas from RTL with no pragmas."""
        # Build RTL without any pragmas
        rtl = (RTLBuilder()
               .comment("Regular comment without pragma")
               .body("/* Block comment")
               .body("   without pragma */")
               .module("empty")
               .port("clk", "input")
               .port("done", "output")
               .comment("Another regular comment")
               .build())
        
        tree = ast_parser.parse_source(rtl)
        pragmas = pragma_handler.extract_pragmas(tree.root_node)
        
        assert len(pragmas) == 0
        assert pragma_handler.pragmas == []
    
    def test_datatype_pragma_with_list_types(self, pragma_handler, ast_parser):
        """Test DATATYPE pragma with list of types."""
        # Build RTL with DATATYPE pragmas using both single and list types
        rtl = (RTLBuilder()
               .pragma("DATATYPE", "in0", "[INT, UINT, FIXED]", "8", "16")
               .pragma("DATATYPE", "in1", "UINT", "8", "32")  # Single type
               .pragma("DATATYPE", "weights", "[FIXED, FLOAT]", "16", "32")
               .module("test_module")
               .port("s_axis_in0_tdata", "input", "16")
               .port("s_axis_in1_tdata", "input", "32") 
               .port("s_axis_weights_tdata", "input", "32")
               .build())
        
        tree = ast_parser.parse_source(rtl)
        pragmas = pragma_handler.extract_pragmas(tree.root_node)
        
        # Find DATATYPE pragmas
        datatype_pragmas = [p for p in pragmas if p.type == PragmaType.DATATYPE]
        assert len(datatype_pragmas) == 3
        
        # Check list type pragma for in0
        list_pragma_in0 = next(p for p in datatype_pragmas if p.parsed_data["interface_name"] == "in0")
        assert list_pragma_in0.parsed_data["base_types"] == ["INT", "UINT", "FIXED"]
        assert list_pragma_in0.parsed_data["min_width"] == 8
        assert list_pragma_in0.parsed_data["max_width"] == 16
        
        # Check single type pragma for in1 (backward compatibility)
        single_pragma = next(p for p in datatype_pragmas if p.parsed_data["interface_name"] == "in1")
        assert single_pragma.parsed_data["base_types"] == ["UINT"]
        assert single_pragma.parsed_data["min_width"] == 8
        assert single_pragma.parsed_data["max_width"] == 32
        
        # Check list type pragma for weights
        list_pragma_weights = next(p for p in datatype_pragmas if p.parsed_data["interface_name"] == "weights")
        assert list_pragma_weights.parsed_data["base_types"] == ["FIXED", "FLOAT"]
        assert list_pragma_weights.parsed_data["min_width"] == 16
        assert list_pragma_weights.parsed_data["max_width"] == 32
    
    def test_datatype_pragma_edge_cases(self, pragma_handler, ast_parser):
        """Test DATATYPE pragma edge cases."""
        # Test empty list
        rtl = (RTLBuilder()
               .pragma("DATATYPE", "in0", "[]", "8", "16")
               .module("test")
               .port("in0", "input", "16")
               .build())
        
        tree = ast_parser.parse_source(rtl)
        pragmas = pragma_handler.extract_pragmas(tree.root_node)
        
        # Empty list should fail validation
        datatype_pragma = next((p for p in pragmas if p.type == PragmaType.DATATYPE), None)
        assert datatype_pragma is None  # Should not create pragma due to validation error
        
        # Test invalid type in list
        rtl = (RTLBuilder()
               .pragma("DATATYPE", "in0", "[INT, INVALID_TYPE]", "8", "16")
               .module("test")
               .port("in0", "input", "16")
               .build())
        
        tree = ast_parser.parse_source(rtl)
        pragmas = pragma_handler.extract_pragmas(tree.root_node)
        
        # Invalid type should fail validation
        datatype_pragma = next((p for p in pragmas if p.type == PragmaType.DATATYPE), None)
        assert datatype_pragma is None  # Should not create pragma due to validation error
    
    def test_datatype_pragma_with_wildcard(self, pragma_handler, ast_parser):
        """Test DATATYPE pragma with wildcard * type."""
        # Build RTL with wildcard DATATYPE pragmas
        rtl = (RTLBuilder()
               .pragma("DATATYPE", "in0", "*", "8", "32")
               .pragma("DATATYPE", "in1", "[*, UINT]", "16", "16")  # * in list
               .pragma("DATATYPE", "in2", "[INT, *]", "4", "8")     # * in list with other type
               .module("test_module")
               .port("s_axis_in0_tdata", "input", "16")
               .port("s_axis_in1_tdata", "input", "16")
               .port("s_axis_in2_tdata", "input", "8")
               .build())
        
        tree = ast_parser.parse_source(rtl)
        pragmas = pragma_handler.extract_pragmas(tree.root_node)
        
        # Find DATATYPE pragmas
        datatype_pragmas = [p for p in pragmas if p.type == PragmaType.DATATYPE]
        assert len(datatype_pragmas) == 3
        
        # Check wildcard expansion to ANY
        wildcard_pragma = next(p for p in datatype_pragmas if p.parsed_data["interface_name"] == "in0")
        assert wildcard_pragma.parsed_data["base_types"] == ["ANY"]
        assert wildcard_pragma.parsed_data["min_width"] == 8
        assert wildcard_pragma.parsed_data["max_width"] == 32
        
        # Check wildcard in list - should become just ANY
        list_pragma1 = next(p for p in datatype_pragmas if p.parsed_data["interface_name"] == "in1")
        assert list_pragma1.parsed_data["base_types"] == ["ANY"]
        assert list_pragma1.parsed_data["min_width"] == 16
        assert list_pragma1.parsed_data["max_width"] == 16
        
        # Check wildcard with other type in list - should still become ANY
        list_pragma2 = next(p for p in datatype_pragmas if p.parsed_data["interface_name"] == "in2")
        assert list_pragma2.parsed_data["base_types"] == ["ANY"]
        assert list_pragma2.parsed_data["min_width"] == 4
        assert list_pragma2.parsed_data["max_width"] == 8