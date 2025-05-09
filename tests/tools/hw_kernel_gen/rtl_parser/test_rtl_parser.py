############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Comprehensive test suite for the RTLParser."""

import os
import pytest
import logging

# Standard imports from the RTLParser module
from brainsmith.tools.hw_kernel_gen.rtl_parser.parser import ParserError, SyntaxError
from brainsmith.tools.hw_kernel_gen.rtl_parser.data import Direction, InterfaceType
from brainsmith.tools.hw_kernel_gen.rtl_parser.pragma import PragmaType
from .test_fixtures import (
    VALID_HEADER_PARAMS_PORTSOPEN,
    VALID_GLOBAL_SIGNALS,
    VALID_AXI_STREAM_IN_INTERFACE,
    # VALID_AXI_STREAM_OUT_INTERFACE, # Let's omit this for a custom example
    VALID_PORTS_CLOSE,
    VALID_MODULE_BODY_CONTENT,
    VALID_ENDMODULE_STATEMENT,
    HEADER_PARAMS_PLACEHOLDER,
    VALID_MIN_INTERFACES, # For comparison or alternative construction
    VALID_MODULE_BODY
)
# No need to import fixtures as they're imported in conftest.py

logger = logging.getLogger(__name__)

# --- Test Classes ---

class TestParserCore:
    """Tests for basic parsing and module structure."""

    def test_empty_module(self, parser, temp_sv_file):
        """Test parsing an empty module raises error due to missing interfaces."""
        content = "module empty_mod; endmodule"
        path = temp_sv_file(content)
        expected_error_msg = r"Module 'empty_mod' is missing a valid Global Control interface \(ap_clk, ap_rst_n\)\."
        with pytest.raises(ParserError, match=expected_error_msg):
             parser.parse_file(path)

    def test_module_selection_single(self, parser, temp_sv_file, valid_module_content):
        """Test selecting the only module present."""
        # Minimal valid interface for parsing to succeed past interface checks
        
        path = temp_sv_file(valid_module_content, "valid_module_example.sv")
        kernel = parser.parse_file(path)
        assert kernel.name == "valid_module"

    def test_module_selection_top_module_pragma(self, parser, temp_sv_file, valid_module_content):
        """Test selecting the module specified by TOP_MODULE pragma."""
        content = """
        module ignore_me (); endmodule

        // @brainsmith TOP_MODULE valid_module
        """+valid_module_content+"""

        module another_ignore (); endmodule
        """
        path = temp_sv_file(content)
        kernel = parser.parse_file(path)
        assert kernel.name == "valid_module"

    def test_module_selection_multiple_no_pragma(self, parser, temp_sv_file):
        """Test parsing when multiple modules exist without TOP_MODULE pragma."""
        content = """
        module first_module (); endmodule
        module second_module (); endmodule
        """
        path = temp_sv_file(content)
        expected_error_msg = (
            r"Multiple modules \(\['first_module', 'second_module'\]\) found .*,"
            r" but no TOP_MODULE pragma specified\."
        )
        with pytest.raises(ParserError, match=expected_error_msg):
            parser.parse_file(path)

    def test_file_not_found(self, parser):
        """Test parsing a non-existent file raises an error."""
        expected_error_msg = r"Failed to read file non_existent_file\.sv: \[Errno 2\]"
        with pytest.raises(ParserError, match=expected_error_msg):
            parser.parse_file("non_existent_file.sv")

    def test_syntax_error(self, parser, temp_sv_file):
        """Test parsing a file with syntax errors raises an error."""
        content = "module syntax_err; wire x = ; endmodule" # Invalid syntax
        path = temp_sv_file(content)
        expected_error_msg = r"Invalid SystemVerilog syntax near line \d+, column \d+"
        # Expect SyntaxError (import fixed above)
        with pytest.raises(SyntaxError, match=expected_error_msg):
            parser.parse_file(path)


class TestParameterParsing:
    """Tests for parameter extraction."""

    def test_no_parameters(self, parser, temp_sv_file, valid_module_placeholder_params):
        content = valid_module_placeholder_params.replace("<PLACEHOLDER>", "")
        path = temp_sv_file(content)
        kernel = parser.parse_file(path)
        assert not kernel.parameters

    def test_simple_parameters(self, parser, temp_sv_file, valid_module_placeholder_params):
        """Tests implicitly typed parameters.""" # <-- Updated docstring
        content = valid_module_placeholder_params.replace("<PLACEHOLDER>", """
            parameter WIDTH = 32,
            parameter DEPTH = 1024,
            parameter NAME = "default_name"
        """)
        path = temp_sv_file(content)
        kernel = parser.parse_file(path)
        assert len(kernel.parameters) == 3
        param_map = {p.name: p for p in kernel.parameters}

        assert "WIDTH" in param_map
        assert param_map["WIDTH"].default_value == "32"
        assert param_map["WIDTH"].param_type is None

        assert "DEPTH" in param_map
        assert param_map["DEPTH"].default_value == "1024"
        assert param_map["DEPTH"].param_type is None

        assert "NAME" in param_map #
        assert param_map["NAME"].default_value == '"default_name"' # Includes quotes
        assert param_map["NAME"].param_type is None 

    def test_parameters_with_types(self, parser, temp_sv_file, valid_module_placeholder_params):
        content = valid_module_placeholder_params.replace("<PLACEHOLDER>", """
            parameter type T = logic, // Type parameter
            parameter int WIDTH = 32
        """)
        path = temp_sv_file(content)
        kernel = parser.parse_file(path)
        # Adjusting expectation to 1 until type parameters are handled
        assert len(kernel.parameters) == 2
        param_map = {p.name: p for p in kernel.parameters}

        assert "T" in param_map
        # Corrected: Use param_type attribute
        assert param_map["T"].param_type == "type"
        assert param_map["T"].default_value == "logic"

        assert "WIDTH" in param_map
        # Corrected: Use param_type attribute
        assert param_map["WIDTH"].param_type == "int"
        assert param_map["WIDTH"].default_value == "32"

    def test_parameter_integer_vector_types(self, parser, temp_sv_file, valid_module_placeholder_params):
        content = valid_module_placeholder_params.replace("<PLACEHOLDER>", """
            parameter bit          P_BIT         = 1'b1,
            parameter logic [7:0]  P_LOGIC_VEC   = 8'hAA,
            parameter reg signed [15:0] P_REG_SIGNED  = -16'd100,
            parameter logic unsigned P_LOGIC_UNS = 32 // Implicit width based on value?
        """)
        path = temp_sv_file(content)
        kernel = parser.parse_file(path)
        assert len(kernel.parameters) == 4
        param_map = {p.name: p for p in kernel.parameters}

        assert param_map["P_BIT"].param_type == "bit"
        assert param_map["P_BIT"].default_value == "1'b1"

        assert param_map["P_LOGIC_VEC"].param_type == "logic [7:0]"
        assert param_map["P_LOGIC_VEC"].default_value == "8'hAA"

        assert param_map["P_REG_SIGNED"].param_type == "reg signed [15:0]"
        assert param_map["P_REG_SIGNED"].default_value == "-16'd100"

        assert param_map["P_LOGIC_UNS"].param_type == "logic unsigned" # Assuming parser captures 'unsigned'
        assert param_map["P_LOGIC_UNS"].default_value == "32"

    def test_parameter_integer_atom_types(self, parser, temp_sv_file, valid_module_placeholder_params):
        content = valid_module_placeholder_params.replace("<PLACEHOLDER>", """
            parameter byte      P_BYTE     = 8'd10,
            parameter shortint  P_SHORT    = 16'd20,
            parameter int       P_INT      = 32, // Already tested, but good to have here
            parameter longint   P_LONG     = 64'd1234567890,
            parameter integer   P_INTEGER  = 99,
            parameter time      P_TIME     = 10ns
        """)
        path = temp_sv_file(content)
        kernel = parser.parse_file(path)
        assert len(kernel.parameters) == 6
        param_map = {p.name: p for p in kernel.parameters}

        assert param_map["P_BYTE"].param_type == "byte"
        assert param_map["P_BYTE"].default_value == "8'd10"
        assert param_map["P_SHORT"].param_type == "shortint"
        assert param_map["P_SHORT"].default_value == "16'd20"
        assert param_map["P_INT"].param_type == "int"
        assert param_map["P_INT"].default_value == "32"
        assert param_map["P_LONG"].param_type == "longint"
        assert param_map["P_LONG"].default_value == "64'd1234567890"
        assert param_map["P_INTEGER"].param_type == "integer"
        assert param_map["P_INTEGER"].default_value == "99"
        assert param_map["P_TIME"].param_type == "time"
        assert param_map["P_TIME"].default_value == "10ns"

    def test_parameter_real_types(self, parser, temp_sv_file, valid_module_placeholder_params):
        content = valid_module_placeholder_params.replace("<PLACEHOLDER>", """
            parameter shortreal P_SREAL = 1.23,
            parameter real      P_REAL  = 3.14159,
            parameter realtime  P_RTIME = 10.5ns
        """)
        path = temp_sv_file(content)
        kernel = parser.parse_file(path)
        assert len(kernel.parameters) == 3
        param_map = {p.name: p for p in kernel.parameters}

        assert param_map["P_SREAL"].param_type == "shortreal"
        assert param_map["P_SREAL"].default_value == "1.23"
        assert param_map["P_REAL"].param_type == "real"
        assert param_map["P_REAL"].default_value == "3.14159"
        assert param_map["P_RTIME"].param_type == "realtime"
        assert param_map["P_RTIME"].default_value == "10.5ns"

    def test_parameter_string_type(self, parser, temp_sv_file, valid_module_placeholder_params):
        content = valid_module_placeholder_params.replace("<PLACEHOLDER>", """
            parameter string P_STRING = "Hello, SystemVerilog!"
        """)
        path = temp_sv_file(content)
        kernel = parser.parse_file(path)
        assert len(kernel.parameters) == 1
        param = kernel.parameters[0]
        assert param.name == "P_STRING"
        assert param.param_type == "string"
        assert param.default_value == '"Hello, SystemVerilog!"' # Includes quotes

    def test_parameter_complex_default(self, parser, temp_sv_file, valid_module_placeholder_params):
        content = valid_module_placeholder_params.replace("<PLACEHOLDER>", """
            parameter WIDTH = 32,
            parameter LSB = WIDTH - 1,
            parameter MSG = { "Part1", "Part2" }
        """)
        path = temp_sv_file(content)
        kernel = parser.parse_file(path)
        assert len(kernel.parameters) == 3
        param_map = {p.name: p for p in kernel.parameters}

        assert param_map["WIDTH"].default_value == "32"
        # Parser likely captures the expression as a string
        assert param_map["LSB"].default_value == "WIDTH - 1"
        assert param_map["MSG"].default_value == '{ "Part1", "Part2" }'

    def test_local_parameters(self, parser, temp_sv_file, valid_module_placeholder_params):
        content =  f"""\
        {VALID_HEADER_PARAMS_PORTSOPEN}
        {VALID_MIN_INTERFACES}
        );
            localparam int LP_WIDTH = 16;
            localparam bit [7:0] LP_NAME = "local_param";

            // Some logic using the local parameters
        endmodule
        """
        path = temp_sv_file(content)
        kernel = parser.parse_file(path)
        assert kernel.name == "valid_module" # Basic check
        for p in kernel.parameters:
            assert not p.name.startswith("LP_")

    def test_parameters_no_default(self, parser, temp_sv_file, valid_module_placeholder_params):
        content = valid_module_placeholder_params.replace("<PLACEHOLDER>", """
            parameter int NO_DEFAULT_INT,
            parameter NO_DEFAULT_IMPLICIT
        """)
        path = temp_sv_file(content)
        kernel = parser.parse_file(path)
        assert len(kernel.parameters) == 2
        param_map = {p.name: p for p in kernel.parameters}
        assert "NO_DEFAULT_INT" in param_map
        assert param_map["NO_DEFAULT_INT"].param_type == "int"
        assert param_map["NO_DEFAULT_INT"].default_value is None
        assert "NO_DEFAULT_IMPLICIT" in param_map
        assert param_map["NO_DEFAULT_IMPLICIT"].param_type is None
        assert param_map["NO_DEFAULT_IMPLICIT"].default_value is None


class TestPortParsing:
    """Tests for port extraction."""

    def test_simple_ports(self, parser, temp_sv_file):
        content = """
        module test (
            input clk,
            input rst,
            output valid
        );
        endmodule
        """
        path = temp_sv_file(content)
        try:
            parser._initial_parse(path)
            parser._extract_kernel_components()
        except (ParserError, SyntaxError) as e:
            pytest.fail(f"Parsing stages 1 or 2 failed unexpectedly: {e}")
        assert len(parser.ports) == 3
        port_map = {p.name: p for p in parser.ports}
        assert port_map["clk"].direction == Direction.INPUT
        assert port_map["rst"].direction == Direction.INPUT
        assert port_map["valid"].direction == Direction.OUTPUT

    def test_ports_with_width(self, parser, temp_sv_file):
        content = """
        module test (
            input logic [31:0] data_in,
            output logic [7:0] data_out,
            inout wire [1:0] bidir
        );
        endmodule
        """
        path = temp_sv_file(content)
        try:
            parser._initial_parse(path)
            parser._extract_kernel_components()
        except (ParserError, SyntaxError) as e:
            pytest.fail(f"Parsing stages 1 or 2 failed unexpectedly: {e}")

        assert parser.name == "test"
        assert not parser.parameters
        assert len(parser.ports) == 3
        port_map = {p.name: p for p in parser.ports}
        # --- MODIFIED: Removed brackets from expected width ---
        assert "data_in" in port_map and port_map["data_in"].width == "31:0" and port_map["data_in"].direction == Direction.INPUT
        assert "data_out" in port_map and port_map["data_out"].width == "7:0" and port_map["data_out"].direction == Direction.OUTPUT
        assert "bidir" in port_map and port_map["bidir"].width == "1:0" and port_map["bidir"].direction == Direction.INOUT
        # --- END MODIFICATION ---

    def test_ports_parametric_width(self, parser, temp_sv_file):
        """Tests ports with widths defined by parameters."""
        content = """
        module test #(parameter WIDTH = 8) (
            input logic [WIDTH-1:0] data_div_width,
            output logic [$clog2(WIDTH):0] addr,
            input logic valid
        );
        endmodule
        """
        path = temp_sv_file(content)
        try:
            parser._initial_parse(path)
            parser._extract_kernel_components()
        except (ParserError, SyntaxError) as e:
            pytest.fail(f"Parsing stages 1 or 2 failed unexpectedly: {e}")

        assert parser.name == "test"
        assert len(parser.parameters) == 1
        assert parser.parameters[0].name == "WIDTH"

        assert len(parser.ports) == 3
        port_map = {p.name: p for p in parser.ports}
        # --- MODIFIED: Removed brackets and fixed 'valid' width ---
        assert "data_div_width" in port_map and port_map["data_div_width"].width == "WIDTH-1:0"
        assert "addr" in port_map and port_map["addr"].width == "$clog2(WIDTH):0"
        assert "valid" in port_map and port_map["valid"].width == '1'
        # --- END MODIFICATION ---

    def test_ansi_ports(self, parser, temp_sv_file):
        """Tests ANSI-style port declarations."""
        content = """
        module test_ansi (
            input logic clk,
            input logic [31:0] data_in,
            output logic data_valid,
            output logic [7:0] data_out
        );
        endmodule
        """
        path = temp_sv_file(content)
        try:
            parser._initial_parse(path)
            parser._extract_kernel_components()
        except (ParserError, SyntaxError) as e:
            pytest.fail(f"Parsing stages 1 or 2 failed unexpectedly: {e}")

        assert parser.name == "test_ansi"
        assert not parser.parameters
        assert len(parser.ports) == 4
        port_map = {p.name: p for p in parser.ports}
        assert "clk" in port_map and port_map["clk"].direction == Direction.INPUT and port_map["clk"].width == '1'
        assert "data_in" in port_map and port_map["data_in"].direction == Direction.INPUT and port_map["data_in"].width == "31:0"
        assert "data_valid" in port_map and port_map["data_valid"].direction == Direction.OUTPUT and port_map["data_valid"].width == '1'
        assert "data_out" in port_map and port_map["data_out"].direction == Direction.OUTPUT and port_map["data_out"].width == "7:0"

    # test_non_ansi_ports - Keep as is, should pass full parse
    # test_mixed_ansi_non_ansi - Already modified
    # test_unassigned_ports - Already modified
    # test_interface_ports - Already modified

class TestPragmaHandling:
    """Tests for pragma extraction and handling."""

    def test_no_pragmas(self, parser, temp_sv_file, valid_module_content):
        path = temp_sv_file(valid_module_content)
        kernel = parser.parse_file(path)
        assert not kernel.pragmas

    def test_supported_pragmas(self, parser, temp_sv_file):
        content = """
        // @brainsmith TOP_MODULE test_module
        // @brainsmith DATATYPE data_in_if T_UINT8
        // @brainsmith DERIVED_PARAMETER hello_world STRIDE
        // @brainsmith WEIGHT in0

        module test_module #(
             parameter KERNEL_SIZE = "3x3",
             parameter STRIDE = 1,
             parameter PADDING = 1,
             parameter ENABLE = 0
        ) (
             input logic ap_clk, input logic ap_rst_n, // Need these for Stage 3 if called
             input logic [7:0] data_in, // Matches DATATYPE pragma, but unassigned by builder
             input logic [31:0] in0_TDATA, input logic in0_TVALID, output logic in0_TREADY // Need AXI stream for Stage 3
        );
        endmodule
        """
        
        path = temp_sv_file(content)
        parser._initial_parse(path)
        for p in parser.pragmas:
            print(f"Pragma: {p.type}, Inputs: {p.inputs}, Line: {p.line_number}")
        try:
            parser._initial_parse(path)
            assert len(parser.pragmas) == 4
            top_pragmas = [p for p in parser.pragmas if p.type == PragmaType.TOP_MODULE]
            datatype_pragmas = [p for p in parser.pragmas if p.type == PragmaType.DATATYPE]
            derived_pragmas = [p for p in parser.pragmas if p.type == PragmaType.DERIVED_PARAMETER]
            weights_pragmas = [p for p in parser.pragmas if p.type == PragmaType.WEIGHT]
            assert len(top_pragmas) == 1
            assert len(datatype_pragmas) == 1
            assert len(derived_pragmas) == 1
            assert len(weights_pragmas) == 1

            parser._extract_kernel_components()
        except (ParserError, SyntaxError) as e:
            pytest.fail(f"Parsing stages 1 or 2 failed unexpectedly: {e}")

        assert parser.name == "test_module"
        assert len(parser.parameters) == 4
        assert len(parser.ports) == 6 # clk, rst_n, data_in, TDATA, TVALID, TREADY
        # We don't call Stage 3, so the unassigned 'data_in' doesn't cause an error

    def test_unsupported_pragmas_ignored(self, parser, temp_sv_file):
        content = """
        // @brainsmith TOP_MODULE test_module
        // @brainsmith RESOURCE DSP 4
        // @brainsmith DATATYPE data_in UINT8

        module test_module (
             input logic ap_clk, input logic ap_rst_n, // Need these
             input logic [7:0] data_in, // Unassigned
             input logic [31:0] in0_TDATA, input logic in0_TVALID, output logic in0_TREADY // Need AXI stream
        );
        endmodule
        """
        path = temp_sv_file(content)
        try:
            parser._initial_parse(path)
            # Check pragmas after Stage 1
            assert len(parser.pragmas) == 2 # TOP_MODULE and DATATYPE
            pragma_types = {p.type for p in parser.pragmas}
            assert pragma_types == {PragmaType.TOP_MODULE, PragmaType.DATATYPE}

            parser._extract_kernel_components()
        except (ParserError, SyntaxError) as e:
            pytest.fail(f"Parsing stages 1 or 2 failed unexpectedly: {e}")

        assert parser.name == "test_module"
        assert not parser.parameters
        assert len(parser.ports) == 6

    def test_malformed_pragmas_ignored(self, parser, temp_sv_file):
        content = """
        // @brainsmith TOP_MODULE test_module
        // @brainsmith DATATYPE data_in // Missing value
        // @brainsmith DERIVED_PARAMETER KERNEL_SIZE
        // @brainsmith INVALID_PRAGMA foo bar
        // @brainsmith // Missing type

        module test_module (
             input logic ap_clk, input logic ap_rst_n, // Need these
             input logic [7:0] data_in, // Unassigned
             input logic [31:0] in0_TDATA, input logic in0_TVALID, output logic in0_TREADY // Need AXI stream
        );
        endmodule
        """
        path = temp_sv_file(content)
        # --- MODIFIED: Call stages 1 & 2 ---
        try:
            parser._initial_parse(path)
            # Check pragmas after Stage 1
            assert len(parser.pragmas) == 1 # Only TOP_MODULE is valid
            assert parser.pragmas[0].type == PragmaType.TOP_MODULE

            parser._extract_kernel_components()
        except (ParserError, SyntaxError) as e:
            pytest.fail(f"Parsing stages 1 or 2 failed unexpectedly: {e}")
        # --- END MODIFICATION ---

        assert parser.name == "test_module"
        assert not parser.parameters
        assert len(parser.ports) == 6

# TestInterfaceAnalysis tests remain unchanged as they test Stage 3 behavior

