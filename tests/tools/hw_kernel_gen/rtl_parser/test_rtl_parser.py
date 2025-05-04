"""Comprehensive test suite for the RTLParser."""

import os
import pytest
import tempfile
import shutil
import logging # Import logging
import collections # Import collections

# Assuming data structures and parser are importable like this
from brainsmith.tools.hw_kernel_gen.rtl_parser.parser import RTLParser, ParserError, SyntaxError
from brainsmith.tools.hw_kernel_gen.rtl_parser.data import HWKernel, Parameter, Port, Direction
from brainsmith.tools.hw_kernel_gen.rtl_parser.interface_types import InterfaceType
from brainsmith.tools.hw_kernel_gen.rtl_parser.pragma import PragmaType
from tree_sitter import Node # Import Node for type hinting

# Get logger for this test module
logger = logging.getLogger(__name__)

# --- Test Fixtures ---

@pytest.fixture(scope="module")
def parser():
    """Provides an RTLParser instance for the test module."""
    # Assuming grammar path is handled internally by RTLParser constructor
    try:
        return RTLParser(debug=True) # Enable debug for more verbose test output if needed
    except Exception as e:
        pytest.fail(f"Failed to initialize RTLParser: {e}")

@pytest.fixture(scope="function")
def temp_sv_file():
    """Creates a temporary directory and a helper function to write SV files."""
    temp_dir = tempfile.mkdtemp()
    files_created = []

    def _create_file(content: str, filename: str = "test.sv") -> str:
        path = os.path.join(temp_dir, filename)
        with open(path, "w") as f:
            f.write(content)
        files_created.append(path)
        print(f"Created temp file: {path}") # Debug print
        return path

    yield _create_file

    # Cleanup: Remove the temporary directory and its contents
    print(f"Cleaning up temp directory: {temp_dir}") # Debug print
    shutil.rmtree(temp_dir)

# --- Helper function to print AST ---
def print_ast_node(node: Node, indent: str = "", logger_func=print):
    """Recursively prints the AST node structure."""
    text_preview = node.text.decode('utf8').replace('\n', '\\n')[:50]
    logger_func(f"{indent}Type: {node.type}, Named: {node.is_named}, Text: '{text_preview}...'")
    for i, child in enumerate(node.children):
        print_ast_node(child, indent + "  ", logger_func=logger_func)

# --- Test Classes ---

class TestDebuggingHelpers: # New class for debug tests
    """Tests specifically for debugging and inspecting parser behavior."""

    @pytest.mark.skip(reason="AST output is too verbose for regular runs")
    def test_print_ast_structure(self, parser, temp_sv_file):
        """Parses a simple module and prints its AST structure for inspection."""
        content = """
        module test_module #(
            parameter WIDTH = 32
        ) (
            input logic clk,
            input logic rst,
            output logic [WIDTH-1:0] data_out
        );
            // Some internal logic
            assign data_out = {WIDTH{1'b0}};
        endmodule
        """
        path = temp_sv_file(content, "ast_test.sv")
        # Use print instead of logger.info for direct output with pytest -s
        print(f"\n--- AST Structure for {path} ---")

        # Read content again (parser needs bytes)
        with open(path, 'rb') as f:
            source_bytes = f.read()

        # Parse directly using the tree-sitter parser instance
        tree = parser.parser.parse(source_bytes)
        root_node = tree.root_node

        # Print the AST structure using the helper, directing output to print
        print_ast_node(root_node, logger_func=print)

        print("--- End AST Structure ---")
        assert True # This test is for printing, always pass if it runs

class TestParserCore:
    """Tests for basic parsing and module structure."""

    def test_empty_module(self, parser, temp_sv_file):
        """Test parsing an empty module."""
        content = "module empty_mod; endmodule"
        path = temp_sv_file(content)
        kernel = parser.parse_file(path)
        assert isinstance(kernel, HWKernel)
        assert kernel.name == "empty_mod"
        assert not kernel.parameters
        assert not kernel.ports
        assert not kernel.pragmas
        assert not kernel.interfaces # Should be empty after analysis

    def test_simple_module_no_ports_params(self, parser, temp_sv_file):
        """Test a simple module without ports or parameters."""
        content = "module simple_mod; initial begin $display(\"Hello\"); end endmodule"
        path = temp_sv_file(content)
        # Expect ParserError because no interfaces (specifically AXI-Stream) are found
        with pytest.raises(ParserError, match=r"requires at least one AXI-Stream interface"):
             parser.parse_file(path)
        # If we didn't have the interface requirement, we'd check:
        # kernel = parser.parse_file(path)
        # assert kernel.name == "simple_mod"
        # assert not kernel.parameters
        # assert not kernel.ports

    def test_module_selection_single(self, parser, temp_sv_file):
        """Test selecting the only module present."""
        # Minimal valid interface for parsing to succeed past interface checks
        content = """
        module target_module (
            input logic ap_clk,
            input logic ap_rst_n,
            // AXI-Stream Input
            input logic [31:0] in0_TDATA,
            input logic        in0_TVALID,
            output logic       in0_TREADY,
            // AXI-Stream Output
            output logic [31:0] out0_TDATA,
            output logic        out0_TVALID,
            input logic         out0_TREADY
        );
        endmodule
        """
        path = temp_sv_file(content)
        kernel = parser.parse_file(path)
        assert kernel.name == "target_module"

    def test_module_selection_top_module_pragma(self, parser, temp_sv_file):
        """Test selecting the module specified by TOP_MODULE pragma."""
        content = """
        module ignore_me (); endmodule

        // @brainsmith TOP_MODULE target_module
        module target_module (
             input logic ap_clk,
             input logic ap_rst_n,
             input logic [31:0] in0_TDATA, input logic in0_TVALID, output logic in0_TREADY,
             output logic [31:0] out0_TDATA, output logic out0_TVALID, input logic out0_TREADY
        );
        endmodule

        module another_ignore (); endmodule
        """
        path = temp_sv_file(content)
        kernel = parser.parse_file(path)
        assert kernel.name == "target_module"


class TestParameterParsing:
    """Tests for parameter extraction."""

    def test_no_parameters(self, parser, temp_sv_file):
        content = """
        module test (
             input logic ap_clk, input logic ap_rst_n,
             input logic [31:0] in0_TDATA, input logic in0_TVALID, output logic in0_TREADY
        );
        endmodule
        """
        path = temp_sv_file(content)
        kernel = parser.parse_file(path)
        assert not kernel.parameters

    def test_simple_parameters(self, parser, temp_sv_file):
        content = """
        module test #(
            parameter WIDTH = 32,
            parameter DEPTH = 1024
        ) (
             input logic ap_clk, input logic ap_rst_n,
             input logic [WIDTH-1:0] in0_TDATA, input logic in0_TVALID, output logic in0_TREADY
        );
        endmodule
        """
        path = temp_sv_file(content)
        kernel = parser.parse_file(path)
        assert len(kernel.parameters) == 2
        param_map = {p.name: p for p in kernel.parameters}
        assert "WIDTH" in param_map
        assert "DEPTH" in param_map
        assert param_map["WIDTH"].param_type == "logic" # Default type assumed by parser
        assert param_map["WIDTH"].default_value == "32"
        assert param_map["DEPTH"].default_value == "1024"

    def test_parameters_with_types(self, parser, temp_sv_file):
        content = """
        module test #(
            parameter int DATA_WIDTH = 64,
            parameter type T = logic [7:0]
        ) (
             input logic ap_clk, input logic ap_rst_n,
             input logic [DATA_WIDTH-1:0] in0_TDATA, input logic in0_TVALID, output logic in0_TREADY
        );
        endmodule
        """
        path = temp_sv_file(content)
        kernel = parser.parse_file(path)
        assert len(kernel.parameters) == 2
        param_map = {p.name: p for p in kernel.parameters}
        assert "DATA_WIDTH" in param_map
        assert "T" in param_map
        assert param_map["DATA_WIDTH"].param_type == "int"
        assert param_map["DATA_WIDTH"].default_value == "64"
        assert param_map["T"].param_type == "type"
        assert param_map["T"].default_value == "logic [7:0]"

    def test_parameters_no_default(self, parser, temp_sv_file):
        content = """
        module test #(
            parameter ADDR_WIDTH
        ) (
             input logic ap_clk, input logic ap_rst_n,
             input logic [ADDR_WIDTH-1:0] in0_TDATA, input logic in0_TVALID, output logic in0_TREADY
        );
        endmodule
        """
        path = temp_sv_file(content)
        kernel = parser.parse_file(path)
        assert len(kernel.parameters) == 1
        assert kernel.parameters[0].name == "ADDR_WIDTH"
        assert kernel.parameters[0].default_value is None

    def test_localparam_ignored(self, parser, temp_sv_file):
        content = """
        module test #(
            parameter WIDTH = 16
        ) (
             input logic ap_clk, input logic ap_rst_n,
             input logic [WIDTH-1:0] in0_TDATA, input logic in0_TVALID, output logic in0_TREADY
        );
            localparam INTERNAL_DEPTH = WIDTH * 2;
        endmodule
        """
        path = temp_sv_file(content)
        kernel = parser.parse_file(path)
        assert len(kernel.parameters) == 1
        assert kernel.parameters[0].name == "WIDTH"


class TestPortParsing:
    """Tests for port extraction."""

    def test_simple_ports(self, parser, temp_sv_file):
        content = """
        module test (
            input logic clk,
            input logic rst,
            output logic valid,
            // Minimal AXI-Stream for validation
             input logic [31:0] in0_TDATA, input logic in0_TVALID, output logic in0_TREADY
        );
            // Assign global signals to satisfy validator for this test focus
            assign ap_clk = clk;
            assign ap_rst_n = rst;
        endmodule
        """
        path = temp_sv_file(content)
        # Modify parser temporarily to skip interface validation for this specific test
        original_builder = parser.interface_builder
        parser.interface_builder = None # Disable builder

        try:
            kernel = parser.parse_file(path)
            assert len(kernel.ports) == 6 # clk, rst, valid, TDATA, TVALID, TREADY
            port_map = {p.name: p for p in kernel.ports}

            assert "clk" in port_map
            assert port_map["clk"].direction == Direction.INPUT
            assert port_map["clk"].width == "1"

            assert "rst" in port_map
            assert port_map["rst"].direction == Direction.INPUT
            assert port_map["rst"].width == "1"

            assert "valid" in port_map
            assert port_map["valid"].direction == Direction.OUTPUT
            assert port_map["valid"].width == "1"

            assert "in0_TDATA" in port_map # Check one of the AXI ports too
            assert port_map["in0_TDATA"].direction == Direction.INPUT
            assert port_map["in0_TDATA"].width == "31:0"

        finally:
             parser.interface_builder = original_builder # Restore builder

    def test_ports_with_width(self, parser, temp_sv_file):
        content = """
        module test (
            input logic [15:0] data_in,
            output logic [63:0] data_out,
            inout wire [7:0] data_io,
            // Minimal AXI-Stream for validation
             input logic ap_clk, input logic ap_rst_n,
             input logic [31:0] in0_TDATA, input logic in0_TVALID, output logic in0_TREADY
        );
        endmodule
        """
        path = temp_sv_file(content)
        kernel = parser.parse_file(path) # Interface validation should pass here
        assert len(kernel.ports) == 8 # 3 test ports + 5 interface ports
        port_map = {p.name: p for p in kernel.ports}

        assert "data_in" in port_map
        assert port_map["data_in"].direction == Direction.INPUT
        assert port_map["data_in"].width == "15:0"

        assert "data_out" in port_map
        assert port_map["data_out"].direction == Direction.OUTPUT
        assert port_map["data_out"].width == "63:0"

        assert "data_io" in port_map
        assert port_map["data_io"].direction == Direction.INOUT
        assert port_map["data_io"].width == "7:0"

    def test_ports_parametric_width(self, parser, temp_sv_file):
        content = """
        module test #(
            parameter WIDTH = 32,
            parameter C = 1
        ) (
            input logic [WIDTH-1:0] data_in,
            output logic [(WIDTH*2)-1:0] data_out,
            input logic [WIDTH/C:0] data_div,
             // Minimal AXI-Stream for validation
             input logic ap_clk, input logic ap_rst_n,
             input logic [31:0] in0_TDATA, input logic in0_TVALID, output logic in0_TREADY
        );
        endmodule
        """
        path = temp_sv_file(content)
        kernel = parser.parse_file(path)
        assert len(kernel.ports) == 8 # 3 test ports + 5 interface ports
        port_map = {p.name: p for p in kernel.ports}

        assert "data_in" in port_map
        assert port_map["data_in"].width == "WIDTH-1:0"

        assert "data_out" in port_map
        assert port_map["data_out"].width == "(WIDTH*2)-1:0" # Parser might capture parens

        assert "data_div" in port_map
        assert port_map["data_div"].width == "WIDTH/C:0"

    def test_ansi_ports(self, parser, temp_sv_file):
        """Test ANSI-style port declarations."""
        content = """
        module test_ansi #(
            parameter WIDTH = 8
        ) (
            input  logic clk,
            input  logic rst,
            output logic [WIDTH-1:0] data_out,
            input  logic [WIDTH-1:0] data_in,
            // Minimal AXI-Stream for validation
             input logic ap_clk, input logic ap_rst_n, // Duplicate global signals ok?
             input logic [31:0] in0_TDATA, input logic in0_TVALID, output logic in0_TREADY
        );
        endmodule
        """
        path = temp_sv_file(content)
        kernel = parser.parse_file(path)
        # Expect 4 test ports + 5 interface ports = 9
        assert len(kernel.ports) == 9
        port_map = {p.name: p for p in kernel.ports}

        assert "clk" in port_map and port_map["clk"].direction == Direction.INPUT
        assert "rst" in port_map and port_map["rst"].direction == Direction.INPUT
        assert "data_out" in port_map and port_map["data_out"].direction == Direction.OUTPUT and port_map["data_out"].width == "WIDTH-1:0"
        assert "data_in" in port_map and port_map["data_in"].direction == Direction.INPUT and port_map["data_in"].width == "WIDTH-1:0"
        assert "ap_clk" in port_map # Check interface ports are also found


class TestPragmaHandling:
    """Tests for pragma extraction and handling."""

    def test_no_pragmas(self, parser, temp_sv_file):
        content = """
        module test (
             input logic ap_clk, input logic ap_rst_n,
             input logic [31:0] in0_TDATA, input logic in0_TVALID, output logic in0_TREADY
        );
            // No pragmas here
        endmodule
        """
        path = temp_sv_file(content)
        kernel = parser.parse_file(path)
        assert not kernel.pragmas

    def test_supported_pragmas(self, parser, temp_sv_file):
        content = """
        // @brainsmith TOP_MODULE test
        // @brainsmith DATATYPE data_in UINT8
        // @brainsmith DERIVED_PARAMETER KERNEL_SIZE 3x3

        module test (
             input logic ap_clk, input logic ap_rst_n,
             input logic [7:0] data_in, // Matches DATATYPE pragma
             input logic [31:0] in0_TDATA, input logic in0_TVALID, output logic in0_TREADY
        );
        endmodule
        """
        path = temp_sv_file(content)
        kernel = parser.parse_file(path)
        assert len(kernel.pragmas) == 3
        pragma_map = {p.type: p for p in kernel.pragmas}

        assert PragmaType.TOP_MODULE.value in pragma_map
        top_pragma = pragma_map[PragmaType.TOP_MODULE.value]
        assert top_pragma.inputs == ["test"]
        assert top_pragma.processed_data == {"module_name": "test"}

        assert PragmaType.DATATYPE.value in pragma_map
        dtype_pragma = pragma_map[PragmaType.DATATYPE.value]
        assert dtype_pragma.inputs == ["data_in", "UINT8"]
        assert dtype_pragma.processed_data == {"port_name": "data_in", "data_type": "UINT8"}

        assert PragmaType.DERIVED_PARAMETER.value in pragma_map
        deriv_pragma = pragma_map[PragmaType.DERIVED_PARAMETER.value]
        assert deriv_pragma.inputs == ["KERNEL_SIZE", "3x3"]
        assert deriv_pragma.processed_data == {"param_name": "KERNEL_SIZE", "value_expr": "3x3"}

    def test_unsupported_pragmas_ignored(self, parser, temp_sv_file):
        content = """
        // @brainsmith TOP_MODULE test
        // @brainsmith RESOURCE DSP 4  (Old/Unsupported)
        // @brainsmith INTERFACE AXI_LITE (Old/Unsupported)
        // @brainsmith TIMING LATENCY 5 (Old/Unsupported)
        // @brainsmith FEATURE F1       (Old/Unsupported)
        // @brainsmith DATATYPE data_in UINT8

        module test (
             input logic ap_clk, input logic ap_rst_n,
             input logic [7:0] data_in,
             input logic [31:0] in0_TDATA, input logic in0_TVALID, output logic in0_TREADY
        );
        endmodule
        """
        path = temp_sv_file(content)
        kernel = parser.parse_file(path)
        # Only TOP_MODULE and DATATYPE should be parsed
        assert len(kernel.pragmas) == 2
        pragma_types = {p.type for p in kernel.pragmas}
        assert pragma_types == {PragmaType.TOP_MODULE.value, PragmaType.DATATYPE.value}

    def test_malformed_pragmas_ignored(self, parser, temp_sv_file):
        content = """
        // @brainsmith TOP_MODULE test
        // @brainsmith DATATYPE data_in // Missing value
        // @brainsmith DERIVED_PARAMETER KERNEL_SIZE
        // @brainsmith INVALID_PRAGMA foo bar
        // @brainsmith // Missing type

        module test (
             input logic ap_clk, input logic ap_rst_n,
             input logic [7:0] data_in,
             input logic [31:0] in0_TDATA, input logic in0_TVALID, output logic in0_TREADY
        );
        endmodule
        """
        path = temp_sv_file(content)
        kernel = parser.parse_file(path)
        # Only the valid TOP_MODULE pragma should be parsed
        assert len(kernel.pragmas) == 1
        assert kernel.pragmas[0].type == PragmaType.TOP_MODULE.value


class TestInterfaceAnalysis:
    """Tests for the integration of interface scanning and validation."""

    def test_valid_global_one_stream(self, parser, temp_sv_file):
        content = """
        module test (
            input logic ap_clk,
            input logic ap_rst_n,
            // AXI-Stream Input
            input logic [31:0] in0_TDATA,
            input logic        in0_TVALID,
            output logic       in0_TREADY,
            input logic        in0_TLAST  // Optional
        );
        endmodule
        """
        path = temp_sv_file(content)
        kernel = parser.parse_file(path)

        assert "global" in kernel.interfaces
        global_if = kernel.interfaces["global"]
        assert global_if.type == InterfaceType.GLOBAL_CONTROL
        assert set(global_if.ports.keys()) == {"ap_clk", "ap_rst_n"}

        assert "in0" in kernel.interfaces
        stream_if = kernel.interfaces["in0"]
        assert stream_if.type == InterfaceType.AXI_STREAM
        assert set(stream_if.ports.keys()) == {"_TDATA", "_TVALID", "_TREADY", "_TLAST"}
        # Check directions based on validator logic (in0 -> input stream)
        assert stream_if.ports["_TDATA"].direction == Direction.INPUT
        assert stream_if.ports["_TVALID"].direction == Direction.INPUT
        assert stream_if.ports["_TREADY"].direction == Direction.OUTPUT # Input stream ready is output
        assert stream_if.ports["_TLAST"].direction == Direction.INPUT

    def test_valid_global_streams_lite(self, parser, temp_sv_file):
        content = """
        module test (
            // Global
            input logic ap_clk,
            input logic ap_rst_n,
            // AXI-Stream Input
            input logic [31:0] s_axis_TDATA,
            input logic        s_axis_TVALID,
            output logic       s_axis_TREADY,
            // AXI-Stream Output
            output logic [63:0] m_axis_TDATA,
            output logic        m_axis_TVALID,
            input logic         m_axis_TREADY,
            output logic        m_axis_TLAST,
             // AXI-Lite
            input  logic [15:0] config_AWADDR, input  logic [2:0] config_AWPROT, input  logic config_AWVALID, output logic config_AWREADY,
            input  logic [31:0] config_WDATA, input  logic [3:0] config_WSTRB, input  logic config_WVALID, output logic config_WREADY,
            output logic [1:0]  config_BRESP, output logic config_BVALID, input  logic config_BREADY,
            input  logic [15:0] config_ARADDR, input  logic [2:0] config_ARPROT, input  logic config_ARVALID, output logic config_ARREADY,
            output logic [31:0] config_RDATA, output logic [1:0] config_RRESP, output logic config_RVALID, input  logic config_RREADY
        );
        endmodule
        """
        path = temp_sv_file(content)
        kernel = parser.parse_file(path)

        assert len(kernel.interfaces) == 4 # global, s_axis, m_axis, config
        assert "global" in kernel.interfaces
        assert "s_axis" in kernel.interfaces and kernel.interfaces["s_axis"].type == InterfaceType.AXI_STREAM
        assert "m_axis" in kernel.interfaces and kernel.interfaces["m_axis"].type == InterfaceType.AXI_STREAM
        assert "config" in kernel.interfaces and kernel.interfaces["config"].type == InterfaceType.AXI_LITE

        # Spot check AXI-Lite
        lite_if = kernel.interfaces["config"]
        assert "config_AWADDR" in lite_if.ports and lite_if.ports["config_AWADDR"].direction == Direction.INPUT
        assert "config_WREADY" in lite_if.ports and lite_if.ports["config_WREADY"].direction == Direction.OUTPUT
        assert "config_ARADDR" in lite_if.ports and lite_if.ports["config_ARADDR"].direction == Direction.INPUT
        assert "config_RDATA" in lite_if.ports and lite_if.ports["config_RDATA"].direction == Direction.OUTPUT

    def test_missing_required_global(self, parser, temp_sv_file):
        content = """
        module test (
            input logic ap_clk,
            // Missing ap_rst_n
             input logic [31:0] in0_TDATA, input logic in0_TVALID, output logic in0_TREADY
        );
        endmodule
        """
        path = temp_sv_file(content)
        with pytest.raises(ParserError, match=r"Validation failed for potential interface 'global'.*Missing required global signals: {'ap_rst_n'}"):
            parser.parse_file(path)

    def test_missing_required_stream(self, parser, temp_sv_file):
        content = """
        module test (
            input logic ap_clk,
            input logic ap_rst_n,
            input logic [31:0] in0_TDATA
            // Missing in0_TVALID, in0_TREADY
        );
        endmodule
        """
        path = temp_sv_file(content)
        with pytest.raises(ParserError, match=r"Validation failed for potential interface 'in0'.*Missing required AXI-Stream signals.*{'_TVALID', '_TREADY'}"):
             parser.parse_file(path)

    def test_incorrect_stream_direction(self, parser, temp_sv_file):
        content = """
        module test (
            input logic ap_clk,
            input logic ap_rst_n,
            // AXI-Stream Input (but TREADY is wrong direction)
            input logic [31:0] in0_TDATA,
            input logic        in0_TVALID,
            input logic        in0_TREADY // Should be output for input stream
        );
        endmodule
        """
        path = temp_sv_file(content)
        with pytest.raises(ParserError, match=r"Validation failed for potential interface 'in0'.*Invalid AXI-Stream signal 'in0_TREADY': Incorrect direction: expected output, got input"):
            parser.parse_file(path)

    def test_unassigned_ports(self, parser, temp_sv_file):
        content = """
        module test (
            input logic ap_clk,
            input logic ap_rst_n,
            input logic [31:0] in0_TDATA,
            input logic        in0_TVALID,
            output logic       in0_TREADY,
            // Unassigned ports
            input logic        enable,
            output logic [7:0] status
        );
        endmodule
        """
        path = temp_sv_file(content)
        with pytest.raises(ParserError, match=r"Module 'test' has 2 ports not assigned to any valid interface: \['enable', 'status'\]"):
            parser.parse_file(path)

    def test_no_axi_stream_interface(self, parser, temp_sv_file):
        content = """
        module test (
            input logic ap_clk,
            input logic ap_rst_n
            // No AXI-Stream interfaces
        );
        endmodule
        """
        path = temp_sv_file(content)
        with pytest.raises(ParserError, match=r"Module 'test' requires at least one AXI-Stream interface, but found none."):
            parser.parse_file(path)

    def test_multiple_axi_lite_interfaces(self, parser, temp_sv_file):
        content = """
        module test (
            input logic ap_clk, input logic ap_rst_n,
            // AXI-Stream
            input logic [31:0] in0_TDATA, input logic in0_TVALID, output logic in0_TREADY,
             // AXI-Lite 1 (config)
            input  logic [15:0] config_AWADDR, input  logic config_AWVALID, output logic config_AWREADY,
            input  logic [31:0] config_WDATA, input  logic config_WVALID, output logic config_WREADY,
            output logic config_BVALID, input  logic config_BREADY,
            input  logic [15:0] config_ARADDR, input  logic config_ARVALID, output logic config_ARREADY,
            output logic [31:0] config_RDATA, output logic config_RVALID, input  logic config_RREADY,
             // AXI-Lite 2 (control) - Not allowed by current rules
            input  logic [15:0] control_AWADDR, input  logic control_AWVALID, output logic control_AWREADY,
            input  logic [31:0] control_WDATA, input  logic control_WVALID, output logic control_WREADY,
            output logic control_BVALID, input  logic control_BREADY,
            input  logic [15:0] control_ARADDR, input  logic control_ARVALID, output logic control_ARREADY,
            output logic [31:0] control_RDATA, output logic control_RVALID, input  logic control_RREADY
        );
        endmodule
        """
        path = temp_sv_file(content)
        # The second AXI-Lite ('control_*') ports will be treated as unassigned first
        with pytest.raises(ParserError, match=r"ports not assigned to any valid interface"):
             parser.parse_file(path)
        # If the scanner/validator were adjusted to allow multiple AXI-Lite groups initially,
        # the post-analysis check in the parser would catch it:
        # with pytest.raises(ParserError, match=r"allows at most one AXI-Lite interface, but found 2"):
        #     parser.parse_file(path)


class TestErrorHandling:
    """Tests for various error conditions."""

    def test_missing_file(self, parser):
        """Test parsing a non-existent file."""
        with pytest.raises(FileNotFoundError):
            parser.parse_file("non_existent_file.sv")

    def test_invalid_syntax(self, parser, temp_sv_file):
        """Test parsing a file with SystemVerilog syntax errors."""
        content = """
        module test (
            input logic clk // Missing semicolon
            output logic [7:0] data
        );
        endmodule
        """
        path = temp_sv_file(content)
        with pytest.raises(SyntaxError, match=r"Invalid SystemVerilog syntax near line"):
            parser.parse_file(path)

    def test_no_module_definition(self, parser, temp_sv_file):
        """Test parsing a file with no module definition."""
        content = "// Just comments\nparameter X = 1;"
        path = temp_sv_file(content)
        with pytest.raises(ParserError, match="No module definition found"):
            parser.parse_file(path)

    def test_multiple_modules_no_pragma(self, parser, temp_sv_file):
        """Test multiple modules without TOP_MODULE pragma."""
        content = """
        module mod1(); endmodule
        module mod2(); endmodule
        """
        path = temp_sv_file(content)
        with pytest.raises(ParserError, match=r"Multiple modules .* found .* but no TOP_MODULE pragma specified"):
            parser.parse_file(path)

    def test_multiple_modules_pragma_mismatch(self, parser, temp_sv_file):
        """Test TOP_MODULE pragma pointing to non-existent module."""
        content = """
        // @brainsmith TOP_MODULE target
        module mod1(); endmodule
        module mod2(); endmodule
        """
        path = temp_sv_file(content)
        with pytest.raises(ParserError, match=r"TOP_MODULE pragma specified 'target', but no such module found"):
            parser.parse_file(path)

    def test_multiple_top_module_pragmas(self, parser, temp_sv_file):
        """Test multiple TOP_MODULE pragmas."""
        content = """
        // @brainsmith TOP_MODULE mod1
        // @brainsmith TOP_MODULE mod2
        module mod1(); endmodule
        module mod2(); endmodule
        """
        path = temp_sv_file(content)
        with pytest.raises(ParserError, match="Multiple TOP_MODULE pragmas found"):
            parser.parse_file(path)

    def test_single_module_pragma_mismatch(self, parser, temp_sv_file):
        """Test single module with mismatching TOP_MODULE pragma."""
        content = """
        // @brainsmith TOP_MODULE wrong_name
        module actual_name (
             input logic ap_clk, input logic ap_rst_n,
             input logic [31:0] in0_TDATA, input logic in0_TVALID, output logic in0_TREADY
        );
        endmodule
        """
        path = temp_sv_file(content)
        with pytest.raises(ParserError, match=r"TOP_MODULE pragma specifies 'wrong_name', but the only module found is 'actual_name'"):
            parser.parse_file(path)

