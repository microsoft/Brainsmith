# Create the new test file /home/tafk/dev/brainsmith/tests/tools/hw_kernel_gen/rtl_parser/debug_port_parsing.py
import pytest
import logging
import os
from tree_sitter import Language, Parser, Node

# Assuming parser and data structures are importable
# Adjust imports based on your project structure if necessary
from brainsmith.tools.hw_kernel_gen.rtl_parser.parser import RTLParser, ParserError, SyntaxError
from brainsmith.tools.hw_kernel_gen.rtl_parser.data import Port, Direction, Parameter # Import Parameter if needed for context
from brainsmith.tools.hw_kernel_gen.rtl_parser.interface_types import InterfaceType # Import if needed

# --- Logging Setup ---
# Configure logging to show debug messages for this test file
log_file = "debug_port_parsing.log"
if os.path.exists(log_file):
    os.remove(log_file) # Clear log file on each run

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file), # Log to file
        logging.StreamHandler()        # Log to console
    ]
)
logger = logging.getLogger(__name__)
# Also configure the parser's logger if it uses standard logging
parser_logger = logging.getLogger('brainsmith.tools.hw_kernel_gen.rtl_parser.parser')
parser_logger.setLevel(logging.DEBUG)
# Add handlers if not already configured by the parser's module
if not parser_logger.handlers:
     parser_logger.addHandler(logging.FileHandler(log_file))
     parser_logger.addHandler(logging.StreamHandler())


# --- Test Fixture for Parser ---
@pytest.fixture(scope="module")
def sv_parser():
    """Fixture to provide an initialized RTLParser instance."""
    # Assuming grammar path logic is handled within RTLParser constructor
    try:
        # Enable debug mode in the parser itself for its internal logs
        parser = RTLParser(debug=True)
        # Ensure the parser's logger is captured by our handlers
        parser_logger.propagate = False # Prevent duplicate messages if root logger also has handlers
        return parser
    except (FileNotFoundError, RuntimeError) as e:
        pytest.fail(f"Failed to initialize RTLParser: {e}")

# --- Helper Function ---
def parse_snippet_and_get_port_node(parser_instance: RTLParser, code_snippet: str) -> Node | None:
    """
    Parses a minimal Verilog module snippet and returns the AST node
    for the first port declaration found.
    """
    # Wrap snippet in a minimal module for valid parsing context
    full_code = f"""
module test_module (
    {code_snippet}
);
endmodule
"""
    logger.debug(f"--- Parsing Snippet ---\n{full_code.strip()}")
    try:
        tree = parser_instance.parser.parse(bytes(full_code, 'utf8'))
        root_node = tree.root_node

        if root_node.has_error:
            logger.error("Syntax error in snippet.")
            # Find and log error location if possible
            error_node = parser_instance._find_first_error_node(root_node)
            if error_node:
                 logger.error(f"Error near line {error_node.start_point[0]}, col {error_node.start_point[1]}")
                 logger.error(f"Error node type: {error_node.type}, text: {error_node.text.decode()}")
            return None

        # Navigate to the port declaration node
        # Path: source_file -> module_declaration -> module_ansi_header -> port_declaration (or similar)
        module_decl = parser_instance._find_child(root_node, ["module_declaration"])
        if not module_decl: logger.error("Could not find module_declaration."); return None

        module_header = parser_instance._find_child(module_decl, ["module_ansi_header"])
        if not module_header: logger.error("Could not find module_ansi_header."); return None

        # Find the first relevant port declaration node within the header
        port_node = parser_instance._find_child(module_header, ["port_declaration", "ansi_port_declaration", "net_port_header", "variable_port_header"])
        if not port_node: logger.error("Could not find port declaration node in header."); return None

        logger.debug(f"Successfully found port node of type: {port_node.type}")
        return port_node

    except Exception as e:
        logger.exception(f"Exception during snippet parsing: {e}")
        return None


# --- Test Class ---
class TestPortParsingDebug:

    def test_simple_ansi_input(self, sv_parser: RTLParser):
        """Test parsing a simple ANSI-style input port."""
        snippet = "input logic clk"
        port_node = parse_snippet_and_get_port_node(sv_parser, snippet)
        assert port_node is not None, "Failed to parse snippet or find port node"

        logger.info(f"--- Testing: {snippet} ---")
        logger.debug("Input Port Node Structure:")
        sv_parser._debug_node(port_node, prefix="  ") # Log the node structure

        parsed_port = sv_parser._parse_port_declaration(port_node)

        logger.debug(f"Parsed Port Object: {parsed_port}")
        assert parsed_port is not None
        assert parsed_port.name == "clk"
        assert parsed_port.direction == Direction.INPUT
        assert parsed_port.width == "1" # Default width

    def test_simple_ansi_output_vector(self, sv_parser: RTLParser):
        """Test parsing an ANSI-style output vector port."""
        snippet = "output logic [7:0] data_out"
        port_node = parse_snippet_and_get_port_node(sv_parser, snippet)
        assert port_node is not None, "Failed to parse snippet or find port node"

        logger.info(f"--- Testing: {snippet} ---")
        logger.debug("Input Port Node Structure:")
        sv_parser._debug_node(port_node, prefix="  ")

        parsed_port = sv_parser._parse_port_declaration(port_node)

        logger.debug(f"Parsed Port Object: {parsed_port}")
        assert parsed_port is not None
        assert parsed_port.name == "data_out"
        assert parsed_port.direction == Direction.OUTPUT
        assert parsed_port.width == "7:0" # Check exact width string

    def test_ansi_input_vector_param_width(self, sv_parser: RTLParser):
        """Test parsing an ANSI-style input vector with parameter width."""
        snippet = "input logic [WIDTH-1:0] data_in"
        port_node = parse_snippet_and_get_port_node(sv_parser, snippet)
        assert port_node is not None, "Failed to parse snippet or find port node"

        logger.info(f"--- Testing: {snippet} ---")
        logger.debug("Input Port Node Structure:")
        sv_parser._debug_node(port_node, prefix="  ")

        parsed_port = sv_parser._parse_port_declaration(port_node)

        logger.debug(f"Parsed Port Object: {parsed_port}")
        assert parsed_port is not None
        assert parsed_port.name == "data_in"
        assert parsed_port.direction == Direction.INPUT
        assert parsed_port.width == "WIDTH-1:0" # Check exact width string

    def test_ansi_inout_wire(self, sv_parser: RTLParser):
        """Test parsing an ANSI-style inout wire port."""
        snippet = "inout wire data_bus" # Type 'wire' instead of 'logic'
        port_node = parse_snippet_and_get_port_node(sv_parser, snippet)
        assert port_node is not None, "Failed to parse snippet or find port node"

        logger.info(f"--- Testing: {snippet} ---")
        logger.debug("Input Port Node Structure:")
        sv_parser._debug_node(port_node, prefix="  ")

        parsed_port = sv_parser._parse_port_declaration(port_node)

        logger.debug(f"Parsed Port Object: {parsed_port}")
        assert parsed_port is not None
        assert parsed_port.name == "data_bus"
        assert parsed_port.direction == Direction.INOUT
        assert parsed_port.width == "1"
        # Add assertion for data_type if it's stored in Port object later
        # assert parsed_port.data_type == "wire"

    def test_ansi_implicit_type(self, sv_parser: RTLParser):
        """Test parsing an ANSI-style port with implicit type (should default to logic)."""
        snippet = "input [3:0] flags" # No explicit type like 'logic' or 'wire'
        port_node = parse_snippet_and_get_port_node(sv_parser, snippet)
        assert port_node is not None, "Failed to parse snippet or find port node"

        logger.info(f"--- Testing: {snippet} ---")
        logger.debug("Input Port Node Structure:")
        sv_parser._debug_node(port_node, prefix="  ")

        parsed_port = sv_parser._parse_port_declaration(port_node)

        logger.debug(f"Parsed Port Object: {parsed_port}")
        assert parsed_port is not None
        assert parsed_port.name == "flags"
        assert parsed_port.direction == Direction.INPUT
        assert parsed_port.width == "3:0"
        # Add assertion for data_type if it's stored in Port object later
        # assert parsed_port.data_type == "logic" # Default

    def test_ansi_port_with_comma(self, sv_parser: RTLParser):
        """Test parsing the first port when there's a comma for the next one."""
        # Provide a valid two-port snippet to test parsing the first one correctly.
        snippet = "input logic clk, input logic rst" # Added a second port
        port_node = parse_snippet_and_get_port_node(sv_parser, snippet)
        assert port_node is not None, "Failed to parse snippet or find port node"

        logger.info(f"--- Testing: {snippet} ---")
        logger.debug("Input Port Node Structure (First Port):")
        sv_parser._debug_node(port_node, prefix="  ")

        parsed_port = sv_parser._parse_port_declaration(port_node)

        logger.debug(f"Parsed Port Object: {parsed_port}")
        assert parsed_port is not None
        # Assertions remain the same, checking the first port (clk)
        assert parsed_port.name == "clk"
        assert parsed_port.direction == Direction.INPUT
        assert parsed_port.width == "1"

    # --- Add more test cases as needed ---
    # Examples:
    # - Ports with complex types (structs, interfaces - might require grammar updates/more complex parsing)
    # - Non-ANSI style ports (would require adjusting parse_snippet_and_get_port_node)
    # - Edge cases with unusual spacing or identifiers
