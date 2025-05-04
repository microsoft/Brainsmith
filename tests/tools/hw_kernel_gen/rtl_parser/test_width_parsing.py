# /home/tafk/dev/brainsmith/tests/tools/hw_kernel_gen/rtl_parser/test_width_parsing.py
import pytest
import logging
from tree_sitter import Node

# Assuming RTLParser and test fixtures are accessible (adjust path if needed)
# Need access to the parser instance to call the protected method and the tree-sitter parser
from brainsmith.tools.hw_kernel_gen.rtl_parser.parser import RTLParser
# Import fixture from conftest or main test file
# from .test_rtl_parser import parser # Example if fixture is in test_rtl_parser.py

# Configure logging for debugging if necessary
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG) # Uncomment for detailed logs

# Helper function to parse a snippet and get the dimension node
def get_dimension_node(parser_instance: RTLParser, type_snippet: str) -> Node | None:
    """
    Parses a minimal type snippet (e.g., 'logic [7:0]') and returns the
    packed_dimension or unpacked_dimension node.
    """
    # Wrap in minimal context for parsing
    # Using a dummy variable declaration context
    full_code = f"module dummy; {type_snippet} dummy_var; endmodule"
    logger.debug(f"Parsing snippet for dimension node: {full_code}")
    try:
        tree = parser_instance.parser.parse(bytes(full_code, 'utf8'))
        root_node = tree.root_node

        if root_node.has_error:
            logger.error("Syntax error in dimension snippet.")
            error_node = parser_instance._find_first_error_node(root_node)
            if error_node:
                 logger.error(f"Error near text: {error_node.text.decode()}")
            return None

        # Navigate to the dimension node
        # Path: source_file -> module_declaration -> module_body -> data_declaration
        #      -> data_type_or_implicit -> data_type -> packed_dimension (or similar)
        # This path might vary slightly based on the exact snippet and grammar
        queue = [root_node]
        visited = {root_node.id}
        dimension_node = None
        while queue:
            current_node = queue.pop(0)
            if current_node.type in ["packed_dimension", "unpacked_dimension"]:
                dimension_node = current_node
                break
            for child in current_node.children:
                if child.id not in visited:
                    visited.add(child.id)
                    queue.append(child)

        if not dimension_node:
            logger.error(f"Could not find dimension node in snippet: {type_snippet}")
            # Optionally print AST for debugging
            # parser_instance._debug_node(root_node, max_depth=10)
            return None

        logger.debug(f"Found dimension node: Type={dimension_node.type}, Text='{dimension_node.text.decode()}'")
        return dimension_node

    except Exception as e:
        logger.exception(f"Exception during dimension snippet parsing: {e}")
        return None


class TestWidthExtraction:
    """Tests focused specifically on the _extract_width_from_dimension method."""

    @pytest.mark.parametrize("type_snippet, expected_width", [
        ("logic [7:0]", "7:0"),
        ("logic [0:0]", "0:0"),
        ("logic [31:0]", "31:0"),
        ("wire signed [15:0]", "15:0"),
        # Parametric widths
        ("logic [WIDTH-1:0]", "WIDTH-1:0"),
        ("logic [(WIDTH*2)-1:0]", "(WIDTH*2)-1:0"), # Assuming parens are kept
        ("logic [WIDTH/C:0]", "WIDTH/C:0"),
        ("logic [$clog2(DEPTH)-1:0]", "$clog2(DEPTH)-1:0"),
        # REMOVED: Single number test case based on invalid syntax
        # ("logic [7]", "7"),
    ])
    def test_packed_dimension_extraction(self, parser: RTLParser, type_snippet, expected_width):
        """Test width extraction from various packed dimension snippets."""
        dimension_node = get_dimension_node(parser, type_snippet)
        assert dimension_node is not None, f"Failed to parse snippet: {type_snippet}"

        extracted_width = parser._extract_width_from_dimension(dimension_node)
        assert extracted_width == expected_width

    def test_no_dimension_node(self, parser: RTLParser):
        """Test the function's behavior when passed None."""
        # The default '1' is typically handled by the caller (_parse_port_declaration)
        # but we can test the direct function call with None
        extracted_width = parser._extract_width_from_dimension(None)
        assert extracted_width == "1"

    # TODO: Add tests for unpacked dimensions if needed, requires adjusting get_dimension_node
    # def test_unpacked_dimension_extraction(self, parser: RTLParser):
    #     type_snippet = "logic [7:0] data [3:0]" # Unpacked part is [3:0]
    #     dimension_node = get_dimension_node(parser, type_snippet) # Needs adjustment
    #     assert dimension_node is not None
    #     extracted_width = parser._extract_width_from_dimension(dimension_node)
    #     assert extracted_width == "3:0"
