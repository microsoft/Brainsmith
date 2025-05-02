"""Test suite for interface analysis functionality."""

import os
import logging
import ctypes
from ctypes import c_void_p, c_char_p, py_object, pythonapi
import pytest
from tree_sitter import Language, Parser

from brainsmith.tools.hw_kernel_gen.rtl_parser.interface import (
    parse_parameter_declaration,
    parse_port_declaration,
    extract_module_header,
    _find_child,
    _debug_node
)
from brainsmith.tools.hw_kernel_gen.rtl_parser.data import Direction

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger(__name__)

# Grammar file path
GRAMMAR_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "sv.so"
)

def create_parser():
    """Create tree-sitter parser with SystemVerilog grammar."""
    # 1. Load the shared object
    lib = ctypes.cdll.LoadLibrary(GRAMMAR_PATH)

    # 2. Get language pointer
    lang_ptr = lib.tree_sitter_verilog
    lang_ptr.restype = c_void_p
    lang_ptr = lang_ptr()

    # 3. Create Python capsule
    PyCapsule_New = pythonapi.PyCapsule_New
    PyCapsule_New.restype = py_object
    PyCapsule_New.argtypes = (c_void_p, c_char_p, c_void_p)
    capsule = PyCapsule_New(lang_ptr, b"tree_sitter.Language", None)

    # 4. Create parser with language
    language = Language(capsule)
    return Parser(language)

@pytest.fixture
def parser():
    """Fixture to create parser."""
    return create_parser()

def parse_sv(parser: Parser, content: str):
    """Parse SystemVerilog content into AST."""
    logger.debug(f"\nParsing content: {content}")
    tree = parser.parse(bytes(content, "utf8"))
    logger.debug("\nFull parse tree:")
    _debug_node(tree.root_node)
    return tree

def dump_node_types(node, types_seen=None, prefix=""):
    """Recursively dump all unique node types in tree."""
    if types_seen is None:
        types_seen = set()
    
    if node is None:
        return types_seen
    
    types_seen.add(node.type)
    text = node.text.decode('utf8').replace('\n', '\\n')
    logger.debug(f"{prefix}{node.type}: '{text}'")
    
    for child in node.children:
        dump_node_types(child, types_seen, prefix + "  ")
    
    return types_seen

class TestPortParsing:
    """Test port declaration parsing."""
    
    def test_basic_port(self, parser):
        """Test basic port parsing."""
        content = "input logic clk;"
        tree = parse_sv(parser, content)
        
        # Print all node types in tree
        logger.debug("\nAll node types in tree:")
        node_types = dump_node_types(tree.root_node)
        logger.debug(f"\nFound node types: {sorted(node_types)}")
        
        # Try finding the node with various possible types
        node_types = [
            "port_declaration",
            "ansi_port_declaration",
            "net_declaration",
            "data_declaration",
            "variable_declaration",
            "data_definition",
            "logic_declaration",
            "variable_port_declaration",
            "data_type_declaration",
            "statement",
            "simple_declaration"
        ]
        logger.debug("\nLooking for port declaration node types: %s", node_types)
        node = _find_child(tree.root_node, node_types)
        if node:
            logger.debug("\nFound node:")
            _debug_node(node)
            logger.debug("\nNode children:")
            for child in node.children:
                logger.debug(f"  {child.type}: '{child.text.decode('utf8')}'")
        else:
            logger.debug("\nNo matching node found in tree")
            logger.debug("\nRoot node children:")
            for child in tree.root_node.children:
                logger.debug(f"  {child.type}: '{child.text.decode('utf8')}'")
        
        # Try parsing the root node's children directly
        for child in tree.root_node.children:
            logger.debug(f"\nTrying to parse node: {child.type}")
            port = parse_port_declaration(child)
            if port is not None:
                logger.debug("Successfully parsed port!")
                break
        
        assert port is not None
        assert port.name == "clk"
        assert port.direction == Direction.INPUT
        assert port.width == "1"
    
    def test_output_port_with_width(self, parser):
        """Test output port with explicit width."""
        content = "output logic [7:0] data;"
        tree = parse_sv(parser, content)
        
        # Print all node types in tree
        logger.debug("\nAll node types in tree:")
        node_types = dump_node_types(tree.root_node)
        logger.debug(f"\nFound node types: {sorted(node_types)}")
        
        node = _find_child(tree.root_node, [
            "port_declaration",
            "ansi_port_declaration",
            "net_declaration",
            "data_declaration",
            "variable_declaration",
            "data_definition",
            "logic_declaration",
            "variable_port_declaration",
            "data_type_declaration",
            "statement",
            "simple_declaration"
        ])
        if node:
            logger.debug("\nFound node:")
            _debug_node(node)
        port = parse_port_declaration(node)
        
        assert port is not None
        assert port.name == "data"
        assert port.direction == Direction.OUTPUT
        assert port.width == "7:0"
    
    def test_parametric_width(self, parser):
        """Test port with parametric width."""
        content = "output logic [WIDTH-1:0] data;"
        tree = parse_sv(parser, content)
        
        # Print all node types in tree
        logger.debug("\nAll node types in tree:")
        node_types = dump_node_types(tree.root_node)
        logger.debug(f"\nFound node types: {sorted(node_types)}")
        
        node = _find_child(tree.root_node, [
            "port_declaration",
            "ansi_port_declaration",
            "net_declaration",
            "data_declaration",
            "variable_declaration",
            "data_definition",
            "logic_declaration",
            "variable_port_declaration",
            "data_type_declaration",
            "statement",
            "simple_declaration"
        ])
        if node:
            logger.debug("\nFound node:")
            _debug_node(node)
        port = parse_port_declaration(node)
        
        assert port is not None
        assert port.name == "data"
        assert port.width == "WIDTH-1:0"
    
    def test_complex_width(self, parser):
        """Test port with complex width expression."""
        content = "output logic [(2*WIDTH/4)-1:0] data;"
        tree = parse_sv(parser, content)
        
        # Print all node types in tree
        logger.debug("\nAll node types in tree:")
        node_types = dump_node_types(tree.root_node)
        logger.debug(f"\nFound node types: {sorted(node_types)}")
        
        node = _find_child(tree.root_node, [
            "port_declaration",
            "ansi_port_declaration",
            "net_declaration",
            "data_declaration",
            "variable_declaration",
            "data_definition",
            "logic_declaration",
            "variable_port_declaration",
            "data_type_declaration",
            "statement", 
            "simple_declaration"
        ])
        if node:
            logger.debug("\nFound node:")
            _debug_node(node)
        port = parse_port_declaration(node)
        
        assert port is not None
        assert port.width == "(2*WIDTH/4)-1:0"