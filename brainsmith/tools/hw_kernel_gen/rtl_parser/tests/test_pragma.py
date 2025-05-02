"""Test suite for pragma processing functionality."""

import os
import ctypes
from ctypes import c_void_p, c_char_p, py_object, pythonapi
import pytest
from tree_sitter import Language, Parser

from brainsmith.tools.hw_kernel_gen.rtl_parser.pragma import (
    PragmaParser,
    PragmaType,
    PragmaError,
    extract_pragmas
)

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

@pytest.fixture
def pragma_parser():
    """Fixture to create pragma parser."""
    return PragmaParser(debug=True)

def parse_sv(parser: Parser, content: str):
    """Parse SystemVerilog content into AST."""
    return parser.parse(bytes(content, "utf8"))

class TestPragmaParser:
    """Test pragma parser functionality."""
    
    def test_interface_pragma(self, parser, pragma_parser):
        """Test parsing of interface pragma."""
        content = "// @brainsmith interface AXI_STREAM"
        tree = parse_sv(parser, content)
        pragma = pragma_parser.parse_comment(tree.root_node, 1)
        
        assert pragma is not None
        assert pragma.type == PragmaType.INTERFACE.value
        assert pragma.inputs == ["AXI_STREAM"]
        
        # Check processed data
        assert "protocol" in pragma.processed_data
        assert pragma.processed_data["protocol"] == "AXI_STREAM"
    
    def test_parameter_pragma(self, parser, pragma_parser):
        """Test parsing of parameter pragma."""
        content = "// @brainsmith parameter STATIC WIDTH"
        tree = parse_sv(parser, content)
        pragma = pragma_parser.parse_comment(tree.root_node, 1)
        
        assert pragma is not None
        assert pragma.type == PragmaType.PARAMETER.value
        assert pragma.inputs == ["STATIC", "WIDTH"]
        
        # Check processed data
        assert "param_type" in pragma.processed_data
        assert pragma.processed_data["param_type"] == "STATIC"
        assert pragma.processed_data["name"] == "WIDTH"
    
    def test_resource_pragma(self, parser, pragma_parser):
        """Test parsing of resource pragma."""
        content = "// @brainsmith resource DSP 4"
        tree = parse_sv(parser, content)
        pragma = pragma_parser.parse_comment(tree.root_node, 1)
        
        assert pragma is not None
        assert pragma.type == PragmaType.RESOURCE.value
        assert pragma.inputs == ["DSP", "4"]
        
        # Check processed data
        assert "resource_type" in pragma.processed_data
        assert pragma.processed_data["resource_type"] == "DSP"
        assert pragma.processed_data["value"] == "4"
    
    def test_timing_pragma(self, parser, pragma_parser):
        """Test parsing of timing pragma."""
        content = "// @brainsmith timing LATENCY 2"
        tree = parse_sv(parser, content)
        pragma = pragma_parser.parse_comment(tree.root_node, 1)
        
        assert pragma is not None
        assert pragma.type == PragmaType.TIMING.value
        assert pragma.inputs == ["LATENCY", "2"]
        
        # Check processed data
        assert "constraint" in pragma.processed_data
        assert pragma.processed_data["constraint"] == "LATENCY"
        assert pragma.processed_data["value"] == "2"
    
    def test_feature_pragma(self, parser, pragma_parser):
        """Test parsing of feature pragma."""
        content = "// @brainsmith feature PIPELINE enabled"
        tree = parse_sv(parser, content)
        pragma = pragma_parser.parse_comment(tree.root_node, 1)
        
        assert pragma is not None
        assert pragma.type == PragmaType.FEATURE.value
        assert pragma.inputs == ["PIPELINE", "enabled"]
        
        # Check processed data
        assert "name" in pragma.processed_data
        assert pragma.processed_data["name"] == "PIPELINE"
        assert pragma.processed_data["enabled"] is True
    
    def test_invalid_pragma_type(self, parser, pragma_parser):
        """Test handling of invalid pragma type."""
        content = "// @brainsmith invalid_type"
        tree = parse_sv(parser, content)
        pragma = pragma_parser.parse_comment(tree.root_node, 1)
        
        assert pragma is None
    
    def test_missing_inputs(self, parser, pragma_parser):
        """Test handling of missing required inputs."""
        content = "// @brainsmith interface"
        tree = parse_sv(parser, content)
        pragma = pragma_parser.parse_comment(tree.root_node, 1)
        
        assert pragma is None

def test_extract_pragmas(parser):
    """Test extraction of all pragmas from file."""
    content = """
    // @brainsmith interface AXI_STREAM
    // @brainsmith parameter STATIC WIDTH
    // Regular comment
    module test_kernel (
        input clk
    );
    // @brainsmith resource DSP 4
    endmodule
    """
    tree = parse_sv(parser, content)
    pragmas = extract_pragmas(tree.root_node)
    
    assert len(pragmas) == 3
    
    # Check pragma types
    types = [p.type for p in pragmas]
    assert PragmaType.INTERFACE.value in types
    assert PragmaType.PARAMETER.value in types
    assert PragmaType.RESOURCE.value in types
    
    # Check line numbers are sequential
    lines = [p.line_number for p in pragmas]
    assert lines == sorted(lines)

def test_pragma_error():
    """Test PragmaError handling."""
    with pytest.raises(PragmaError):
        raise PragmaError("Test error")

def test_pragma_type_values():
    """Test PragmaType enumeration values."""
    assert PragmaType.INTERFACE.value == "interface"
    assert PragmaType.PARAMETER.value == "parameter"
    assert PragmaType.RESOURCE.value == "resource"
    assert PragmaType.TIMING.value == "timing"
    assert PragmaType.FEATURE.value == "feature"