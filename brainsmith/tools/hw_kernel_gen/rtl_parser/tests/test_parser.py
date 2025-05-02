"""Test suite for RTL Parser."""

import os
import pytest

from brainsmith.tools.hw_kernel_gen.rtl_parser import (
    RTLParser,
    HWKernel,
    Parameter,
    Port,
    Pragma
)
from brainsmith.tools.hw_kernel_gen.rtl_parser.parser import ParserError, SyntaxError

# Grammar file path
GRAMMAR_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "sv.so"
)

# Test fixtures directory
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

def create_test_file(content: str, name: str = "test.sv") -> str:
    """Create a test file with given content.
    
    Args:
        content: File content
        name: File name
        
    Returns:
        Path to created file
    """
    if not os.path.exists(FIXTURES_DIR):
        os.makedirs(FIXTURES_DIR)
    
    path = os.path.join(FIXTURES_DIR, name)
    with open(path, "w") as f:
        f.write(content)
    return path

@pytest.fixture
def parser():
    """Create RTLParser instance."""
    return RTLParser(grammar_path=GRAMMAR_PATH, debug=True)

def test_basic_module(parser):
    """Test parsing of basic module."""
    content = """
    module test_kernel (
        input logic clk,
        input logic rst_n,
        output logic [7:0] data
    );
    endmodule
    """
    
    path = create_test_file(content)
    kernel = parser.parse_file(path)
    
    assert kernel.name == "test_kernel"
    assert len(kernel.ports) == 3
    
    # Check ports
    port_map = {p.name: p for p in kernel.ports}
    assert "clk" in port_map
    assert "rst_n" in port_map
    assert "data" in port_map
    
    assert port_map["data"].width == "7:0"

def test_module_with_parameters(parser):
    """Test parsing of module with parameters."""
    content = """
    module test_kernel #(
        parameter WIDTH = 32,
        parameter SIGNED = 0
    ) (
        input logic clk,
        output logic [WIDTH-1:0] data
    );
    endmodule
    """
    
    path = create_test_file(content)
    kernel = parser.parse_file(path)
    
    assert len(kernel.parameters) == 2
    
    # Check parameters
    param_map = {p.name: p for p in kernel.parameters}
    assert "WIDTH" in param_map
    assert "SIGNED" in param_map
    
    assert param_map["WIDTH"].default_value == "32"
    assert param_map["SIGNED"].default_value == "0"

def test_module_with_pragmas(parser):
    """Test parsing of module with pragmas."""
    content = """
    // @brainsmith interface AXI_STREAM
    // @brainsmith parameter STATIC WIDTH
    module test_kernel #(
        parameter WIDTH = 32
    ) (
        input logic clk,
        output logic [WIDTH-1:0] data
    );
    endmodule
    """
    
    path = create_test_file(content)
    kernel = parser.parse_file(path)
    
    assert len(kernel.pragmas) == 2
    
    # Check pragmas
    pragmas = {p.type: p for p in kernel.pragmas}
    assert "interface" in pragmas
    assert "parameter" in pragmas
    
    interface_pragma = pragmas["interface"]
    assert interface_pragma.inputs == ["AXI_STREAM"]
    
    param_pragma = pragmas["parameter"]
    assert param_pragma.inputs == ["STATIC", "WIDTH"]

def test_invalid_syntax(parser):
    """Test handling of invalid SystemVerilog syntax."""
    content = """
    module test_kernel (
        input logic clk,
        output logic [7:0] data
    // Missing endmodule
    """
    
    path = create_test_file(content)
    with pytest.raises(SyntaxError):
        parser.parse_file(path)

def test_missing_file(parser):
    """Test handling of missing file."""
    with pytest.raises(FileNotFoundError):
        parser.parse_file("nonexistent.sv")

def test_empty_module(parser):
    """Test parsing of empty module."""
    content = """
    module empty;
    endmodule
    """
    
    path = create_test_file(content)
    kernel = parser.parse_file(path)
    
    assert kernel.name == "empty"
    assert len(kernel.parameters) == 0
    assert len(kernel.ports) == 0
    assert len(kernel.pragmas) == 0

def test_local_parameters(parser):
    """Test that local parameters are ignored."""
    content = """
    module test_kernel #(
        parameter WIDTH = 32,
        localparam DEPTH = WIDTH * 2
    ) (
        input logic clk
    );
    endmodule
    """
    
    path = create_test_file(content)
    kernel = parser.parse_file(path)
    
    assert len(kernel.parameters) == 1
    assert kernel.parameters[0].name == "WIDTH"

def test_complex_port_widths(parser):
    """Test parsing of complex port width expressions."""
    content = """
    module test_kernel #(
        parameter WIDTH = 32
    ) (
        input logic clk,
        output logic [WIDTH-1:0] data,
        output logic [2*WIDTH-1:0] wide_data,
        output logic [(WIDTH/2)-1:0] narrow_data
    );
    endmodule
    """
    
    path = create_test_file(content)
    kernel = parser.parse_file(path)
    
    port_map = {p.name: p for p in kernel.ports}
    
    assert port_map["data"].width == "WIDTH-1:0"
    assert port_map["wide_data"].width == "2*WIDTH-1:0"
    assert port_map["narrow_data"].width == "(WIDTH/2)-1:0"

def test_multiple_pragmas_same_type(parser):
    """Test handling of multiple pragmas of same type."""
    content = """
    // @brainsmith interface AXI_STREAM
    // @brainsmith interface AXI_LITE
    module test_kernel (
        input logic clk
    );
    endmodule
    """
    
    path = create_test_file(content)
    kernel = parser.parse_file(path)
    
    interface_pragmas = [p for p in kernel.pragmas if p.type == "interface"]
    assert len(interface_pragmas) == 2
    
    protocols = [p.inputs[0] for p in interface_pragmas]
    assert "AXI_STREAM" in protocols
    assert "AXI_LITE" in protocols

def test_cleanup():
    """Clean up test fixtures."""
    import shutil
    if os.path.exists(FIXTURES_DIR):
        shutil.rmtree(FIXTURES_DIR)