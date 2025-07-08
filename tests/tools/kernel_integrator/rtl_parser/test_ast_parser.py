############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Unit tests for the AST Parser component.

Tests the Tree-sitter SystemVerilog parsing foundation including:
- Valid SystemVerilog syntax parsing
- Syntax error detection and reporting
- AST node traversal and queries
- Performance with large files
"""

import pytest
from pathlib import Path

from brainsmith.tools.kernel_integrator.rtl_parser.ast_parser import ASTParser
from brainsmith.tools.kernel_integrator.rtl_parser.ast_parser import SyntaxError

from .utils.rtl_builder import RTLBuilder, create_minimal_module
from .utils.ast_patterns import ASTPatterns


class TestASTParser:
    """Test cases for AST Parser functionality."""
    
    def test_parse_minimal_module(self, ast_parser):
        """Test parsing a minimal valid module."""
        rtl = create_minimal_module()
        
        # Parse the RTL
        tree = ast_parser.parse_source(rtl)
        
        # Verify tree structure
        assert tree is not None
        assert tree.root_node is not None
        assert tree.root_node.type == "source_file"
        
        # Find module declaration
        modules = ast_parser.find_modules(tree)
        assert len(modules) == 1
        assert modules[0].type == "module_declaration"
    
    def test_parse_module_with_parameters(self, ast_parser):
        """Test parsing module with parameter declarations."""
        rtl = (RTLBuilder()
               .module("parametric")
               .parameter("WIDTH", "32")
               .parameter("DEPTH", "16", "integer")
               .parameter("SIGNED", "0", None)  # No type specified
               .port("clk", "input")
               .port("data", "input", "WIDTH")
               .build())
        
        tree = ast_parser.parse_source(rtl)
        modules = ast_parser.find_modules(tree)
        assert len(modules) == 1
        
        # Check parameter parsing
        module = modules[0]
        # Parameters are inside module_ansi_header
        header = ast_parser.find_child(module, ["module_ansi_header"])
        assert header is not None
        
        # Find parameter port list within header
        param_list = ast_parser.find_child(header, ["parameter_port_list"])
        assert param_list is not None
        
        # Count parameters - looking for parameter_port_declaration nodes
        params = ast_parser.find_children(param_list, ["parameter_port_declaration"])
        assert len(params) == 3
    
    def test_parse_module_with_ports(self, ast_parser):
        """Test parsing module with various port types."""
        rtl = (RTLBuilder()
               .module("port_test")
               .port("clk", "input")
               .port("rst_n", "input")
               .port("data_in", "input", "32")
               .port("data_out", "output", "32")
               .port("valid", "output")
               .port("bidir", "inout", "8")
               .build())
        
        tree = ast_parser.parse_source(rtl)
        modules = ast_parser.find_modules(tree)
        
        # Find port declarations
        module = modules[0]
        # Ports are inside module_ansi_header
        header = ast_parser.find_child(module, ["module_ansi_header"])
        assert header is not None
        
        port_list = ast_parser.find_child(header, ["list_of_port_declarations"])
        assert port_list is not None
        
        # Count ports - looking for ansi_port_declaration nodes
        ports = ast_parser.find_children(port_list, ["ansi_port_declaration"])
        assert len(ports) >= 6  # Each port should be separate
    
    def test_parse_syntax_error(self, ast_parser):
        """Test detection of syntax errors."""
        # Use ASTPatterns for malformed module
        rtl = ASTPatterns.malformed_module("missing_semicolon")
        
        tree = ast_parser.parse_source(rtl)
        
        # Check for syntax errors
        syntax_error = ast_parser.check_syntax_errors(tree)
        assert syntax_error is not None
        assert "syntax" in str(syntax_error).lower()
    
    def test_parse_incomplete_module(self, ast_parser):
        """Test parsing incomplete module (missing endmodule)."""
        # Use ASTPatterns for incomplete module
        rtl = ASTPatterns.malformed_module("incomplete_module")
        
        tree = ast_parser.parse_source(rtl)
        
        # Check for syntax errors
        syntax_error = ast_parser.check_syntax_errors(tree)
        assert syntax_error is not None
    
    def test_parse_large_file_performance(self, ast_parser, performance_timer):
        """Test parsing performance with large file."""
        # Generate a large module with many ports and parameters
        builder = RTLBuilder().module("large_module")
        
        # Add 100 parameters
        for i in range(100):
            builder.parameter(f"PARAM_{i}", str(i))
        
        # Add 200 ports
        for i in range(100):
            builder.port(f"input_{i}", "input", "32")
        for i in range(100):
            builder.port(f"output_{i}", "output", "32")
        
        # Add some body content
        for i in range(50):
            builder.assign(f"output_{i}", f"input_{i}")
        
        rtl = builder.build()
        
        # Time the parsing
        with performance_timer() as timer:
            tree = ast_parser.parse_source(rtl)
        
        # Should parse in under 100ms even for large files
        timer.assert_under(0.1)
        
        # Verify it parsed correctly
        modules = ast_parser.find_modules(tree)
        assert len(modules) == 1
    
    def test_parse_nested_modules(self, ast_parser):
        """Test parsing file with multiple modules."""
        # Build multi-module RTL using RTLBuilder
        rtl_top = (RTLBuilder()
                   .module("top")
                   .port("clk", "input")
                   .port("done", "output")
                   .body("// Implementation")
                   .build())
        
        rtl_sub = (RTLBuilder()
                   .module("sub")
                   .port("clk", "input")
                   .port("data", "input", "8")
                   .body("// Implementation")
                   .build())
        
        # Combine modules
        rtl = rtl_top + "\n\n" + rtl_sub
        
        tree = ast_parser.parse_source(rtl)
        modules = ast_parser.find_modules(tree)
        
        assert len(modules) == 2
        
        # Get module names
        names = []
        for module in modules:
            # Check in module header first
            header = ast_parser.find_child(module, ["module_ansi_header", "module_nonansi_header"])
            if header:
                name_node = ast_parser.find_child(header, ["simple_identifier"])
            else:
                name_node = ast_parser.find_child(module, ["simple_identifier"])
            
            if name_node:
                names.append(name_node.text.decode('utf8'))
        
        assert "top" in names
        assert "sub" in names
    
    def test_parse_with_comments(self, ast_parser):
        """Test parsing preserves comments for pragma extraction."""
        # Use ASTPatterns for comment variations
        rtl = ASTPatterns.comment_variations()
        
        tree = ast_parser.parse_source(rtl)
        
        # Comments should be accessible in the tree
        root = tree.root_node
        
        # Tree-sitter includes comments in the AST
        comment_count = 0
        
        def count_comments(node):
            nonlocal comment_count
            if node.type == "comment":
                comment_count += 1
            for child in node.children:
                count_comments(child)
        
        count_comments(root)
        assert comment_count >= 4  # Should find all comments
    
    def test_parse_complex_expressions(self, ast_parser):
        """Test parsing complex parameter expressions."""
        # Build module with complex expressions using RTLBuilder
        rtl = (RTLBuilder()
               .module("complex")
               .parameter("WIDTH", "32")
               .parameter("DEPTH", "WIDTH * 2")
               .parameter("SIZE", "(WIDTH + 7) / 8")
               .parameter("MASK", "(1 << WIDTH) - 1")
               .port("clk", "input")
               .port("data", "input", "WIDTH-1:0")
               .port("addr", "input", "$clog2(DEPTH)-1:0")
               .build())
        
        tree = ast_parser.parse_source(rtl)
        modules = ast_parser.find_modules(tree)
        assert len(modules) == 1
        
        # Should parse without syntax errors
        assert tree.root_node.has_error is False
    
    def test_find_child_by_type(self, ast_parser):
        """Test find_child_by_type utility method."""
        rtl = create_minimal_module()
        tree = ast_parser.parse_source(rtl)
        
        # Find module
        module = ast_parser.find_child(tree.root_node, ["module_declaration"])
        assert module is not None
        
        # Find something that doesn't exist
        nonexistent = ast_parser.find_child(tree.root_node, ["class_declaration"])
        assert nonexistent is None
    
    def test_find_children_by_type(self, ast_parser):
        """Test find_children_by_type utility method."""
        # Build module with multiple ports using RTLBuilder
        rtl = (RTLBuilder()
               .module("multi_port")
               .port("a", "input")
               .port("b", "input")
               .port("c", "input")
               .port("x", "output")
               .port("y", "output")
               .build())
        
        tree = ast_parser.parse_source(rtl)
        module = ast_parser.find_modules(tree)[0]
        
        # Find all port declarations
        header = ast_parser.find_child(module, ["module_ansi_header", "module_nonansi_header"])
        assert header is not None
        
        port_list = ast_parser.find_child(header, ["list_of_port_declarations"])
        assert port_list is not None
        
        ports = ast_parser.find_children(port_list, ["ansi_port_declaration"])
        
        # Should find multiple ports
        assert len(ports) >= 2
    
    def test_get_node_text(self, ast_parser):
        """Test extracting text from AST nodes."""
        # Use RTLBuilder for simple module
        rtl = (RTLBuilder()
               .module("test_name")
               .port("clk", "input")
               .build())
        tree = ast_parser.parse_source(rtl)
        
        module = ast_parser.find_modules(tree)[0]
        
        # Check in module header
        header = ast_parser.find_child(module, ["module_ansi_header", "module_nonansi_header"])
        if header:
            name_node = ast_parser.find_child(header, ["simple_identifier"])
        else:
            name_node = ast_parser.find_child(module, ["simple_identifier"])
        
        assert name_node is not None
        name_text = name_node.text.decode('utf8')
        assert name_text == "test_name"
    
    def test_parse_generate_blocks(self, ast_parser):
        """Test parsing generate blocks."""
        # Use ASTPatterns for generate blocks
        rtl = ASTPatterns.generate_blocks()
        
        tree = ast_parser.parse_source(rtl)
        assert tree.root_node.has_error is False
        
        # Should handle generate blocks without issues
        modules = ast_parser.find_modules(tree)
        assert len(modules) == 1
    
    def test_parse_interfaces(self, ast_parser):
        """Test parsing SystemVerilog interfaces (not module interfaces)."""
        # Use ASTPatterns for interfaces
        rtl = ASTPatterns.syntax_edge_case("interfaces")
        
        # Should parse without errors even with SV interfaces
        tree = ast_parser.parse_source(rtl)
        
        # Find the module (not the interface)
        modules = ast_parser.find_modules(tree)
        assert len(modules) == 1
        
        # Get module name
        module = modules[0]
        header = ast_parser.find_child(module, ["module_ansi_header", "module_nonansi_header"])
        if header:
            name_node = ast_parser.find_child(header, ["simple_identifier"])
        else:
            name_node = ast_parser.find_child(module, ["simple_identifier"])
            
        assert name_node is not None
        assert name_node.text.decode('utf8') == "using_interface"
    
    def test_empty_file(self, ast_parser):
        """Test parsing empty file."""
        tree = ast_parser.parse_source("")
        assert tree is not None
        assert tree.root_node is not None
        
        modules = ast_parser.find_modules(tree)
        assert len(modules) == 0
    
    def test_unicode_content(self, ast_parser):
        """Test parsing files with unicode characters."""
        # Use ASTPatterns for unicode identifiers
        rtl = ASTPatterns.unicode_identifiers()
        
        # Should handle unicode in comments gracefully
        tree = ast_parser.parse_source(rtl)
        modules = ast_parser.find_modules(tree)
        assert len(modules) == 1