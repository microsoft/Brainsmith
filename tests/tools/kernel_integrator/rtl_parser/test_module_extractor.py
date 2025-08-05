############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Unit tests for the Module Extractor component.

Tests the SystemVerilog module extraction including:
- Module selection from multi-module files
- TOP_MODULE pragma handling
- Parameter extraction with types and defaults
- Port extraction with directions and widths
- Module name extraction
"""

import pytest
from pathlib import Path

from brainsmith.tools.kernel_integrator.rtl_parser.module_extractor import ModuleExtractor
from brainsmith.tools.kernel_integrator.types.rtl import Parameter, Port, PragmaType
from brainsmith.tools.kernel_integrator.rtl_parser.pragmas.module import TopModulePragma
from brainsmith.tools.kernel_integrator.types.rtl import PortDirection

from .utils.rtl_builder import RTLBuilder, create_minimal_module
from .utils.ast_patterns import ASTPatterns
from .utils.pragma_patterns import PragmaPatterns


class TestModuleExtractor:
    """Test cases for Module Extractor functionality."""
    
    def test_extract_single_module(self, module_extractor, ast_parser):
        """Test extracting components from a single module."""
        rtl = (RTLBuilder()
               .module("single_module")
               .parameter("WIDTH", "32")
               .parameter("DEPTH", "16")
               .port("clk", "input")
               .port("data_in", "input", "WIDTH")
               .port("data_out", "output", "WIDTH")
               .build())
        
        tree = ast_parser.parse_source(rtl)
        modules = ast_parser.find_modules(tree)
        assert len(modules) == 1
        
        # Extract module header
        name, param_nodes, port_nodes = module_extractor.extract_module_header(modules[0])
        
        assert name == "single_module"
        assert param_nodes is not None
        assert len(param_nodes) == 2
        assert port_nodes is not None
        assert len(port_nodes) == 3
    
    def test_extract_multi_module_with_pragma(self, module_extractor, ast_parser):
        """Test module selection with TOP_MODULE pragma."""
        # Build multi-module RTL with TOP_MODULE pragma
        rtl_selected = (RTLBuilder()
                        .comment("@brainsmith TOP_MODULE selected_module")
                        .module("selected_module")
                        .port("clk", "input")
                        .port("done", "output")
                        .build())
        
        rtl_other = (RTLBuilder()
                     .module("other_module")
                     .port("clk", "input")
                     .port("data", "input", "8")
                     .build())
        
        rtl = rtl_selected + "\n\n" + rtl_other
        
        tree = ast_parser.parse_source(rtl)
        modules = ast_parser.find_modules(tree)
        assert len(modules) == 2
        
        # Create mock pragmas
        pragmas = [
            TopModulePragma(
                type=PragmaType.TOP_MODULE,
                inputs=["selected_module"],
                line_number=1
            )
        ]
        
        # Select module based on pragma
        selected = module_extractor.select_target_module(modules, pragmas, "test.sv", target_module=None)
        assert selected is not None
        
        name, _, _ = module_extractor.extract_module_header(selected)
        assert name == "selected_module"
    
    def test_extract_module_by_name(self, module_extractor, ast_parser):
        """Test explicit module selection by name."""
        # Build multiple modules for selection test
        modules_rtl = []
        for name, port in [("first_module", "a"), ("target_module", "b"), ("third_module", "c")]:
            modules_rtl.append(
                RTLBuilder()
                .module(name)
                .port(port, "input")
                .build()
            )
        
        rtl = "\n\n".join(modules_rtl)
        
        tree = ast_parser.parse_source(rtl)
        modules = ast_parser.find_modules(tree)
        assert len(modules) == 3
        
        # Select by explicit name
        selected = module_extractor.select_target_module(modules, [], "test.sv", target_module="target_module")
        assert selected is not None
        
        name, _, _ = module_extractor.extract_module_header(selected)
        assert name == "target_module"
    
    def test_extract_parameters_all_types(self, module_extractor, ast_parser):
        """Test extracting parameters with various types and defaults."""
        # Build module with various parameter types
        rtl = (RTLBuilder()
               .module("param_test")
               .parameter("WIDTH", "32", "integer")
               .parameter("DEPTH", "16", "integer")
               .parameter("SIGNED", "1")
               .parameter("SCALE", "1.5", "real")
               .parameter("MODE", '"DEFAULT"', "string")
               .parameter("MASK", "8'hFF", "[7:0]")
               .parameter("FLAGS", "4'b1010", "logic [3:0]")
               .port("clk", "input")
               .build())
        
        tree = ast_parser.parse_source(rtl)
        module = ast_parser.find_modules(tree)[0]
        
        name, param_nodes, _ = module_extractor.extract_module_header(module)
        assert name == "param_test"
        assert len(param_nodes) == 7
        
        # Extract parameter details - method expects module node
        params = module_extractor.extract_parameters(module)
        assert len(params) == 7
        
        # Check specific parameters
        width_param = next(p for p in params if p.name == "WIDTH")
        assert width_param.default_value == "32"
        assert width_param.param_type == "integer"
        
        signed_param = next(p for p in params if p.name == "SIGNED")
        assert signed_param.default_value == "1"
        # RTLBuilder defaults to "integer" type even when not specified
        assert signed_param.param_type == "integer"
        
        scale_param = next(p for p in params if p.name == "SCALE")
        assert scale_param.default_value == "1.5"
        assert scale_param.param_type == "real"
    
    def test_extract_ports_all_directions(self, module_extractor, ast_parser):
        """Test extracting ports with all directions and widths."""
        # Build module with various port types
        rtl = (RTLBuilder()
               .module("port_test")
               .parameter("WIDTH", "32")
               .parameter("ADDR_WIDTH", "16")
               .port("clk", "input")
               .port("rst_n", "input")
               .port("data_in", "input", "WIDTH-1:0")
               .port("addr", "input", "ADDR_WIDTH-1:0")
               .port("data_out", "output", "WIDTH-1:0")
               .port("valid", "output reg")
               .port("status", "output", "8")
               .port("bidir_bus", "inout", "16")
               .build())
        
        tree = ast_parser.parse_source(rtl)
        module = ast_parser.find_modules(tree)[0]
        
        name, _, port_nodes = module_extractor.extract_module_header(module)
        assert name == "port_test"
        assert len(port_nodes) == 8
        
        # Extract port details - method expects module node
        ports = module_extractor.extract_ports(module)
        assert len(ports) == 8
        
        # Check specific ports
        clk_port = next(p for p in ports if p.name == "clk")
        assert clk_port.direction == PortDirection.INPUT
        assert clk_port.width == "1"
        
        data_in_port = next(p for p in ports if p.name == "data_in")
        assert data_in_port.direction == PortDirection.INPUT
        assert data_in_port.width == "WIDTH-1:0"
        
        valid_port = next(p for p in ports if p.name == "valid")
        assert valid_port.direction == PortDirection.OUTPUT
        
        bidir_port = next(p for p in ports if p.name == "bidir_bus")
        assert bidir_port.direction == PortDirection.INOUT
        # RTLBuilder converts "16" to "16-1:0"
        assert bidir_port.width == "16-1:0"
    
    def test_extract_empty_module(self, module_extractor, ast_parser):
        """Test extracting from module with no parameters or ports."""
        # Build empty module using RTLBuilder
        rtl = (RTLBuilder()
               .module("empty_module")
               .body("// No parameters or ports")
               .build())
        
        tree = ast_parser.parse_source(rtl)
        module = ast_parser.find_modules(tree)[0]
        
        name, param_nodes, port_nodes = module_extractor.extract_module_header(module)
        
        assert name == "empty_module"
        assert param_nodes == []
        assert port_nodes == []
        
        # Extract should handle empty module
        params = module_extractor.extract_parameters(module)
        ports = module_extractor.extract_ports(module)
        
        assert params == []
        assert ports == []
    
    def test_module_selection_priority(self, module_extractor, ast_parser):
        """Test module selection priority: explicit > pragma > single."""
        # Build multi-module RTL with pragma
        rtl_pragma = (RTLBuilder()
                      .comment("@brainsmith TOP_MODULE pragma_module")
                      .module("pragma_module")
                      .port("a", "input")
                      .build())
        
        rtl_explicit = (RTLBuilder()
                        .module("explicit_module")
                        .port("b", "input")
                        .build())
        
        rtl = rtl_pragma + "\n\n" + rtl_explicit
        
        tree = ast_parser.parse_source(rtl)
        modules = ast_parser.find_modules(tree)
        
        pragmas = [
            TopModulePragma(
                type=PragmaType.TOP_MODULE,
                inputs=["pragma_module"],
                line_number=1
            )
        ]
        
        # Test 1: Explicit target overrides pragma
        selected = module_extractor.select_target_module(modules, pragmas, "test.sv", target_module="explicit_module")
        name, _, _ = module_extractor.extract_module_header(selected)
        assert name == "explicit_module"
        
        # Test 2: Pragma used when no explicit target
        selected = module_extractor.select_target_module(modules, pragmas, "test.sv", target_module=None)
        name, _, _ = module_extractor.extract_module_header(selected)
        assert name == "pragma_module"
        
        # Test 3: Single module auto-selected
        single_rtl = (RTLBuilder()
                      .module("single")
                      .port("clk", "input")
                      .build())
        single_tree = ast_parser.parse_source(single_rtl)
        single_modules = ast_parser.find_modules(single_tree)
        
        selected = module_extractor.select_target_module(single_modules, [], "test.sv", target_module=None)
        name, _, _ = module_extractor.extract_module_header(selected)
        assert name == "single"
    
    def test_module_selection_errors(self, module_extractor, ast_parser):
        """Test error cases in module selection."""
        # Build two modules for error testing
        rtl = (RTLBuilder()
               .module("module_a")
               .port("a", "input")
               .build() + "\n\n" +
               RTLBuilder()
               .module("module_b")
               .port("b", "input")
               .build())
        
        tree = ast_parser.parse_source(rtl)
        modules = ast_parser.find_modules(tree)
        
        # Test 1: No modules found
        with pytest.raises(ValueError, match="Internal error"):
            module_extractor.select_target_module([], [], "test.sv", target_module=None)
        
        # Test 2: Target module not found
        with pytest.raises(ValueError, match="not found"):
            module_extractor.select_target_module(modules, [], "test.sv", target_module="nonexistent")
        
        # Test 3: Multiple modules without selection criteria
        with pytest.raises(ValueError, match="Multiple modules"):
            module_extractor.select_target_module(modules, [], "test.sv", target_module=None)
    
    def test_extract_complex_parameter_expressions(self, module_extractor, ast_parser):
        """Test extracting parameters with complex expressions."""
        # Build module with complex parameter expressions
        rtl = (RTLBuilder()
               .module("complex_params")
               .parameter("WIDTH", "32")
               .parameter("DEPTH", "WIDTH * 2")
               .parameter("SIZE", "(WIDTH + 7) / 8")
               .parameter("MASK", "(1 << WIDTH) - 1")
               .parameter("HALF", "WIDTH >> 1")
               .parameter("DOUBLE", "WIDTH << 1")
               .parameter("CHECK", "(WIDTH == 32) ? 1 : 0")
               .port("clk", "input")
               .build())
        
        tree = ast_parser.parse_source(rtl)
        module = ast_parser.find_modules(tree)[0]
        
        _, param_nodes, _ = module_extractor.extract_module_header(module)
        params = module_extractor.extract_parameters(module)
        
        assert len(params) == 7
        
        # Check that expressions are preserved
        depth_param = next(p for p in params if p.name == "DEPTH")
        assert depth_param.default_value == "WIDTH * 2"
        
        size_param = next(p for p in params if p.name == "SIZE")
        assert size_param.default_value == "(WIDTH + 7) / 8"
        
        check_param = next(p for p in params if p.name == "CHECK")
        assert check_param.default_value == "(WIDTH == 32) ? 1 : 0"
    
    def test_extract_array_ports(self, module_extractor, ast_parser):
        """Test extracting ports with array dimensions."""
        # Build module with array ports
        rtl = (RTLBuilder()
               .module("array_ports")
               .port("single_vector", "input", "32")
               .port("byte_array", "input", "[7:0]", "[0:15]")
               .port("mem_array", "output reg", "[31:0]", "[0:1023]")
               .port("param_width", "input", "WIDTH-1:0")
               .port("reversed_range", "output", "[0:31]")
               .build())
        
        tree = ast_parser.parse_source(rtl)
        module = ast_parser.find_modules(tree)[0]
        
        _, _, port_nodes = module_extractor.extract_module_header(module)
        ports = module_extractor.extract_ports(module)
        
        assert len(ports) == 5
        
        # Check vector widths - RTLBuilder converts "32" to "32-1:0"
        single = next(p for p in ports if p.name == "single_vector")
        assert single.width == "32-1:0"
        
        param = next(p for p in ports if p.name == "param_width")
        assert param.width == "WIDTH-1:0"
        
        reversed = next(p for p in ports if p.name == "reversed_range")
        assert reversed.width == "0:31"
    
    def test_nonansi_module_header(self, module_extractor, ast_parser):
        """Test extracting from non-ANSI style module."""
        # Non-ANSI style module - build manually since RTLBuilder doesn't support it
        rtl = """module nonansi(clk, rst, data_in, data_out);
    parameter WIDTH = 8;
    
    input clk;
    input rst;
    input [WIDTH-1:0] data_in;
    output [WIDTH-1:0] data_out;
    
    // Implementation
endmodule"""
        
        tree = ast_parser.parse_source(rtl)
        module = ast_parser.find_modules(tree)[0]
        
        # Module extractor should handle non-ANSI style
        name, param_nodes, port_nodes = module_extractor.extract_module_header(module)
        
        assert name == "nonansi"
        # Non-ANSI style parameters are in module body, not header
        assert param_nodes is not None
        # Port extraction might be limited for non-ANSI
        assert port_nodes is not None