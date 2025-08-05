############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Test cases for AST serialization and comparison.

This module tests the initial phase of RTL parsing by:
- Parsing RTL files into AST
- Serializing AST to text format
- Comparing with expected output
- Providing debugging utilities
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from brainsmith.tools.kernel_integrator.rtl_parser.ast_parser import ASTParser
from .utils.ast_serializer import ASTSerializer, ASTDiffer


class TestASTSerialization:
    """Test cases for AST serialization functionality."""
    
    @pytest.fixture
    def ast_comparison_dir(self, test_data_dir):
        """Path to AST comparison test data."""
        return test_data_dir / "ast_comparison"
    
    @pytest.fixture
    def output_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_simple_module_serialization(self, ast_parser, ast_comparison_dir, output_dir):
        """Test serializing a simple module to AST text format."""
        # Read RTL file
        rtl_file = ast_comparison_dir / "simple_module.sv"
        with open(rtl_file, 'r') as f:
            rtl_content = f.read()
        
        # Parse to AST
        tree = ast_parser.parse_source(rtl_content)
        assert tree is not None
        assert not tree.root_node.has_error
        
        # Serialize to different formats
        serializer = ASTSerializer(max_text_length=30)
        
        # Tree format
        tree_output = serializer.serialize_tree(tree, format="tree")
        tree_file = output_dir / "simple_module_tree.txt"
        with open(tree_file, 'w') as f:
            f.write(tree_output)
        
        # Verify tree output contains expected elements
        assert "module_declaration" in tree_output
        assert "simple_module" in tree_output
        assert "clk" in tree_output
        assert "rst_n" in tree_output
        assert "data_out" in tree_output
        
        # JSON format
        json_output = serializer.serialize_tree(tree, format="json")
        json_file = output_dir / "simple_module.json"
        with open(json_file, 'w') as f:
            f.write(json_output)
        
        # Compact format
        compact_output = serializer.serialize_tree(tree, format="compact")
        compact_file = output_dir / "simple_module_compact.txt"
        with open(compact_file, 'w') as f:
            f.write(compact_output)
        
        print(f"\nGenerated AST files in: {output_dir}")
        print(f"Tree format preview:\n{tree_output[:500]}...")
    
    def test_parameterized_module_serialization(self, ast_parser, ast_comparison_dir, output_dir):
        """Test serializing a parameterized module with complex structure."""
        # Read RTL file
        rtl_file = ast_comparison_dir / "parameterized_module.sv"
        with open(rtl_file, 'r') as f:
            rtl_content = f.read()
        
        # Parse to AST
        tree = ast_parser.parse_source(rtl_content)
        assert not tree.root_node.has_error
        
        # Serialize with custom settings
        serializer = ASTSerializer(
            max_depth=10,  # Limit depth for readability
            max_text_length=40,
            include_positions=True
        )
        
        # Generate tree output
        tree_output = serializer.serialize_tree(tree, format="tree")
        output_file = output_dir / "parameterized_module_tree.txt"
        serializer.serialize_to_file(tree, str(output_file), format="tree")
        
        # Verify key elements
        assert "parameter_port_list" in tree_output
        assert "WIDTH" in tree_output
        assert "DEPTH" in tree_output
        assert "always_ff" in tree_output
        assert "posedge clk" in tree_output
    
    def test_module_with_pragmas_serialization(self, ast_parser, ast_comparison_dir, output_dir):
        """Test serializing a module with pragma comments."""
        # Read RTL file
        rtl_file = ast_comparison_dir / "module_with_pragmas.sv"
        with open(rtl_file, 'r') as f:
            rtl_content = f.read()
        
        # Parse to AST
        tree = ast_parser.parse_source(rtl_content)
        assert not tree.root_node.has_error
        
        # Serialize without excluding comments
        serializer = ASTSerializer(
            max_text_length=50,
            exclude_types=set()  # Don't exclude any node types
        )
        
        tree_output = serializer.serialize_tree(tree, format="tree")
        output_file = output_dir / "module_with_pragmas_tree.txt"
        with open(output_file, 'w') as f:
            f.write(tree_output)
        
        # Verify pragmas are captured as comments
        assert "comment" in tree_output
        assert "brainsmith:pragma:INTERFACE" in tree_output
        assert "brainsmith:pragma:DATATYPE" in tree_output
        assert "brainsmith:pragma:BDIM" in tree_output
        assert "brainsmith:pragma:SDIM" in tree_output
    
    def test_malformed_module_serialization(self, ast_parser, ast_comparison_dir, output_dir):
        """Test serializing a malformed module to see error nodes."""
        # Read RTL file
        rtl_file = ast_comparison_dir / "malformed_module.sv"
        with open(rtl_file, 'r') as f:
            rtl_content = f.read()
        
        # Parse to AST (will have errors)
        tree = ast_parser.parse_source(rtl_content)
        assert tree.root_node.has_error  # Should have syntax errors
        
        # Serialize to show error nodes
        serializer = ASTSerializer(include_positions=True)
        tree_output = serializer.serialize_tree(tree, format="tree")
        
        output_file = output_dir / "malformed_module_tree.txt"
        with open(output_file, 'w') as f:
            f.write(tree_output)
        
        # The output should indicate errors
        assert "ERROR" in tree_output or tree.root_node.has_error
    
    def test_ast_comparison(self, ast_parser, ast_comparison_dir):
        """Test comparing two AST trees for differences."""
        # Parse two different modules
        rtl_file1 = ast_comparison_dir / "simple_module.sv"
        rtl_file2 = ast_comparison_dir / "parameterized_module.sv"
        
        with open(rtl_file1, 'r') as f:
            rtl1 = f.read()
        with open(rtl_file2, 'r') as f:
            rtl2 = f.read()
        
        tree1 = ast_parser.parse_source(rtl1)
        tree2 = ast_parser.parse_source(rtl2)
        
        # Compare trees
        differ = ASTDiffer(ignore_positions=True)
        differences = differ.compare_trees(tree1, tree2)
        
        # Should find differences
        assert len(differences) > 0
        
        # Compare same tree to itself
        no_differences = differ.compare_trees(tree1, tree1)
        assert len(no_differences) == 0
    
    def test_serialization_with_depth_limit(self, ast_parser, ast_comparison_dir, output_dir):
        """Test serialization with depth limits."""
        # Read complex RTL file
        rtl_file = ast_comparison_dir / "parameterized_module.sv"
        with open(rtl_file, 'r') as f:
            rtl_content = f.read()
        
        tree = ast_parser.parse_source(rtl_content)
        
        # Serialize with different depth limits
        for depth in [1, 3, 5, None]:
            serializer = ASTSerializer(max_depth=depth)
            output = serializer.serialize_tree(tree, format="tree")
            
            filename = f"depth_{depth if depth else 'unlimited'}.txt"
            with open(output_dir / filename, 'w') as f:
                f.write(output)
            
            # Deeper serialization should have more content
            if depth == 1:
                shallow_len = len(output)
            elif depth is None:
                assert len(output) > shallow_len
    
    def test_exclude_node_types(self, ast_parser, ast_comparison_dir, output_dir):
        """Test excluding specific node types from serialization."""
        # Read RTL file
        rtl_file = ast_comparison_dir / "module_with_pragmas.sv"
        with open(rtl_file, 'r') as f:
            rtl_content = f.read()
        
        tree = ast_parser.parse_source(rtl_content)
        
        # Serialize excluding comments
        serializer = ASTSerializer(exclude_types={"comment"})
        output = serializer.serialize_tree(tree, format="tree")
        
        # Comments should be excluded
        assert "comment" not in output
        # But other content should be present
        assert "module_declaration" in output
        assert "matrix_multiply" in output
    
    def test_generate_reference_files(self, ast_parser, ast_comparison_dir, output_dir):
        """Generate reference AST files for all test modules.
        
        This test can be used to generate expected output files for comparison.
        """
        serializer = ASTSerializer(max_text_length=50, include_positions=True)
        
        # Process all .sv files in the comparison directory
        for sv_file in ast_comparison_dir.glob("*.sv"):
            if "malformed" in sv_file.name:
                continue  # Skip malformed files for reference generation
                
            with open(sv_file, 'r') as f:
                rtl_content = f.read()
            
            tree = ast_parser.parse_source(rtl_content)
            
            # Generate reference files in different formats
            base_name = sv_file.stem
            
            # Tree format
            tree_ref = output_dir / f"{base_name}_expected.tree"
            serializer.serialize_to_file(tree, str(tree_ref), format="tree")
            
            # JSON format
            json_ref = output_dir / f"{base_name}_expected.json"
            serializer.serialize_to_file(tree, str(json_ref), format="json")
            
            # Compact format
            compact_ref = output_dir / f"{base_name}_expected.compact"
            serializer.serialize_to_file(tree, str(compact_ref), format="compact")
        
        print(f"\nGenerated reference files in: {output_dir}")
    
    def test_custom_serialization_format(self, ast_parser, ast_comparison_dir):
        """Test using AST serializer for custom analysis."""
        # Read RTL file
        rtl_file = ast_comparison_dir / "parameterized_module.sv"
        with open(rtl_file, 'r') as f:
            rtl_content = f.read()
        
        tree = ast_parser.parse_source(rtl_content)
        
        # Create custom serializer for specific analysis
        # For example, only show module structure without implementation details
        serializer = ASTSerializer(
            exclude_types={"comment", "always_ff", "blocking_assignment", "nonblocking_assignment"},
            max_text_length=20
        )
        
        structure_output = serializer.serialize_tree(tree, format="tree")
        
        # Should show module structure but not implementation
        assert "module_declaration" in structure_output
        assert "parameter_port_list" in structure_output
        assert "list_of_port_declarations" in structure_output
        # But not implementation details
        assert "always_ff" not in structure_output