#!/usr/bin/env python3
############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Simple demo script to run AST serialization without pytest.

This script demonstrates the AST serialization functionality by:
1. Parsing an RTL file
2. Serializing the AST to different formats
3. Saving the output to files
"""

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from brainsmith.tools.kernel_integrator.rtl_parser.ast_parser import ASTParser
from tests.tools.kernel_integrator.rtl_parser.utils.ast_serializer import ASTSerializer


def main():
    """Run AST serialization demo."""
    # Setup paths
    test_dir = Path(__file__).parent
    fixtures_dir = test_dir / "fixtures" / "ast_comparison"
    output_dir = test_dir / "ast_output"
    output_dir.mkdir(exist_ok=True)
    
    # Initialize parser
    ast_parser = ASTParser()
    
    # Test file
    rtl_file = fixtures_dir / "simple_module.sv"
    
    print(f"Reading RTL file: {rtl_file}")
    with open(rtl_file, 'r') as f:
        rtl_content = f.read()
    
    # Parse to AST
    print("Parsing RTL to AST...")
    tree = ast_parser.parse_source(rtl_content)
    
    if tree.root_node.has_error:
        print("WARNING: AST contains syntax errors")
    
    # Create serializer
    serializer = ASTSerializer(
        max_text_length=40,
        include_positions=True
    )
    
    # Generate different formats
    print("\nGenerating AST outputs...")
    
    # Tree format
    tree_output = serializer.serialize_tree(tree, format="tree")
    tree_file = output_dir / "simple_module.tree"
    with open(tree_file, 'w') as f:
        f.write(tree_output)
    print(f"✓ Tree format: {tree_file}")
    
    # JSON format
    json_output = serializer.serialize_tree(tree, format="json")
    json_file = output_dir / "simple_module.json"
    with open(json_file, 'w') as f:
        f.write(json_output)
    print(f"✓ JSON format: {json_file}")
    
    # Compact format
    compact_output = serializer.serialize_tree(tree, format="compact")
    compact_file = output_dir / "simple_module.compact"
    with open(compact_file, 'w') as f:
        f.write(compact_output)
    print(f"✓ Compact format: {compact_file}")
    
    # Show preview of tree format
    print("\n=== Tree Format Preview ===")
    lines = tree_output.split('\n')
    for line in lines[:20]:  # Show first 20 lines
        print(line)
    if len(lines) > 20:
        print(f"... ({len(lines) - 20} more lines)")
    
    print(f"\nAll outputs saved to: {output_dir}")
    
    # Process other test files
    print("\n=== Processing other test files ===")
    for sv_file in ["parameterized_module.sv", "module_with_pragmas.sv"]:
        rtl_file = fixtures_dir / sv_file
        if rtl_file.exists():
            print(f"\nProcessing: {sv_file}")
            with open(rtl_file, 'r') as f:
                rtl_content = f.read()
            
            tree = ast_parser.parse_source(rtl_content)
            
            # Save tree format
            output_file = output_dir / f"{rtl_file.stem}.tree"
            serializer.serialize_to_file(tree, str(output_file), format="tree")
            print(f"✓ Saved: {output_file}")


if __name__ == "__main__":
    main()