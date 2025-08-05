#!/usr/bin/env python
"""Compare AST outputs after migration to tree-sitter-systemverilog."""

import json
from pathlib import Path
from brainsmith.tools.kernel_integrator.rtl_parser.ast_parser import ASTParser
from tests.tools.kernel_integrator.rtl_parser.utils.ast_serializer import ASTSerializer


def compare_ast_outputs():
    """Compare current AST outputs with ground truth."""
    fixtures_dir = Path(__file__).parent / "fixtures" / "ast_comparison"
    ground_truth_dir = Path(__file__).parent / "ground_truth"
    
    # Initialize parser and serializer
    parser = ASTParser()
    serializer = ASTSerializer(max_text_length=50)
    
    # Process all .sv files
    sv_files = list(fixtures_dir.glob("*.sv"))
    print(f"Comparing {len(sv_files)} SystemVerilog files")
    
    all_match = True
    
    for sv_file in sv_files:
        print(f"\n{'='*60}")
        print(f"Checking: {sv_file.name}")
        
        # Read and parse the file
        with open(sv_file, 'r') as f:
            content = f.read()
        
        tree = parser.parse_source(content)
        
        # Generate current output
        current_tree = serializer.serialize_tree(tree, format='tree')
        current_json = serializer.serialize_tree(tree, format='json')
        
        # Load ground truth
        ground_truth_tree_file = ground_truth_dir / f"{sv_file.stem}_ground_truth.tree"
        ground_truth_json_file = ground_truth_dir / f"{sv_file.stem}_ground_truth.json"
        
        if not ground_truth_tree_file.exists():
            print(f"  WARNING: No ground truth found for {sv_file.name}")
            continue
        
        with open(ground_truth_tree_file, 'r') as f:
            ground_truth_tree = f.read()
        
        with open(ground_truth_json_file, 'r') as f:
            ground_truth_json = f.read()
        
        # Compare tree format
        if current_tree == ground_truth_tree:
            print("  ✓ Tree format matches")
        else:
            print("  ✗ Tree format differs")
            all_match = False
            
            # Show first difference
            lines_current = current_tree.split('\n')
            lines_ground = ground_truth_tree.split('\n')
            
            for i, (curr, ground) in enumerate(zip(lines_current, lines_ground)):
                if curr != ground:
                    print(f"    First difference at line {i+1}:")
                    print(f"    Ground truth: {ground[:100]}")
                    print(f"    Current:      {curr[:100]}")
                    break
        
        # Compare JSON structure
        json_current = json.loads(current_json)
        json_ground = json.loads(ground_truth_json)
        
        if json_current == json_ground:
            print("  ✓ JSON structure matches")
        else:
            print("  ✗ JSON structure differs")
            all_match = False
            
            # Basic analysis
            def count_nodes(node):
                count = 1
                for child in node.get('children', []):
                    count += count_nodes(child)
                return count
            
            nodes_current = count_nodes(json_current)
            nodes_ground = count_nodes(json_ground)
            
            print(f"    Node count - Ground truth: {nodes_ground}, Current: {nodes_current}")
            
            if json_current.get('type') != json_ground.get('type'):
                print(f"    Root type differs - Ground: {json_ground.get('type')}, Current: {json_current.get('type')}")
    
    print(f"\n{'='*60}")
    if all_match:
        print("✓ All AST outputs match ground truth!")
    else:
        print("✗ Some AST outputs differ from ground truth")
        print("\nThis may be expected if the new grammar has improvements.")
        print("Please review the differences carefully.")
    
    return all_match


if __name__ == "__main__":
    all_match = compare_ast_outputs()
    exit(0 if all_match else 1)