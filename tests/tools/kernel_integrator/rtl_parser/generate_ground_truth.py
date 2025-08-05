#!/usr/bin/env python
"""Generate ground truth AST outputs for all test fixtures before migration."""

import json
from pathlib import Path
from brainsmith.tools.kernel_integrator.rtl_parser.ast_parser import ASTParser
from tests.tools.kernel_integrator.rtl_parser.utils.ast_serializer import ASTSerializer

def generate_ground_truth():
    """Generate ground truth files for all SystemVerilog test fixtures."""
    fixtures_dir = Path(__file__).parent / "fixtures" / "ast_comparison"
    ground_truth_dir = Path(__file__).parent / "ground_truth"
    ground_truth_dir.mkdir(exist_ok=True)
    
    # Initialize parser and serializer
    parser = ASTParser()
    serializer = ASTSerializer(max_text_length=50)
    
    # Process all .sv files
    sv_files = list(fixtures_dir.glob("*.sv"))
    print(f"Found {len(sv_files)} SystemVerilog files to process")
    
    results = {}
    
    for sv_file in sv_files:
        print(f"\nProcessing: {sv_file.name}")
        
        try:
            # Read and parse the file
            with open(sv_file, 'r') as f:
                content = f.read()
            
            tree = parser.parse_source(content)
            
            # Generate outputs in multiple formats
            tree_output = serializer.serialize_tree(tree, format='tree')
            json_output = serializer.serialize_tree(tree, format='json')
            compact_output = serializer.serialize_tree(tree, format='compact')
            
            # Save tree format to file
            tree_file = ground_truth_dir / f"{sv_file.stem}_ground_truth.tree"
            with open(tree_file, 'w') as f:
                f.write(tree_output)
            print(f"  Saved: {tree_file.name}")
            
            # Save JSON format for programmatic comparison
            json_file = ground_truth_dir / f"{sv_file.stem}_ground_truth.json"
            with open(json_file, 'w') as f:
                f.write(json_output)
            print(f"  Saved: {json_file.name}")
            
            # Collect summary info
            results[sv_file.name] = {
                "has_errors": tree.root_node.has_error,
                "root_type": tree.root_node.type,
                "node_count": count_nodes(tree.root_node),
                "tree_file": str(tree_file),
                "json_file": str(json_file),
                "compact_preview": compact_output[:200] + "..." if len(compact_output) > 200 else compact_output
            }
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results[sv_file.name] = {"error": str(e)}
    
    # Save summary
    summary_file = ground_truth_dir / "ground_truth_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")
    
    # Create README
    readme_content = f"""# Ground Truth AST Outputs

Generated before migrating to py-tree-sitter v0.25.0 with tree-sitter-systemverilog.

## Files Generated

"""
    for sv_name, info in results.items():
        if "error" not in info:
            readme_content += f"### {sv_name}\n"
            readme_content += f"- Root type: `{info['root_type']}`\n"
            readme_content += f"- Has errors: {info['has_errors']}\n"
            readme_content += f"- Node count: {info['node_count']}\n"
            readme_content += f"- Files: `{Path(info['tree_file']).name}`, `{Path(info['json_file']).name}`\n\n"
    
    readme_file = ground_truth_dir / "README.md"
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    
    print(f"\nREADME saved to: {readme_file}")
    print("\nGround truth generation complete!")
    
    return results

def count_nodes(node):
    """Count total nodes in AST."""
    count = 1
    for child in node.children:
        count += count_nodes(child)
    return count

if __name__ == "__main__":
    generate_ground_truth()