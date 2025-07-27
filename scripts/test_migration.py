#!/usr/bin/env python3
"""
Test migration script for segment-based execution tree.

This script compares the old step-based tree with the new segment-based tree
to ensure they represent the same execution paths.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from brainsmith.core.blueprint_parser import BlueprintParser
from brainsmith.core import execution_tree as tree_v1
from brainsmith.core import execution_tree_v2 as tree_v2
from brainsmith.core import tree_builder as builder_v1
from brainsmith.core import tree_builder_v2 as builder_v2


def get_all_paths_v1(node):
    """Get all paths in v1 tree (step-based)."""
    if not node.children:
        return [[node]]
    
    paths = []
    for child in node.children:
        child_paths = get_all_paths_v1(child)
        for path in child_paths:
            paths.append([node] + path)
    
    return paths


def get_all_paths_v2(root):
    """Get all paths in v2 tree (segment-based)."""
    leaves = tree_v2.get_leaf_segments(root)
    paths = []
    
    for leaf in leaves:
        path = leaf.get_path()
        paths.append(path)
    
    return paths


def path_to_steps_v1(path):
    """Convert v1 path to list of steps."""
    steps = []
    for node in path:
        if node.step_name != "root":
            steps.append({
                "name": node.step_name,
                "config": node.config
            })
    return steps


def path_to_steps_v2(path):
    """Convert v2 path to list of steps."""
    steps = []
    for segment in path:
        steps.extend(segment.segment_steps)
    return steps


def compare_trees(tree1, tree2):
    """Compare two trees to ensure they represent the same execution paths."""
    print("Comparing trees...")
    
    # Get all paths
    paths_v1 = get_all_paths_v1(tree1)
    paths_v2 = get_all_paths_v2(tree2)
    
    print(f"V1 paths: {len(paths_v1)}")
    print(f"V2 paths: {len(paths_v2)}")
    
    if len(paths_v1) != len(paths_v2):
        print("❌ Different number of paths!")
        return False
    
    # Compare each path
    all_match = True
    for i, (path1, path2) in enumerate(zip(paths_v1, paths_v2)):
        steps1 = path_to_steps_v1(path1)
        steps2 = path_to_steps_v2(path2)
        
        # Compare step counts
        if len(steps1) != len(steps2):
            print(f"❌ Path {i}: Different step counts ({len(steps1)} vs {len(steps2)})")
            all_match = False
            continue
        
        # Compare each step
        for j, (s1, s2) in enumerate(zip(steps1, steps2)):
            # Extract comparable information
            name1 = s1["name"]
            name2 = s2.get("name", s2.get("stage_name", "unknown"))
            
            # Handle stage names
            if name1.startswith("stage_") and "stage_name" in s2:
                name1 = name1[6:]  # Remove "stage_" prefix
            
            if name1 != name2 and not (name1 == "infer_kernels" and "kernel_backends" in s2):
                print(f"❌ Path {i}, Step {j}: Names don't match ({name1} vs {name2})")
                all_match = False
    
    if all_match:
        print("✅ All paths match!")
    
    return all_match


def compare_statistics(tree1, tree2):
    """Compare tree statistics."""
    print("\nComparing statistics...")
    
    stats1 = tree_v1.get_tree_stats(tree1)
    stats2 = tree_v2.get_tree_stats(tree2)
    
    print(f"\nV1 Statistics:")
    print(f"  - Total paths: {stats1['total_paths']}")
    print(f"  - Total nodes: {stats1['total_nodes']}")
    print(f"  - Max depth: {stats1['max_depth']}")
    print(f"  - Sharing factor: {stats1['sharing_factor']}x")
    
    print(f"\nV2 Statistics:")
    print(f"  - Total paths: {stats2['total_paths']}")
    print(f"  - Total segments: {stats2['total_segments']}")
    print(f"  - Max depth: {stats2['max_depth']}")
    print(f"  - Segment efficiency: {stats2['segment_efficiency']}%")
    
    # Key invariant: same number of paths
    if stats1['total_paths'] != stats2['total_paths']:
        print("❌ Different number of paths!")
        return False
    
    # Expected: fewer nodes in v2
    reduction = (stats1['total_nodes'] - stats2['total_segments']) / stats1['total_nodes'] * 100
    print(f"\n✅ Node reduction: {reduction:.1f}%")
    
    return True


def test_blueprint(blueprint_path, model_path):
    """Test a specific blueprint."""
    print(f"\nTesting blueprint: {blueprint_path}")
    print("=" * 60)
    
    # Parse blueprint
    parser = BlueprintParser()
    design_space = parser.parse(blueprint_path, model_path)
    
    # Build both trees
    print("\nBuilding v1 tree (step-based)...")
    tree1 = builder_v1.build_execution_tree(design_space)
    
    print("Building v2 tree (segment-based)...")
    tree2 = builder_v2.build_execution_tree(design_space)
    
    # Compare trees
    paths_match = compare_trees(tree1, tree2)
    stats_match = compare_statistics(tree1, tree2)
    
    # Visual comparison
    print("\nV1 Tree Structure:")
    tree_v1.print_tree(tree1)
    
    print("\nV2 Tree Structure:")
    tree_v2.print_tree(tree2)
    
    return paths_match and stats_match


def main():
    """Run migration tests."""
    # Test with example blueprint
    test_dir = Path(__file__).parent.parent / "tests" / "blueprints"
    
    # Find test blueprints
    test_files = [
        "simple_linear.yaml",
        "simple_branching.yaml", 
        "complex_branching.yaml",
        "inheritance_base.yaml"
    ]
    
    model_path = "/tmp/test_model.onnx"  # Dummy path for testing
    
    all_passed = True
    
    for test_file in test_files:
        blueprint_path = test_dir / test_file
        if blueprint_path.exists():
            passed = test_blueprint(str(blueprint_path), model_path)
            all_passed = all_passed and passed
        else:
            print(f"⚠️  Skipping {test_file} (not found)")
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All migration tests passed!")
    else:
        print("❌ Some tests failed")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())