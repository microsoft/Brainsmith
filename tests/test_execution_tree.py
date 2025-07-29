# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Tests for Execution Tree Implementation

These tests verify that the execution tree correctly:
1. Shares common prefixes
2. Creates correct number of paths
3. Handles all branching scenarios

No mocks - uses real transforms from the registry.
"""

import pytest
from typing import List, Type

from brainsmith.core.execution_tree import (
    ExecutionSegment, count_leaves, count_nodes, get_tree_stats, print_tree, get_leaf_segments
)
from brainsmith.core.design_space import DesignSpace
from brainsmith.core.tree_builder import TreeBuilder
from brainsmith.core.config import ForgeConfig
from brainsmith.core.plugins import get_transform, get_backend
from brainsmith.core.plugins.registry import get_registry


def setup_module():
    """Ensure registry is initialized with real transforms."""
    # This happens automatically on import, but let's be explicit
    registry = get_registry()
    # Registry should be auto-populated on import


def build_tree_from_design_space(design_space):
    """Helper to build tree from design space for tests."""
    builder = TreeBuilder()
    forge_config = ForgeConfig(clock_ns=5.0)  # Use default config for tests
    return builder.build_tree(design_space, forge_config)


def get_real_transforms():
    """Get real transform classes from registry."""
    # Get some common QONNX transforms
    fold_constants = get_transform("FoldConstants")
    remove_identity = get_transform("RemoveIdentityOps")
    remove_unused = get_transform("RemoveUnusedTensors")
    infer_shapes = get_transform("InferShapes")
    
    # Get some FINN transforms
    round_thresholds = get_transform("RoundAndClipThresholds")
    absorb_sign_bias = get_transform("AbsorbSignBiasIntoMultiThreshold")
    
    return {
        "fold_constants": fold_constants,
        "remove_identity": remove_identity,
        "remove_unused": remove_unused,
        "infer_shapes": infer_shapes,
        "round_thresholds": round_thresholds,
        "absorb_sign_bias": absorb_sign_bias
    }


def get_real_backends():
    """Get real backend classes from registry."""
    # Get FINN backends
    mvau_hls = get_backend("MVAU_hls")
    mvau_rtl = get_backend("MVAU_rtl")
    thresholding_hls = get_backend("Thresholding_hls")
    
    return {
        "mvau_hls": mvau_hls,
        "mvau_rtl": mvau_rtl,
        "thresholding_hls": thresholding_hls
    }


def test_execution_node_deduplication():
    """Test that nodes are created correctly."""
    transforms = get_real_transforms()
    root = ExecutionSegment(segment_steps=[], branch_decision=None)
    
    # Create children with add_child
    child1 = root.add_child("step1_opt1", [{"transforms": [transforms["fold_constants"]]}])
    child2 = root.add_child("step1_opt2", [{"transforms": [transforms["fold_constants"]]}])
    
    assert child1 is not child2  # Different children
    assert len(root.children) == 2
    
    # Check segment IDs
    assert child1.segment_id == "step1_opt1"
    assert child2.segment_id == "step1_opt2"


def test_simple_linear_tree():
    """Test tree building with no branching."""
    transforms = get_real_transforms()
    backends = get_real_backends()
    
    # Create simple design space using new API
    design_space = DesignSpace(
        model_path="test.onnx",
        steps=["cleanup", "fold_constants", "infer_kernels"],  # Direct steps, no variations
        kernel_backends=[("MVAU", [backends["mvau_hls"]])],
    )
    
    tree = build_tree_from_design_space(design_space)
    
    # With segment-based trees, all linear steps are consolidated
    assert count_leaves(tree) == 1
    
    # Debug: print tree structure
    print("\nTree structure:")
    print_tree(tree)
    stats = get_tree_stats(tree)
    print(f"\nStats: {stats}")
    
    # In segment-based trees, linear steps are consolidated into one segment
    # The entire linear flow becomes a single segment in root
    assert count_nodes(tree) == 0  # All steps are in root segment


def test_simple_optional_stage():
    """Test tree building with a simple optional stage."""
    # Create design space with one optional step using new API
    design_space = DesignSpace(
        model_path="test.onnx",
        steps=[
            ["fold_constants", "~"]  # 2 options: do or skip
        ],
        kernel_backends=[],
    )
    
    tree = build_tree_from_design_space(design_space)
    
    # Should have 2 paths: one with fold_constants, one without
    assert count_leaves(tree) == 2


def test_branching_tree():
    """Test tree building with branching."""
    # Create design space with branching using new API
    design_space = DesignSpace(
        model_path="test.onnx",
        steps=[
            "fold_constants",  # No branch
            ["remove_identity", "remove_unused"],  # 2 options
            ["infer_shapes", "~"],  # 2 options (do or skip)
        ],
        kernel_backends=[],
    )
    
    tree = build_tree_from_design_space(design_space)
    
    # Should have 2 * 2 = 4 paths
    assert count_leaves(tree) == 4
    
    # Verify efficiency: segments reduce redundant computation
    stats = get_tree_stats(tree)
    assert stats['segment_efficiency'] > 0  # Some sharing is happening


def test_complex_tree_with_sharing():
    """Test complex tree with multiple branching points."""
    transforms = get_real_transforms()
    
    # Design space that creates a tree with sharing
    design_space = DesignSpace(
        model_path="test.onnx",
        steps=[
            "fold_constants",  # A
            ["remove_identity", "remove_unused"],  # B (2 options)
            "infer_shapes",  # C1 - required
            ["round_thresholds", "~"],  # C2 - optional
        ],
        kernel_backends=[],
    )
    
    tree = build_tree_from_design_space(design_space)
    
    # Paths: A->B1->C1->C2a, A->B1->C1->C2b, A->B2->C1->C2a, A->B2->C1->C2b
    assert count_leaves(tree) == 4
    
    # Verify branching structure
    stats = get_tree_stats(tree)
    assert stats['total_paths'] == 4


def test_empty_stages():
    """Test handling of empty transform stages."""
    design_space = DesignSpace(
        model_path="test.onnx",
        steps=[
            ["~"],  # Only skip option
            "fold_constants",  # Normal step
        ],
        kernel_backends=[],
    )
    
    tree = build_tree_from_design_space(design_space)
    
    # Even with only skip option, we create a branch
    # Tree should have one child for the skip branch
    assert count_nodes(tree) == 1  # Root plus one child
    assert count_leaves(tree) == 1  # Only one path through the tree


def test_real_finn_pipeline():
    """Test with realistic FINN transform pipeline."""
    registry = get_registry()
    
    # Build realistic steps using new API
    design_space = DesignSpace(
        model_path="test.onnx",
        steps=[
            "qonnx_to_finn",
            # Cleanup stage with variations
            "cleanup",
            ["remove_unused", "~"],  # Optional
            "fold_constants",
            # Streamline stage with variations
            ["absorb_sign_bias", "absorb_add_bias", "absorb_mul_bias"],  # Choose one
            ["round_thresholds", "~"],  # Optional
            "infer_kernels",
            "create_dataflow"
        ],
        kernel_backends=[
            ("MVAU", [get_backend("MVAU_hls"), get_backend("MVAU_rtl")]),
            ("Thresholding", [get_backend("Thresholding_hls")])
        ],
    )
    
    tree = build_tree_from_design_space(design_space)
    
    # cleanup: 1 * 2 * 1 = 2 combinations
    # streamline: 3 * 2 = 6 combinations
    # Total paths: 2 * 6 = 12
    assert count_leaves(tree) == 12
    
    # Verify efficiency through segment reuse
    stats = get_tree_stats(tree)
    assert stats['segment_efficiency'] > 0  # Segments reduce redundancy
    
    # Check that kernels are properly attached
    kernel_nodes = []
    
    def find_kernel_steps(node, results):
        for step in node.segment_steps:
            if isinstance(step, dict) and step.get("name") == "infer_kernels":
                results.append(node)
                break
        for child in node.children.values():
            find_kernel_steps(child, results)
    
    find_kernel_steps(tree, kernel_nodes)
    # Since we have branching before infer_kernels, multiple nodes will have it
    assert len(kernel_nodes) > 0
    
    # Check leaf nodes have complete pipelines
    leaves = get_leaf_segments(tree)
    for leaf in leaves:
        steps = leaf.get_all_steps()
        # Should have kernel inference step somewhere in the pipeline
        has_kernel_step = any(
            s == "infer_kernels" if isinstance(s, str) else 
            (isinstance(s, dict) and s.get("name") == "infer_kernels")
            for s in steps
        )
        assert has_kernel_step


def test_tree_stats():
    """Test tree statistics calculation."""
    # Create a tree with known structure
    root = ExecutionSegment(segment_steps=[], branch_decision=None)
    a = root.add_child("a", [{"name": "step_a"}])
    b1 = a.add_child("b1", [{"name": "step_b", "variant": 1}])
    b2 = a.add_child("b2", [{"name": "step_b", "variant": 2}])
    c1 = b1.add_child("c1", [{"name": "step_c"}])
    c2 = b2.add_child("c2", [{"name": "step_c"}])
    
    stats = get_tree_stats(root)
    
    assert stats['total_paths'] == 2  # Two leaves
    assert stats['total_segments'] == 5  # a, b1, b2, c1, c2
    assert stats['max_depth'] == 3    # root -> a -> b -> c
    assert stats['segment_efficiency'] >= 0  # Some efficiency from sharing


def test_segment_id_generation():
    """Test that segment IDs are generated correctly."""
    root = ExecutionSegment(segment_steps=[], branch_decision=None)
    assert root.segment_id == "root"
    
    child1 = root.add_child("opt1", [{"name": "step1"}])
    assert child1.segment_id == "opt1"
    
    grandchild = child1.add_child("opt2", [{"name": "step2"}])
    assert grandchild.segment_id == "opt1/opt2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])