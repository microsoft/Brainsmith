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
    ExecutionNode, count_leaves, count_nodes, get_tree_stats, print_tree, get_leaf_segments
)
from brainsmith.core.design_space import DesignSpace, GlobalConfig, TransformStage
from brainsmith.core.blueprint_parser import BlueprintParser
from brainsmith.core.plugins.registry import get_registry


def setup_module():
    """Ensure registry is initialized with real transforms."""
    # This happens automatically on import, but let's be explicit
    registry = get_registry()
    # Verify we have transforms
    if not registry.transforms:
        from brainsmith.core.plugins.framework_adapters import initialize_framework_integrations
        initialize_framework_integrations()


def build_tree_from_design_space(design_space):
    """Helper to build tree from design space for tests."""
    parser = BlueprintParser()
    from brainsmith.core.explorer.utils import StageWrapperFactory
    from brainsmith.core.plugins.registry import BrainsmithPluginRegistry
    registry = BrainsmithPluginRegistry()
    wrapper_factory = StageWrapperFactory(registry)
    return parser._build_execution_tree(design_space, wrapper_factory)


def get_real_transforms():
    """Get real transform classes from registry."""
    registry = get_registry()
    
    # Get some common QONNX transforms
    fold_constants = registry.get_transform("FoldConstants")
    remove_identity = registry.get_transform("RemoveIdentityOps")
    remove_unused = registry.get_transform("RemoveUnusedTensors")
    infer_shapes = registry.get_transform("InferShapes")
    
    # Get some FINN transforms
    round_thresholds = registry.get_transform("RoundAndClipThresholds")
    absorb_sign_bias = registry.get_transform("AbsorbSignBiasIntoMultiThreshold")
    
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
    registry = get_registry()
    
    # Get FINN backends
    mvau_hls = registry.get_backend("MVAU_hls")
    mvau_rtl = registry.get_backend("MVAU_rtl")
    thresholding_hls = registry.get_backend("Thresholding_hls")
    
    return {
        "mvau_hls": mvau_hls,
        "mvau_rtl": mvau_rtl,
        "thresholding_hls": thresholding_hls
    }


def test_transform_stage_combinations():
    """Test that TransformStage generates correct combinations."""
    transforms = get_real_transforms()
    
    # Single required transform
    stage1 = TransformStage("test1", [[transforms["fold_constants"]]])
    assert len(stage1.get_combinations()) == 1
    assert stage1.get_combinations()[0] == [transforms["fold_constants"]]
    
    # Optional transform (with None)
    stage2 = TransformStage("test2", [[transforms["remove_identity"], None]])
    combos = stage2.get_combinations()
    assert len(combos) == 2
    assert [transforms["remove_identity"]] in combos
    assert [] in combos  # None becomes empty list
    
    # Multiple steps
    stage3 = TransformStage("test3", [
        [transforms["fold_constants"]],  # Required
        [transforms["remove_identity"], transforms["remove_unused"]],  # Choose one
        [None, transforms["infer_shapes"]]  # Optional
    ])
    combos = stage3.get_combinations()
    assert len(combos) == 4  # 1 * 2 * 2


def test_execution_node_deduplication():
    """Test that nodes are created correctly."""
    transforms = get_real_transforms()
    root = ExecutionNode(segment_steps=[], branch_decision=None)
    
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
    
    # Create simple design space
    design_space = DesignSpace(
        model_path="test.onnx",
        transform_stages={
            "imports": TransformStage("imports", [[transforms["fold_constants"]]]),
            "cleanup": TransformStage("cleanup", [[transforms["remove_identity"]]]),
        },
        kernel_backends=[("MVAU", [backends["mvau_hls"]])],
        build_pipeline=["start", "{imports}", "{cleanup}", "infer_kernels", "end"],
        global_config=GlobalConfig()
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
    transforms = get_real_transforms()
    
    # Create design space with one optional stage
    design_space = DesignSpace(
        model_path="test.onnx",
        transform_stages={
            "opt": TransformStage("opt", [
                [transforms["fold_constants"], None]
            ]),  # 2 options: do or skip
        },
        kernel_backends=[],
        build_pipeline=["{opt}"],
        global_config=GlobalConfig()
    )
    
    tree = build_tree_from_design_space(design_space)
    
    # Should have 2 paths: one with fold_constants, one without
    assert count_leaves(tree) == 2
    

def test_branching_tree():
    """Test tree building with branching."""
    transforms = get_real_transforms()
    
    # Create design space with branching
    design_space = DesignSpace(
        model_path="test.onnx",
        transform_stages={
            "stage1": TransformStage("stage1", [[transforms["fold_constants"]]]),  # No branch
            "stage2": TransformStage("stage2", [
                [transforms["remove_identity"], transforms["remove_unused"]]
            ]),  # 2 options for one step
            "stage3": TransformStage("stage3", [
                [transforms["infer_shapes"], None]
            ]),  # 2 options (do or skip)
        },
        kernel_backends=[],
        build_pipeline=["{stage1}", "{stage2}", "{stage3}"],
        global_config=GlobalConfig()
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
        transform_stages={
            "A": TransformStage("A", [[transforms["fold_constants"]]]),
            "B": TransformStage("B", [
                [transforms["remove_identity"], transforms["remove_unused"]]
            ]),
            "C": TransformStage("C", [
                [transforms["infer_shapes"]],  # Required
                [transforms["round_thresholds"], None]  # Optional
            ]),
        },
        kernel_backends=[],
        build_pipeline=["{A}", "{B}", "{C}"],
        global_config=GlobalConfig()
    )
    
    tree = build_tree_from_design_space(design_space)
    
    # Paths: A->B1->C1, A->B1->C2, A->B2->C1, A->B2->C2
    assert count_leaves(tree) == 4
    
    # Find stage_A node (should have 1)
    stage_a_nodes = []
    
    def find_nodes_with_stage(node, stage_name, results):
        # Check if this node has steps from the given stage
        for step in node.segment_steps:
            if isinstance(step, dict) and step.get('stage_name') == stage_name:
                results.append(node)
                break
        for child in node.children.values():
            find_nodes_with_stage(child, stage_name, results)
    
    find_nodes_with_stage(tree, "A", stage_a_nodes)
    # Root contains stage A, so we find it there
    assert len(stage_a_nodes) >= 1
    
    # Verify branching structure
    stats = get_tree_stats(tree)
    assert stats['total_paths'] == 4


def test_empty_stages():
    """Test handling of empty transform stages."""
    transforms = get_real_transforms()
    
    design_space = DesignSpace(
        model_path="test.onnx",
        transform_stages={
            "empty": TransformStage("empty", [[None]]),  # Only skip option
            "normal": TransformStage("normal", [[transforms["fold_constants"]]]),
        },
        kernel_backends=[],
        build_pipeline=["{empty}", "{normal}"],
        global_config=GlobalConfig()
    )
    
    tree = build_tree_from_design_space(design_space)
    
    # Empty stage gets included in root segment
    # Tree should have one segment with both stages
    assert count_nodes(tree) == 0  # Just root, no child segments
    assert len(tree.segment_steps) > 0  # Root has steps


def test_real_finn_pipeline():
    """Test with realistic FINN transform pipeline."""
    registry = get_registry()
    
    # Build realistic stages
    design_space = DesignSpace(
        model_path="test.onnx",
        transform_stages={
            "cleanup": TransformStage("cleanup", [
                [registry.get_transform("RemoveIdentityOps")],
                [registry.get_transform("RemoveUnusedTensors"), None],  # Optional
                [registry.get_transform("FoldConstants")],
            ]),
            "streamline": TransformStage("streamline", [
                [
                    registry.get_transform("AbsorbSignBiasIntoMultiThreshold"),
                    registry.get_transform("AbsorbAddIntoMultiThreshold"),
                    registry.get_transform("AbsorbMulIntoMultiThreshold")
                ],  # Choose one absorb variant
                [registry.get_transform("RoundAndClipThresholds"), None],  # Optional
            ]),
        },
        kernel_backends=[
            ("MVAU", [registry.get_backend("MVAU_hls"), registry.get_backend("MVAU_rtl")]),
            ("Thresholding", [registry.get_backend("Thresholding_hls")])
        ],
        build_pipeline=[
            "qonnx_to_finn",
            "{cleanup}",
            "{streamline}",
            "infer_kernels",
            "create_dataflow"
        ],
        global_config=GlobalConfig()
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
            isinstance(s, dict) and s.get("name") == "infer_kernels" 
            for s in steps
        )
        assert has_kernel_step


def test_tree_stats():
    """Test tree statistics calculation."""
    # Create a tree with known structure
    root = ExecutionNode(segment_steps=[], branch_decision=None)
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
    root = ExecutionNode(segment_steps=[], branch_decision=None)
    assert root.segment_id == "root"
    
    child1 = root.add_child("opt1", [{"name": "step1"}])
    assert child1.segment_id == "opt1"
    
    grandchild = child1.add_child("opt2", [{"name": "step2"}])
    assert grandchild.segment_id == "opt1/opt2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])