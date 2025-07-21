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
    ExecutionNode, TransformStage, count_leaves, count_nodes, get_tree_stats
)
from brainsmith.core.design_space import DesignSpace, GlobalConfig
from brainsmith.core.tree_builder import build_execution_tree
from brainsmith.core.plugins.registry import get_registry


def setup_module():
    """Ensure registry is initialized with real transforms."""
    # This happens automatically on import, but let's be explicit
    registry = get_registry()
    # Verify we have transforms
    if not registry.transforms:
        from brainsmith.core.plugins.framework_adapters import initialize_framework_integrations
        initialize_framework_integrations()


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
    """Test that nodes with same config are deduplicated."""
    transforms = get_real_transforms()
    root = ExecutionNode("root", {})
    
    # Create child with same config twice
    child1 = root.find_or_create_child("step1", {"transforms": [transforms["fold_constants"]]})
    child2 = root.find_or_create_child("step1", {"transforms": [transforms["fold_constants"]]})
    
    assert child1 is child2  # Same object
    assert len(root.children) == 1
    
    # Different config creates new child
    child3 = root.find_or_create_child("step1", {"transforms": [transforms["remove_identity"]]})
    assert child3 is not child1
    assert len(root.children) == 2


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
    
    tree = build_execution_tree(design_space)
    
    # Should be linear: root -> start -> imports -> cleanup -> kernels -> end
    assert count_leaves(tree) == 1
    assert count_nodes(tree) == 5  # Not counting root


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
            ]),  # 2 options
            "stage3": TransformStage("stage3", [
                [transforms["infer_shapes"], None]
            ]),  # 2 options
        },
        kernel_backends=[],
        build_pipeline=["{stage1}", "{stage2}", "{stage3}"],
        global_config=GlobalConfig()
    )
    
    tree = build_execution_tree(design_space)
    
    # Should have 2 * 2 = 4 paths
    assert count_leaves(tree) == 4
    
    # Verify sharing: stage1 should be shared
    stats = get_tree_stats(tree)
    assert stats['sharing_factor'] > 1.0


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
    
    tree = build_execution_tree(design_space)
    
    # Paths: A->B1->C1, A->B1->C2, A->B2->C1, A->B2->C2
    assert count_leaves(tree) == 4
    
    # Find stage_A node (should have 1)
    stage_a_nodes = []
    
    def find_nodes_by_name(node, name, results):
        if node.step_name == name:
            results.append(node)
        for child in node.children:
            find_nodes_by_name(child, name, results)
    
    find_nodes_by_name(tree, "stage_A", stage_a_nodes)
    assert len(stage_a_nodes) == 1  # Shared!
    
    # Stage A should have 2 children (B options)
    assert len(stage_a_nodes[0].children) == 2


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
    
    tree = build_execution_tree(design_space)
    
    # Empty stage should not create a node
    # root -> normal
    assert count_nodes(tree) == 1
    assert tree.children[0].step_name == "stage_normal"


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
    
    tree = build_execution_tree(design_space)
    
    # cleanup: 1 * 2 * 1 = 2 combinations
    # streamline: 3 * 2 = 6 combinations
    # Total paths: 2 * 6 = 12
    assert count_leaves(tree) == 12
    
    # Verify sharing is happening
    stats = get_tree_stats(tree)
    assert stats['sharing_factor'] > 1.0
    
    # Check that kernels are properly attached
    kernel_nodes = []
    
    def find_kernel_nodes(node, results):
        if node.step_name == "infer_kernels":
            results.append(node)
        for child in node.children:
            find_kernel_nodes(child, results)
    
    find_kernel_nodes(tree, kernel_nodes)
    assert len(kernel_nodes) > 0
    
    # All kernel nodes should have the same kernel config
    first_config = kernel_nodes[0].config["kernel_backends"]
    for node in kernel_nodes:
        assert node.config["kernel_backends"] == first_config


def test_tree_stats():
    """Test tree statistics calculation."""
    # Create a tree with known structure
    root = ExecutionNode("root", {})
    a = root.find_or_create_child("a", {})
    b1 = a.find_or_create_child("b", {"variant": 1})
    b2 = a.find_or_create_child("b", {"variant": 2})
    c1 = b1.find_or_create_child("c", {})
    c2 = b2.find_or_create_child("c", {})
    
    stats = get_tree_stats(root)
    
    assert stats['total_paths'] == 2  # Two leaves
    assert stats['total_nodes'] == 5  # a, b1, b2, c1, c2
    assert stats['max_depth'] == 3    # root -> a -> b -> c
    assert stats['sharing_factor'] > 1.0  # Some sharing happening


def test_config_key_generation():
    """Test that config keys correctly identify equivalent configurations."""
    transforms = get_real_transforms()
    backends = get_real_backends()
    
    node = ExecutionNode("test", {})
    
    # Transform configs
    key1 = node._make_config_key({"transforms": [transforms["fold_constants"]]})
    key2 = node._make_config_key({"transforms": [transforms["fold_constants"]]})
    key3 = node._make_config_key({"transforms": [transforms["remove_identity"]]})
    
    assert key1 == key2  # Same transform
    assert key1 != key3  # Different transform
    
    # Backend configs
    key4 = node._make_config_key({
        "kernel_backends": [("MVAU", [backends["mvau_hls"]])]
    })
    key5 = node._make_config_key({
        "kernel_backends": [("MVAU", [backends["mvau_hls"]])]
    })
    key6 = node._make_config_key({
        "kernel_backends": [("MVAU", [backends["mvau_rtl"]])]
    })
    
    assert key4 == key5  # Same backend
    assert key4 != key6  # Different backend


if __name__ == "__main__":
    pytest.main([__file__, "-v"])