"""
Integration tests for DSE (Design Space Exploration) execution.

Tests the DSE tree construction, segment execution, and result aggregation.
"""

import pytest
from pathlib import Path

from brainsmith.core.design.builder import DSETreeBuilder
from brainsmith.core.dse import SegmentResult
from brainsmith.core.dse.utils import share_artifacts_at_branch
from tests.utils.tree_assertions import (
    TreeAssertions,
    ExpectedTreeStructure,
    ExpectedExecutionLevel,
    calculate_segment_efficiency
)
from tests.utils.test_constants import (
    SINGLE_BRANCH_EFFICIENCY_WITH_SEGMENTS,
    SINGLE_BRANCH_EFFICIENCY_WITHOUT_SEGMENTS,
    MULTI_LEVEL_TOTAL_NODES,
    MULTI_LEVEL_TOTAL_LEAVES,
    MULTI_LEVEL_TOTAL_PATHS,
    MULTI_LEVEL_LEVEL_2_START_INDEX,
    DETERMINISM_TEST_ITERATIONS,
    NO_EFFICIENCY,
    MOCK_EXECUTION_TIME,
    DEFAULT_MAX_COMBINATIONS
)

# Import test fixtures
from tests.fixtures.dse_fixtures import (
    simple_design_space,
    branching_design_space,
    multi_branch_design_space,
    blueprint_config,
    base_finn_config
)


class TestDSETreeConstruction:
    """Test suite for DSE tree construction."""
    
    def test_build_linear_tree(self, simple_design_space, blueprint_config):
        """Test building a tree from linear steps."""
        builder = DSETreeBuilder()
        tree = builder.build_tree(simple_design_space, blueprint_config)
        
        # Verify tree structure
        assert tree.root is not None
        assert tree.count_nodes() == 1  # Just the root node
        assert tree.count_leaves() == 1
        assert not tree.root.is_branch_point
        assert tree.root.is_leaf
        
        # Verify steps
        steps = tree.root.steps
        assert len(steps) == 3
        assert steps[0]["name"] == "test_step"
        assert steps[1]["name"] == "test_step1"
        assert steps[2]["name"] == "test_step2"
        
        # Verify tree statistics
        stats = tree.get_statistics()
        assert stats['total_paths'] == 1  # Linear tree has only one path
        assert stats['total_segments'] == 1  # Just root segment
        # For linear tree, efficiency is 0% because no sharing possible
        assert stats['segment_efficiency'] == 0.0
        
    def test_build_single_branch_tree(self, branching_design_space, blueprint_config):
        """Test building a tree with one branch point."""
        builder = DSETreeBuilder()
        tree = builder.build_tree(branching_design_space, blueprint_config)
        
        # Calculate expected efficiency: 1 - (5 transforms with segments / 6 without) = 16.7%
        expected_efficiency = calculate_segment_efficiency(
            SINGLE_BRANCH_EFFICIENCY_WITH_SEGMENTS, 
            SINGLE_BRANCH_EFFICIENCY_WITHOUT_SEGMENTS
        )
        
        # Verify tree structure using helper
        expected = ExpectedTreeStructure(
            total_nodes=3,  # Root + 2 child branches
            total_leaves=2,
            total_paths=2,   # Two branches
            total_segments=3,  # Root + 2 children
            segment_efficiency=expected_efficiency
        )
        TreeAssertions.assert_complete_tree_validation(tree, expected)
        
        # Check root only has the first transform
        assert len(tree.root.steps) == 1
        assert tree.root.steps[0]["name"] == "test_step"
        
        # Check children structure
        assert len(tree.root.children) == 2
        child_names = [child.branch_choice for child in tree.root.children.values()]
        assert set(child_names) == {"test_step1", "test_step2"}
        
        # Each child should have their branch transform + the step
        for child in tree.root.children.values():
            assert len(child.steps) == 2
            assert child.steps[0]["name"] in ["test_step1", "test_step2"]
            assert child.steps[1]["name"] == "test_step3"
            
    def test_build_multi_level_tree(self, multi_branch_design_space, blueprint_config):
        """Test building a tree with multiple branch levels."""
        builder = DSETreeBuilder()
        tree = builder.build_tree(multi_branch_design_space, blueprint_config)
        
        # Should have: root + 2 first level branches + 2×2 second level branches = 7 nodes
        assert tree.count_nodes() == MULTI_LEVEL_TOTAL_NODES
        assert tree.count_leaves() == MULTI_LEVEL_TOTAL_LEAVES
        
        # Get execution order and verify BFS
        execution_order = tree.get_execution_order()
        assert len(execution_order) == MULTI_LEVEL_TOTAL_NODES
        
        # Root should be first
        assert execution_order[0] == tree.root
        assert execution_order[0].segment_id == "root"
        
        # Level 1 nodes should come before level 2
        level1_ids = {node.segment_id for node in execution_order[1:3]}
        assert level1_ids == {"test_step1", "test_step2"}
        
        # Level 2 nodes should include skip branches
        level2_ids = {node.segment_id for node in execution_order[MULTI_LEVEL_LEVEL_2_START_INDEX:]}
        # The skip branches might have different naming based on step numbering
        assert any("skip" in id for id in level2_ids), f"No skip branches found in {level2_ids}"
        assert any("test_step" in id for id in level2_ids), f"No test_step branches found in {level2_ids}"
        
        # Verify parent-child relationships
        for child in tree.root.children.values():
            assert child.parent == tree.root
            # Check grandchildren
            for grandchild in child.children.values():
                assert grandchild.parent == child
                assert grandchild.is_leaf  # These should be leaves
                
        # Verify tree statistics
        stats = tree.get_statistics()
        assert stats['total_paths'] == MULTI_LEVEL_TOTAL_PATHS  # 2 branches * 2 sub-branches
        assert stats['total_segments'] == MULTI_LEVEL_TOTAL_NODES  # 1 root + 2 level1 + 4 level2
        # Complex calculation - just verify it's positive (indicates sharing benefit)
        assert stats['segment_efficiency'] > 0
        
    def test_segment_id_generation(self, multi_branch_design_space, blueprint_config):
        """Test that segment IDs are generated deterministically."""
        builder = DSETreeBuilder()
        
        # Build tree multiple times
        trees = []
        for _ in range(DETERMINISM_TEST_ITERATIONS):
            tree = builder.build_tree(multi_branch_design_space, blueprint_config)
            trees.append(tree)
        
        # Collect all segment IDs from each tree
        all_segment_ids = []
        for tree in trees:
            segments = tree.get_execution_order()
            segment_ids = [seg.segment_id for seg in segments]
            all_segment_ids.append(segment_ids)
        
        # All runs should produce identical segment IDs
        assert all(ids == all_segment_ids[0] for ids in all_segment_ids)
        
        # Verify ID format
        assert all_segment_ids[0][0] == "root"
        for segment_id in all_segment_ids[0][1:]:
            assert "/" in segment_id or segment_id in ["test_step1", "test_step2"]
            
    def test_build_empty_design_space(self, blueprint_config):
        """Test building a tree from empty design space (edge case)."""
        from brainsmith.core.design.space import DesignSpace
        
        # Create empty design space
        empty_design_space = DesignSpace(
            model_path="test_model.onnx",
            steps=[],
            kernel_backends=[],
            max_combinations=DEFAULT_MAX_COMBINATIONS
        )
        
        builder = DSETreeBuilder()
        tree = builder.build_tree(empty_design_space, blueprint_config)
        
        # Verify empty tree structure
        assert tree.root is not None
        assert tree.count_nodes() == 1  # Just the root
        assert tree.count_leaves() == 1
        assert tree.root.is_leaf
        assert not tree.root.is_branch_point
        assert len(tree.root.steps) == 0
        
        # Verify statistics for empty tree
        stats = tree.get_statistics()
        assert stats['total_paths'] == 1  # Empty path is still a path
        assert stats['total_segments'] == 1  # Just root
        assert stats['segment_efficiency'] == NO_EFFICIENCY  # No transforms = no efficiency



class TestArtifactManagement:
    """Test suite for artifact management."""
    
    def test_artifact_sharing_logic(self, branching_design_space, blueprint_config, tmp_path):
        """Test artifact sharing function works correctly."""
        # Build tree to get structure
        builder = DSETreeBuilder()
        tree = builder.build_tree(branching_design_space, blueprint_config)
        
        # Create mock result and directories
        root_dir = tmp_path / "root"
        root_dir.mkdir()
        marker = root_dir / "test_artifact.txt"
        marker.write_text("root content")
        
        # Create mock segment result
        root_result = SegmentResult(
            segment_id="root",
            success=True,
            output_model=root_dir / "model.onnx",
            output_dir=root_dir,
            execution_time=MOCK_EXECUTION_TIME
        )
        
        # Test artifact sharing
        child_segments = list(tree.root.children.values())
        share_artifacts_at_branch(root_result, child_segments, tmp_path)
        
        # Verify copies were made
        for child in child_segments:
            child_file = tmp_path / child.segment_id / "test_artifact.txt"
            assert child_file.exists()
            assert child_file.read_text() == "root content"


class TestStepRangeExecution:
    """Test suite for step range control during DSE execution."""

    def test_apply_step_slicing_in_dse(self, simple_onnx_model, blueprint_config):
        """Test that step slicing is applied before tree building."""
        from brainsmith.core.design.space import DesignSpace, slice_steps
        from brainsmith.core.design.builder import DSETreeBuilder

        # Create design space with 5 steps
        design_space = DesignSpace(
            model_path=str(simple_onnx_model),
            steps=["test_step", "test_step1", "test_step2", "test_step3", "export_to_build"],
            kernel_backends=[],
            max_combinations=100
        )

        # Slice to middle 3 steps
        design_space.steps = slice_steps(design_space.steps, "test_step1", "test_step3")

        # Build tree with sliced steps
        builder = DSETreeBuilder()
        tree = builder.build_tree(design_space, blueprint_config)

        # Verify tree only contains sliced steps
        all_steps = tree.root.get_all_steps()
        assert len(all_steps) == 3
        assert all_steps[0]["name"] == "test_step1"
        assert all_steps[1]["name"] == "test_step2"
        assert all_steps[2]["name"] == "test_step3"

    def test_slice_preserves_branches_in_tree(self, simple_onnx_model, blueprint_config):
        """Test that slicing preserves branch structure."""
        from brainsmith.core.design.space import DesignSpace, slice_steps
        from brainsmith.core.design.builder import DSETreeBuilder

        # Create design space with branches
        design_space = DesignSpace(
            model_path=str(simple_onnx_model),
            steps=["test_step", ["test_step1", "test_step2"], "test_step3", "export_to_build"],
            kernel_backends=[],
            max_combinations=100
        )

        # Slice to include the branch
        design_space.steps = slice_steps(design_space.steps, "test_step", "test_step3")

        # Build tree
        builder = DSETreeBuilder()
        tree = builder.build_tree(design_space, blueprint_config)

        # Verify branch structure is preserved
        assert tree.root.is_branch_point
        assert len(tree.root.children) == 2
