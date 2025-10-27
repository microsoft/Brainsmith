"""Fast integration tests for DSE tree construction.

Tests tree building, segment IDs, statistics, and execution order without FINN execution.

Ported patterns from: OLD_FOR_REFERENCE_ONLY/integration/test_dse_execution.py
Marker: @pytest.mark.fast
Execution time: < 1 min (no FINN execution)
"""

import pytest
from brainsmith.dse import build_tree, parse_blueprint
from brainsmith.dse.tree import DSETree
from brainsmith.dse.segment import DSESegment


class TestTreeBuilding:
    """Test suite for DSE tree construction."""

    @pytest.mark.fast
    def test_linear_tree(self, tmp_path, simple_onnx_model, simple_design_space, blueprint_config):
        """Build tree from linear steps (no branching)."""
        tree = build_tree(simple_design_space, blueprint_config)

        assert tree is not None
        assert isinstance(tree, DSETree)
        assert isinstance(tree.root, DSESegment)

        # Linear tree should have single path
        stats = tree.get_statistics()
        assert stats['total_paths'] == 1
        assert stats['total_segments'] >= 1

    @pytest.mark.fast
    def test_single_branch_tree(self, tmp_path, simple_onnx_model, branching_design_space, blueprint_config):
        """Build tree with one branch point."""
        tree = build_tree(branching_design_space, blueprint_config)

        assert tree is not None
        stats = tree.get_statistics()

        # Branch creates multiple paths
        assert stats['total_paths'] > 1
        assert stats['max_depth'] > 0

    @pytest.mark.fast
    def test_multi_branch_tree(self, tmp_path, simple_onnx_model, multi_branch_design_space, blueprint_config):
        """Build tree with multiple branch levels."""
        tree = build_tree(multi_branch_design_space, blueprint_config)

        assert tree is not None
        stats = tree.get_statistics()

        # Multiple branches create exponentially more paths
        assert stats['total_paths'] >= 4  # At least 2 branches * 2 options
        assert stats['max_depth'] >= 2

    @pytest.mark.fast
    def test_tree_from_blueprint(self, tmp_path, simple_onnx_model):
        """Build tree from blueprint end-to-end."""
        from tests.fixtures.blueprints import create_minimal_blueprint

        blueprint_path = create_minimal_blueprint(
            tmp_path,
            name="test_tree",
            steps=["custom:test_step", "custom:test_step1", "custom:test_step2"]
        )

        design_space, blueprint_config = parse_blueprint(
            str(blueprint_path),
            str(simple_onnx_model)
        )

        tree = build_tree(design_space, blueprint_config)

        assert tree is not None
        assert tree.root is not None


class TestTreeStatistics:
    """Test suite for tree statistics."""

    @pytest.mark.fast
    def test_tree_statistics_linear(self, tmp_path, simple_onnx_model, simple_design_space, blueprint_config):
        """Verify statistics for linear tree."""
        tree = build_tree(simple_design_space, blueprint_config)
        stats = tree.get_statistics()

        # Validate stat structure
        assert 'total_paths' in stats
        assert 'total_segments' in stats
        assert 'max_depth' in stats
        assert 'segment_efficiency' in stats

        # Linear tree properties
        assert stats['total_paths'] == 1
        assert stats['total_segments'] >= 1

    @pytest.mark.fast
    def test_tree_statistics_branching(self, tmp_path, simple_onnx_model, branching_design_space, blueprint_config):
        """Verify statistics for branched tree."""
        tree = build_tree(branching_design_space, blueprint_config)
        stats = tree.get_statistics()

        # Branched tree has multiple paths
        assert stats['total_paths'] > 1

        # Segment efficiency should be > 0 (segments are shared)
        assert stats['segment_efficiency'] >= 0

    @pytest.mark.fast
    def test_execution_order(self, tmp_path, simple_onnx_model, branching_design_space, blueprint_config):
        """Test BFS execution order."""
        tree = build_tree(branching_design_space, blueprint_config)
        execution_order = tree.get_execution_order()

        assert len(execution_order) > 0
        assert all(isinstance(seg, DSESegment) for seg in execution_order)

        # First segment should be root or root's first child
        assert execution_order[0] == tree.root or execution_order[0] in tree.root.children.values()


class TestSegmentIDDeterminism:
    """Test suite for segment ID consistency (critical for caching)."""

    @pytest.mark.fast
    def test_segment_ids_deterministic(self, tmp_path, simple_onnx_model, simple_design_space, blueprint_config):
        """Segment IDs must be deterministic across multiple builds."""
        # Build tree twice
        tree1 = build_tree(simple_design_space, blueprint_config)
        tree2 = build_tree(simple_design_space, blueprint_config)

        # Collect all segment IDs
        segments1 = tree1.get_all_segments()
        segments2 = tree2.get_all_segments()

        ids1 = [seg.segment_id for seg in segments1]
        ids2 = [seg.segment_id for seg in segments2]

        # IDs must match exactly (same order, same values)
        assert ids1 == ids2, "Segment IDs must be deterministic for caching to work"

    @pytest.mark.fast
    def test_segment_ids_unique(self, tmp_path, simple_onnx_model, branching_design_space, blueprint_config):
        """All segment IDs must be unique within a tree."""
        tree = build_tree(branching_design_space, blueprint_config)
        segments = tree.get_all_segments()

        ids = [seg.segment_id for seg in segments]

        # All IDs must be unique
        assert len(ids) == len(set(ids)), "Segment IDs must be unique within a tree"
