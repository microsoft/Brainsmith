"""Fast integration tests for blueprint parsing.

Tests blueprint YAML parsing, inheritance, and step operations without FINN execution.

Ported from: OLD_FOR_REFERENCE_ONLY/integration/test_blueprint_parser.py
Marker: @pytest.mark.fast
Execution time: < 1 min (no FINN execution)
"""


import pytest

from brainsmith.dse import parse_blueprint
from brainsmith.dse.types import OutputType
from tests.fixtures.dse.blueprints import (
    create_base_steps_blueprint,
    create_branch_points_blueprint,
    create_extends_blueprint,
    create_full_blueprint,
    create_inheritance_grandparent,
    create_inheritance_parent,
    create_minimal_blueprint,
    create_step_insert_after_blueprint,
    create_step_insert_end_blueprint,
    create_step_insert_start_blueprint,
    create_step_remove_blueprint,
    create_step_replace_blueprint,
)


class TestBasicParsing:
    """Test suite for basic blueprint parsing."""

    @pytest.mark.fast
    def test_parse_minimal_blueprint(self, tmp_path, simple_onnx_model):
        """Test parsing a minimal valid blueprint."""
        blueprint_path = create_minimal_blueprint(
            tmp_path,
            name="test_minimal",
            steps=["custom:test_step", "custom:test_step1", "custom:test_step2"],
        )

        design_space, blueprint_config = parse_blueprint(
            str(blueprint_path), str(simple_onnx_model)
        )

        assert design_space.model_path == str(simple_onnx_model)
        assert design_space.steps == ["custom:test_step", "custom:test_step1", "custom:test_step2"]
        assert blueprint_config.clock_ns == 5.0

    @pytest.mark.fast
    def test_parse_full_blueprint(self, tmp_path, simple_onnx_model):
        """Test parsing a complete blueprint with all options."""
        blueprint_path = create_full_blueprint(
            tmp_path,
            name="test_full",
            description="Full blueprint test",
            clock_ns=3.5,
            output="bitfile",
            board="Pynq-Z1",
            steps=["custom:test_step"],
        )

        design_space, blueprint_config = parse_blueprint(
            str(blueprint_path), str(simple_onnx_model)
        )

        assert blueprint_config.clock_ns == 3.5
        assert blueprint_config.output == OutputType.BITFILE
        assert blueprint_config.board == "Pynq-Z1"

    @pytest.mark.fast
    def test_blueprint_config_defaults(self, tmp_path, simple_onnx_model):
        """Test that blueprint config uses proper defaults."""
        minimal_path = create_minimal_blueprint(
            tmp_path, name="minimal", steps=["custom:test_step"]
        )

        _, blueprint_config = parse_blueprint(str(minimal_path), str(simple_onnx_model))

        # Verify defaults
        assert blueprint_config.output == OutputType.ESTIMATES
        assert blueprint_config.board is None


class TestBlueprintInheritance:
    """Test suite for blueprint inheritance."""

    @pytest.mark.fast
    def test_single_level_inheritance(self, tmp_path, simple_onnx_model):
        """Test single level blueprint inheritance."""
        # Create parent blueprint
        create_inheritance_parent(
            tmp_path,
            name="parent",
            steps=["custom:test_step", "custom:test_step1"],
            clock_ns=10.0,
            board="Pynq-Z1",
        )

        # Create child blueprint extending parent
        child_path = create_extends_blueprint(
            tmp_path,
            name="child",
            extends="parent.yaml",
            clock_ns=3.0,
            steps=["custom:test_step1", "custom:test_step2"],
        )

        design_space, blueprint_config = parse_blueprint(str(child_path), str(simple_onnx_model))

        # Child overrides
        assert blueprint_config.clock_ns == 3.0
        assert design_space.steps == ["custom:test_step1", "custom:test_step2"]

        # Parent values inherited
        assert blueprint_config.board == "Pynq-Z1"

    @pytest.mark.fast
    def test_multi_level_inheritance(self, tmp_path, simple_onnx_model):
        """Test multi-level inheritance chain (grandparent → parent → child)."""
        # Create grandparent
        create_inheritance_grandparent(
            tmp_path, name="grandparent", clock_ns=10.0, board="ZCU102", steps=["custom:test_step"]
        )

        # Create parent extending grandparent
        create_extends_blueprint(
            tmp_path,
            name="parent",
            extends="grandparent.yaml",
            clock_ns=7.0,
            steps=["custom:test_step1"],
        )

        # Create child extending parent
        child_path = create_extends_blueprint(
            tmp_path, name="child", extends="parent.yaml", steps=["custom:test_step2"]
        )

        design_space, blueprint_config = parse_blueprint(str(child_path), str(simple_onnx_model))

        # Verify inheritance chain
        assert design_space.steps == ["custom:test_step2"]  # Child override
        assert blueprint_config.clock_ns == 7.0  # Parent override
        assert blueprint_config.board == "ZCU102"  # Grandparent value

    @pytest.mark.fast
    def test_inheritance_with_step_lists(self, tmp_path, simple_onnx_model):
        """Test inheritance behavior when overriding step lists."""
        # Parent has steps list
        create_inheritance_parent(
            tmp_path,
            name="parent",
            steps=["custom:test_step", "custom:test_step1", "custom:test_step2"],
        )

        # Child replaces entire steps list
        child_path = create_extends_blueprint(
            tmp_path,
            name="child",
            extends="parent.yaml",
            steps=["custom:test_step1", "custom:test_step2"],  # Complete replacement
        )

        design_space, _ = parse_blueprint(str(child_path), str(simple_onnx_model))

        # Verify list replacement (not merge)
        assert design_space.steps == ["custom:test_step1", "custom:test_step2"]


class TestStepOperations:
    """Test suite for step operation features."""

    @pytest.mark.fast
    def test_insert_after(self, tmp_path, simple_onnx_model):
        """Test insert_after step operation."""
        create_base_steps_blueprint(tmp_path)
        after_path = create_step_insert_after_blueprint(tmp_path)

        design_space, _ = parse_blueprint(str(after_path), str(simple_onnx_model))

        # Verify insertion after target step
        expected = [
            "custom:test_step",
            "custom:test_identity_step",
            "custom:test_step1",
            "custom:test_step2",
        ]
        assert design_space.steps == expected

    @pytest.mark.fast
    def test_insert_at_start(self, tmp_path, simple_onnx_model):
        """Test insert_at_start step operation."""
        create_base_steps_blueprint(tmp_path)
        start_path = create_step_insert_start_blueprint(tmp_path)

        design_space, _ = parse_blueprint(str(start_path), str(simple_onnx_model))

        # Verify insertion at beginning
        expected = [
            "custom:test_transform_sequence_step",
            "custom:test_step",
            "custom:test_step1",
            "custom:test_step2",
        ]
        assert design_space.steps == expected

    @pytest.mark.fast
    def test_insert_at_end(self, tmp_path, simple_onnx_model):
        """Test insert_at_end step operation."""
        create_base_steps_blueprint(tmp_path)
        end_path = create_step_insert_end_blueprint(tmp_path)

        design_space, _ = parse_blueprint(str(end_path), str(simple_onnx_model))

        # Verify insertion at end
        expected = [
            "custom:test_step",
            "custom:test_step1",
            "custom:test_step2",
            "custom:test_identity_step",
        ]
        assert design_space.steps == expected

    @pytest.mark.fast
    def test_replace_step(self, tmp_path, simple_onnx_model):
        """Test replace step operation."""
        create_base_steps_blueprint(tmp_path)
        replace_path = create_step_replace_blueprint(tmp_path)

        design_space, _ = parse_blueprint(str(replace_path), str(simple_onnx_model))

        # Verify replacement with order preservation
        expected = ["custom:test_step", "custom:test_identity_step", "custom:test_step2"]
        assert design_space.steps == expected

    @pytest.mark.fast
    def test_remove_step(self, tmp_path, simple_onnx_model):
        """Test remove step operation."""
        create_base_steps_blueprint(tmp_path)
        remove_path = create_step_remove_blueprint(tmp_path)

        design_space, _ = parse_blueprint(str(remove_path), str(simple_onnx_model))

        # Verify step removal
        expected = ["custom:test_step", "custom:test_step2"]
        assert design_space.steps == expected


class TestBranchPoints:
    """Test suite for branch point parsing."""

    @pytest.mark.fast
    def test_branch_point_syntax(self, tmp_path, simple_onnx_model):
        """Test parsing branch points in blueprint YAML."""
        blueprint_path = create_branch_points_blueprint(tmp_path)

        design_space, _ = parse_blueprint(str(blueprint_path), str(simple_onnx_model))

        # Verify branch structure
        assert design_space.steps[0] == "custom:test_step"
        assert isinstance(design_space.steps[1], list)
        assert len(design_space.steps[1]) == 3  # test_step1, test_step2, ~

    @pytest.mark.fast
    def test_branch_with_skip(self, tmp_path, simple_onnx_model):
        """Test branch points with skip option (~)."""
        blueprint_path = create_branch_points_blueprint(tmp_path)

        design_space, _ = parse_blueprint(str(blueprint_path), str(simple_onnx_model))

        # Verify skip option preserved
        branch_point = design_space.steps[1]
        assert isinstance(branch_point, list)
        assert "~" in branch_point or None in branch_point

    @pytest.mark.fast
    def test_branch_without_skip(self, tmp_path, simple_onnx_model):
        """Test branch points without skip option."""
        # Create blueprint with branch but no skip
        blueprint_yaml = """
name: test_branch_no_skip
clock_ns: 5.0
design_space:
  steps:
    - custom:test_step
    - [custom:test_step1, custom:test_step2]
    - custom:test_identity_step
"""
        blueprint_path = tmp_path / "branch_no_skip.yaml"
        blueprint_path.write_text(blueprint_yaml)

        design_space, _ = parse_blueprint(str(blueprint_path), str(simple_onnx_model))

        # Verify branch structure without skip
        assert design_space.steps[0] == "custom:test_step"
        branch = design_space.steps[1]
        assert isinstance(branch, list)
        assert len(branch) == 2
        assert "custom:test_step1" in branch
        assert "custom:test_step2" in branch
        assert "~" not in branch
