"""Unit tests for step slicing utilities."""

import pytest
from brainsmith.dse.design_space import slice_steps, find_step_index


class TestFindStepIndex:
    """Tests for find_step_index function."""

    def test_find_linear_step(self):
        """Find step in linear sequence."""
        steps = ["step1", "step2", "step3"]
        assert find_step_index(steps, "step1") == 0
        assert find_step_index(steps, "step2") == 1
        assert find_step_index(steps, "step3") == 2

    def test_find_step_in_branch_point(self):
        """Find step within branch point list."""
        steps = ["step1", ["step2a", "step2b"], "step3"]
        assert find_step_index(steps, "step2a") == 1
        assert find_step_index(steps, "step2b") == 1

    def test_find_step_with_skip(self):
        """Find step in branch with skip operator."""
        steps = ["step1", ["step2", "~"], "step3"]
        assert find_step_index(steps, "step2") == 1
        assert find_step_index(steps, "~") == 1

    def test_find_nonexistent_step(self):
        """Raise error for nonexistent step."""
        steps = ["step1", "step2", "step3"]
        with pytest.raises(ValueError, match="Step 'invalid' not found"):
            find_step_index(steps, "invalid")

    def test_error_message_shows_available_steps(self):
        """Error message includes available steps."""
        steps = ["step1", ["step2a", "step2b"], "step3"]
        with pytest.raises(ValueError) as exc:
            find_step_index(steps, "invalid")
        assert "step1" in str(exc.value)
        assert "[step2a, step2b]" in str(exc.value)
        assert "step3" in str(exc.value)


class TestSliceSteps:
    """Tests for slice_steps function."""

    def test_slice_with_start_and_stop(self):
        """Slice from start to stop (inclusive)."""
        steps = ["step1", "step2", "step3", "step4"]
        result = slice_steps(steps, "step2", "step3")
        assert result == ["step2", "step3"]

    def test_slice_with_start_only(self):
        """Slice from start to end."""
        steps = ["step1", "step2", "step3", "step4"]
        result = slice_steps(steps, "step2", None)
        assert result == ["step2", "step3", "step4"]

    def test_slice_with_stop_only(self):
        """Slice from beginning to stop."""
        steps = ["step1", "step2", "step3", "step4"]
        result = slice_steps(steps, None, "step3")
        assert result == ["step1", "step2", "step3"]

    def test_slice_preserves_branch_points(self):
        """Slicing preserves branch structure."""
        steps = ["step1", ["step2a", "step2b"], "step3", "step4"]
        result = slice_steps(steps, "step1", "step3")
        assert result == ["step1", ["step2a", "step2b"], "step3"]

    def test_slice_single_step(self):
        """Slice to single step (start == stop)."""
        steps = ["step1", "step2", "step3"]
        result = slice_steps(steps, "step2", "step2")
        assert result == ["step2"]

    def test_slice_branch_point_by_member(self):
        """Can slice to branch point using any member step."""
        steps = ["step1", ["step2a", "step2b"], "step3"]
        result = slice_steps(steps, "step2a", "step3")
        assert result == [["step2a", "step2b"], "step3"]

    def test_slice_invalid_range(self):
        """Error when start comes after stop."""
        steps = ["step1", "step2", "step3"]
        with pytest.raises(ValueError, match="Invalid step range"):
            slice_steps(steps, "step3", "step1")

    def test_slice_none_none_returns_all(self):
        """None/None returns full list."""
        steps = ["step1", "step2", "step3"]
        result = slice_steps(steps, None, None)
        assert result == steps
