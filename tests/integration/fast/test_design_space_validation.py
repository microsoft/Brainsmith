"""Fast integration tests for design space validation.

Tests design space utilities: step slicing, combination limits, step indexing.

Ported from: OLD_FOR_REFERENCE_ONLY/unit/test_step_slicing.py
Marker: @pytest.mark.fast
Execution time: < 1 min (pure Python validation)
"""

import pytest
from brainsmith.dse.design_space import _slice_steps, _find_step_index, GlobalDesignSpace


class TestFindStepIndex:
    """Tests for _find_step_index function."""

    @pytest.mark.fast
    def test_find_linear_step(self):
        """Find step in linear sequence."""
        steps = ["step1", "step2", "step3"]
        assert _find_step_index(steps, "step1") == 0
        assert _find_step_index(steps, "step2") == 1
        assert _find_step_index(steps, "step3") == 2

    @pytest.mark.fast
    def test_find_step_in_branch_point(self):
        """Find step within branch point list."""
        steps = ["step1", ["step2a", "step2b"], "step3"]
        assert _find_step_index(steps, "step2a") == 1
        assert _find_step_index(steps, "step2b") == 1

    @pytest.mark.fast
    def test_find_nonexistent_step(self):
        """Raise error for nonexistent step."""
        steps = ["step1", "step2", "step3"]
        with pytest.raises(ValueError, match="Step 'invalid' not found"):
            _find_step_index(steps, "invalid")

    @pytest.mark.fast
    def test_error_message_shows_available_steps(self):
        """Error message includes available steps."""
        steps = ["step1", ["step2a", "step2b"], "step3"]
        with pytest.raises(ValueError) as exc:
            _find_step_index(steps, "invalid")
        assert "step1" in str(exc.value)
        assert "[step2a, step2b]" in str(exc.value) or "step2a" in str(exc.value)
        assert "step3" in str(exc.value)


class TestSliceSteps:
    """Tests for _slice_steps function."""

    @pytest.mark.fast
    def test_slice_with_start_and_stop(self):
        """Slice from start to stop (inclusive)."""
        steps = ["step1", "step2", "step3", "step4"]
        result = _slice_steps(steps, "step2", "step3")
        assert result == ["step2", "step3"]

    @pytest.mark.fast
    def test_slice_with_start_only(self):
        """Slice from start to end."""
        steps = ["step1", "step2", "step3", "step4"]
        result = _slice_steps(steps, "step2", None)
        assert result == ["step2", "step3", "step4"]

    @pytest.mark.fast
    def test_slice_with_stop_only(self):
        """Slice from beginning to stop."""
        steps = ["step1", "step2", "step3", "step4"]
        result = _slice_steps(steps, None, "step3")
        assert result == ["step1", "step2", "step3"]

    @pytest.mark.fast
    def test_slice_preserves_branch_points(self):
        """Slicing preserves branch structure."""
        steps = ["step1", ["step2a", "step2b"], "step3", "step4"]
        result = _slice_steps(steps, "step1", "step3")
        assert result == ["step1", ["step2a", "step2b"], "step3"]

    @pytest.mark.fast
    def test_slice_single_step(self):
        """Slice to single step (start == stop)."""
        steps = ["step1", "step2", "step3"]
        result = _slice_steps(steps, "step2", "step2")
        assert result == ["step2"]

    @pytest.mark.fast
    def test_slice_branch_point_by_member(self):
        """Can slice to branch point using any member step."""
        steps = ["step1", ["step2a", "step2b"], "step3"]
        result = _slice_steps(steps, "step2a", "step3")
        assert result == [["step2a", "step2b"], "step3"]

    @pytest.mark.fast
    def test_slice_invalid_range(self):
        """Error when start comes after stop."""
        steps = ["step1", "step2", "step3"]
        with pytest.raises(ValueError, match="Invalid step range"):
            _slice_steps(steps, "step3", "step1")


class TestDesignSpaceCombinationLimits:
    """Tests for design space size validation."""

    @pytest.mark.fast
    def test_linear_space_within_limit(self, simple_onnx_model):
        """Linear design space is always valid."""
        design_space = GlobalDesignSpace(
            model_path=str(simple_onnx_model),
            steps=["step1", "step2", "step3", "step4", "step5"],
            kernel_backends=[],
            max_combinations=100
        )
        # Should not raise - linear space has 1 combination
        assert design_space is not None

    @pytest.mark.fast
    def test_branching_space_within_limit(self, simple_onnx_model):
        """Branching design space under limit."""
        design_space = GlobalDesignSpace(
            model_path=str(simple_onnx_model),
            steps=[
                "step1",
                ["opt1", "opt2"],  # 2 options
                "step3",
                ["opt_a", "opt_b", "opt_c"],  # 3 options
                "step5"
                # Total: 2 * 3 = 6 combinations
            ],
            kernel_backends=[],
            max_combinations=100
        )
        assert design_space is not None

    @pytest.mark.fast
    def test_exceeds_combination_limit(self, simple_onnx_model):
        """Reject design space exceeding combination limit."""
        with pytest.raises(ValueError, match="Design space too large"):
            GlobalDesignSpace(
                model_path=str(simple_onnx_model),
                steps=[
                    ["opt1", "opt2", "opt3", "opt4"],  # 4 options
                    ["opt1", "opt2", "opt3", "opt4"],  # 4 options
                    ["opt1", "opt2", "opt3", "opt4"],  # 4 options
                    # Total: 4^3 = 64 combinations
                ],
                kernel_backends=[],
                max_combinations=50  # Limit: 50
            )

    @pytest.mark.fast
    def test_combination_count_calculation(self, simple_onnx_model):
        """Verify combination count calculation."""
        # 2 * 3 * 2 = 12 combinations
        design_space = GlobalDesignSpace(
            model_path=str(simple_onnx_model),
            steps=[
                ["opt1", "opt2"],           # 2
                ["opt_a", "opt_b", "opt_c"], # 3
                ["final1", "final2"]        # 2
            ],
            kernel_backends=[],
            max_combinations=20
        )
        assert design_space is not None

        # Exactly at limit should pass
        design_space_at_limit = GlobalDesignSpace(
            model_path=str(simple_onnx_model),
            steps=[
                ["opt1", "opt2"],           # 2
                ["opt_a", "opt_b", "opt_c"], # 3
                ["final1", "final2"]        # 2
            ],
            kernel_backends=[],
            max_combinations=12  # Exactly 12 combinations
        )
        assert design_space_at_limit is not None

        # One over limit should fail
        with pytest.raises(ValueError, match="Design space too large"):
            GlobalDesignSpace(
                model_path=str(simple_onnx_model),
                steps=[
                    ["opt1", "opt2"],           # 2
                    ["opt_a", "opt_b", "opt_c"], # 3
                    ["final1", "final2"]        # 2
                ],
                kernel_backends=[],
                max_combinations=11  # 11 < 12
            )
