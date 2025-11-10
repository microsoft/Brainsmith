"""Fast integration tests for design space validation.

Tests design space combination limits validation.

Marker: @pytest.mark.fast
Execution time: < 1 min (pure Python validation)
"""

import pytest
from brainsmith.dse.design_space import GlobalDesignSpace


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
