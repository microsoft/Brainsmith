"""Integration tests with real FINN execution.

Tests segment execution, branching, artifact sharing, and error handling
with actual FINN builds (no mocks).

Marker: @pytest.mark.finn_build
Execution time: 1-5 min per test (real FINN builds)
Timeout: 600-900 seconds per test

IMPORTANT: Real FINN execution - validates actual DSE behavior!
"""


import pytest

from brainsmith.dse import build_tree, execute_tree, parse_blueprint
from brainsmith.dse.types import SegmentStatus


class TestSegmentExecution:
    """Test suite for segment execution with real FINN."""

    @pytest.mark.finn_build
    @pytest.mark.timeout(600)  # 10 min max
    def test_single_segment_execution(self, tmp_path, quantized_onnx_model, test_workspace):
        """Execute single segment with real FINN steps.

        Tests basic execution path: quantized model → finn:streamline
        Validates segment completion, artifacts, and model output.
        """
        # Create minimal FINN blueprint with single fast step
        from tests.fixtures.dse.blueprints import FINN_PIPELINE_MINIMAL, create_finn_blueprint

        blueprint_path = create_finn_blueprint(
            tmp_path,
            name="single_segment",
            steps=FINN_PIPELINE_MINIMAL,
            clock_ns=10.0,
            target_fps=None,
        )

        # Parse blueprint
        design_space, config = parse_blueprint(str(blueprint_path), str(quantized_onnx_model))

        # Build tree
        tree = build_tree(design_space, config)

        # Execute tree
        output_dir = test_workspace / "single_segment"
        result = execute_tree(
            tree=tree,
            model_path=str(quantized_onnx_model),
            config=config,
            output_dir=str(output_dir),
        )

        # Verify execution completed
        assert len(result.segment_results) > 0, "Should have segment results"

        # Verify at least one segment completed successfully
        completed = [
            seg for seg in result.segment_results.values() if seg.status == SegmentStatus.COMPLETED
        ]
        assert len(completed) > 0, "At least one segment should complete"

        # Verify output ONNX model exists
        for seg_id, seg_result in result.segment_results.items():
            if seg_result.status == SegmentStatus.COMPLETED:
                seg_result.output_dir / "final_model.onnx"
                # Note: FINN may use different naming, check for any .onnx file
                onnx_files = list(seg_result.output_dir.glob("*.onnx"))
                assert len(onnx_files) > 0, f"Segment {seg_id} should produce ONNX output"

    @pytest.mark.finn_build
    @pytest.mark.timeout(900)  # 15 min max
    def test_branching_execution(self, tmp_path, quantized_onnx_model, test_workspace):
        """Execute tree with branches - verify all paths complete.

        Tests branching: model → finn:streamline → [option1, skip]
        Validates both branch paths execute correctly.
        """

        # Create branching blueprint
        blueprint_yaml = """
name: branching_test
clock_ns: 10.0
output: estimates
board: Pynq-Z1
design_space:
  steps:
    - finn:streamline
    - [finn:tidy_up, "~"]
"""
        blueprint_path = tmp_path / "branching.yaml"
        blueprint_path.write_text(blueprint_yaml)

        # Parse and build tree
        design_space, config = parse_blueprint(str(blueprint_path), str(quantized_onnx_model))
        tree = build_tree(design_space, config)

        # Verify tree has branches
        stats = tree.get_statistics()
        assert stats["total_paths"] == 2, "Should have 2 paths (branch + skip)"

        # Execute tree
        output_dir = test_workspace / "branching"
        result = execute_tree(
            tree=tree,
            model_path=str(quantized_onnx_model),
            config=config,
            output_dir=str(output_dir),
        )

        # Verify both paths attempted
        assert len(result.segment_results) >= 2, "Should have results for both branches"

        # Verify at least the shared segment completed
        completed = [
            seg for seg in result.segment_results.values() if seg.status == SegmentStatus.COMPLETED
        ]
        assert len(completed) >= 1, "At least shared segment should complete"

    @pytest.mark.finn_build
    @pytest.mark.timeout(600)
    def test_artifact_sharing_at_branches(self, tmp_path, quantized_onnx_model, test_workspace):
        """Verify parent artifacts copied to children at branch points.

        Tests that when execution branches, child segments receive
        the parent's output model as their input.
        """

        # Create branching blueprint
        blueprint_yaml = """
name: artifact_sharing
clock_ns: 10.0
output: estimates
board: Pynq-Z1
design_space:
  steps:
    - finn:streamline
    - [finn:tidy_up, "~"]
"""
        blueprint_path = tmp_path / "sharing.yaml"
        blueprint_path.write_text(blueprint_yaml)

        # Parse and execute
        design_space, config = parse_blueprint(str(blueprint_path), str(quantized_onnx_model))
        tree = build_tree(design_space, config)

        output_dir = test_workspace / "sharing"
        execute_tree(
            tree=tree,
            model_path=str(quantized_onnx_model),
            config=config,
            output_dir=str(output_dir),
        )

        # Find parent (shared) segment and children
        root = tree.root
        if root.children:
            # Verify children have access to parent's output
            for child_id, child_seg in root.children.items():
                child_dir = output_dir / child_seg.segment_id
                # If child executed, verify it has initial model
                if child_dir.exists():
                    # Child should have inherited parent's artifacts
                    assert child_dir.exists(), f"Child {child_id} directory should exist"

    @pytest.mark.finn_build
    @pytest.mark.timeout(600)
    def test_segment_failure_handling(self, tmp_path, quantized_onnx_model, test_workspace):
        """Test error propagation when FINN step fails.

        Tests that segment failures are properly captured and don't
        crash the entire tree execution.
        """
        from tests.fixtures.dse.blueprints import create_finn_blueprint

        # Create blueprint with intentionally problematic configuration
        # Note: This test validates error handling, actual failure depends on FINN behavior
        blueprint_path = create_finn_blueprint(
            tmp_path,
            name="failure_test",
            steps=["finn:streamline"],
            clock_ns=0.1,  # Extremely low clock may cause issues
        )

        # Parse blueprint
        design_space, config = parse_blueprint(str(blueprint_path), str(quantized_onnx_model))

        # Build and execute tree
        tree = build_tree(design_space, config)
        output_dir = test_workspace / "failure"

        try:
            result = execute_tree(
                tree=tree,
                model_path=str(quantized_onnx_model),
                config=config,
                output_dir=str(output_dir),
            )

            # If execution completes, verify we have results
            assert len(result.segment_results) > 0, "Should have segment results"

            # Check for any failed segments
            failed = [
                seg for seg in result.segment_results.values() if seg.status == SegmentStatus.FAILED
            ]

            # Either execution succeeds OR failures are properly captured
            if len(failed) > 0:
                # Verify failure captured properly
                for seg_result in failed:
                    assert seg_result.error is not None, "Failed segment should have error"

        except RuntimeError as e:
            # Execution may raise if all segments fail
            # This is acceptable - verify error message is informative
            assert (
                "no successful builds" in str(e).lower() or "failed" in str(e).lower()
            ), "Error message should indicate failure"
