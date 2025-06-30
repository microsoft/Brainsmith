"""Unit tests for the explorer engine."""

import pytest
from datetime import datetime
from unittest.mock import Mock, call

from brainsmith.core.phase2.explorer import ExplorerEngine
from brainsmith.core.phase2.data_structures import (
    BuildResult,
    BuildStatus,
)
from brainsmith.core.phase2.hooks import ExplorationHook
from brainsmith.core.phase2.interfaces import BuildRunnerInterface, MockBuildRunner
from brainsmith.core.phase1.data_structures import (
    DesignSpace,
    HWCompilerSpace,
    ProcessingSpace,
    SearchConfig,
    SearchStrategy,
    GlobalConfig,
    BuildMetrics,
)


class TestExplorerEngine:
    """Test the ExplorerEngine class."""
    
    @pytest.fixture
    def simple_design_space(self):
        """Create a simple design space for testing."""
        return DesignSpace(
            model_path="/path/to/model.onnx",
            hw_compiler_space=HWCompilerSpace(
                kernels=[["Gemm", "Conv"]],  # Mutually exclusive kernels
                transforms={"default": ["quantize"]},
                build_steps=["synth"],
                config_flags={}
            ),
            processing_space=ProcessingSpace(),
            search_config=SearchConfig(
                strategy=SearchStrategy.EXHAUSTIVE,
                max_evaluations=None,
                timeout_minutes=None
            ),
            global_config=GlobalConfig()
        )
    
    def test_explorer_initialization(self):
        """Test explorer engine initialization."""
        build_runner_factory = Mock(return_value=MockBuildRunner())
        hooks = [Mock(spec=ExplorationHook)]
        
        explorer = ExplorerEngine(build_runner_factory, hooks)
        
        assert explorer.build_runner_factory == build_runner_factory
        assert explorer.hooks == hooks
        assert explorer.progress_tracker is None
        assert explorer.exploration_results is None
    
    def test_explore_simple_space(self, simple_design_space):
        """Test exploring a simple design space."""
        # Create mock build runner
        mock_runner = MockBuildRunner(success_rate=1.0, simulate_delay=False)
        build_runner_factory = Mock(return_value=mock_runner)
        
        # Create explorer
        explorer = ExplorerEngine(build_runner_factory)
        
        # Run exploration
        results = explorer.explore(simple_design_space)
        
        # Check results
        assert results.design_space_id.startswith("dse_")
        assert results.total_combinations == 2  # 2 kernels * 1 transform
        assert results.evaluated_count == 2
        assert results.success_count == 2
        assert results.failure_count == 0
        
        # Check that best config was found
        assert results.best_config is not None
        assert len(results.pareto_optimal) > 0
    
    def test_explore_with_failures(self, simple_design_space):
        """Test exploration with some build failures."""
        # Create mock build runner with 50% success rate
        mock_runner = MockBuildRunner(success_rate=0.5, simulate_delay=False)
        build_runner_factory = Mock(return_value=mock_runner)
        
        # Create explorer
        explorer = ExplorerEngine(build_runner_factory)
        
        # Run exploration
        results = explorer.explore(simple_design_space)
        
        # Check that we have both successes and failures
        assert results.evaluated_count == 2
        assert results.success_count + results.failure_count == 2
        
        # If we have successes, best config should be found
        if results.success_count > 0:
            assert results.best_config is not None
    
    def test_explore_with_max_evaluations(self, simple_design_space):
        """Test exploration with max evaluations limit."""
        # Set max evaluations to 1
        simple_design_space.search_config.max_evaluations = 1
        
        mock_runner = MockBuildRunner(success_rate=1.0, simulate_delay=False)
        build_runner_factory = Mock(return_value=mock_runner)
        
        explorer = ExplorerEngine(build_runner_factory)
        results = explorer.explore(simple_design_space)
        
        # Should only evaluate 1 configuration
        assert results.evaluated_count == 1
        assert results.total_combinations == 2
    
    def test_explore_with_hooks(self, simple_design_space):
        """Test exploration with hooks."""
        # Create mock hook
        mock_hook = Mock(spec=ExplorationHook)
        
        mock_runner = MockBuildRunner(success_rate=1.0, simulate_delay=False)
        build_runner_factory = Mock(return_value=mock_runner)
        
        explorer = ExplorerEngine(build_runner_factory, hooks=[mock_hook])
        results = explorer.explore(simple_design_space)
        
        # Check hook calls
        assert mock_hook.on_exploration_start.called
        assert mock_hook.on_combinations_generated.called
        assert mock_hook.on_build_complete.call_count == 2  # 2 builds
        assert mock_hook.on_exploration_complete.called
        
        # Check hook was called with correct arguments
        mock_hook.on_exploration_start.assert_called_once()
        args = mock_hook.on_exploration_start.call_args[0]
        assert args[0] == simple_design_space
        assert args[1] == results
    
    def test_explore_with_resume(self, simple_design_space):
        """Test exploration with resume functionality."""
        mock_runner = MockBuildRunner(success_rate=1.0, simulate_delay=False)
        build_runner_factory = Mock(return_value=mock_runner)
        
        # First exploration
        explorer1 = ExplorerEngine(build_runner_factory)
        results1 = explorer1.explore(simple_design_space)
        
        # Get ID of first config
        first_config_id = results1.evaluations[0].config_id
        
        # Resume from first config
        explorer2 = ExplorerEngine(build_runner_factory)
        results2 = explorer2.explore(simple_design_space, resume_from=first_config_id)
        
        # Should only evaluate remaining configs
        assert results2.evaluated_count == 1  # Only the second config
        assert results2.total_combinations == 2
    
    def test_evaluate_config_success(self, simple_design_space):
        """Test evaluating a successful configuration."""
        # Create mock build runner
        mock_runner = Mock(spec=BuildRunnerInterface)
        mock_result = BuildResult(
            config_id="test_001",
            status=BuildStatus.SUCCESS
        )
        mock_result.metrics = BuildMetrics(
            throughput=1000.0,
            latency=10.0,
            clock_frequency=250.0,
            lut_utilization=0.6,
            dsp_utilization=0.4,
            bram_utilization=0.3,
            total_power=10.0,
            accuracy=0.98
        )
        mock_runner.run.return_value = mock_result
        
        build_runner_factory = Mock(return_value=mock_runner)
        explorer = ExplorerEngine(build_runner_factory)
        
        # Create a config
        from brainsmith.core.phase2.data_structures import BuildConfig
        config = BuildConfig(
            id="test_001",
            design_space_id="dse_test",
            kernels=[],
            transforms={},
            preprocessing=[],
            postprocessing=[],
            build_steps=[],
            config_flags={},
            global_config=GlobalConfig()
        )
        
        # Evaluate
        result = explorer._evaluate_config(config, mock_runner)
        
        assert result.status == BuildStatus.SUCCESS
        assert result.metrics is not None
        assert result.metrics.throughput == 1000.0
    
    def test_evaluate_config_exception(self, simple_design_space):
        """Test evaluating a configuration that throws an exception."""
        # Create mock build runner that throws
        mock_runner = Mock(spec=BuildRunnerInterface)
        mock_runner.run.side_effect = Exception("Unexpected error!")
        
        build_runner_factory = Mock(return_value=mock_runner)
        explorer = ExplorerEngine(build_runner_factory)
        
        # Create a config
        from brainsmith.core.phase2.data_structures import BuildConfig
        config = BuildConfig(
            id="test_001",
            design_space_id="dse_test",
            kernels=[],
            transforms={},
            preprocessing=[],
            postprocessing=[],
            build_steps=[],
            config_flags={},
            global_config=GlobalConfig()
        )
        
        # Evaluate - should handle exception gracefully
        result = explorer._evaluate_config(config, mock_runner)
        
        assert result.status == BuildStatus.FAILED
        assert "Unexpected error" in result.error_message
    
    def test_fire_hook_exception_handling(self, simple_design_space):
        """Test that hook exceptions are handled gracefully."""
        # Create hook that throws
        bad_hook = Mock(spec=ExplorationHook)
        bad_hook.on_exploration_start.side_effect = Exception("Hook error!")
        
        # Create good hook
        good_hook = Mock(spec=ExplorationHook)
        
        mock_runner = MockBuildRunner(success_rate=1.0, simulate_delay=False)
        build_runner_factory = Mock(return_value=mock_runner)
        
        # Explorer with both hooks
        explorer = ExplorerEngine(
            build_runner_factory,
            hooks=[bad_hook, good_hook]
        )
        
        # Should complete exploration despite bad hook
        results = explorer.explore(simple_design_space)
        
        assert results.evaluated_count > 0
        
        # Good hook should still be called
        assert good_hook.on_exploration_start.called
        assert good_hook.on_exploration_complete.called