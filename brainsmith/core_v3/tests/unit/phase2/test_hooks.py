"""Unit tests for the hook system."""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

from brainsmith.core_v3.phase2.hooks import (
    ExplorationHook,
    LoggingHook,
    CachingHook,
    HookRegistry,
)
from brainsmith.core_v3.phase2.data_structures import (
    BuildConfig,
    BuildResult,
    BuildStatus,
    ExplorationResults,
)
from brainsmith.core_v3.phase1.data_structures import (
    DesignSpace,
    HWCompilerSpace,
    ProcessingSpace,
    SearchConfig,
    SearchStrategy,
    GlobalConfig,
    BuildMetrics,
)


class TestExplorationHook:
    """Test the ExplorationHook abstract base class."""
    
    def test_abstract_methods(self):
        """Test that ExplorationHook is abstract."""
        with pytest.raises(TypeError):
            ExplorationHook()
    
    def test_concrete_implementation(self):
        """Test creating a concrete hook implementation."""
        class ConcreteHook(ExplorationHook):
            def on_exploration_start(self, design_space, exploration_results):
                pass
            
            def on_combinations_generated(self, configs):
                pass
            
            def on_build_complete(self, config, result):
                pass
            
            def on_exploration_complete(self, exploration_results):
                pass
        
        # Should be able to instantiate
        hook = ConcreteHook()
        assert isinstance(hook, ExplorationHook)


class TestLoggingHook:
    """Test the LoggingHook class."""
    
    @pytest.fixture
    def sample_design_space(self):
        """Create a sample design space."""
        return DesignSpace(
            model_path="/path/to/model.onnx",
            hw_compiler_space=HWCompilerSpace(
                kernels=["Gemm"],
                transforms={"default": ["quantize"]},
                build_steps=["synth"],
                config_flags={}
            ),
            processing_space=ProcessingSpace(),
            search_config=SearchConfig(
                strategy=SearchStrategy.EXHAUSTIVE,
                max_evaluations=10,
                timeout_minutes=30
            ),
            global_config=GlobalConfig()
        )
    
    def test_logging_hook_creation(self):
        """Test creating a logging hook."""
        hook = LoggingHook(log_level="DEBUG")
        assert hook.log_level == "DEBUG"
        assert hook.log_file is None
    
    def test_on_exploration_start(self, sample_design_space, caplog):
        """Test logging exploration start."""
        # Set up logging to capture at INFO level
        import logging
        caplog.set_level(logging.INFO, logger="brainsmith.core_v3.phase2.hooks")
        
        hook = LoggingHook()
        results = ExplorationResults(
            design_space_id="dse_test123",
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_combinations=100
        )
        
        hook.on_exploration_start(sample_design_space, results)
        
        assert "DESIGN SPACE EXPLORATION STARTED" in caplog.text
        assert "dse_test123" in caplog.text
        assert "exhaustive" in caplog.text
        assert "100" in caplog.text
    
    def test_on_build_complete_success(self, caplog):
        """Test logging successful build completion."""
        import logging
        caplog.set_level(logging.INFO, logger="brainsmith.core_v3.phase2.hooks")
        
        hook = LoggingHook()
        
        config = BuildConfig(
            id="config_001",
            design_space_id="dse_test",
            kernels=[],
            transforms={},
            preprocessing=[],
            postprocessing=[],
            build_steps=[],
            config_flags={},
            global_config=GlobalConfig(),
            combination_index=0,
            total_combinations=10
        )
        
        result = BuildResult(
            config_id="config_001",
            status=BuildStatus.SUCCESS
        )
        result.metrics = BuildMetrics(
            throughput=1234.56,
            latency=12.34,
            clock_frequency=250.0,
            lut_utilization=0.6,
            dsp_utilization=0.4,
            bram_utilization=0.3,
            total_power=10.0,
            accuracy=0.98
        )
        
        hook.on_build_complete(config, result)
        
        assert "✅" in caplog.text
        assert "config_001" in caplog.text
        assert "1/10" in caplog.text
        assert "1234.56" in caplog.text
        assert "12.34" in caplog.text
    
    def test_on_build_complete_failure(self, caplog):
        """Test logging failed build completion."""
        import logging
        caplog.set_level(logging.INFO, logger="brainsmith.core_v3.phase2.hooks")
        
        hook = LoggingHook()
        
        config = BuildConfig(
            id="config_002",
            design_space_id="dse_test",
            kernels=[],
            transforms={},
            preprocessing=[],
            postprocessing=[],
            build_steps=[],
            config_flags={},
            global_config=GlobalConfig(),
            combination_index=1,
            total_combinations=10
        )
        
        result = BuildResult(
            config_id="config_002",
            status=BuildStatus.FAILED,
            error_message="Timing constraints not met"
        )
        
        hook.on_build_complete(config, result)
        
        assert "❌" in caplog.text
        assert "config_002" in caplog.text
        assert "Timing constraints not met" in caplog.text
    
    def test_on_exploration_complete(self, caplog):
        """Test logging exploration completion."""
        import logging
        caplog.set_level(logging.INFO, logger="brainsmith.core_v3.phase2.hooks")
        
        hook = LoggingHook()
        
        results = ExplorationResults(
            design_space_id="dse_test123",
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        # Add some failed results
        for i in range(3):
            result = BuildResult(
                config_id=f"config_{i:03d}",
                status=BuildStatus.FAILED,
                error_message="Timing constraints not met" if i < 2 else "Synthesis failed"
            )
            results.evaluations.append(result)
        
        results.update_counts()
        
        hook.on_exploration_complete(results)
        
        assert "DESIGN SPACE EXPLORATION COMPLETED" in caplog.text
        assert "Failure Summary (3 failures)" in caplog.text
        assert "Timing constraints not met: 2 occurrences" in caplog.text
        assert "Synthesis failed: 1 occurrences" in caplog.text


class TestCachingHook:
    """Test the CachingHook class."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_caching_hook_creation(self, temp_cache_dir):
        """Test creating a caching hook."""
        hook = CachingHook(cache_dir=str(temp_cache_dir))
        assert hook.cache_dir == temp_cache_dir
        assert hook.cache_file is None
    
    def test_on_exploration_start_new(self, temp_cache_dir):
        """Test starting exploration with no existing cache."""
        hook = CachingHook(cache_dir=str(temp_cache_dir))
        
        design_space = DesignSpace(
            model_path="/path/to/model.onnx",
            hw_compiler_space=HWCompilerSpace(
                kernels=["Gemm"],
                transforms={"default": ["quantize"]},
                build_steps=["synth"],
                config_flags={}
            ),
            processing_space=ProcessingSpace(),
            search_config=SearchConfig(strategy=SearchStrategy.EXHAUSTIVE),
            global_config=GlobalConfig()
        )
        
        results = ExplorationResults(
            design_space_id="dse_test123",
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        hook.on_exploration_start(design_space, results)
        
        assert hook.design_space_id == "dse_test123"
        # Cache file should be in exploration directory
        expected_path = Path(design_space.global_config.working_directory) / "dse_test123" / "exploration_cache.jsonl"
        assert hook.cache_file == expected_path
        assert hook.cache_file.parent.exists()
    
    def test_on_build_complete_caching(self, temp_cache_dir):
        """Test caching build results."""
        hook = CachingHook(cache_dir=str(temp_cache_dir))
        hook.cache_file = temp_cache_dir / "test_results.jsonl"
        
        config = BuildConfig(
            id="config_001",
            design_space_id="dse_test",
            kernels=[],
            transforms={},
            preprocessing=[],
            postprocessing=[],
            build_steps=[],
            config_flags={},
            global_config=GlobalConfig()
        )
        
        result = BuildResult(
            config_id="config_001",
            status=BuildStatus.SUCCESS
        )
        result.metrics = BuildMetrics(
            throughput=1000.0,
            latency=10.0,
            clock_frequency=250.0,
            lut_utilization=0.6,
            dsp_utilization=0.4,
            bram_utilization=0.3,
            total_power=10.0,
            accuracy=0.98
        )
        result.complete(BuildStatus.SUCCESS)
        
        hook.on_build_complete(config, result)
        
        # Check cache file was written
        assert hook.cache_file.exists()
        
        # Check content
        with open(hook.cache_file, "r") as f:
            line = f.readline()
            cached = json.loads(line)
            
            assert cached["config_id"] == "config_001"
            assert cached["status"] == "success"
            assert cached["metrics"]["throughput"] == 1000.0
    
    def test_load_cached_results(self, temp_cache_dir):
        """Test loading previously cached results."""
        # Create cache file with some results
        cache_file = temp_cache_dir / "dse_test123_results.jsonl"
        
        cached_entries = [
            {
                "config_id": "config_001",
                "status": "success",
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": 10.5,
                "metrics": {
                    "throughput": 1000.0,
                    "latency": 10.0,
                    "clock_frequency": 250.0,
                    "lut_utilization": 0.6,
                    "dsp_utilization": 0.4,
                    "bram_utilization": 0.3,
                    "total_power": 10.0,
                    "accuracy": 0.98,
                    "custom": {}
                }
            },
            {
                "config_id": "config_002",
                "status": "failed",
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": 5.0,
                "error_message": "Build failed"
            }
        ]
        
        with open(cache_file, "w") as f:
            for entry in cached_entries:
                f.write(json.dumps(entry) + "\n")
        
        # Create hook and load results
        hook = CachingHook(cache_dir=str(temp_cache_dir))
        hook.cache_file = cache_file
        
        results = ExplorationResults(
            design_space_id="dse_test123",
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        hook._load_cached_results(results)
        
        assert len(results.evaluations) == 2
        assert results.evaluations[0].config_id == "config_001"
        assert results.evaluations[0].status == BuildStatus.SUCCESS
        assert results.evaluations[0].metrics.throughput == 1000.0
        assert results.evaluations[1].config_id == "config_002"
        assert results.evaluations[1].status == BuildStatus.FAILED
    
    def test_on_exploration_complete_summary(self, temp_cache_dir):
        """Test saving exploration summary."""
        hook = CachingHook(cache_dir=str(temp_cache_dir))
        hook.design_space_id = "dse_test123"
        hook.cache_file = temp_cache_dir / "dse_test123_results.jsonl"
        
        # Create config for best result
        config = BuildConfig(
            id="config_best",
            design_space_id="dse_test123",
            kernels=[],
            transforms={},
            preprocessing=[],
            postprocessing=[],
            build_steps=[],
            config_flags={},
            global_config=GlobalConfig()
        )
        
        results = ExplorationResults(
            design_space_id="dse_test123",
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_combinations=100,
            evaluated_count=50,
            success_count=45,
            failure_count=5,
            skipped_count=0
        )
        results.best_config = config
        results.pareto_optimal = [config]
        
        hook.on_exploration_complete(results)
        
        # Check summary file in exploration directory
        summary_file = hook.cache_file.parent / "exploration_summary.json"
        assert summary_file.exists()
        
        with open(summary_file, "r") as f:
            summary = json.load(f)
            
            assert summary["design_space_id"] == "dse_test123"
            assert summary["total_combinations"] == 100
            assert summary["evaluated_count"] == 50
            assert summary["success_count"] == 45
            assert summary["best_config_id"] == "config_best"
            assert len(summary["pareto_optimal_ids"]) == 1


class TestHookRegistry:
    """Test the HookRegistry class."""
    
    def test_hook_registry_creation(self):
        """Test creating a hook registry."""
        registry = HookRegistry()
        assert len(registry.hooks) == 0
    
    def test_register_hook(self):
        """Test registering hooks."""
        registry = HookRegistry()
        
        hook1 = LoggingHook()
        hook2 = CachingHook()
        
        registry.register(hook1)
        registry.register(hook2)
        
        assert len(registry.hooks) == 2
        assert hook1 in registry.hooks
        assert hook2 in registry.hooks
    
    def test_unregister_hook(self):
        """Test unregistering hooks."""
        registry = HookRegistry()
        
        hook1 = LoggingHook()
        hook2 = CachingHook()
        
        registry.register(hook1)
        registry.register(hook2)
        
        # Unregister hook1
        registry.unregister(hook1)
        
        assert len(registry.hooks) == 1
        assert hook1 not in registry.hooks
        assert hook2 in registry.hooks
        
        # Try unregistering non-existent hook
        registry.unregister(hook1)  # Should not raise
    
    def test_get_all_hooks(self):
        """Test getting all hooks."""
        registry = HookRegistry()
        
        hook1 = LoggingHook()
        hook2 = CachingHook()
        
        registry.register(hook1)
        registry.register(hook2)
        
        all_hooks = registry.get_all()
        
        # Should return a copy
        assert len(all_hooks) == 2
        assert hook1 in all_hooks
        assert hook2 in all_hooks
        
        # Modifying returned list shouldn't affect registry
        all_hooks.clear()
        assert len(registry.hooks) == 2