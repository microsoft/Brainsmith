"""Unit tests for Phase 2 data structures."""

import pytest
from datetime import datetime, timedelta

from brainsmith.core.phase2.data_structures import (
    BuildConfig,
    BuildResult,
    BuildStatus,
    ExplorationResults,
)
from brainsmith.core.phase1.data_structures import (
    BuildMetrics,
    ProcessingStep,
    GlobalConfig,
    OutputStage,
)


class TestBuildConfig:
    """Test BuildConfig data structure."""
    
    def test_build_config_creation(self):
        """Test creating a BuildConfig."""
        config = BuildConfig(
            id="test_001",
            design_space_id="dse_12345",
            kernels=[("Gemm", ["rtl", "hls"]), ("Conv", ["hls"])],
            transforms={"default": ["quantize", "fold"]},
            preprocessing=[ProcessingStep("resize", "transform", {"size": 224})],
            postprocessing=[ProcessingStep("softmax", "activation", {})],
            build_steps=["synth", "opt", "place", "route"],
            config_flags={"target": "xcu250"},
            global_config=GlobalConfig(
                output_stage=OutputStage.RTL,
                working_directory="/tmp/work"
            ),
            combination_index=5,
            total_combinations=100
        )
        
        assert config.id == "test_001"
        assert config.design_space_id == "dse_12345"
        assert len(config.kernels) == 2
        assert config.kernels[0] == ("Gemm", ["rtl", "hls"])
        assert len(config.transforms["default"]) == 2
        assert config.transforms["default"] == ["quantize", "fold"]
        assert config.combination_index == 5
        assert config.total_combinations == 100
    
    def test_build_config_str(self):
        """Test BuildConfig string representation."""
        config = BuildConfig(
            id="test_001",
            design_space_id="dse_12345",
            kernels=[("Gemm", ["rtl"]), ("Conv", ["hls"])],
            transforms={"default": ["quantize"]},
            preprocessing=[],
            postprocessing=[],
            build_steps=[],
            config_flags={},
            global_config=GlobalConfig(),
            output_dir="/tmp/brainsmith/dse_12345/builds/config_0",
            combination_index=0,
            total_combinations=10
        )
        
        config_str = str(config)
        assert "test_001" in config_str
        assert "1/10" in config_str
        assert "Gemm[rtl]" in config_str
        assert "Conv[hls]" in config_str
        assert "Output: /tmp/brainsmith/dse_12345/builds/config_0" in config_str
    
    def test_build_config_to_dict(self):
        """Test BuildConfig serialization."""
        config = BuildConfig(
            id="test_001",
            design_space_id="dse_12345",
            kernels=[("Gemm", ["rtl"])],
            transforms={"default": ["quantize"]},
            preprocessing=[ProcessingStep("resize", "transform", {"size": 224})],
            postprocessing=[],
            build_steps=["synth"],
            config_flags={"opt": True},
            global_config=GlobalConfig(),
            output_dir="/tmp/brainsmith/dse_12345/builds/config_0",
            combination_index=0,
            total_combinations=1
        )
        
        config_dict = config.to_dict()
        assert config_dict["id"] == "test_001"
        assert config_dict["kernels"] == [("Gemm", ["rtl"])]
        assert config_dict["transforms"] == {"default": ["quantize"]}
        assert len(config_dict["preprocessing"]) == 1
        assert config_dict["preprocessing"][0]["name"] == "resize"
        assert "timestamp" in config_dict
        assert config_dict["output_dir"] == "/tmp/brainsmith/dse_12345/builds/config_0"


class TestBuildResult:
    """Test BuildResult data structure."""
    
    def test_build_result_creation(self):
        """Test creating a BuildResult."""
        result = BuildResult(
            config_id="test_001",
            status=BuildStatus.RUNNING
        )
        
        assert result.config_id == "test_001"
        assert result.status == BuildStatus.RUNNING
        assert result.metrics is None
        assert result.error_message is None
        assert isinstance(result.start_time, datetime)
    
    def test_build_result_complete_success(self):
        """Test completing a successful build."""
        result = BuildResult(
            config_id="test_001",
            status=BuildStatus.RUNNING
        )
        
        # Add metrics
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
        
        # Complete the build
        result.complete(BuildStatus.SUCCESS)
        
        assert result.status == BuildStatus.SUCCESS
        assert result.end_time is not None
        assert result.duration_seconds > 0
        assert result.error_message is None
    
    def test_build_result_complete_failure(self):
        """Test completing a failed build."""
        result = BuildResult(
            config_id="test_001",
            status=BuildStatus.RUNNING
        )
        
        # Complete with error
        result.complete(BuildStatus.FAILED, "Timing constraints not met")
        
        assert result.status == BuildStatus.FAILED
        assert result.end_time is not None
        assert result.error_message == "Timing constraints not met"
        assert result.metrics is None
    
    def test_build_result_str(self):
        """Test BuildResult string representation."""
        result = BuildResult(
            config_id="test_001",
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
        
        result_str = str(result)
        assert "test_001" in result_str
        assert "success" in result_str
        assert "1234.56" in result_str
        assert "12.34" in result_str


class TestExplorationResults:
    """Test ExplorationResults data structure."""
    
    def test_exploration_results_creation(self):
        """Test creating ExplorationResults."""
        start_time = datetime.now()
        results = ExplorationResults(
            design_space_id="dse_12345",
            start_time=start_time,
            end_time=start_time + timedelta(hours=1)
        )
        
        assert results.design_space_id == "dse_12345"
        assert results.start_time == start_time
        assert results.total_combinations == 0
        assert results.evaluated_count == 0
        assert len(results.evaluations) == 0
    
    def test_add_and_get_config(self):
        """Test adding and retrieving configurations."""
        results = ExplorationResults(
            design_space_id="dse_12345",
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        # Create and add config
        config = BuildConfig(
            id="test_001",
            design_space_id="dse_12345",
            kernels=[],
            transforms={},
            preprocessing=[],
            postprocessing=[],
            build_steps=[],
            config_flags={},
            global_config=GlobalConfig()
        )
        
        results.add_config(config)
        
        # Retrieve config
        retrieved = results.get_config("test_001")
        assert retrieved is not None
        assert retrieved.id == "test_001"
        
        # Try non-existent config
        assert results.get_config("non_existent") is None
    
    def test_get_successful_results(self):
        """Test filtering successful results."""
        results = ExplorationResults(
            design_space_id="dse_12345",
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        # Add various results
        for i in range(5):
            status = BuildStatus.SUCCESS if i % 2 == 0 else BuildStatus.FAILED
            result = BuildResult(
                config_id=f"test_{i:03d}",
                status=status
            )
            results.evaluations.append(result)
        
        successful = results.get_successful_results()
        assert len(successful) == 3
        assert all(r.status == BuildStatus.SUCCESS for r in successful)
    
    def test_update_counts(self):
        """Test updating summary counts."""
        results = ExplorationResults(
            design_space_id="dse_12345",
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        # Add various results
        statuses = [
            BuildStatus.SUCCESS,
            BuildStatus.SUCCESS,
            BuildStatus.FAILED,
            BuildStatus.SKIPPED,
            BuildStatus.SUCCESS,
        ]
        
        for i, status in enumerate(statuses):
            result = BuildResult(
                config_id=f"test_{i:03d}",
                status=status
            )
            results.evaluations.append(result)
        
        results.update_counts()
        
        assert results.evaluated_count == 5
        assert results.success_count == 3
        assert results.failure_count == 1
        assert results.skipped_count == 1
    
    def test_get_summary_string(self):
        """Test generating summary string."""
        results = ExplorationResults(
            design_space_id="dse_12345",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=30),
            total_combinations=100
        )
        
        # Add some results
        for i in range(10):
            status = BuildStatus.SUCCESS if i < 7 else BuildStatus.FAILED
            result = BuildResult(
                config_id=f"test_{i:03d}",
                status=status
            )
            if status == BuildStatus.SUCCESS:
                result.metrics = BuildMetrics(
                    throughput=1000.0 + i * 10,
                    latency=10.0 - i * 0.1,
                    clock_frequency=250.0,
                    lut_utilization=0.6,
                    dsp_utilization=0.4,
                    bram_utilization=0.3,
                    total_power=10.0,
                    accuracy=0.98
                )
            results.evaluations.append(result)
        
        results.update_counts()
        
        # Set best config
        config = BuildConfig(
            id="test_006",
            design_space_id="dse_12345",
            kernels=[],
            transforms={},
            preprocessing=[],
            postprocessing=[],
            build_steps=[],
            config_flags={},
            global_config=GlobalConfig()
        )
        results.add_config(config)
        results.best_config = config
        
        summary = results.get_summary_string()
        
        assert "dse_12345" in summary
        assert "100" in summary  # total combinations
        assert "10" in summary   # evaluated
        assert "7" in summary    # successful
        assert "70.0%" in summary  # success rate
        assert "test_006" in summary  # best config