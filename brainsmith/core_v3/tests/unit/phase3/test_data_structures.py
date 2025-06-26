"""
Unit tests for Phase 3 data structures.
"""

import pytest
from datetime import datetime
import time

from brainsmith.core_v3.phase3.data_structures import (
    BuildStatus,
    BuildMetrics,
    BuildResult,
)


class TestBuildStatus:
    """Test BuildStatus enum."""
    
    def test_enum_values(self):
        """Test that BuildStatus has correct values."""
        assert BuildStatus.SUCCESS.value == "success"
        assert BuildStatus.FAILED.value == "failed"
        assert BuildStatus.TIMEOUT.value == "timeout"
        assert BuildStatus.SKIPPED.value == "skipped"
    
    def test_enum_members(self):
        """Test that all expected members exist."""
        members = [status.name for status in BuildStatus]
        assert "SUCCESS" in members
        assert "FAILED" in members
        assert "TIMEOUT" in members
        assert "SKIPPED" in members


class TestBuildMetrics:
    """Test BuildMetrics dataclass."""
    
    def test_creation_with_defaults(self):
        """Test creating BuildMetrics with default values."""
        metrics = BuildMetrics()
        
        # Performance metrics
        assert metrics.throughput is None
        assert metrics.latency is None
        assert metrics.clock_frequency is None
        
        # Resource metrics
        assert metrics.lut_utilization is None
        assert metrics.dsp_utilization is None
        assert metrics.bram_utilization is None
        assert metrics.uram_utilization is None
        assert metrics.total_power is None
        
        # Quality metrics
        assert metrics.accuracy is None
        
        # Raw metrics
        assert metrics.raw_metrics == {}
    
    def test_creation_with_values(self):
        """Test creating BuildMetrics with specific values."""
        metrics = BuildMetrics(
            throughput=1000.0,
            latency=10.5,
            clock_frequency=200.0,
            lut_utilization=0.75,
            dsp_utilization=0.5,
            bram_utilization=0.3,
            uram_utilization=0.1,
            total_power=15.5,
            accuracy=0.95,
            raw_metrics={"test": "value"}
        )
        
        assert metrics.throughput == 1000.0
        assert metrics.latency == 10.5
        assert metrics.clock_frequency == 200.0
        assert metrics.lut_utilization == 0.75
        assert metrics.dsp_utilization == 0.5
        assert metrics.bram_utilization == 0.3
        assert metrics.uram_utilization == 0.1
        assert metrics.total_power == 15.5
        assert metrics.accuracy == 0.95
        assert metrics.raw_metrics == {"test": "value"}
    
    def test_partial_metrics(self):
        """Test creating BuildMetrics with only some values."""
        metrics = BuildMetrics(
            throughput=500.0,
            lut_utilization=0.6
        )
        
        assert metrics.throughput == 500.0
        assert metrics.lut_utilization == 0.6
        assert metrics.latency is None
        assert metrics.dsp_utilization is None


class TestBuildResult:
    """Test BuildResult dataclass."""
    
    def test_creation_with_defaults(self):
        """Test creating BuildResult with default values."""
        result = BuildResult(config_id="test_config_001")
        
        assert result.config_id == "test_config_001"
        assert result.status == BuildStatus.SKIPPED
        assert result.metrics is None
        assert isinstance(result.start_time, datetime)
        assert result.end_time is None
        assert result.duration_seconds == 0.0
        assert result.artifacts == {}
        assert result.logs == {}
        assert result.error_message is None
    
    def test_complete_success(self):
        """Test marking build as successfully completed."""
        result = BuildResult(config_id="test_config_001")
        
        # Simulate some work
        time.sleep(0.1)
        
        # Complete the build
        result.complete(BuildStatus.SUCCESS)
        
        assert result.status == BuildStatus.SUCCESS
        assert result.end_time is not None
        assert result.duration_seconds > 0.0
        assert result.error_message is None
    
    def test_complete_failure(self):
        """Test marking build as failed."""
        result = BuildResult(config_id="test_config_001")
        
        # Complete with failure
        result.complete(BuildStatus.FAILED, "Test error message")
        
        assert result.status == BuildStatus.FAILED
        assert result.end_time is not None
        assert result.duration_seconds >= 0.0
        assert result.error_message == "Test error message"
    
    def test_is_successful(self):
        """Test is_successful method."""
        result = BuildResult(config_id="test_config_001")
        
        # Initially not successful
        assert not result.is_successful()
        
        # After successful completion
        result.complete(BuildStatus.SUCCESS)
        assert result.is_successful()
        
        # Failed result
        failed_result = BuildResult(config_id="test_config_002")
        failed_result.complete(BuildStatus.FAILED)
        assert not failed_result.is_successful()
    
    def test_has_metrics(self):
        """Test has_metrics method."""
        result = BuildResult(config_id="test_config_001")
        
        # No metrics initially
        assert not result.has_metrics()
        
        # Add metrics but not successful
        result.metrics = BuildMetrics(throughput=1000.0)
        assert not result.has_metrics()
        
        # Mark as successful
        result.complete(BuildStatus.SUCCESS)
        assert result.has_metrics()
        
        # Failed result with metrics
        failed_result = BuildResult(config_id="test_config_002")
        failed_result.metrics = BuildMetrics(throughput=500.0)
        failed_result.complete(BuildStatus.FAILED)
        assert not failed_result.has_metrics()
    
    def test_duration_calculation(self):
        """Test that duration is calculated correctly."""
        result = BuildResult(config_id="test_config_001")
        start = result.start_time
        
        # Wait a bit
        time.sleep(0.2)
        
        # Complete
        result.complete(BuildStatus.SUCCESS)
        
        # Check duration is reasonable
        assert result.duration_seconds >= 0.2
        assert result.duration_seconds < 1.0  # Should be less than 1 second
        
        # Manual calculation should match
        manual_duration = (result.end_time - start).total_seconds()
        assert abs(result.duration_seconds - manual_duration) < 0.01
    
    def test_with_artifacts_and_logs(self):
        """Test BuildResult with artifacts and logs."""
        result = BuildResult(
            config_id="test_config_001",
            artifacts={
                "performance_report": "/path/to/perf.json",
                "resource_report": "/path/to/resources.json"
            },
            logs={
                "build_log": "/path/to/build.log",
                "error_log": "Error occurred at line 42"
            }
        )
        
        assert len(result.artifacts) == 2
        assert result.artifacts["performance_report"] == "/path/to/perf.json"
        assert len(result.logs) == 2
        assert "Error occurred" in result.logs["error_log"]