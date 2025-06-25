"""Unit tests for the progress tracker."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from brainsmith.core_v3.phase2.progress import ProgressTracker
from brainsmith.core_v3.phase2.data_structures import BuildResult, BuildStatus


class TestProgressTracker:
    """Test the ProgressTracker class."""
    
    def test_progress_tracker_creation(self):
        """Test creating a progress tracker."""
        tracker = ProgressTracker(total_configs=100)
        
        assert tracker.total_configs == 100
        assert tracker.completed == 0
        assert tracker.successful == 0
        assert tracker.failed == 0
        assert tracker.skipped == 0
        assert isinstance(tracker.start_time, datetime)
    
    def test_update_with_success(self):
        """Test updating progress with successful build."""
        tracker = ProgressTracker(total_configs=10)
        
        result = BuildResult(
            config_id="config_001",
            status=BuildStatus.SUCCESS
        )
        result.complete(BuildStatus.SUCCESS)
        
        tracker.update(result)
        
        assert tracker.completed == 1
        assert tracker.successful == 1
        assert tracker.failed == 0
        assert tracker.total_build_time > 0
        assert tracker.successful_build_time > 0
    
    def test_update_with_failure(self):
        """Test updating progress with failed build."""
        tracker = ProgressTracker(total_configs=10)
        
        result = BuildResult(
            config_id="config_001",
            status=BuildStatus.FAILED
        )
        result.complete(BuildStatus.FAILED, "Error occurred")
        
        tracker.update(result)
        
        assert tracker.completed == 1
        assert tracker.successful == 0
        assert tracker.failed == 1
        assert tracker.failed_build_time > 0
    
    def test_update_with_skipped(self):
        """Test updating progress with skipped build."""
        tracker = ProgressTracker(total_configs=10)
        
        result = BuildResult(
            config_id="config_001",
            status=BuildStatus.SKIPPED
        )
        result.duration_seconds = 0.1
        
        tracker.update(result)
        
        assert tracker.completed == 1
        assert tracker.skipped == 1
        assert tracker.successful == 0
        assert tracker.failed == 0
    
    def test_get_eta_no_data(self):
        """Test ETA calculation with no completed builds."""
        tracker = ProgressTracker(total_configs=10)
        
        eta = tracker.get_eta()
        assert eta is None
    
    def test_get_eta_with_data(self):
        """Test ETA calculation with completed builds."""
        tracker = ProgressTracker(total_configs=10)
        
        # Simulate some completed builds
        for i in range(5):
            result = BuildResult(
                config_id=f"config_{i:03d}",
                status=BuildStatus.SUCCESS
            )
            result.duration_seconds = 10.0  # 10 seconds each
            tracker.update(result)
        
        eta = tracker.get_eta()
        
        assert eta is not None
        assert isinstance(eta, datetime)
        
        # ETA should be approximately 50 seconds from now
        # (5 remaining * 10 seconds each)
        expected_eta = datetime.now() + timedelta(seconds=50)
        
        # Allow 1 second tolerance
        assert abs((eta - expected_eta).total_seconds()) < 1
    
    def test_get_summary(self):
        """Test getting progress summary."""
        tracker = ProgressTracker(total_configs=10)
        
        # Add some results
        for i in range(6):
            status = BuildStatus.SUCCESS if i < 4 else BuildStatus.FAILED
            result = BuildResult(
                config_id=f"config_{i:03d}",
                status=status
            )
            result.duration_seconds = 5.0
            tracker.update(result)
        
        summary = tracker.get_summary()
        
        assert "6/10 (60.0%)" in summary  # Progress
        assert "Success: 4 (66.7%)" in summary  # Success rate
        assert "Failed: 2" in summary
        assert "Avg build: 5.0s" in summary
    
    def test_get_detailed_summary(self):
        """Test getting detailed progress summary."""
        tracker = ProgressTracker(total_configs=20)
        
        # Add various results
        for i in range(10):
            if i < 7:
                status = BuildStatus.SUCCESS
                duration = 5.0
            elif i < 9:
                status = BuildStatus.FAILED
                duration = 3.0
            else:
                status = BuildStatus.SKIPPED
                duration = 0.1
            
            result = BuildResult(
                config_id=f"config_{i:03d}",
                status=status
            )
            result.duration_seconds = duration
            tracker.update(result)
        
        summary = tracker.get_detailed_summary()
        
        assert "Total Configurations: 20" in summary
        assert "Completed: 10 (50.0%)" in summary
        assert "Successful: 7" in summary
        assert "Failed: 2" in summary
        assert "Skipped: 1" in summary
        assert "Average Build Time:" in summary
        assert "Builds per minute:" in summary
    
    def test_format_duration(self):
        """Test duration formatting."""
        tracker = ProgressTracker(total_configs=1)
        
        # Test seconds
        assert tracker._format_duration(30) == "30s"
        assert tracker._format_duration(59) == "59s"
        
        # Test minutes
        assert tracker._format_duration(60) == "1.0m"
        assert tracker._format_duration(90) == "1.5m"
        assert tracker._format_duration(3599) == "60.0m"
        
        # Test hours
        assert tracker._format_duration(3600) == "1.0h"
        assert tracker._format_duration(5400) == "1.5h"
        assert tracker._format_duration(7200) == "2.0h"
    
    def test_get_progress_bar(self):
        """Test progress bar generation."""
        tracker = ProgressTracker(total_configs=10)
        
        # No progress
        bar = tracker.get_progress_bar(width=20)
        assert bar == "[" + "░" * 20 + "] 0/10"
        
        # 50% progress
        for i in range(5):
            result = BuildResult(
                config_id=f"config_{i:03d}",
                status=BuildStatus.SUCCESS
            )
            tracker.update(result)
        
        bar = tracker.get_progress_bar(width=20)
        assert bar == "[" + "█" * 10 + "░" * 10 + "] 5/10"
        
        # 100% progress
        for i in range(5, 10):
            result = BuildResult(
                config_id=f"config_{i:03d}",
                status=BuildStatus.SUCCESS
            )
            tracker.update(result)
        
        bar = tracker.get_progress_bar(width=20)
        assert bar == "[" + "█" * 20 + "] 10/10"
    
    def test_get_progress_bar_empty(self):
        """Test progress bar with zero total configs."""
        tracker = ProgressTracker(total_configs=0)
        
        bar = tracker.get_progress_bar(width=20)
        assert bar == "[" + " " * 20 + "]"
    
    def test_timing_statistics(self):
        """Test tracking of timing statistics."""
        tracker = ProgressTracker(total_configs=10)
        
        # Add builds with different durations
        # 3 successful builds: 10s, 12s, 14s
        for i, duration in enumerate([10, 12, 14]):
            result = BuildResult(
                config_id=f"success_{i}",
                status=BuildStatus.SUCCESS
            )
            result.duration_seconds = duration
            tracker.update(result)
        
        # 2 failed builds: 5s, 7s
        for i, duration in enumerate([5, 7]):
            result = BuildResult(
                config_id=f"failed_{i}",
                status=BuildStatus.FAILED
            )
            result.duration_seconds = duration
            tracker.update(result)
        
        assert tracker.completed == 5
        assert tracker.successful == 3
        assert tracker.failed == 2
        
        # Check timing totals
        assert tracker.total_build_time == 48.0  # 10+12+14+5+7
        assert tracker.successful_build_time == 36.0  # 10+12+14
        assert tracker.failed_build_time == 12.0  # 5+7