"""
Metrics Tests - DSE Metrics System

Tests the DSEMetrics, PerformanceMetrics, and ResourceMetrics classes.
Validates calculations, serialization, and data handling.
"""

import pytest
import json
from typing import Dict, Any

# Import metrics components
try:
    from brainsmith.core.metrics import (
        DSEMetrics, PerformanceMetrics, ResourceMetrics, 
        create_metrics, compare_metrics
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False


@pytest.mark.core
class TestPerformanceMetrics:
    """Test PerformanceMetrics class functionality."""
    
    def test_performance_metrics_creation(self):
        """Test creating PerformanceMetrics instance."""
        if not METRICS_AVAILABLE:
            pytest.skip("Metrics not available")
        
        perf = PerformanceMetrics()
        
        # Default values should be None
        assert perf.throughput_ops_sec is None
        assert perf.latency_ms is None
        assert perf.clock_frequency_mhz is None
        assert perf.target_fps is None
        assert perf.achieved_fps is None
    
    def test_performance_metrics_with_values(self):
        """Test PerformanceMetrics with actual values."""
        if not METRICS_AVAILABLE:
            pytest.skip("Metrics not available")
        
        perf = PerformanceMetrics(
            throughput_ops_sec=500.0,
            latency_ms=20.0,
            clock_frequency_mhz=150.0,
            target_fps=30.0,
            achieved_fps=28.5
        )
        
        assert perf.throughput_ops_sec == 500.0
        assert perf.latency_ms == 20.0
        assert perf.clock_frequency_mhz == 150.0
        assert perf.target_fps == 30.0
        assert perf.achieved_fps == 28.5
    
    def test_fps_efficiency_calculation(self):
        """Test FPS efficiency calculation."""
        if not METRICS_AVAILABLE:
            pytest.skip("Metrics not available")
        
        perf = PerformanceMetrics(target_fps=30.0, achieved_fps=28.5)
        efficiency = perf.get_fps_efficiency()
        
        assert efficiency is not None
        assert abs(efficiency - (28.5 / 30.0)) < 0.001
    
    def test_fps_efficiency_with_none_values(self):
        """Test FPS efficiency with None values."""
        if not METRICS_AVAILABLE:
            pytest.skip("Metrics not available")
        
        perf = PerformanceMetrics()
        efficiency = perf.get_fps_efficiency()
        assert efficiency is None
        
        # Test with zero target
        perf = PerformanceMetrics(target_fps=0.0, achieved_fps=10.0)
        efficiency = perf.get_fps_efficiency()
        assert efficiency is None


@pytest.mark.core
class TestResourceMetrics:
    """Test ResourceMetrics class functionality."""
    
    def test_resource_metrics_creation(self):
        """Test creating ResourceMetrics instance."""
        if not METRICS_AVAILABLE:
            pytest.skip("Metrics not available")
        
        resources = ResourceMetrics()
        
        # Default values should be None
        assert resources.lut_utilization_percent is None
        assert resources.dsp_utilization_percent is None
        assert resources.bram_utilization_percent is None
        assert resources.estimated_power_w is None
    
    def test_resource_metrics_with_values(self):
        """Test ResourceMetrics with actual values."""
        if not METRICS_AVAILABLE:
            pytest.skip("Metrics not available")
        
        resources = ResourceMetrics(
            lut_utilization_percent=45.0,
            dsp_utilization_percent=32.0,
            bram_utilization_percent=28.0,
            estimated_power_w=12.5
        )
        
        assert resources.lut_utilization_percent == 45.0
        assert resources.dsp_utilization_percent == 32.0
        assert resources.bram_utilization_percent == 28.0
        assert resources.estimated_power_w == 12.5
    
    def test_resource_efficiency_calculation(self):
        """Test resource efficiency calculation."""
        if not METRICS_AVAILABLE:
            pytest.skip("Metrics not available")
        
        resources = ResourceMetrics(
            lut_utilization_percent=45.0,
            dsp_utilization_percent=32.0,
            bram_utilization_percent=28.0
        )
        
        efficiency = resources.get_resource_efficiency()
        expected = (45.0 + 32.0 + 28.0) / 3.0
        
        assert efficiency is not None
        assert abs(efficiency - expected) < 0.001
    
    def test_resource_efficiency_with_partial_data(self):
        """Test resource efficiency with partial data."""
        if not METRICS_AVAILABLE:
            pytest.skip("Metrics not available")
        
        # Only LUT utilization
        resources = ResourceMetrics(lut_utilization_percent=45.0)
        efficiency = resources.get_resource_efficiency()
        assert efficiency == 45.0
        
        # No utilization data
        resources = ResourceMetrics(estimated_power_w=12.5)
        efficiency = resources.get_resource_efficiency()
        assert efficiency is None


@pytest.mark.core
class TestDSEMetrics:
    """Test DSEMetrics class functionality."""
    
    def test_dse_metrics_creation(self):
        """Test creating DSEMetrics instance."""
        if not METRICS_AVAILABLE:
            pytest.skip("Metrics not available")
        
        metrics = DSEMetrics()
        
        # Check default values
        assert isinstance(metrics.performance, PerformanceMetrics)
        assert isinstance(metrics.resources, ResourceMetrics)
        assert metrics.build_success is False
        assert metrics.build_time_seconds == 0.0
        assert metrics.design_point_id == ""
        assert isinstance(metrics.configuration, dict)
        assert len(metrics.configuration) == 0
    
    def test_dse_metrics_with_values(self):
        """Test DSEMetrics with configured values."""
        if not METRICS_AVAILABLE:
            pytest.skip("Metrics not available")
        
        metrics = DSEMetrics()
        metrics.design_point_id = "test_point_1"
        metrics.build_success = True
        metrics.build_time_seconds = 120.5
        metrics.configuration = {"pe_conv": 8, "simd_conv": 4}
        
        # Set performance metrics
        metrics.performance.throughput_ops_sec = 500.0
        metrics.performance.latency_ms = 20.0
        
        # Set resource metrics
        metrics.resources.lut_utilization_percent = 45.0
        metrics.resources.dsp_utilization_percent = 32.0
        
        assert metrics.design_point_id == "test_point_1"
        assert metrics.build_success is True
        assert metrics.build_time_seconds == 120.5
        assert metrics.configuration["pe_conv"] == 8
        assert metrics.performance.throughput_ops_sec == 500.0
        assert metrics.resources.lut_utilization_percent == 45.0
    
    def test_optimization_score_calculation(self):
        """Test optimization score calculation."""
        if not METRICS_AVAILABLE:
            pytest.skip("Metrics not available")
        
        metrics = DSEMetrics()
        metrics.build_success = True
        metrics.performance.throughput_ops_sec = 500.0
        metrics.resources.lut_utilization_percent = 45.0
        metrics.resources.dsp_utilization_percent = 32.0
        metrics.resources.bram_utilization_percent = 28.0
        
        score = metrics.get_optimization_score()
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        
        # Score should be higher for successful builds
        assert score > 0.0
    
    def test_optimization_score_failed_build(self):
        """Test optimization score for failed build."""
        if not METRICS_AVAILABLE:
            pytest.skip("Metrics not available")
        
        metrics = DSEMetrics()
        metrics.build_success = False  # Failed build
        metrics.performance.throughput_ops_sec = 1000.0  # High performance
        
        score = metrics.get_optimization_score()
        
        # Should still have some score from performance, but lower due to failure
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    def test_optimization_score_no_data(self):
        """Test optimization score with no performance data."""
        if not METRICS_AVAILABLE:
            pytest.skip("Metrics not available")
        
        metrics = DSEMetrics()
        # No performance or resource data set
        
        score = metrics.get_optimization_score()
        
        assert score == 0.0  # Should be zero with no data


@pytest.mark.core
class TestDSEMetricsSerialization:
    """Test DSEMetrics serialization and deserialization."""
    
    def test_to_dict_conversion(self):
        """Test converting DSEMetrics to dictionary."""
        if not METRICS_AVAILABLE:
            pytest.skip("Metrics not available")
        
        metrics = DSEMetrics()
        metrics.design_point_id = "test_point_1"
        metrics.build_success = True
        metrics.performance.throughput_ops_sec = 500.0
        metrics.resources.lut_utilization_percent = 45.0
        
        result_dict = metrics.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["design_point_id"] == "test_point_1"
        assert result_dict["build_success"] is True
        assert "performance" in result_dict
        assert "resources" in result_dict
        assert "optimization_score" in result_dict
        
        # Check nested structure
        assert result_dict["performance"]["throughput_ops_sec"] == 500.0
        assert result_dict["resources"]["lut_utilization_percent"] == 45.0
    
    def test_to_json_conversion(self):
        """Test converting DSEMetrics to JSON string."""
        if not METRICS_AVAILABLE:
            pytest.skip("Metrics not available")
        
        metrics = DSEMetrics()
        metrics.design_point_id = "test_point_1"
        metrics.build_success = True
        
        json_str = metrics.to_json()
        
        assert isinstance(json_str, str)
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["design_point_id"] == "test_point_1"
        assert parsed["build_success"] is True
    
    def test_from_dict_creation(self):
        """Test creating DSEMetrics from dictionary."""
        if not METRICS_AVAILABLE:
            pytest.skip("Metrics not available")
        
        data = {
            "design_point_id": "test_point_1",
            "build_success": True,
            "build_time_seconds": 120.5,
            "configuration": {"pe_conv": 8},
            "performance": {
                "throughput_ops_sec": 500.0,
                "latency_ms": 20.0,
                "clock_frequency_mhz": 150.0
            },
            "resources": {
                "lut_utilization_percent": 45.0,
                "dsp_utilization_percent": 32.0,
                "estimated_power_w": 12.5
            }
        }
        
        metrics = DSEMetrics.from_dict(data)
        
        assert metrics.design_point_id == "test_point_1"
        assert metrics.build_success is True
        assert metrics.build_time_seconds == 120.5
        assert metrics.configuration["pe_conv"] == 8
        assert metrics.performance.throughput_ops_sec == 500.0
        assert metrics.performance.latency_ms == 20.0
        assert metrics.resources.lut_utilization_percent == 45.0
        assert metrics.resources.estimated_power_w == 12.5
    
    def test_round_trip_serialization(self):
        """Test round-trip serialization (to_dict -> from_dict)."""
        if not METRICS_AVAILABLE:
            pytest.skip("Metrics not available")
        
        # Create original metrics
        original = DSEMetrics()
        original.design_point_id = "test_point_1"
        original.build_success = True
        original.performance.throughput_ops_sec = 500.0
        original.resources.lut_utilization_percent = 45.0
        
        # Convert to dict and back
        data = original.to_dict()
        restored = DSEMetrics.from_dict(data)
        
        # Compare key values
        assert restored.design_point_id == original.design_point_id
        assert restored.build_success == original.build_success
        assert restored.performance.throughput_ops_sec == original.performance.throughput_ops_sec
        assert restored.resources.lut_utilization_percent == original.resources.lut_utilization_percent


@pytest.mark.core
class TestMetricsUtilities:
    """Test metrics utility functions."""
    
    def test_create_metrics_basic(self):
        """Test create_metrics utility function."""
        if not METRICS_AVAILABLE:
            pytest.skip("Metrics not available")
        
        metrics = create_metrics("test_point_1")
        
        assert isinstance(metrics, DSEMetrics)
        assert metrics.design_point_id == "test_point_1"
        assert isinstance(metrics.configuration, dict)
        assert len(metrics.configuration) == 0
    
    def test_create_metrics_with_configuration(self):
        """Test create_metrics with configuration."""
        if not METRICS_AVAILABLE:
            pytest.skip("Metrics not available")
        
        config = {"pe_conv": 8, "simd_conv": 4}
        metrics = create_metrics("test_point_1", config)
        
        assert metrics.design_point_id == "test_point_1"
        assert metrics.configuration == config
    
    def test_compare_metrics_empty_list(self):
        """Test compare_metrics with empty list."""
        if not METRICS_AVAILABLE:
            pytest.skip("Metrics not available")
        
        best = compare_metrics([])
        
        assert isinstance(best, DSEMetrics)
        assert best.design_point_id == ""
    
    def test_compare_metrics_single_item(self):
        """Test compare_metrics with single item."""
        if not METRICS_AVAILABLE:
            pytest.skip("Metrics not available")
        
        metrics = create_metrics("test_point_1")
        metrics.build_success = True
        
        best = compare_metrics([metrics])
        
        assert best.design_point_id == "test_point_1"
    
    def test_compare_metrics_multiple_items(self):
        """Test compare_metrics with multiple items."""
        if not METRICS_AVAILABLE:
            pytest.skip("Metrics not available")
        
        # Create multiple metrics with different scores
        metrics1 = create_metrics("point_1")
        metrics1.build_success = True
        metrics1.performance.throughput_ops_sec = 300.0
        
        metrics2 = create_metrics("point_2")
        metrics2.build_success = True
        metrics2.performance.throughput_ops_sec = 500.0  # Higher performance
        
        metrics3 = create_metrics("point_3")
        metrics3.build_success = False  # Failed build
        metrics3.performance.throughput_ops_sec = 600.0
        
        best = compare_metrics([metrics1, metrics2, metrics3])
        
        # Should select metrics2 (successful build with high performance)
        assert best.design_point_id == "point_2"
    
    def test_compare_metrics_all_failed_builds(self):
        """Test compare_metrics when all builds failed."""
        if not METRICS_AVAILABLE:
            pytest.skip("Metrics not available")
        
        metrics1 = create_metrics("point_1")
        metrics1.build_success = False
        
        metrics2 = create_metrics("point_2")
        metrics2.build_success = False
        
        best = compare_metrics([metrics1, metrics2])
        
        # Should return first one when all failed
        assert best.design_point_id == "point_1"


@pytest.mark.core
class TestMetricsValidation:
    """Test metrics validation and edge cases."""
    
    def test_negative_performance_values(self):
        """Test behavior with negative performance values."""
        if not METRICS_AVAILABLE:
            pytest.skip("Metrics not available")
        
        metrics = DSEMetrics()
        metrics.performance.throughput_ops_sec = -100.0  # Invalid
        metrics.performance.latency_ms = -5.0  # Invalid
        
        # Should still calculate score (implementation dependent)
        score = metrics.get_optimization_score()
        assert isinstance(score, float)
    
    def test_extreme_resource_utilization(self):
        """Test behavior with extreme resource utilization values."""
        if not METRICS_AVAILABLE:
            pytest.skip("Metrics not available")
        
        metrics = DSEMetrics()
        metrics.resources.lut_utilization_percent = 150.0  # Over 100%
        metrics.resources.dsp_utilization_percent = -10.0  # Negative
        
        efficiency = metrics.resources.get_resource_efficiency()
        assert isinstance(efficiency, float)
        # Implementation should handle extreme values gracefully
    
    def test_very_large_numbers(self):
        """Test metrics with very large numbers."""
        if not METRICS_AVAILABLE:
            pytest.skip("Metrics not available")
        
        metrics = DSEMetrics()
        metrics.performance.throughput_ops_sec = 1e12  # Very large
        metrics.build_time_seconds = 1e6  # Very long build time
        
        score = metrics.get_optimization_score()
        assert isinstance(score, float)
        assert not (score != score)  # Check for NaN


# Helper functions for metrics testing
def create_test_metrics(point_id: str, throughput: float = 500.0, 
                       lut_util: float = 45.0, success: bool = True) -> 'DSEMetrics':
    """Helper to create test metrics with standard values."""
    if not METRICS_AVAILABLE:
        return None
    
    metrics = create_metrics(point_id)
    metrics.build_success = success
    metrics.performance.throughput_ops_sec = throughput
    metrics.resources.lut_utilization_percent = lut_util
    return metrics


def assert_metrics_valid(metrics: 'DSEMetrics'):
    """Helper to assert metrics are valid."""
    if not METRICS_AVAILABLE:
        return
    
    assert isinstance(metrics, DSEMetrics)
    assert isinstance(metrics.performance, PerformanceMetrics)
    assert isinstance(metrics.resources, ResourceMetrics)
    assert isinstance(metrics.design_point_id, str)
    assert isinstance(metrics.build_success, bool)
    assert isinstance(metrics.build_time_seconds, (int, float))
    assert isinstance(metrics.configuration, dict)


def assert_score_in_range(score: float, min_score: float = 0.0, max_score: float = 1.0):
    """Helper to assert optimization score is in valid range."""
    assert isinstance(score, float)
    assert min_score <= score <= max_score
    assert not (score != score)  # Check for NaN