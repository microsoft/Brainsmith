"""
Comprehensive test suite for the unified BrainSmith Data module.

Tests the consolidation of metrics and analysis modules while ensuring
North Star principles are maintained and functionality is preserved.
"""

import pytest
import json
import os
import tempfile
from unittest.mock import Mock, patch
from typing import Dict, Any, List

# Import the unified data module
try:
    from brainsmith.data import (
        collect_build_metrics, collect_dse_metrics, summarize_data, compare_results,
        filter_data, validate_data, export_for_analysis, to_pandas, to_csv, to_json,
        create_report, BuildMetrics, PerformanceData, ResourceData, QualityData,
        BuildData, DataSummary, ComparisonResult
    )
except ImportError:
    pytest.skip("BrainSmith data module not available", allow_module_level=True)


class TestDataTypes:
    """Test the unified data types work correctly."""
    
    def test_build_metrics_creation(self):
        """Test BuildMetrics can be created and converted to dict."""
        metrics = BuildMetrics(
            model_path='test_model.onnx',
            blueprint_path='test_blueprint.yaml'
        )
        
        assert metrics.model_path == 'test_model.onnx'
        assert metrics.blueprint_path == 'test_blueprint.yaml'
        assert metrics.is_successful()  # Default is successful
        
        # Test dict conversion
        data_dict = metrics.to_dict()
        assert isinstance(data_dict, dict)
        assert 'performance' in data_dict
        assert 'resources' in data_dict
        assert 'quality' in data_dict
        assert 'build' in data_dict
        
    def test_performance_data(self):
        """Test PerformanceData functionality."""
        perf = PerformanceData(
            throughput_ops_sec=1000.0,
            latency_ms=5.0,
            clock_freq_mhz=100.0
        )
        
        assert perf.throughput_ops_sec == 1000.0
        assert perf.latency_ms == 5.0
        assert perf.clock_freq_mhz == 100.0
        
        # Test efficiency ratio calculation
        efficiency = perf.get_efficiency_ratio()
        assert efficiency == 10.0  # 1000 / 100
        
    def test_resource_data(self):
        """Test ResourceData functionality."""
        resources = ResourceData(
            lut_utilization_percent=50.0,
            dsp_utilization_percent=30.0,
            bram_utilization_percent=20.0
        )
        
        # Test total utilization calculation
        total_util = resources.get_total_utilization()
        expected = (50.0 + 30.0 + 20.0) / 3
        assert abs(total_util - expected) < 0.01
        
    def test_data_summary(self):
        """Test DataSummary success rate calculation."""
        summary = DataSummary(
            metric_count=100,
            successful_builds=85,
            failed_builds=15
        )
        
        assert summary.success_rate == 0.85
        
        # Test with zero metrics
        empty_summary = DataSummary()
        assert empty_summary.success_rate == 0.0


class TestDataCollection:
    """Test data collection from various result formats."""
    
    def test_collect_build_metrics_from_dict(self):
        """Test collecting metrics from dictionary result format."""
        build_result = {
            'performance': {
                'throughput_ops_sec': 1500.0,
                'latency_ms': 3.5,
                'clock_freq_mhz': 150.0
            },
            'resources': {
                'lut_utilization_percent': 65.0,
                'dsp_utilization_percent': 45.0,
                'lut_count': 5000,
                'dsp_count': 100
            },
            'quality': {
                'accuracy_percent': 96.5,
                'f1_score': 0.94
            },
            'build_info': {
                'build_success': True,
                'build_time_seconds': 120.0,
                'target_device': 'xc7z020clg400-1'
            }
        }
        
        metrics = collect_build_metrics(
            build_result, 
            'test_model.onnx', 
            'test_blueprint.yaml',
            {'pe': 16, 'simd': 8}
        )
        
        # Validate extracted data
        assert metrics.performance.throughput_ops_sec == 1500.0
        assert metrics.performance.latency_ms == 3.5
        assert metrics.resources.lut_utilization_percent == 65.0
        assert metrics.resources.dsp_count == 100
        assert metrics.quality.accuracy_percent == 96.5
        assert metrics.build.build_success is True
        assert metrics.build.target_device == 'xc7z020clg400-1'
        assert metrics.parameters == {'pe': 16, 'simd': 8}
        
    def test_collect_build_metrics_from_object(self):
        """Test collecting metrics from object with attributes."""
        # Mock build result object
        mock_result = Mock()
        mock_performance = Mock()
        mock_performance.throughput_ops_sec = 2000.0
        mock_performance.latency_ms = 2.0
        mock_result.performance = mock_performance
        
        mock_resources = Mock()
        mock_resources.lut_utilization_percent = 70.0
        mock_result.resources = mock_resources
        
        metrics = collect_build_metrics(mock_result)
        
        assert metrics.performance.throughput_ops_sec == 2000.0
        assert metrics.resources.lut_utilization_percent == 70.0
        
    def test_collect_dse_metrics(self):
        """Test collecting metrics from DSE results."""
        # Mock DSE results
        dse_results = []
        for i in range(3):
            result = Mock()
            result.design_parameters = {'pe': 4 * (i + 1), 'simd': 2 * (i + 1)}
            result.objective_values = [1000 * (i + 1), 5.0 - i]  # throughput, latency
            dse_results.append(result)
        
        metrics_list = collect_dse_metrics(dse_results)
        
        assert len(metrics_list) == 3
        
        # Check first result
        assert metrics_list[0].parameters == {'pe': 4, 'simd': 2}
        assert metrics_list[0].performance.throughput_ops_sec == 1000
        assert metrics_list[0].performance.latency_ms == 5.0
        assert metrics_list[0].metadata['dse_index'] == 0
        
        # Check scaling
        assert metrics_list[2].parameters == {'pe': 12, 'simd': 6}
        assert metrics_list[2].performance.throughput_ops_sec == 3000
        
    def test_collect_build_metrics_error_handling(self):
        """Test error handling in metrics collection."""
        # Test with None result
        metrics = collect_build_metrics(None)
        assert metrics.build.build_success is False
        
        # Test with malformed result
        bad_result = {"broken": "data"}
        metrics = collect_build_metrics(bad_result)
        # Should not crash, should create empty metrics
        assert isinstance(metrics, BuildMetrics)


class TestDataAnalysis:
    """Test data analysis and processing functions."""
    
    def test_summarize_data(self):
        """Test statistical summarization of metrics."""
        # Create test metrics
        metrics_list = []
        for i in range(5):
            metrics = BuildMetrics()
            metrics.performance.throughput_ops_sec = 1000 + i * 200
            metrics.performance.latency_ms = 5 - i * 0.5
            metrics.resources.lut_utilization_percent = 50 + i * 5
            metrics.build.build_success = i < 4  # One failure
            metrics_list.append(metrics)
        
        summary = summarize_data(metrics_list)
        
        assert summary.metric_count == 5
        assert summary.successful_builds == 4
        assert summary.failed_builds == 1
        assert summary.success_rate == 0.8
        
        # Check performance statistics (only successful builds)
        assert summary.avg_throughput == 1300.0  # (1000+1200+1400+1600)/4
        assert summary.max_throughput == 1600.0
        assert summary.min_throughput == 1000.0
        
    def test_compare_results(self):
        """Test metrics comparison functionality."""
        # Create two different metrics
        metrics_a = BuildMetrics()
        metrics_a.performance.throughput_ops_sec = 1000.0
        metrics_a.resources.lut_utilization_percent = 60.0
        
        metrics_b = BuildMetrics()  
        metrics_b.performance.throughput_ops_sec = 1500.0  # 50% better
        metrics_b.resources.lut_utilization_percent = 50.0  # 16.7% better
        
        comparison = compare_results(metrics_a, metrics_b)
        
        assert 'throughput' in comparison.improvement_ratios
        assert comparison.improvement_ratios['throughput'] == 1.5
        assert 'throughput' in comparison.metrics_b_better
        assert '50.0% higher' in comparison.metrics_b_better['throughput']
        
    def test_filter_data(self):
        """Test data filtering functionality."""
        # Create test metrics with varying performance
        metrics_list = []
        for i in range(10):
            metrics = BuildMetrics()
            metrics.performance.throughput_ops_sec = 500 + i * 100
            metrics.resources.lut_utilization_percent = 40 + i * 5
            metrics.build.build_success = i % 2 == 0  # Every other one succeeds
            metrics_list.append(metrics)
        
        # Filter for high performance, low resource usage, successful builds
        filtered = filter_data(metrics_list, {
            'min_throughput': 1000,
            'max_lut_utilization': 70,
            'build_success': True
        })
        
        # Should get metrics with index 6 only (success + criteria)
        # Index 6: throughput=1100, lut=70, success=True ✓
        # Index 8: throughput=1300, lut=80, success=True ✗ (lut > 70)
        assert len(filtered) == 1
        
        for metrics in filtered:
            assert metrics.performance.throughput_ops_sec >= 1000
            assert metrics.resources.lut_utilization_percent <= 70
            assert metrics.build.build_success is True
            
    def test_validate_data(self):
        """Test data validation functionality."""
        # Valid metrics - use consistent throughput/latency
        valid_metrics = BuildMetrics()
        valid_metrics.performance.throughput_ops_sec = 200.0  # 1000ms / 5ms = 200 ops/sec
        valid_metrics.performance.latency_ms = 5.0
        valid_metrics.resources.lut_utilization_percent = 50.0
        valid_metrics.quality.accuracy_percent = 95.0
        
        issues = validate_data(valid_metrics)
        assert len(issues) == 0
        
        # Invalid metrics
        invalid_metrics = BuildMetrics()
        invalid_metrics.build.build_success = False
        invalid_metrics.performance.throughput_ops_sec = 1000.0  # Performance data for failed build
        invalid_metrics.resources.lut_utilization_percent = 150.0  # Out of range
        invalid_metrics.quality.accuracy_percent = -5.0  # Out of range
        
        issues = validate_data(invalid_metrics)
        assert len(issues) > 0
        assert any('Performance data present for failed build' in issue for issue in issues)
        assert any('LUT utilization out of range' in issue for issue in issues)
        assert any('Accuracy out of range' in issue for issue in issues)


class TestDataExport:
    """Test data export functionality."""
    
    def test_export_for_analysis_dict(self):
        """Test exporting data as dictionary."""
        metrics = BuildMetrics()
        metrics.performance.throughput_ops_sec = 1000.0
        
        result = export_for_analysis(metrics, 'dict')
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert 'performance' in result[0]
        assert result[0]['performance']['throughput_ops_sec'] == 1000.0
        
    def test_to_json(self):
        """Test JSON export functionality."""
        metrics = BuildMetrics()
        metrics.performance.throughput_ops_sec = 1000.0
        metrics.model_path = 'test_model.onnx'
        
        json_str = to_json(metrics)
        
        # Validate JSON structure
        data = json.loads(json_str)
        assert 'export_timestamp' in data
        assert 'data_count' in data
        assert data['data_count'] == 1
        assert 'data' in data
        assert data['data'][0]['model_path'] == 'test_model.onnx'
        
    def test_to_csv(self):
        """Test CSV export functionality."""
        metrics_list = []
        for i in range(3):
            metrics = BuildMetrics()
            metrics.performance.throughput_ops_sec = 1000 + i * 100
            metrics.resources.lut_utilization_percent = 50 + i * 10
            metrics.parameters = {'pe': 4 * (i + 1)}
            metrics_list.append(metrics)
        
        csv_content = to_csv(metrics_list)
        
        lines = csv_content.strip().split('\n')
        assert len(lines) == 4  # Header + 3 data rows
        
        # Check header contains expected columns
        header = lines[0]
        assert 'performance_throughput_ops_sec' in header
        assert 'resources_lut_utilization_percent' in header
        assert 'parameters_pe' in header
        
        # Check data row - values are correctly formatted in CSV
        first_row = lines[1].split(',')
        assert '1000' in first_row  # throughput (as integer or float)
        assert '50' in first_row     # LUT utilization (as integer or float)
        assert '4' in first_row      # PE parameter
        
        # Verify specific columns contain expected values
        header_cols = header.split(',')
        throughput_idx = header_cols.index('performance_throughput_ops_sec')
        lut_idx = header_cols.index('resources_lut_utilization_percent')
        pe_idx = header_cols.index('parameters_pe')
        
        assert first_row[throughput_idx] == '1000'
        assert first_row[lut_idx] == '50'
        assert first_row[pe_idx] == '4'
        
    @pytest.mark.skipif(True, reason="pandas optional dependency")
    def test_to_pandas(self):
        """Test pandas export functionality."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not available")
        
        metrics_list = []
        for i in range(3):
            metrics = BuildMetrics()
            metrics.performance.throughput_ops_sec = 1000 + i * 100  
            metrics_list.append(metrics)
        
        df = to_pandas(metrics_list)
        
        assert df is not None
        assert len(df) == 3
        assert 'performance_throughput_ops_sec' in df.columns
        assert df['performance_throughput_ops_sec'].iloc[0] == 1000.0
        assert df['performance_throughput_ops_sec'].iloc[2] == 1200.0
        
    def test_create_report(self):
        """Test report generation."""
        # Single metrics report
        metrics = BuildMetrics()
        metrics.performance.throughput_ops_sec = 1500.0
        metrics.resources.lut_utilization_percent = 65.0
        metrics.model_path = 'test_model.onnx'
        
        report = create_report(metrics, 'markdown')
        
        assert '# FPGA Data Report' in report
        assert 'Single Configuration Report' in report
        assert 'test_model.onnx' in report
        assert '1500.0' in report
        assert '65.0' in report
        
        # Summary report  
        summary = DataSummary(
            metric_count=50,
            successful_builds=45,
            avg_throughput=1200.0
        )
        
        summary_report = create_report(summary, 'markdown')
        
        assert 'Summary Statistics' in summary_report
        assert '50' in summary_report  # metric count
        assert '90.0%' in summary_report  # success rate
        assert '1200.0' in summary_report  # avg throughput


class TestIntegration:
    """Test integration with other BrainSmith modules."""
    
    @patch('brainsmith.data.functions.log_data_event')
    def test_hooks_integration(self, mock_log_event):
        """Test integration with hooks module."""
        build_result = {'performance': {'throughput_ops_sec': 1000.0}}
        
        collect_build_metrics(build_result, 'model.onnx', 'blueprint.yaml')
        
        # Verify events were logged
        assert mock_log_event.call_count >= 2
        call_args = [call[0][0] for call in mock_log_event.call_args_list]  # Extract event names
        assert 'data_collection_start' in call_args
        assert 'data_collection_complete' in call_args
        
    def test_backwards_compatibility(self):
        """Test backwards compatibility shims work."""
        from brainsmith.data import expose_analysis_data
        
        # Create mock DSE results
        dse_results = [{'objective_values': [1000, 5]}]
        
        with pytest.warns(DeprecationWarning):
            result = expose_analysis_data(dse_results)
        
        # Legacy function returns raw input (list in this case)
        assert isinstance(result, list)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_data_handling(self):
        """Test handling of empty or None data."""
        # Empty metrics list
        summary = summarize_data([])
        assert summary.metric_count == 0
        assert summary.success_rate == 0.0
        
        # None DSE results
        metrics_list = collect_dse_metrics(None)
        assert metrics_list == []
        
        # Empty data export
        result = export_for_analysis([], 'dict')
        assert result == []
        
    def test_malformed_data_handling(self):
        """Test handling of malformed input data."""
        # Malformed build result
        bad_result = "this is not a valid result"
        metrics = collect_build_metrics(bad_result)
        assert isinstance(metrics, BuildMetrics)
        
        # Missing attributes - Mock objects should be filtered out
        incomplete_result = Mock()
        # No performance, resources, etc. attributes
        metrics = collect_build_metrics(incomplete_result)
        # Mock objects should be filtered out, so values should be None
        assert metrics.performance.throughput_ops_sec is None
        assert metrics.resources.lut_utilization_percent is None
        
    def test_file_operations(self):
        """Test file save/load operations."""
        metrics = BuildMetrics()
        metrics.performance.throughput_ops_sec = 1000.0
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            # Test JSON file save
            json_content = to_json(metrics, temp_path)
            assert os.path.exists(temp_path)
            
            # Verify file contents
            with open(temp_path, 'r') as f:
                saved_data = json.load(f)
            assert saved_data['data_count'] == 1
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])