"""
Comprehensive Test Suite for Metrics Simplification

Tests the North Star aligned metrics implementation including:
- Simple data types and functions
- Integration with streamlined modules
- Data export for external analysis tools
- Backwards compatibility
"""

import pytest
import json
import time
from unittest.mock import Mock, patch
from typing import Dict, Any, List

# Import simplified metrics functions and types
try:
    from brainsmith.data import (
        # Core functions
        collect_build_metrics,
        collect_performance_metrics,
        collect_resource_metrics,
        collect_quality_metrics,
        summarize_metrics,
        compare_metrics,
        filter_metrics,
        validate_metrics,
        
        # Export functions
        export_metrics,
        export_to_pandas,
        export_to_csv,
        export_to_json,
        create_metrics_report,
        
        # Data types
        BuildMetrics,
        PerformanceData,
        ResourceData,
        QualityData,
        BuildData,
        MetricsSummary,
        MetricsConfiguration
    )
    METRICS_AVAILABLE = True
except ImportError as e:
    METRICS_AVAILABLE = False
    print(f"Metrics module not available: {e}")


@pytest.mark.skipif(not METRICS_AVAILABLE, reason="Metrics module not available")
class TestMetricsDataTypes:
    """Test simple metrics data types."""
    
    def test_performance_data_creation(self):
        """Test PerformanceData creation and methods."""
        perf = PerformanceData(
            throughput_ops_sec=1000.0,
            latency_ms=2.5,
            clock_freq_mhz=200.0
        )
        
        assert perf.throughput_ops_sec == 1000.0
        assert perf.latency_ms == 2.5
        assert perf.clock_freq_mhz == 200.0
        
        # Test efficiency ratio calculation
        efficiency = perf.get_efficiency_ratio()
        assert efficiency == 5.0  # 1000 / 200
        
        # Test dictionary conversion
        data_dict = perf.to_dict()
        assert data_dict['throughput_ops_sec'] == 1000.0
        assert data_dict['latency_ms'] == 2.5
    
    def test_resource_data_creation(self):
        """Test ResourceData creation and methods."""
        resources = ResourceData(
            lut_utilization_percent=75.0,
            dsp_utilization_percent=60.0,
            bram_utilization_percent=45.0,
            lut_count=50000,
            dsp_count=300
        )
        
        assert resources.lut_utilization_percent == 75.0
        assert resources.lut_count == 50000
        
        # Test total utilization calculation
        total_util = resources.get_total_utilization()
        expected = (75.0 + 60.0 + 45.0) / 3.0  # Only 3 non-None values
        assert abs(total_util - expected) < 0.1
    
    def test_build_metrics_creation(self):
        """Test BuildMetrics complete container."""
        metrics = BuildMetrics(
            model_path='test_model.onnx',
            blueprint_path='test_blueprint.yaml',
            parameters={'pe_count': 4, 'simd_factor': 2}
        )
        
        assert metrics.model_path == 'test_model.onnx'
        assert metrics.parameters['pe_count'] == 4
        assert metrics.is_successful() == True  # Default build success
        
        # Test JSON serialization
        json_str = metrics.to_json()
        data = json.loads(json_str)
        assert data['model_path'] == 'test_model.onnx'
        assert data['parameters']['pe_count'] == 4
        
        # Test from_dict creation
        metrics_copy = BuildMetrics.from_dict(data)
        assert metrics_copy.model_path == metrics.model_path
        assert metrics_copy.parameters == metrics.parameters


@pytest.mark.skipif(not METRICS_AVAILABLE, reason="Metrics module not available")
class TestMetricsCoreFunctions:
    """Test core metrics collection functions."""
    
    def test_collect_build_metrics_with_dict_result(self):
        """Test metrics collection from dictionary build result."""
        mock_result = {
            'performance': {
                'throughput_ops_sec': 1500.0,
                'latency_ms': 3.2,
                'clock_freq_mhz': 250.0
            },
            'resources': {
                'lut_utilization_percent': 80.0,
                'dsp_utilization_percent': 65.0,
                'lut_count': 60000
            },
            'quality': {
                'accuracy_percent': 95.5
            },
            'build_info': {
                'build_success': True,
                'build_time_seconds': 45.0
            }
        }
        
        metrics = collect_build_metrics(
            mock_result,
            model_path='test.onnx',
            blueprint_path='test.yaml',
            parameters={'pe_count': 8}
        )
        
        assert metrics.performance.throughput_ops_sec == 1500.0
        assert metrics.resources.lut_utilization_percent == 80.0
        assert metrics.quality.accuracy_percent == 95.5
        assert metrics.build.build_success == True
        assert metrics.model_path == 'test.onnx'
        assert metrics.parameters['pe_count'] == 8
    
    def test_collect_build_metrics_with_object_result(self):
        """Test metrics collection from object build result."""
        # Create mock result with object attributes
        mock_result = Mock()
        
        # Performance object
        mock_perf = Mock()
        mock_perf.throughput_ops_sec = 2000.0
        mock_perf.latency_ms = 1.8
        mock_result.performance = mock_perf
        
        # Resources object
        mock_resources = Mock()
        mock_resources.lut_utilization_percent = 70.0
        mock_resources.dsp_utilization_percent = 55.0
        mock_result.resources = mock_resources
        
        metrics = collect_build_metrics(mock_result, model_path='mock.onnx')
        
        assert metrics.performance.throughput_ops_sec == 2000.0
        assert metrics.performance.latency_ms == 1.8
        assert metrics.resources.lut_utilization_percent == 70.0
        assert metrics.model_path == 'mock.onnx'
    
    def test_collect_performance_metrics(self):
        """Test performance metrics extraction."""
        mock_result = {
            'performance': {
                'throughput_ops_sec': 1200.0,
                'cycles_per_inference': 500,
                'clock_freq_mhz': 200.0
            }
        }
        
        perf = collect_performance_metrics(mock_result)
        
        assert perf.throughput_ops_sec == 1200.0
        assert perf.cycles_per_inference == 500
        assert perf.clock_freq_mhz == 200.0
        
        # Test derived metric calculation
        expected_inference_time = (500 / (200.0 * 1e6)) * 1000
        assert abs(perf.inference_time_ms - expected_inference_time) < 0.001
    
    def test_collect_resource_metrics(self):
        """Test resource metrics extraction."""
        mock_result = {
            'resources': {
                'lut_count': 45000,
                'dsp_count': 200,
                'lut_utilization_percent': 65.0
            }
        }
        
        resources = collect_resource_metrics(mock_result)
        
        assert resources.lut_count == 45000
        assert resources.dsp_count == 200
        assert resources.lut_utilization_percent == 65.0
    
    def test_summarize_metrics(self):
        """Test metrics summarization functionality."""
        # Create test metrics list
        metrics_list = []
        
        for i in range(5):
            metrics = BuildMetrics()
            metrics.performance.throughput_ops_sec = 1000.0 + i * 100
            metrics.resources.lut_utilization_percent = 60.0 + i * 5
            metrics.build.build_success = i < 4  # One failed build
            metrics_list.append(metrics)
        
        summary = summarize_metrics(metrics_list)
        
        assert summary.metric_count == 5
        assert summary.successful_builds == 4
        assert summary.failed_builds == 1
        assert summary.get_success_rate() == 0.8
        
        # Test averages (only successful builds)
        expected_avg_throughput = sum(1000.0 + i * 100 for i in range(4)) / 4
        assert abs(summary.avg_throughput - expected_avg_throughput) < 0.1
    
    def test_compare_metrics(self):
        """Test metrics comparison functionality."""
        metrics_a = BuildMetrics()
        metrics_a.performance.throughput_ops_sec = 1000.0
        metrics_a.resources.lut_utilization_percent = 70.0
        
        metrics_b = BuildMetrics()
        metrics_b.performance.throughput_ops_sec = 1200.0
        metrics_b.resources.lut_utilization_percent = 60.0
        
        comparison = compare_metrics(metrics_a, metrics_b)
        
        assert 'improvement_ratios' in comparison
        assert 'metrics_b_better' in comparison
        
        # B should be better in throughput
        throughput_ratio = comparison['improvement_ratios']['throughput']
        assert throughput_ratio == 1.2
    
    def test_filter_metrics(self):
        """Test metrics filtering functionality."""
        metrics_list = []
        
        for i in range(5):
            metrics = BuildMetrics()
            metrics.performance.throughput_ops_sec = 800.0 + i * 200  # 800, 1000, 1200, 1400, 1600
            metrics.resources.lut_utilization_percent = 50.0 + i * 10  # 50, 60, 70, 80, 90
            metrics.build.build_success = True
            metrics_list.append(metrics)
        
        # Filter for high throughput, low resource usage
        filtered = filter_metrics(metrics_list, {
            'min_throughput': 1100,
            'max_lut_utilization': 85,
            'build_success': True
        })
        
        # Should get metrics with throughput >= 1100 and LUT <= 85%
        # That's metrics[2] (1200, 70%) and metrics[3] (1400, 80%)
        assert len(filtered) == 2
        assert all(m.performance.throughput_ops_sec >= 1100 for m in filtered)
        assert all(m.resources.lut_utilization_percent <= 85 for m in filtered)
    
    def test_validate_metrics(self):
        """Test metrics validation functionality."""
        # Valid metrics
        valid_metrics = BuildMetrics()
        valid_metrics.performance.throughput_ops_sec = 1000.0
        valid_metrics.performance.latency_ms = 1.0  # Consistent: 1000 ops/sec = 1ms latency
        valid_metrics.resources.lut_utilization_percent = 75.0
        valid_metrics.build.build_success = True
        
        issues = validate_metrics(valid_metrics)
        assert len(issues) == 0
        
        # Invalid metrics
        invalid_metrics = BuildMetrics()
        invalid_metrics.performance.throughput_ops_sec = 1000.0
        invalid_metrics.performance.latency_ms = 5.0  # Inconsistent: should be 1ms
        invalid_metrics.resources.lut_utilization_percent = 150.0  # Out of range
        invalid_metrics.build.build_success = False
        
        issues = validate_metrics(invalid_metrics)
        assert len(issues) > 0
        assert any('utilization out of range' in issue for issue in issues)


@pytest.mark.skipif(not METRICS_AVAILABLE, reason="Metrics module not available")
class TestMetricsExport:
    """Test metrics data export functionality."""
    
    def test_export_to_dict(self):
        """Test dictionary export."""
        metrics = BuildMetrics(
            model_path='test.onnx',
            parameters={'pe_count': 4}
        )
        metrics.performance.throughput_ops_sec = 1500.0
        
        result = export_metrics(metrics, 'dict')
        
        assert isinstance(result, dict)
        assert result['model_path'] == 'test.onnx'
        assert result['parameters']['pe_count'] == 4
        assert result['performance']['throughput_ops_sec'] == 1500.0
    
    def test_export_to_json(self):
        """Test JSON export."""
        metrics = BuildMetrics(model_path='test.onnx')
        metrics.performance.throughput_ops_sec = 1200.0
        
        json_str = export_metrics(metrics, 'json')
        
        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert 'metrics' in data
        assert len(data['metrics']) == 1
        assert data['metrics'][0]['model_path'] == 'test.onnx'
    
    def test_export_to_csv(self):
        """Test CSV export."""
        metrics_list = []
        for i in range(3):
            metrics = BuildMetrics(model_path=f'model_{i}.onnx')
            metrics.performance.throughput_ops_sec = 1000.0 + i * 100
            metrics.resources.lut_utilization_percent = 60.0 + i * 10
            metrics_list.append(metrics)
        
        csv_str = export_metrics(metrics_list, 'csv')
        
        assert isinstance(csv_str, str)
        lines = csv_str.strip().split('\n')
        assert len(lines) == 4  # Header + 3 data rows
        
        # Check header contains expected fields
        header = lines[0]
        assert 'model_path' in header
        assert 'performance_throughput_ops_sec' in header
        assert 'resources_lut_utilization_percent' in header
    
    @pytest.mark.skipif(True, reason="pandas optional dependency")
    def test_export_to_pandas(self):
        """Test pandas DataFrame export (if pandas available)."""
        try:
            import pandas as pd
            
            metrics_list = []
            for i in range(3):
                metrics = BuildMetrics(model_path=f'model_{i}.onnx')
                metrics.performance.throughput_ops_sec = 1000.0 + i * 100
                metrics_list.append(metrics)
            
            df = export_metrics(metrics_list, 'pandas')
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 3
            assert 'model_path' in df.columns
            assert 'performance_throughput_ops_sec' in df.columns
            
        except ImportError:
            pytest.skip("pandas not available")
    
    def test_create_metrics_report(self):
        """Test metrics report generation."""
        metrics_list = []
        for i in range(3):
            metrics = BuildMetrics(model_path=f'model_{i}.onnx')
            metrics.performance.throughput_ops_sec = 1000.0 + i * 100
            metrics.build.build_success = True
            metrics_list.append(metrics)
        
        # Test markdown report
        md_report = create_metrics_report(metrics_list, 'markdown')
        assert isinstance(md_report, str)
        assert '# FPGA Metrics Report' in md_report
        assert 'Total configurations: 3' in md_report
        assert 'Success rate: 100.0%' in md_report
        
        # Test text report
        text_report = create_metrics_report(metrics_list, 'text')
        assert isinstance(text_report, str)
        assert 'FPGA Metrics Report' in text_report


@pytest.mark.skipif(not METRICS_AVAILABLE, reason="Metrics module not available")
class TestModuleIntegration:
    """Test integration with streamlined BrainSmith modules."""
    
    @patch('brainsmith.data.functions.HOOKS_AVAILABLE', True)
    @patch('brainsmith.data.functions.log_metrics_event')
    def test_hooks_integration(self, mock_log_event):
        """Test integration with hooks module."""
        mock_result = {'performance': {'throughput_ops_sec': 1000.0}}
        
        metrics = collect_build_metrics(mock_result, model_path='test.onnx')
        
        # Verify hooks events were logged
        assert mock_log_event.call_count >= 2
        calls = [call[0][0] for call in mock_log_event.call_args_list]
        assert 'metrics_collection_start' in calls
        assert 'metrics_collection_complete' in calls
    
    def test_core_integration_compatibility(self):
        """Test compatibility with core module result formats."""
        # Simulate core.forge() result format
        mock_core_result = {
            'build_success': True,
            'performance': {
                'throughput_ops_sec': 1500.0,
                'latency_ms': 2.0
            },
            'resources': {
                'lut_utilization_percent': 75.0,
                'dsp_utilization_percent': 60.0
            },
            'build_time': 30.0
        }
        
        metrics = collect_build_metrics(
            mock_core_result,
            model_path='core_test.onnx',
            blueprint_path='core_blueprint.yaml'
        )
        
        assert metrics.performance.throughput_ops_sec == 1500.0
        assert metrics.resources.lut_utilization_percent == 75.0
        assert metrics.build.build_success == True
        assert metrics.model_path == 'core_test.onnx'
    
    def test_dse_integration_compatibility(self):
        """Test compatibility with DSE result formats."""
        # Simulate DSE parameter sweep results
        dse_results = []
        
        for pe_count in [1, 2, 4, 8]:
            mock_result = {
                'performance': {
                    'throughput_ops_sec': pe_count * 250.0,
                    'latency_ms': 4.0 / pe_count
                },
                'resources': {
                    'lut_utilization_percent': pe_count * 15.0,
                    'dsp_utilization_percent': pe_count * 12.0
                }
            }
            
            metrics = collect_build_metrics(
                mock_result,
                parameters={'pe_count': pe_count, 'simd_factor': 2}
            )
            dse_results.append(metrics)
        
        # Test DSE summary
        summary = summarize_metrics(dse_results)
        assert summary.metric_count == 4
        assert summary.successful_builds == 4
        assert summary.max_throughput == 2000.0  # 8 * 250
        assert summary.min_latency == 0.5  # 4.0 / 8
        
        # Test filtering for good configurations
        good_configs = filter_metrics(dse_results, {
            'min_throughput': 750,  # pe_count >= 4
            'max_lut_utilization': 80  # pe_count <= 4
        })
        assert len(good_configs) == 1  # Only pe_count=4


@pytest.mark.skipif(not METRICS_AVAILABLE, reason="Metrics module not available")
class TestBackwardsCompatibility:
    """Test backwards compatibility with enterprise interfaces."""
    
    def test_deprecated_class_warnings(self):
        """Test that deprecated classes trigger warnings."""
        with pytest.warns(DeprecationWarning):
            from brainsmith.metrics import MetricsCollector
            collector = MetricsCollector()
        
        with pytest.warns(DeprecationWarning):
            from brainsmith.metrics import MetricsRegistry
            registry = MetricsRegistry()
        
        with pytest.warns(DeprecationWarning):
            from brainsmith.metrics import MetricsExporter
            exporter = MetricsExporter()
    
    def test_deprecated_manager_compatibility(self):
        """Test basic compatibility with MetricsManager interface."""
        with pytest.warns(DeprecationWarning):
            from brainsmith.metrics import MetricsManager
            manager = MetricsManager()
        
        # Test basic compatibility methods
        assert hasattr(manager, 'collect_manual')
        assert hasattr(manager, 'export_metrics')
        
        # Test that methods return sensible defaults
        collections = manager.collect_manual()
        assert isinstance(collections, list)


@pytest.mark.skipif(not METRICS_AVAILABLE, reason="Metrics module not available")
class TestPerformanceAndEdgeCases:
    """Test performance characteristics and edge cases."""
    
    def test_large_metrics_list_performance(self):
        """Test performance with large metrics lists."""
        import time
        
        # Create large metrics list
        metrics_list = []
        for i in range(100):
            metrics = BuildMetrics()
            metrics.performance.throughput_ops_sec = 1000.0 + i
            metrics.resources.lut_utilization_percent = 50.0 + (i % 40)
            metrics_list.append(metrics)
        
        # Test summarization performance
        start_time = time.time()
        summary = summarize_metrics(metrics_list)
        summarize_time = time.time() - start_time
        
        assert summary.metric_count == 100
        assert summarize_time < 1.0  # Should be fast
        
        # Test export performance
        start_time = time.time()
        csv_data = export_metrics(metrics_list, 'csv')
        export_time = time.time() - start_time
        
        assert len(csv_data) > 1000  # Should have substantial data
        assert export_time < 2.0  # Should be reasonably fast
    
    def test_empty_metrics_handling(self):
        """Test handling of empty metrics lists."""
        empty_list = []
        
        summary = summarize_metrics(empty_list)
        assert summary.metric_count == 0
        assert summary.get_success_rate() == 0.0
        
        csv_data = export_metrics(empty_list, 'csv')
        assert csv_data == ""
        
        json_data = export_metrics(empty_list, 'json')
        data = json.loads(json_data)
        assert data['data_count'] == 0
    
    def test_invalid_data_handling(self):
        """Test handling of invalid or corrupted data."""
        # Test with None values
        metrics = BuildMetrics()
        metrics.performance.throughput_ops_sec = None
        metrics.resources.lut_utilization_percent = None
        
        # Should not crash
        summary = summarize_metrics([metrics])
        assert summary.metric_count == 1
        
        # Test export with None values
        csv_data = export_metrics(metrics, 'csv')
        assert isinstance(csv_data, str)
        
        # Test validation with missing data
        issues = validate_metrics(metrics)
        assert isinstance(issues, list)
    
    def test_memory_usage_with_large_datasets(self):
        """Test memory efficiency with large datasets."""
        # Create metrics with substantial data
        metrics_list = []
        for i in range(50):
            metrics = BuildMetrics()
            metrics.parameters = {f'param_{j}': j for j in range(20)}  # 20 parameters each
            metrics.metadata = {f'meta_{j}': f'value_{j}' for j in range(10)}  # 10 metadata items
            metrics_list.append(metrics)
        
        # Test that export completes without memory issues
        try:
            csv_data = export_metrics(metrics_list, 'csv')
            json_data = export_metrics(metrics_list, 'json')
            
            assert len(csv_data) > 5000
            assert len(json_data) > 10000
            
        except MemoryError:
            pytest.fail("Memory error during large dataset export")


if __name__ == "__main__":
    # Run basic tests if executed directly
    if METRICS_AVAILABLE:
        print("✅ Running metrics simplification tests...")
        
        # Test basic functionality
        test_types = TestMetricsDataTypes()
        test_types.test_performance_data_creation()
        test_types.test_resource_data_creation()
        test_types.test_build_metrics_creation()
        print("✅ Data types tests passed")
        
        test_functions = TestMetricsCoreFunctions()
        test_functions.test_collect_build_metrics_with_dict_result()
        test_functions.test_summarize_metrics()
        print("✅ Core functions tests passed")
        
        test_export = TestMetricsExport()
        test_export.test_export_to_dict()
        test_export.test_export_to_json()
        test_export.test_export_to_csv()
        print("✅ Export functions tests passed")
        
        print("🎉 All basic metrics tests passed!")
        
    else:
        print("❌ Metrics module not available for testing")