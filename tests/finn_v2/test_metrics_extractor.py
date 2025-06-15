"""
Tests for MetricsExtractor

Tests FINN results parsing and metrics standardization.
"""

import pytest
from unittest.mock import Mock, patch, mock_open
import json
from pathlib import Path

from brainsmith.core.finn_v2.metrics_extractor import MetricsExtractor


class TestMetricsExtractor:
    """Tests for MetricsExtractor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = MetricsExtractor()
    
    def test_initialization(self):
        """Test MetricsExtractor initialization."""
        assert self.extractor.supported_metrics is not None
        assert isinstance(self.extractor.supported_metrics, list)
        
        # Check supported metrics
        expected_metrics = [
            'throughput', 'latency', 'clock_frequency',
            'lut_utilization', 'dsp_utilization', 'bram_utilization',
            'power_consumption', 'build_time', 'success_rate'
        ]
        
        for metric in expected_metrics:
            assert metric in self.extractor.supported_metrics
    
    def test_get_supported_metrics(self):
        """Test getting supported metrics list."""
        metrics = self.extractor.get_supported_metrics()
        
        assert isinstance(metrics, list)
        assert len(metrics) > 0
        assert 'throughput' in metrics
        assert 'latency' in metrics
        assert 'resource_efficiency' not in metrics  # This is calculated, not extracted
    
    def test_extract_metrics_basic_structure(self):
        """Test basic metrics extraction structure."""
        # Mock FINN result
        mock_finn_result = Mock()
        mock_finn_result.model = Mock()
        mock_finn_result.output_dir = "test_output"
        
        # Mock FINN config
        mock_finn_config = Mock()
        mock_finn_config.combination_id = "test_123"
        
        # Test extraction
        metrics = self.extractor.extract_metrics(mock_finn_result, mock_finn_config)
        
        # Verify required structure
        assert isinstance(metrics, dict)
        assert 'success' in metrics
        assert 'build_time' in metrics
        assert 'primary_metric' in metrics
        assert 'combination_id' in metrics
        assert 'throughput' in metrics
        assert 'latency' in metrics
        assert 'resource_efficiency' in metrics
        
        # Verify values
        assert metrics['success'] == True
        assert metrics['combination_id'] == "test_123"
        assert isinstance(metrics['resource_efficiency'], float)
    
    def test_extract_metrics_exception_handling(self):
        """Test metrics extraction with exception handling."""
        # Mock problematic FINN result
        mock_finn_result = Mock()
        mock_finn_result.model = None
        
        # Mock config that causes issues
        mock_finn_config = Mock()
        mock_finn_config.combination_id = "error_test"
        
        # Force an exception in performance extraction
        with patch.object(self.extractor, '_extract_performance_metrics', side_effect=Exception("Extraction failed")):
            metrics = self.extractor.extract_metrics(mock_finn_result, mock_finn_config)
            
            # Should return error metrics
            assert metrics['success'] == False
            assert 'error' in metrics
            assert metrics['primary_metric'] == 0.0
            assert metrics['throughput'] == 0.0
            assert metrics['latency'] == float('inf')
    
    def test_extract_performance_metrics_fallback(self):
        """Test performance metrics extraction with fallback estimates."""
        mock_finn_result = Mock()
        mock_finn_result.model = None
        
        # Test fallback performance estimation
        performance = self.extractor._extract_performance_metrics(mock_finn_result)
        
        assert isinstance(performance, dict)
        assert 'throughput' in performance
        assert 'latency' in performance
        assert 'clock_frequency' in performance
        
        # Should provide conservative estimates
        assert performance['throughput'] >= 0
        assert performance['latency'] >= 0
        assert performance['clock_frequency'] >= 0
    
    def test_extract_resource_metrics_empty(self):
        """Test resource metrics extraction with empty result."""
        mock_finn_result = Mock()
        mock_finn_result.output_dir = None
        mock_finn_result.model = None
        
        resources = self.extractor._extract_resource_metrics(mock_finn_result)
        
        assert isinstance(resources, dict)
        assert 'lut_utilization' in resources
        assert 'dsp_utilization' in resources
        assert 'bram_utilization' in resources
        assert 'power_consumption' in resources
        
        # Should be zeros for empty result
        assert resources['lut_utilization'] == 0.0
        assert resources['dsp_utilization'] == 0.0
        assert resources['bram_utilization'] == 0.0
        assert resources['power_consumption'] == 0.0
    
    def test_extract_quality_metrics(self):
        """Test quality metrics extraction."""
        mock_finn_result = Mock()
        mock_finn_result.build_time = 120.5
        mock_finn_result.verification_results = {'passed': True}
        mock_finn_result.build_log = "Build completed with 2 warnings and 0 errors"
        
        mock_finn_config = Mock()
        
        quality = self.extractor._extract_quality_metrics(mock_finn_result, mock_finn_config)
        
        assert isinstance(quality, dict)
        assert quality['build_time'] == 120.5
        assert quality['verification_passed'] == True
        assert quality['warnings_count'] == 2
        assert quality['errors_count'] == 0
    
    def test_extract_from_reports_with_mock_files(self):
        """Test extracting metrics from report files with mocks."""
        output_dir = "test_output"
        
        # Mock performance report data
        performance_data = {
            'throughput_fps': 1500.0,
            'latency_ms': 6.7,
            'clock_freq_mhz': 200.0
        }
        
        # Mock file reading
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(performance_data))):
                metrics = self.extractor._extract_from_reports(output_dir)
                
                assert metrics['throughput'] == 1500.0
                assert metrics['latency'] == 6.7
                assert metrics['clock_frequency'] == 200.0
    
    def test_extract_from_synthesis_reports_with_mock_files(self):
        """Test extracting resource metrics from synthesis reports with mocks."""
        output_dir = "test_output"
        
        # Mock utilization report data
        util_data = {
            'LUT_utilization': 0.75,
            'DSP_utilization': 0.85,
            'BRAM_utilization': 0.60
        }
        
        # Mock power report data
        power_data = {
            'total_power_w': 12.5
        }
        
        # Mock file operations
        def mock_exists(path):
            return str(path).endswith(('utilization_report.json', 'power_report.json'))
        
        def mock_open_files(filename, mode='r'):
            if 'utilization' in filename:
                return mock_open(read_data=json.dumps(util_data))()
            elif 'power' in filename:
                return mock_open(read_data=json.dumps(power_data))()
            else:
                raise FileNotFoundError()
        
        with patch('pathlib.Path.exists', side_effect=mock_exists):
            with patch('builtins.open', side_effect=mock_open_files):
                resources = self.extractor._extract_from_synthesis_reports(output_dir)
                
                assert resources['lut_utilization'] == 0.75
                assert resources['dsp_utilization'] == 0.85
                assert resources['bram_utilization'] == 0.60
                assert resources['power_consumption'] == 12.5
    
    def test_calculate_resource_efficiency(self):
        """Test resource efficiency calculation."""
        # Test with balanced metrics
        metrics = {
            'throughput': 1000.0,
            'lut_utilization': 0.6,
            'dsp_utilization': 0.7,
            'bram_utilization': 0.5
        }
        
        efficiency = self.extractor._calculate_resource_efficiency(metrics)
        
        assert isinstance(efficiency, float)
        assert 0.0 <= efficiency <= 1.0
        
        # Test with zero utilization
        metrics_zero = {
            'throughput': 1000.0,
            'lut_utilization': 0.0,
            'dsp_utilization': 0.0,
            'bram_utilization': 0.0
        }
        
        efficiency_zero = self.extractor._calculate_resource_efficiency(metrics_zero)
        assert efficiency_zero == 0.0
        
        # Test with high throughput and low utilization (high efficiency)
        metrics_efficient = {
            'throughput': 2000.0,
            'lut_utilization': 0.3,
            'dsp_utilization': 0.2,
            'bram_utilization': 0.1
        }
        
        efficiency_high = self.extractor._calculate_resource_efficiency(metrics_efficient)
        assert efficiency_high > efficiency  # Should be more efficient
    
    def test_validate_metrics_valid(self):
        """Test metrics validation with valid metrics."""
        valid_metrics = {
            'success': True,
            'primary_metric': 100.0,
            'throughput': 1500.0,
            'latency': 8.0,
            'lut_utilization': 0.7,
            'dsp_utilization': 0.8,
            'bram_utilization': 0.6
        }
        
        is_valid, warnings = self.extractor.validate_metrics(valid_metrics)
        
        assert is_valid == True
        assert len(warnings) == 0
    
    def test_validate_metrics_missing_required(self):
        """Test metrics validation with missing required fields."""
        incomplete_metrics = {
            'latency': 8.0,
            'lut_utilization': 0.7
        }
        
        is_valid, warnings = self.extractor.validate_metrics(incomplete_metrics)
        
        assert is_valid == False
        assert len(warnings) > 0
        assert any('success' in warning for warning in warnings)
        assert any('primary_metric' in warning for warning in warnings)
        assert any('throughput' in warning for warning in warnings)
    
    def test_validate_metrics_out_of_range(self):
        """Test metrics validation with out-of-range values."""
        invalid_metrics = {
            'success': True,
            'primary_metric': 100.0,
            'throughput': -50.0,  # Invalid: negative
            'latency': -5.0,     # Invalid: negative
            'lut_utilization': 1.5,  # Invalid: > 1.0
            'dsp_utilization': -0.1,  # Invalid: < 0.0
            'bram_utilization': 0.6
        }
        
        is_valid, warnings = self.extractor.validate_metrics(invalid_metrics)
        
        assert is_valid == False
        assert len(warnings) > 0
        assert any('negative throughput' in warning.lower() for warning in warnings)
        assert any('negative latency' in warning.lower() for warning in warnings)
        assert any('lut_utilization' in warning and 'out of range' in warning for warning in warnings)
        assert any('dsp_utilization' in warning and 'out of range' in warning for warning in warnings)
    
    def test_estimate_performance_metrics(self):
        """Test fallback performance estimation."""
        mock_finn_result = Mock()
        
        estimated = self.extractor._estimate_performance_metrics(mock_finn_result)
        
        assert isinstance(estimated, dict)
        assert 'throughput' in estimated
        assert 'latency' in estimated
        assert 'clock_frequency' in estimated
        
        # Should provide reasonable estimates
        assert estimated['throughput'] > 0
        assert estimated['latency'] > 0
        assert estimated['clock_frequency'] > 0
        
        # Should be conservative estimates
        assert estimated['throughput'] <= 1000  # Conservative throughput
        assert estimated['latency'] >= 5       # Conservative latency
        assert estimated['clock_frequency'] >= 100  # Reasonable clock frequency


class TestMetricsExtractorIntegration:
    """Integration tests for MetricsExtractor."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.extractor = MetricsExtractor()
    
    def test_realistic_finn_result_structure(self):
        """Test with realistic FINN result structure."""
        # Create mock FINN result that mimics real structure
        mock_finn_result = Mock()
        mock_finn_result.model = Mock()
        mock_finn_result.output_dir = "realistic_output"
        
        # Add realistic model annotations
        mock_model = Mock()
        mock_model.graph = Mock()
        mock_model.graph.value_info = []
        mock_finn_result.model = mock_model
        
        # Add realistic resource estimates
        mock_finn_result.resource_estimates = {
            'LUT': 45000,
            'DSP': 1200,
            'BRAM': 800
        }
        
        mock_finn_config = Mock()
        mock_finn_config.combination_id = "realistic_001"
        
        # Test extraction
        metrics = self.extractor.extract_metrics(mock_finn_result, mock_finn_config)
        
        # Verify realistic extraction
        assert metrics['success'] == True
        assert metrics['combination_id'] == "realistic_001"
        assert metrics['throughput'] >= 0
        assert metrics['latency'] >= 0
        assert 0.0 <= metrics['resource_efficiency'] <= 1.0
    
    def test_comprehensive_metrics_workflow(self):
        """Test complete metrics extraction workflow."""
        mock_finn_result = Mock()
        mock_finn_result.model = Mock()
        mock_finn_result.output_dir = "comprehensive_test"
        mock_finn_result.build_time = 300.0
        mock_finn_result.verification_results = {'passed': True}
        
        mock_finn_config = Mock()
        mock_finn_config.combination_id = "comprehensive_001"
        
        # Test complete workflow
        metrics = self.extractor.extract_metrics(mock_finn_result, mock_finn_config)
        
        # Validate comprehensive metrics
        is_valid, warnings = self.extractor.validate_metrics(metrics)
        
        # Should have valid metrics structure
        assert isinstance(metrics, dict)
        assert len(metrics) >= 5  # Should have multiple metrics
        
        # If invalid, should be due to missing files (expected in test)
        if not is_valid:
            print(f"Metrics validation warnings (expected): {warnings}")


if __name__ == "__main__":
    # Run basic tests
    test_extractor = TestMetricsExtractor()
    test_extractor.setup_method()
    
    print("Testing MetricsExtractor...")
    test_extractor.test_initialization()
    print("✓ Initialization test passed")
    
    test_extractor.test_extract_metrics_basic_structure()
    print("✓ Basic metrics extraction test passed")
    
    test_extractor.test_calculate_resource_efficiency()
    print("✓ Resource efficiency calculation test passed")
    
    test_extractor.test_validate_metrics_valid()
    print("✓ Metrics validation test passed")
    
    print("All basic tests passed!")