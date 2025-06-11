"""
Comprehensive Test Suite for DSE Simplification

Tests the simplified DSE module functions and integration with streamlined BrainSmith modules.
Validates North Star alignment and practical functionality.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import simplified DSE functions
from brainsmith.dse import (
    parameter_sweep,
    batch_evaluate,
    find_best_result,
    compare_results,
    sample_design_space,
    generate_parameter_grid,
    create_parameter_samples,
    export_results,
    estimate_runtime,
    count_parameter_combinations,
    validate_parameter_space,
    DSEResult,
    ParameterSet,
    ComparisonResult,
    DSEConfiguration
)

# Test data and fixtures
@pytest.fixture
def sample_parameters():
    """Sample parameter space for testing."""
    return {
        'pe_count': [1, 2, 4],
        'simd_factor': [1, 2],
        'precision': [8, 16]
    }

@pytest.fixture
def mock_metrics():
    """Mock metrics object that mimics core.metrics.DSEMetrics."""
    metrics = Mock()
    metrics.performance = Mock()
    metrics.performance.throughput_ops_sec = 1000.0
    metrics.performance.latency_ms = 10.0
    metrics.resources = Mock()
    metrics.resources.lut_utilization_percent = 50.0
    metrics.resources.dsp_utilization_percent = 30.0
    metrics.to_dict.return_value = {
        'performance': {
            'throughput_ops_sec': 1000.0,
            'latency_ms': 10.0
        },
        'resources': {
            'lut_utilization_percent': 50.0,
            'dsp_utilization_percent': 30.0
        }
    }
    return metrics

@pytest.fixture
def sample_dse_results(mock_metrics):
    """Sample DSE results for testing."""
    results = []
    
    # Create varied results
    params_and_perf = [
        ({'pe_count': 1, 'precision': 8}, 500.0),
        ({'pe_count': 2, 'precision': 8}, 1000.0),
        ({'pe_count': 4, 'precision': 8}, 1800.0),
        ({'pe_count': 2, 'precision': 16}, 900.0),
        ({'pe_count': 4, 'precision': 16}, 1600.0)
    ]
    
    for params, throughput in params_and_perf:
        metrics = Mock()
        metrics.performance = Mock()
        metrics.performance.throughput_ops_sec = throughput
        metrics.performance.latency_ms = 1000.0 / throughput  # Inverse relationship
        metrics.resources = Mock()
        metrics.resources.lut_utilization_percent = params['pe_count'] * 12.5
        metrics.to_dict.return_value = {
            'performance': {
                'throughput_ops_sec': throughput,
                'latency_ms': 1000.0 / throughput
            },
            'resources': {
                'lut_utilization_percent': params['pe_count'] * 12.5
            }
        }
        
        result = DSEResult(
            parameters=params,
            metrics=metrics,
            build_success=True,
            build_time=5.0
        )
        results.append(result)
    
    return results


class TestParameterSpaceHandling:
    """Test parameter space generation and validation."""
    
    def test_generate_parameter_grid(self, sample_parameters):
        """Test parameter grid generation."""
        grid = generate_parameter_grid(sample_parameters)
        
        # Should generate all combinations: 3 * 2 * 2 = 12
        assert len(grid) == 12
        
        # Check that all combinations are present
        pe_counts = [combo['pe_count'] for combo in grid]
        assert set(pe_counts) == {1, 2, 4}
        
        simd_factors = [combo['simd_factor'] for combo in grid]
        assert set(simd_factors) == {1, 2}
        
        precisions = [combo['precision'] for combo in grid]
        assert set(precisions) == {8, 16}
    
    def test_generate_parameter_grid_empty(self):
        """Test grid generation with empty parameters."""
        grid = generate_parameter_grid({})
        assert grid == [{}]
    
    def test_count_parameter_combinations(self, sample_parameters):
        """Test parameter combination counting."""
        count = count_parameter_combinations(sample_parameters)
        assert count == 12  # 3 * 2 * 2
    
    def test_validate_parameter_space_valid(self, sample_parameters):
        """Test parameter space validation with valid input."""
        is_valid, errors = validate_parameter_space(sample_parameters)
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_parameter_space_invalid(self):
        """Test parameter space validation with invalid input."""
        # Empty parameter space
        is_valid, errors = validate_parameter_space({})
        assert not is_valid
        assert "Parameter space is empty" in errors
        
        # Parameter with no values
        invalid_params = {'pe_count': []}
        is_valid, errors = validate_parameter_space(invalid_params)
        assert not is_valid
        assert any("no values" in error for error in errors)
    
    def test_create_parameter_samples_random(self, sample_parameters):
        """Test random parameter sampling."""
        samples = create_parameter_samples(sample_parameters, 'random', n_samples=5, seed=42)
        
        assert len(samples) == 5
        
        # Check that all samples have required parameters
        for sample in samples:
            assert 'pe_count' in sample
            assert 'simd_factor' in sample
            assert 'precision' in sample
            
            # Check values are in valid range
            assert sample['pe_count'] in [1, 2, 4]
            assert sample['simd_factor'] in [1, 2]
            assert sample['precision'] in [8, 16]
    
    def test_create_parameter_samples_lhs(self):
        """Test Latin Hypercube sampling."""
        # Use numeric parameters for LHS
        numeric_params = {
            'pe_count': [1, 2, 4, 8],
            'frequency': [100, 150, 200]
        }
        
        samples = create_parameter_samples(numeric_params, 'lhs', n_samples=10, seed=42)
        
        assert len(samples) == 10
        
        # Check parameter coverage (LHS should provide good coverage)
        pe_counts = [s['pe_count'] for s in samples]
        frequencies = [s['frequency'] for s in samples]
        
        # Should have variety in values
        assert len(set(pe_counts)) > 1
        assert len(set(frequencies)) > 1


class TestDSECoreFunction:
    """Test the core parameter_sweep function."""
    
    @patch('brainsmith.dse.functions.forge')
    @patch('brainsmith.dse.functions.load_blueprint_yaml')
    def test_parameter_sweep_basic(self, mock_load_blueprint, mock_forge, sample_parameters, mock_metrics):
        """Test basic parameter sweep functionality."""
        # Setup mocks
        mock_load_blueprint.return_value = {'name': 'test_blueprint'}
        mock_forge.return_value = {
            'metrics': mock_metrics,
            'build_success': True,
            'build_time': 5.0
        }
        
        # Run parameter sweep
        results = parameter_sweep('test_model.onnx', 'test_blueprint.yaml', sample_parameters)
        
        # Verify results
        assert len(results) == 12  # 3 * 2 * 2 combinations
        assert all(isinstance(r, DSEResult) for r in results)
        assert all(r.build_success for r in results)
        
        # Verify forge was called for each combination
        assert mock_forge.call_count == 12
    
    @patch('brainsmith.dse.functions.forge')
    @patch('brainsmith.dse.functions.load_blueprint_yaml')
    def test_parameter_sweep_with_failures(self, mock_load_blueprint, mock_forge, sample_parameters):
        """Test parameter sweep with some build failures."""
        # Setup mocks
        mock_load_blueprint.return_value = {'name': 'test_blueprint'}
        
        # Make some builds fail
        def mock_forge_side_effect(*args, **kwargs):
            if kwargs.get('pe_count') == 4:  # Fail high PE counts
                raise RuntimeError("Build failed")
            return {
                'metrics': Mock(),
                'build_success': True,
                'build_time': 3.0
            }
        
        mock_forge.side_effect = mock_forge_side_effect
        
        # Run with continue_on_failure=True (default)
        config = DSEConfiguration(continue_on_failure=True)
        results = parameter_sweep('test_model.onnx', 'test_blueprint.yaml', sample_parameters, config)
        
        # Should have results for all combinations, some failed
        assert len(results) == 12
        successful_results = [r for r in results if r.build_success]
        failed_results = [r for r in results if not r.build_success]
        
        assert len(successful_results) == 8  # 4 combinations had pe_count=4
        assert len(failed_results) == 4
    
    def test_parameter_sweep_integration_hooks(self, sample_parameters):
        """Test integration with hooks system."""
        with patch('brainsmith.dse.functions.log_optimization_event') as mock_log_opt, \
             patch('brainsmith.dse.functions.log_dse_event') as mock_log_dse, \
             patch('brainsmith.dse.functions.forge') as mock_forge, \
             patch('brainsmith.dse.functions.load_blueprint_yaml') as mock_load_blueprint:
            
            # Setup mocks
            mock_load_blueprint.return_value = {'name': 'test_blueprint'}
            mock_forge.return_value = {
                'metrics': Mock(),
                'build_success': True,
                'build_time': 2.0
            }
            
            # Run parameter sweep
            results = parameter_sweep('test_model.onnx', 'test_blueprint.yaml', sample_parameters)
            
            # Verify hooks integration
            assert mock_log_opt.call_count >= 2  # Start and complete events
            
            # Check start event
            start_call = mock_log_opt.call_args_list[0]
            assert start_call[0][0] == 'dse_start'
            assert 'total_combinations' in start_call[0][1]
            assert start_call[0][1]['total_combinations'] == 12


class TestResultAnalysis:
    """Test result analysis and comparison functions."""
    
    def test_find_best_result_maximize(self, sample_dse_results):
        """Test finding best result by maximizing a metric."""
        best = find_best_result(sample_dse_results, 'performance.throughput_ops_sec', 'maximize')
        
        assert best is not None
        assert best.metrics.performance.throughput_ops_sec == 1800.0
        assert best.parameters['pe_count'] == 4
        assert best.parameters['precision'] == 8
    
    def test_find_best_result_minimize(self, sample_dse_results):
        """Test finding best result by minimizing a metric."""
        best = find_best_result(sample_dse_results, 'performance.latency_ms', 'minimize')
        
        assert best is not None
        # Should be the one with highest throughput (lowest latency)
        assert best.metrics.performance.throughput_ops_sec == 1800.0
    
    def test_find_best_result_no_successful(self):
        """Test finding best result with no successful builds."""
        failed_results = [
            DSEResult(
                parameters={'pe_count': 1},
                metrics=Mock(),
                build_success=False,
                build_time=0.0
            )
        ]
        
        best = find_best_result(failed_results, 'performance.throughput_ops_sec')
        assert best is None
    
    def test_compare_results_single_metric(self, sample_dse_results):
        """Test comparing results with single metric."""
        comparison = compare_results(sample_dse_results, ['performance.throughput_ops_sec'])
        
        assert isinstance(comparison, ComparisonResult)
        assert comparison.best_result is not None
        assert len(comparison.ranking) == 5
        assert comparison.comparison_metric == 'weighted_composite_1_metrics'
        
        # Results should be ranked by throughput (highest first)
        throughputs = [r.metrics.performance.throughput_ops_sec for r in comparison.ranking]
        assert throughputs == sorted(throughputs, reverse=True)
    
    def test_compare_results_multiple_metrics(self, sample_dse_results):
        """Test comparing results with multiple metrics and weights."""
        metrics = ['performance.throughput_ops_sec', 'resources.lut_utilization_percent']
        weights = [0.7, 0.3]  # Prioritize throughput over resource usage
        
        comparison = compare_results(sample_dse_results, metrics, weights)
        
        assert comparison.best_result is not None
        assert len(comparison.ranking) == 5
        assert 'weights_used' in comparison.summary_stats
        assert comparison.summary_stats['weights_used'] == weights


class TestDataExport:
    """Test data export functionality for external analysis tools."""
    
    def test_export_to_pandas(self, sample_dse_results):
        """Test export to pandas DataFrame."""
        try:
            import pandas as pd
            
            df = export_results(sample_dse_results, 'pandas')
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 5
            
            # Check that parameters are columns
            assert 'pe_count' in df.columns
            assert 'precision' in df.columns
            
            # Check that metrics are flattened
            assert 'performance_throughput_ops_sec' in df.columns
            assert 'resources_lut_utilization_percent' in df.columns
            
            # Check build info
            assert 'build_success' in df.columns
            assert 'build_time' in df.columns
            
        except ImportError:
            pytest.skip("pandas not available")
    
    def test_export_to_csv(self, sample_dse_results):
        """Test export to CSV format."""
        csv_data = export_results(sample_dse_results, 'csv')
        
        assert isinstance(csv_data, str)
        assert 'pe_count' in csv_data
        assert 'build_success' in csv_data
        
        # Should have header line plus data lines
        lines = csv_data.strip().split('\n')
        assert len(lines) == 6  # 1 header + 5 data rows
    
    def test_export_to_json(self, sample_dse_results):
        """Test export to JSON format."""
        json_data = export_results(sample_dse_results, 'json')
        
        assert isinstance(json_data, str)
        
        import json
        data = json.loads(json_data)
        
        assert isinstance(data, list)
        assert len(data) == 5
        
        # Check structure
        first_result = data[0]
        assert 'parameters' in first_result
        assert 'metrics' in first_result
        assert 'build_success' in first_result
    
    def test_export_to_file(self, sample_dse_results):
        """Test export to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_results.csv')
            
            export_results(sample_dse_results, 'csv', filepath)
            
            assert os.path.exists(filepath)
            
            with open(filepath, 'r') as f:
                content = f.read()
                assert 'pe_count' in content


class TestBatchEvaluation:
    """Test batch evaluation functionality."""
    
    @patch('brainsmith.dse.functions._evaluate_single_configuration')
    def test_batch_evaluate_success(self, mock_evaluate, mock_metrics):
        """Test successful batch evaluation."""
        # Setup mock
        mock_evaluate.return_value = DSEResult(
            parameters={'pe_count': 2},
            metrics=mock_metrics,
            build_success=True,
            build_time=3.0
        )
        
        model_list = ['model1.onnx', 'model2.onnx', 'model3.onnx']
        parameters = {'pe_count': 2, 'precision': 8}
        
        results = batch_evaluate(model_list, 'blueprint.yaml', parameters)
        
        assert len(results) == 3
        assert all(path in results for path in model_list)
        assert all(r.build_success for r in results.values())
        
        # Should call evaluate for each model
        assert mock_evaluate.call_count == 3
    
    @patch('brainsmith.dse.functions._evaluate_single_configuration')
    def test_batch_evaluate_with_failures(self, mock_evaluate):
        """Test batch evaluation with some failures."""
        def mock_evaluate_side_effect(model_path, blueprint_path, params):
            if 'model2' in model_path:
                raise RuntimeError("Build failed")
            return DSEResult(
                parameters=params,
                metrics=Mock(),
                build_success=True,
                build_time=2.0
            )
        
        mock_evaluate.side_effect = mock_evaluate_side_effect
        
        model_list = ['model1.onnx', 'model2.onnx', 'model3.onnx']
        parameters = {'pe_count': 2}
        config = DSEConfiguration(continue_on_failure=True)
        
        results = batch_evaluate(model_list, 'blueprint.yaml', parameters, config)
        
        assert len(results) == 3
        assert results['model1.onnx'].build_success
        assert not results['model2.onnx'].build_success  # Failed
        assert results['model3.onnx'].build_success


class TestDesignSpaceSampling:
    """Test design space sampling functions."""
    
    def test_sample_design_space_random(self, sample_parameters):
        """Test random design space sampling."""
        samples = sample_design_space(sample_parameters, 'random', n_samples=20, seed=42)
        
        assert len(samples) == 20
        
        # Check that samples are valid
        for sample in samples:
            assert sample['pe_count'] in [1, 2, 4]
            assert sample['simd_factor'] in [1, 2]
            assert sample['precision'] in [8, 16]
    
    def test_sample_design_space_lhs(self):
        """Test LHS design space sampling."""
        # Use parameters suitable for LHS
        numeric_params = {
            'pe_count': [1, 8],  # Will be treated as range
            'frequency': [100, 200]
        }
        
        samples = sample_design_space(numeric_params, 'lhs', n_samples=10, seed=42)
        
        assert len(samples) == 10
        
        # LHS should provide good coverage
        pe_counts = [s['pe_count'] for s in samples]
        frequencies = [s['frequency'] for s in samples]
        
        # Should have variety in sampled values
        assert len(set(pe_counts)) > 3
        assert len(set(frequencies)) > 3


class TestIntegrationWithStreamlinedModules:
    """Test integration with other simplified BrainSmith modules."""
    
    def test_blueprint_integration(self):
        """Test integration with simplified blueprint functions."""
        with patch('brainsmith.dse.functions.load_blueprint_yaml') as mock_load:
            mock_load.return_value = {
                'name': 'test_blueprint',
                'build_steps': ['step1', 'step2'],
                'objectives': {'throughput': {'direction': 'maximize'}}
            }
            
            # This should not raise an error
            sample_design_space({'pe_count': [1, 2]}, 'random', 5)
            
            # Blueprint loading should work in parameter_sweep context
            # (tested implicitly through other parameter_sweep tests)
    
    def test_metrics_integration(self, mock_metrics):
        """Test integration with simplified metrics system."""
        # Test DSEResult with metrics
        result = DSEResult(
            parameters={'pe_count': 4},
            metrics=mock_metrics,
            build_success=True,
            build_time=5.0
        )
        
        # Should be able to access metrics
        assert result.metrics.performance.throughput_ops_sec == 1000.0
        
        # Should be able to convert to dict
        result_dict = result.to_dict()
        assert 'metrics' in result_dict
        assert result_dict['metrics']['performance']['throughput_ops_sec'] == 1000.0
    
    def test_hooks_integration(self):
        """Test integration with hooks system."""
        with patch('brainsmith.dse.functions.log_optimization_event') as mock_log:
            # Test that DSE events are logged
            sample_design_space({'pe_count': [1, 2]}, 'random', 3)
            
            # Should log sampling event
            assert any(call[0][0] == 'design_space_sampled' for call in mock_log.call_args_list)


class TestEstimationAndUtilities:
    """Test runtime estimation and utility functions."""
    
    def test_estimate_runtime(self):
        """Test runtime estimation."""
        param_combinations = [
            {'pe_count': 1, 'precision': 8},
            {'pe_count': 2, 'precision': 8},
            {'pe_count': 4, 'precision': 8}
        ]
        
        estimated_time = estimate_runtime(param_combinations, benchmark_time=10.0)
        assert estimated_time == 30.0  # 3 combinations * 10s each
    
    def test_dse_configuration(self):
        """Test DSE configuration object."""
        config = DSEConfiguration(
            max_parallel=4,
            timeout_seconds=1800,
            continue_on_failure=False,
            export_format='json'
        )
        
        assert config.max_parallel == 4
        assert config.timeout_seconds == 1800
        assert not config.continue_on_failure
        assert config.export_format == 'json'
        
        # Test conversion to dict
        config_dict = config.to_dict()
        assert config_dict['max_parallel'] == 4
        assert config_dict['export_format'] == 'json'


class TestBackwardsCompatibility:
    """Test backwards compatibility with deprecated interfaces."""
    
    def test_deprecated_warnings(self):
        """Test that deprecated functions issue warnings."""
        from brainsmith.dse import create_dse_engine, DSEInterface, DSEEngine
        
        with pytest.warns(DeprecationWarning):
            create_dse_engine()
        
        with pytest.warns(DeprecationWarning):
            DSEInterface()
        
        with pytest.warns(DeprecationWarning):
            DSEEngine()


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_results_handling(self):
        """Test handling of empty result lists."""
        # Empty results should not crash functions
        best = find_best_result([], 'performance.throughput_ops_sec')
        assert best is None
        
        comparison = compare_results([], ['performance.throughput_ops_sec'])
        assert comparison.best_result is None
        assert len(comparison.ranking) == 0
    
    def test_invalid_metric_names(self, sample_dse_results):
        """Test handling of invalid metric names."""
        # Should handle non-existent metrics gracefully
        best = find_best_result(sample_dse_results, 'nonexistent.metric')
        # Should still return a result (the first successful one)
        assert best is not None
    
    def test_mixed_success_failure_results(self):
        """Test handling results with mixed success/failure."""
        results = [
            DSEResult({'pe_count': 1}, Mock(), True, 5.0),
            DSEResult({'pe_count': 2}, Mock(), False, 0.0),  # Failed
            DSEResult({'pe_count': 4}, Mock(), True, 8.0)
        ]
        
        # Should filter to successful results only
        best = find_best_result(results, 'performance.throughput_ops_sec')
        assert best is not None
        assert best.build_success


# Integration test that demonstrates the full workflow
def test_full_dse_workflow():
    """
    Integration test demonstrating the complete simplified DSE workflow.
    This test shows how simple the new API is compared to the old enterprise framework.
    """
    # Define parameter space
    parameters = {
        'pe_count': [1, 2, 4],
        'simd_factor': [1, 2],
        'precision': [8, 16]
    }
    
    with patch('brainsmith.dse.functions.forge') as mock_forge, \
         patch('brainsmith.dse.functions.load_blueprint_yaml') as mock_load_blueprint:
        
        # Setup mocks
        mock_load_blueprint.return_value = {'name': 'test_blueprint'}
        
        def mock_forge_implementation(**kwargs):
            # Simulate different performance based on parameters
            pe_count = kwargs.get('pe_count', 1)
            precision = kwargs.get('precision', 8)
            
            throughput = pe_count * 400 + (16 - precision) * 50
            
            metrics = Mock()
            metrics.performance = Mock()
            metrics.performance.throughput_ops_sec = throughput
            metrics.performance.latency_ms = 1000.0 / throughput
            metrics.to_dict.return_value = {
                'performance': {
                    'throughput_ops_sec': throughput,
                    'latency_ms': 1000.0 / throughput
                }
            }
            
            return {
                'metrics': metrics,
                'build_success': True,
                'build_time': 3.0 + pe_count * 0.5
            }
        
        mock_forge.side_effect = mock_forge_implementation
        
        # Step 1: Run parameter sweep
        results = parameter_sweep('model.onnx', 'blueprint.yaml', parameters)
        assert len(results) == 12
        
        # Step 2: Find best result
        best = find_best_result(results, 'performance.throughput_ops_sec', 'maximize')
        assert best is not None
        assert best.parameters['pe_count'] == 4  # Highest PE count should be best
        
        # Step 3: Compare results across multiple metrics
        comparison = compare_results(results, ['performance.throughput_ops_sec'])
        assert len(comparison.ranking) == 12
        
        # Step 4: Export for external analysis
        try:
            df = export_results(results, 'pandas')
            assert len(df) == 12
            assert 'pe_count' in df.columns
        except ImportError:
            # If pandas not available, test CSV export
            csv_data = export_results(results, 'csv')
            assert 'pe_count' in csv_data
        
        # This workflow demonstrates:
        # - Simple function calls (no enterprise objects)
        # - Integration with all streamlined modules
        # - Data export for external analysis tools
        # - Practical FPGA DSE workflows
        print("✅ Full DSE workflow completed successfully!")


if __name__ == '__main__':
    # Run the tests
    pytest.main([__file__, '-v'])