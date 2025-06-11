"""
Comprehensive tests for the simplified automation module.

Tests the North Star simplification of brainsmith.automation with
focus on integration with forge(), hooks, and blueprint systems.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from brainsmith.automation import parameter_sweep, batch_process, find_best, aggregate_stats


class TestParameterSweep:
    """Test parameter_sweep function."""
    
    @patch('brainsmith.automation.sweep.forge')
    @patch('brainsmith.automation.sweep.track_parameter')
    def test_parameter_sweep_basic(self, mock_track_parameter, mock_forge):
        """Test basic parameter sweep functionality."""
        # Mock forge to return realistic results
        mock_forge.return_value = {
            'metrics': {
                'performance': {
                    'throughput': 100.0,
                    'latency': 10.0,
                    'power': 5.0
                }
            },
            'dataflow_graph': Mock()
        }
        
        # Run parameter sweep
        param_ranges = {
            'pe_count': [4, 8],
            'simd_width': [2, 4]
        }
        
        results = parameter_sweep(
            "test_model.onnx",
            "test_blueprint.yaml", 
            param_ranges,
            max_workers=1  # Sequential for predictable testing
        )
        
        # Verify results
        assert len(results) == 4  # 2 x 2 combinations
        assert all('sweep_info' in result for result in results)
        assert all(result['sweep_info']['success'] for result in results)
        
        # Verify forge was called correctly
        assert mock_forge.call_count == 4
        
        # Verify parameter tracking
        assert mock_track_parameter.call_count == 8  # 2 params x 4 combinations
    
    @patch('brainsmith.automation.sweep.forge')
    def test_parameter_sweep_error_handling(self, mock_forge):
        """Test parameter sweep error handling."""
        # Mock forge to raise exception
        mock_forge.side_effect = ValueError("Test error")
        
        param_ranges = {'pe_count': [4, 8]}
        
        results = parameter_sweep(
            "test_model.onnx",
            "test_blueprint.yaml",
            param_ranges,
            max_workers=1
        )
        
        # Verify error handling
        assert len(results) == 2
        assert all(not result['sweep_info']['success'] for result in results)
        assert all('error' in result['sweep_info'] for result in results)
    
    def test_parameter_sweep_progress_callback(self):
        """Test progress callback functionality."""
        progress_calls = []
        
        def progress_callback(completed, total, params):
            progress_calls.append((completed, total, params))
        
        with patch('brainsmith.automation.sweep.forge') as mock_forge:
            mock_forge.return_value = {'metrics': {'performance': {}}}
            
            parameter_sweep(
                "test_model.onnx",
                "test_blueprint.yaml",
                {'pe_count': [4, 8]},
                max_workers=1,
                progress_callback=progress_callback
            )
        
        # Verify progress tracking
        assert len(progress_calls) == 2
        assert progress_calls[0] == (1, 2, {'pe_count': 4})
        assert progress_calls[1] == (2, 2, {'pe_count': 8})


class TestBatchProcess:
    """Test batch_process function."""
    
    @patch('brainsmith.automation.batch.forge')
    def test_batch_process_basic(self, mock_forge):
        """Test basic batch processing functionality."""
        # Mock forge to return different results
        mock_forge.side_effect = [
            {
                'metrics': {
                    'performance': {'throughput': 100.0}
                }
            },
            {
                'metrics': {
                    'performance': {'throughput': 150.0}
                }
            }
        ]
        
        # Run batch processing
        model_blueprint_pairs = [
            ("model1.onnx", "blueprint1.yaml"),
            ("model2.onnx", "blueprint2.yaml")
        ]
        
        results = batch_process(
            model_blueprint_pairs,
            max_workers=1
        )
        
        # Verify results
        assert len(results) == 2
        assert all('batch_info' in result for result in results)
        assert all(result['batch_info']['success'] for result in results)
        
        # Verify forge was called correctly
        assert mock_forge.call_count == 2
        
        # Verify proper ordering
        assert results[0]['batch_info']['model_path'] == "model1.onnx"
        assert results[1]['batch_info']['model_path'] == "model2.onnx"
    
    @patch('brainsmith.automation.batch.forge')
    def test_batch_process_with_common_config(self, mock_forge):
        """Test batch processing with common configuration."""
        mock_forge.return_value = {'metrics': {'performance': {}}}
        
        common_config = {
            'objectives': {'throughput': {'direction': 'maximize'}},
            'constraints': {'max_power': 10.0}
        }
        
        batch_process(
            [("model1.onnx", "blueprint1.yaml")],
            common_config=common_config,
            max_workers=1
        )
        
        # Verify common config was passed to forge
        mock_forge.assert_called_once_with(
            model_path="model1.onnx",
            blueprint_path="blueprint1.yaml",
            **common_config
        )


class TestFindBest:
    """Test find_best function."""
    
    def test_find_best_maximize(self):
        """Test finding best result by maximizing metric."""
        results = [
            {
                'sweep_info': {'success': True},
                'metrics': {
                    'performance': {'throughput': 100.0}
                }
            },
            {
                'sweep_info': {'success': True},
                'metrics': {
                    'performance': {'throughput': 150.0}
                }
            },
            {
                'sweep_info': {'success': True},
                'metrics': {
                    'performance': {'throughput': 120.0}
                }
            }
        ]
        
        best = find_best(results, metric='throughput', maximize=True)
        
        # Verify best result
        assert best is not None
        assert best['metrics']['performance']['throughput'] == 150.0
        assert 'optimization_info' in best
        assert best['optimization_info']['optimized_metric'] == 'throughput'
        assert best['optimization_info']['maximize'] is True
    
    def test_find_best_minimize(self):
        """Test finding best result by minimizing metric."""
        results = [
            {
                'sweep_info': {'success': True},
                'metrics': {
                    'performance': {'latency': 10.0}
                }
            },
            {
                'sweep_info': {'success': True},
                'metrics': {
                    'performance': {'latency': 5.0}
                }
            }
        ]
        
        best = find_best(results, metric='latency', maximize=False)
        
        # Verify best result
        assert best is not None
        assert best['metrics']['performance']['latency'] == 5.0
        assert best['optimization_info']['maximize'] is False
    
    def test_find_best_no_successful_results(self):
        """Test find_best with no successful results."""
        results = [
            {
                'sweep_info': {'success': False, 'error': 'Test error'}
            }
        ]
        
        best = find_best(results, metric='throughput')
        
        # Verify no result returned
        assert best is None
    
    def test_find_best_empty_results(self):
        """Test find_best with empty results list."""
        best = find_best([], metric='throughput')
        assert best is None


class TestAggregateStats:
    """Test aggregate_stats function."""
    
    def test_aggregate_stats_basic(self):
        """Test basic statistical aggregation."""
        results = [
            {
                'sweep_info': {'success': True},
                'metrics': {
                    'performance': {
                        'throughput': 100.0,
                        'latency': 10.0
                    }
                }
            },
            {
                'sweep_info': {'success': True},
                'metrics': {
                    'performance': {
                        'throughput': 150.0,
                        'latency': 8.0
                    }
                }
            },
            {
                'sweep_info': {'success': False, 'error': 'Test error'}
            }
        ]
        
        stats = aggregate_stats(results)
        
        # Verify statistics
        assert stats['total_runs'] == 3
        assert stats['successful_runs'] == 2
        assert stats['success_rate'] == 2/3
        
        # Verify metric statistics
        throughput_stats = stats['aggregated_metrics']['throughput']
        assert throughput_stats['mean'] == 125.0
        assert throughput_stats['min'] == 100.0
        assert throughput_stats['max'] == 150.0
        assert throughput_stats['count'] == 2
        
        latency_stats = stats['aggregated_metrics']['latency']
        assert latency_stats['mean'] == 9.0
        assert latency_stats['min'] == 8.0
        assert latency_stats['max'] == 10.0
    
    def test_aggregate_stats_no_successful_results(self):
        """Test aggregate_stats with no successful results."""
        results = [
            {
                'sweep_info': {'success': False, 'error': 'Test error'}
            }
        ]
        
        stats = aggregate_stats(results)
        
        # Verify error handling
        assert stats['total_runs'] == 1
        assert stats['successful_runs'] == 0
        assert stats['success_rate'] == 0.0
        assert 'error' in stats
    
    def test_aggregate_stats_empty_results(self):
        """Test aggregate_stats with empty results."""
        stats = aggregate_stats([])
        assert 'error' in stats


class TestIntegration:
    """Test integration between functions."""
    
    @patch('brainsmith.automation.sweep.forge')
    def test_parameter_sweep_to_find_best_workflow(self, mock_forge):
        """Test complete workflow from parameter sweep to finding best result."""
        # Mock forge to return varying results
        mock_forge.side_effect = [
            {
                'metrics': {
                    'performance': {'throughput': 100.0}
                }
            },
            {
                'metrics': {
                    'performance': {'throughput': 150.0}
                }
            }
        ]
        
        # Run parameter sweep
        results = parameter_sweep(
            "test_model.onnx",
            "test_blueprint.yaml",
            {'pe_count': [4, 8]},
            max_workers=1
        )
        
        # Find best result
        best = find_best(results, metric='throughput', maximize=True)
        
        # Generate statistics
        stats = aggregate_stats(results)
        
        # Verify complete workflow
        assert len(results) == 2
        assert best['metrics']['performance']['throughput'] == 150.0
        assert stats['success_rate'] == 1.0
        assert stats['aggregated_metrics']['throughput']['max'] == 150.0
    
    @patch('brainsmith.automation.batch.forge')  
    def test_batch_to_aggregate_workflow(self, mock_forge):
        """Test workflow from batch processing to statistics."""
        # Mock forge results
        mock_forge.side_effect = [
            {
                'metrics': {
                    'performance': {'power': 5.0}
                }
            },
            {
                'metrics': {
                    'performance': {'power': 7.0}
                }
            }
        ]
        
        # Run batch processing
        results = batch_process([
            ("model1.onnx", "blueprint1.yaml"),
            ("model2.onnx", "blueprint2.yaml")
        ], max_workers=1)
        
        # Aggregate statistics
        stats = aggregate_stats(results)
        
        # Verify workflow
        assert len(results) == 2
        assert stats['aggregated_metrics']['power']['mean'] == 6.0


if __name__ == '__main__':
    pytest.main([__file__])