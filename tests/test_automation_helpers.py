"""
Test suite for Automation Helpers

Tests the simplified automation module that provides parameter sweep, 
batch processing, and result aggregation utilities.
"""

import unittest
from unittest.mock import Mock, patch
from typing import Dict, List, Any

# Import components to test
try:
    from brainsmith.automation import (
        parameter_sweep,
        grid_search,
        random_search,
        batch_process,
        multi_objective_runs,
        configuration_sweep,
        aggregate_results,
        find_best_result,
        find_top_results,
        save_automation_results
    )
except ImportError as e:
    print(f"Warning: Could not import automation helpers: {e}")
    parameter_sweep = None


class TestParameterSweep(unittest.TestCase):
    """Test cases for parameter sweep functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if parameter_sweep is None:
            self.skipTest("Automation helpers not available")
    
    @patch('brainsmith.automation.parameter_sweep.forge')
    def test_parameter_sweep_basic(self, mock_forge):
        """Test basic parameter sweep functionality."""
        # Mock forge function
        mock_forge.return_value = {
            'metrics': {'performance': {'throughput': 100.0, 'latency': 10.0}},
            'dataflow_graph': {'onnx_model': 'mock_model'}
        }
        
        # Run parameter sweep
        results = parameter_sweep(
            "test_model.onnx",
            "test_blueprint.yaml",
            {'pe_count': [4, 8], 'simd_width': [2, 4]}
        )
        
        # Verify results
        self.assertEqual(len(results), 4)  # 2 x 2 combinations
        self.assertEqual(mock_forge.call_count, 4)
        
        # Check result structure
        for result in results:
            self.assertIn('sweep_parameters', result)
            self.assertIn('sweep_index', result)
            self.assertIn('success', result)
            self.assertTrue(result['success'])
    
    @patch('brainsmith.automation.parameter_sweep.forge')
    def test_parameter_sweep_with_failures(self, mock_forge):
        """Test parameter sweep with some failures."""
        # Mock forge to fail on second call
        def side_effect(*args, **kwargs):
            if mock_forge.call_count == 2:
                raise Exception("Mock failure")
            return {
                'metrics': {'performance': {'throughput': 100.0}},
                'dataflow_graph': {'onnx_model': 'mock_model'}
            }
        
        mock_forge.side_effect = side_effect
        
        # Run parameter sweep
        results = parameter_sweep(
            "test_model.onnx",
            "test_blueprint.yaml",
            {'pe_count': [4, 8]}
        )
        
        # Verify results
        self.assertEqual(len(results), 2)
        self.assertEqual(sum(1 for r in results if r.get('success', False)), 1)
        self.assertEqual(sum(1 for r in results if not r.get('success', True)), 1)
    
    @patch('brainsmith.automation.parameter_sweep.forge')
    def test_grid_search(self, mock_forge):
        """Test grid search optimization."""
        # Mock forge with different throughput values
        def side_effect(*args, **kwargs):
            # Return different throughput based on call count
            throughput = 100.0 + mock_forge.call_count * 50.0
            return {
                'metrics': {'performance': {'throughput': throughput}},
                'dataflow_graph': {'onnx_model': 'mock_model'}
            }
        
        mock_forge.side_effect = side_effect
        
        # Run grid search
        best_result = grid_search(
            "test_model.onnx",
            "test_blueprint.yaml",
            {'pe_count': [4, 8, 16]},
            metric='throughput',
            maximize=True
        )
        
        # Verify best result
        self.assertIn('grid_search', best_result)
        self.assertEqual(best_result['grid_search']['optimization_metric'], 'throughput')
        self.assertTrue(best_result['grid_search']['maximize'])
        self.assertEqual(best_result['grid_search']['total_combinations'], 3)
    
    @patch('brainsmith.automation.parameter_sweep.forge')
    def test_random_search(self, mock_forge):
        """Test random search functionality."""
        mock_forge.return_value = {
            'metrics': {'performance': {'throughput': 150.0}},
            'dataflow_graph': {'onnx_model': 'mock_model'}
        }
        
        # Run random search
        best_result = random_search(
            "test_model.onnx",
            "test_blueprint.yaml",
            {
                'pe_count': [4, 8, 16, 32],
                'frequency': (100, 200)  # Range
            },
            n_iterations=5,
            random_seed=42
        )
        
        # Verify results
        self.assertIn('random_search', best_result)
        self.assertEqual(best_result['random_search']['total_iterations'], 5)
        self.assertEqual(mock_forge.call_count, 5)


class TestBatchProcessing(unittest.TestCase):
    """Test cases for batch processing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if batch_process is None:
            self.skipTest("Automation helpers not available")
    
    @patch('brainsmith.automation.batch_processing.forge')
    def test_batch_process_basic(self, mock_forge):
        """Test basic batch processing."""
        mock_forge.return_value = {
            'metrics': {'performance': {'throughput': 120.0}},
            'dataflow_graph': {'onnx_model': 'mock_model'}
        }
        
        # Run batch processing
        results = batch_process([
            ("model1.onnx", "blueprint1.yaml"),
            ("model2.onnx", "blueprint2.yaml"),
            ("model3.onnx", "blueprint3.yaml")
        ])
        
        # Verify results
        self.assertEqual(len(results), 3)
        self.assertEqual(mock_forge.call_count, 3)
        
        # Check result structure
        for i, result in enumerate(results):
            self.assertIn('batch_info', result)
            self.assertEqual(result['batch_info']['batch_index'], i)
            self.assertTrue(result['batch_info']['success'])
    
    @patch('brainsmith.automation.batch_processing.forge')
    def test_multi_objective_runs(self, mock_forge):
        """Test multi-objective runs."""
        mock_forge.return_value = {
            'metrics': {'performance': {'throughput': 100.0, 'power': 15.0}},
            'dataflow_graph': {'onnx_model': 'mock_model'}
        }
        
        # Run multi-objective runs
        results = multi_objective_runs(
            "test_model.onnx",
            "test_blueprint.yaml",
            [
                {'throughput': {'direction': 'maximize'}},
                {'power': {'direction': 'minimize'}},
                {'latency': {'direction': 'minimize'}}
            ]
        )
        
        # Verify results
        self.assertEqual(len(results), 3)
        self.assertEqual(mock_forge.call_count, 3)
        
        # Check objective info
        for i, result in enumerate(results):
            self.assertIn('multi_objective_info', result)
            self.assertEqual(result['multi_objective_info']['run_index'], i)
    
    @patch('brainsmith.automation.batch_processing.forge')
    def test_configuration_sweep(self, mock_forge):
        """Test configuration sweep."""
        mock_forge.return_value = {
            'metrics': {'performance': {'throughput': 130.0}},
            'dataflow_graph': {'onnx_model': 'mock_model'}
        }
        
        # Run configuration sweep
        results = configuration_sweep(
            "test_model.onnx",
            ["config1.yaml", "config2.yaml", "config3.yaml"]
        )
        
        # Verify results
        self.assertEqual(len(results), 3)
        self.assertEqual(mock_forge.call_count, 3)
        
        # Check config info
        for i, result in enumerate(results):
            self.assertIn('config_sweep_info', result)
            self.assertEqual(result['config_sweep_info']['config_index'], i)


class TestResultAnalysis(unittest.TestCase):
    """Test cases for result analysis utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        if aggregate_results is None:
            self.skipTest("Automation helpers not available")
        
        # Create mock results
        self.mock_results = [
            {
                'success': True,
                'metrics': {'performance': {'throughput': 100.0, 'power': 10.0}},
                'sweep_parameters': {'pe_count': 4}
            },
            {
                'success': True,
                'metrics': {'performance': {'throughput': 150.0, 'power': 15.0}},
                'sweep_parameters': {'pe_count': 8}
            },
            {
                'success': True,
                'metrics': {'performance': {'throughput': 200.0, 'power': 20.0}},
                'sweep_parameters': {'pe_count': 16}
            },
            {
                'success': False,
                'error': 'Mock failure',
                'sweep_parameters': {'pe_count': 32}
            }
        ]
    
    def test_aggregate_results(self):
        """Test result aggregation."""
        aggregated = aggregate_results(self.mock_results)
        
        # Verify aggregation
        self.assertEqual(aggregated['total_runs'], 4)
        self.assertEqual(aggregated['successful_runs'], 3)
        self.assertEqual(aggregated['success_rate'], 0.75)
        
        # Check aggregated metrics
        self.assertIn('aggregated_metrics', aggregated)
        metrics = aggregated['aggregated_metrics']
        
        # Check throughput statistics
        self.assertIn('throughput', metrics)
        self.assertEqual(metrics['throughput']['mean'], 150.0)  # (100+150+200)/3
        self.assertEqual(metrics['throughput']['min'], 100.0)
        self.assertEqual(metrics['throughput']['max'], 200.0)
    
    def test_find_best_result(self):
        """Test finding best result."""
        best_result = find_best_result(
            self.mock_results,
            metric='throughput',
            maximize=True
        )
        
        # Verify best result
        self.assertIsNotNone(best_result)
        self.assertEqual(best_result['metrics']['performance']['throughput'], 200.0)
        self.assertIn('optimization_info', best_result)
        self.assertEqual(best_result['optimization_info']['optimized_metric'], 'throughput')
    
    def test_find_top_results(self):
        """Test finding top N results."""
        top_results = find_top_results(
            self.mock_results,
            n=2,
            metric='throughput'
        )
        
        # Verify top results
        self.assertEqual(len(top_results), 2)
        
        # Check ranking
        self.assertEqual(top_results[0]['ranking_info']['rank'], 1)
        self.assertEqual(top_results[1]['ranking_info']['rank'], 2)
        
        # Check ordering (descending by throughput)
        throughput1 = top_results[0]['metrics']['performance']['throughput']
        throughput2 = top_results[1]['metrics']['performance']['throughput']
        self.assertGreater(throughput1, throughput2)
    
    def test_empty_results(self):
        """Test analysis with empty results."""
        aggregated = aggregate_results([])
        self.assertIn('error', aggregated)
        
        best_result = find_best_result([])
        self.assertIsNone(best_result)
        
        top_results = find_top_results([])
        self.assertEqual(len(top_results), 0)
    
    @patch('builtins.open')
    @patch('json.dump')
    def test_save_automation_results(self, mock_json_dump, mock_open):
        """Test saving automation results."""
        # Mock file operations
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Save results
        save_automation_results(
            self.mock_results,
            "test_results.json",
            include_analysis=True
        )
        
        # Verify file operations
        mock_open.assert_called_once()
        mock_json_dump.assert_called_once()
        
        # Check saved data structure
        saved_data = mock_json_dump.call_args[0][0]
        self.assertIn('automation_results', saved_data)
        self.assertIn('summary', saved_data)
        self.assertIn('aggregated_analysis', saved_data)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration test scenarios for automation helpers."""
    
    def setUp(self):
        """Set up test fixtures."""
        if parameter_sweep is None:
            self.skipTest("Automation helpers not available")
    
    @patch('brainsmith.automation.parameter_sweep.forge')
    def test_complete_parameter_sweep_workflow(self, mock_forge):
        """Test complete parameter sweep workflow."""
        # Mock forge with varied results
        def side_effect(*args, **kwargs):
            call_count = mock_forge.call_count
            return {
                'metrics': {
                    'performance': {
                        'throughput': 100.0 + call_count * 25.0,
                        'power': 10.0 + call_count * 2.0
                    }
                },
                'dataflow_graph': {'onnx_model': f'mock_model_{call_count}'}
            }
        
        mock_forge.side_effect = side_effect
        
        # Run parameter sweep
        results = parameter_sweep(
            "test_model.onnx",
            "test_blueprint.yaml",
            {
                'pe_count': [4, 8, 16],
                'simd_width': [2, 4]
            }
        )
        
        # Analyze results
        aggregated = aggregate_results(results)
        best_result = find_best_result(results, metric='throughput', maximize=True)
        top_results = find_top_results(results, n=3, metric='throughput')
        
        # Verify complete workflow
        self.assertEqual(len(results), 6)  # 3 x 2 combinations
        self.assertEqual(aggregated['successful_runs'], 6)
        self.assertIsNotNone(best_result)
        self.assertEqual(len(top_results), 3)
    
    @patch('brainsmith.automation.batch_processing.forge')
    def test_batch_processing_with_analysis(self, mock_forge):
        """Test batch processing followed by analysis."""
        # Mock forge with different results
        mock_forge.side_effect = [
            {
                'metrics': {'performance': {'throughput': 120.0, 'latency': 8.0}},
                'dataflow_graph': {'onnx_model': 'model1'}
            },
            {
                'metrics': {'performance': {'throughput': 180.0, 'latency': 6.0}},
                'dataflow_graph': {'onnx_model': 'model2'}
            },
            {
                'metrics': {'performance': {'throughput': 160.0, 'latency': 7.0}},
                'dataflow_graph': {'onnx_model': 'model3'}
            }
        ]
        
        # Run batch processing
        results = batch_process([
            ("model1.onnx", "blueprint1.yaml"),
            ("model2.onnx", "blueprint2.yaml"),
            ("model3.onnx", "blueprint3.yaml")
        ])
        
        # Analyze results
        best_throughput = find_best_result(results, metric='throughput', maximize=True)
        best_latency = find_best_result(results, metric='latency', maximize=False)
        
        # Verify analysis
        self.assertEqual(best_throughput['metrics']['performance']['throughput'], 180.0)
        self.assertEqual(best_latency['metrics']['performance']['latency'], 6.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)