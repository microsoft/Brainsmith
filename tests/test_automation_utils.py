"""
Test suite for Automation Utils

Tests the utility functions that work independently of forge() integration.
"""

import unittest
from unittest.mock import Mock, patch

# Import components to test
try:
    from brainsmith.automation.utils import (
        aggregate_results,
        find_best_result,
        find_top_results,
        generate_parameter_combinations,
        compare_automation_runs
    )
except ImportError as e:
    print(f"Warning: Could not import automation utils: {e}")
    aggregate_results = None


class TestAutomationUtils(unittest.TestCase):
    """Test cases for automation utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        if aggregate_results is None:
            self.skipTest("Automation utils not available")
        
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
    
    def test_generate_parameter_combinations(self):
        """Test parameter combination generation."""
        combinations = generate_parameter_combinations({
            'pe_count': [4, 8],
            'simd_width': [2, 4]
        })
        
        # Verify combinations
        self.assertEqual(len(combinations), 4)  # 2 x 2
        
        expected_combinations = [
            {'pe_count': 4, 'simd_width': 2},
            {'pe_count': 4, 'simd_width': 4},
            {'pe_count': 8, 'simd_width': 2},
            {'pe_count': 8, 'simd_width': 4}
        ]
        
        for expected in expected_combinations:
            self.assertIn(expected, combinations)
    
    def test_generate_parameter_combinations_empty(self):
        """Test parameter combinations with empty input."""
        combinations = generate_parameter_combinations({})
        self.assertEqual(combinations, [{}])
    
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
        self.assertEqual(metrics['throughput']['count'], 3)
    
    def test_find_best_result_maximize(self):
        """Test finding best result (maximize)."""
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
        self.assertTrue(best_result['optimization_info']['maximize'])
    
    def test_find_best_result_minimize(self):
        """Test finding best result (minimize)."""
        best_result = find_best_result(
            self.mock_results,
            metric='power',
            maximize=False
        )
        
        # Verify best result (lowest power)
        self.assertIsNotNone(best_result)
        self.assertEqual(best_result['metrics']['performance']['power'], 10.0)
        self.assertFalse(best_result['optimization_info']['maximize'])
    
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
        
        # Check values
        self.assertEqual(throughput1, 200.0)
        self.assertEqual(throughput2, 150.0)
    
    def test_compare_automation_runs(self):
        """Test comparing two automation runs."""
        # Create second set of results with higher throughput
        results2 = [
            {
                'success': True,
                'metrics': {'performance': {'throughput': 180.0}},
            },
            {
                'success': True,
                'metrics': {'performance': {'throughput': 220.0}},
            },
            {
                'success': True,
                'metrics': {'performance': {'throughput': 260.0}},
            }
        ]
        
        comparison = compare_automation_runs(
            self.mock_results,
            results2,
            metric='throughput'
        )
        
        # Verify comparison
        self.assertEqual(comparison['metric'], 'throughput')
        self.assertEqual(comparison['run1']['count'], 3)
        self.assertEqual(comparison['run2']['count'], 3)
        self.assertGreater(comparison['run2']['mean'], comparison['run1']['mean'])
        self.assertEqual(comparison['better_run'], 'run2')
        self.assertGreater(comparison['improvement_percent'], 0)
    
    def test_empty_results(self):
        """Test analysis with empty results."""
        aggregated = aggregate_results([])
        self.assertIn('error', aggregated)
        
        best_result = find_best_result([])
        self.assertIsNone(best_result)
        
        top_results = find_top_results([])
        self.assertEqual(len(top_results), 0)
    
    def test_no_successful_results(self):
        """Test analysis with no successful results."""
        failed_results = [
            {'success': False, 'error': 'Failed'},
            {'error': 'Also failed'}
        ]
        
        aggregated = aggregate_results(failed_results)
        self.assertEqual(aggregated['successful_runs'], 0)
        self.assertEqual(aggregated['success_rate'], 0.0)
        self.assertIn('error', aggregated)
        
        best_result = find_best_result(failed_results)
        self.assertIsNone(best_result)


if __name__ == '__main__':
    unittest.main(verbosity=2)