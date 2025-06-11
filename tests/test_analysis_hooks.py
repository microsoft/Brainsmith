"""
Test suite for Analysis Hooks Framework

Tests the hooks-based analysis module that exposes data for external tools
instead of providing custom analysis implementations.
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch

# Import components to test
try:
    from brainsmith.data import (
        collect_dse_metrics,
        export_for_analysis,
        register_analyzer,
        get_registered_analyzers,
        get_raw_data,
        export_to_dataframe,
        pandas_adapter,
        scipy_adapter,
        sklearn_adapter
    )
except ImportError as e:
    print(f"Warning: Could not import analysis hooks: {e}")
    expose_analysis_data = None


class TestAnalysisHooks(unittest.TestCase):
    """Test cases for analysis hooks functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if expose_analysis_data is None:
            self.skipTest("Analysis hooks not available")
        
        # Create mock DSE results
        self.mock_dse_results = [
            Mock(
                design_parameters={'pe': 4, 'simd': 2},
                objective_values=[100.0, 50.0],
                constraint_violations=[],
                metadata={'iteration': 0}
            ),
            Mock(
                design_parameters={'pe': 8, 'simd': 4},
                objective_values=[200.0, 25.0],
                constraint_violations=[],
                metadata={'iteration': 1}
            ),
            Mock(
                design_parameters={'pe': 16, 'simd': 8},
                objective_values=[400.0, 12.5],
                constraint_violations=[],
                metadata={'iteration': 2}
            )
        ]
    
    def test_expose_analysis_data(self):
        """Test data exposure functionality."""
        analysis_data = expose_analysis_data(self.mock_dse_results)
        
        # Verify structure
        self.assertIsInstance(analysis_data, dict)
        self.assertIn('solutions', analysis_data)
        self.assertIn('metrics', analysis_data)
        self.assertIn('pareto_frontier', analysis_data)
        self.assertIn('metadata', analysis_data)
        
        # Verify solutions data
        solutions = analysis_data['solutions']
        self.assertEqual(len(solutions), 3)
        
        for i, solution in enumerate(solutions):
            self.assertEqual(solution['id'], i)
            self.assertIn('parameters', solution)
            self.assertIn('objectives', solution)
            self.assertIn('metadata', solution)
        
        # Verify metrics data
        metrics = analysis_data['metrics']
        self.assertIn('objective_0', metrics)
        self.assertIn('objective_1', metrics)
        
        # Check metric arrays
        self.assertTrue(isinstance(metrics['objective_0'], np.ndarray))
        self.assertEqual(len(metrics['objective_0']), 3)
        np.testing.assert_array_equal(metrics['objective_0'], [100.0, 200.0, 400.0])
        np.testing.assert_array_equal(metrics['objective_1'], [50.0, 25.0, 12.5])
    
    def test_expose_analysis_data_empty(self):
        """Test data exposure with empty results."""
        analysis_data = expose_analysis_data([])
        
        self.assertEqual(analysis_data['solutions'], [])
        self.assertEqual(analysis_data['metrics'], {})
        self.assertEqual(analysis_data['pareto_frontier'], [])
    
    def test_register_analyzer(self):
        """Test external analyzer registration."""
        def custom_analyzer(data):
            return {'custom_metric': np.mean(data)}
        
        # Register analyzer
        register_analyzer('custom', custom_analyzer)
        
        # Verify registration
        analyzers = get_registered_analyzers()
        self.assertIn('custom', analyzers)
        self.assertEqual(analyzers['custom'], custom_analyzer)
    
    def test_get_raw_data(self):
        """Test raw data extraction."""
        raw_data = get_raw_data(self.mock_dse_results)
        
        self.assertIsInstance(raw_data, dict)
        self.assertIn('objective_0', raw_data)
        self.assertIn('objective_1', raw_data)
        
        # Verify arrays
        np.testing.assert_array_equal(raw_data['objective_0'], [100.0, 200.0, 400.0])
        np.testing.assert_array_equal(raw_data['objective_1'], [50.0, 25.0, 12.5])
    
    @patch('pandas.DataFrame')
    def test_export_to_dataframe(self, mock_dataframe):
        """Test pandas DataFrame export."""
        mock_df = Mock()
        mock_dataframe.return_value = mock_df
        
        # Mock pandas import
        with patch('builtins.__import__') as mock_import:
            mock_pandas = Mock()
            mock_pandas.DataFrame = mock_dataframe
            
            def import_side_effect(name, *args, **kwargs):
                if name == 'pandas':
                    return mock_pandas
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = import_side_effect
            
            result = export_to_dataframe(self.mock_dse_results)
            
            # Should have called pandas DataFrame
            mock_dataframe.assert_called_once()
            self.assertEqual(result, mock_df)
    
    def test_export_to_dataframe_no_pandas(self):
        """Test DataFrame export without pandas."""
        with patch('builtins.__import__', side_effect=ImportError("No pandas")):
            result = export_to_dataframe(self.mock_dse_results)
            self.assertIsNone(result)


class TestExternalToolAdapters(unittest.TestCase):
    """Test cases for external tool adapters."""
    
    def setUp(self):
        """Set up test fixtures."""
        if expose_analysis_data is None:
            self.skipTest("Analysis hooks not available")
        
        # Create analysis data
        mock_results = [
            Mock(
                design_parameters={'pe': 4, 'simd': 2},
                objective_values=[100.0, 50.0],
                constraint_violations=[],
                metadata={'iteration': 0}
            ),
            Mock(
                design_parameters={'pe': 8, 'simd': 4}, 
                objective_values=[200.0, 25.0],
                constraint_violations=[],
                metadata={'iteration': 1}
            )
        ]
        
        self.analysis_data = expose_analysis_data(mock_results)
    
    @patch('pandas.DataFrame')
    def test_pandas_adapter(self, mock_dataframe):
        """Test pandas adapter."""
        mock_df = Mock()
        mock_dataframe.return_value = mock_df
        
        with patch('builtins.__import__') as mock_import:
            mock_pandas = Mock()
            mock_pandas.DataFrame = mock_dataframe
            
            def import_side_effect(name, *args, **kwargs):
                if name == 'pandas':
                    return mock_pandas
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = import_side_effect
            
            result = pandas_adapter(self.analysis_data)
            
            # Should have called pandas DataFrame with flattened data
            mock_dataframe.assert_called_once()
            call_args = mock_dataframe.call_args[0][0]  # First argument (rows)
            
            # Verify structure of passed data
            self.assertIsInstance(call_args, list)
            self.assertEqual(len(call_args), 2)  # Two solutions
            
            # Check first row
            first_row = call_args[0]
            self.assertEqual(first_row['solution_id'], 0)
            self.assertEqual(first_row['param_pe'], 4)
            self.assertEqual(first_row['param_simd'], 2)
            self.assertEqual(first_row['objective_0'], 100.0)
            self.assertEqual(first_row['objective_1'], 50.0)
    
    def test_scipy_adapter(self):
        """Test scipy adapter."""
        result = scipy_adapter(self.analysis_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('arrays', result)
        self.assertIn('sample_size', result)
        self.assertIn('metric_names', result)
        self.assertIn('metadata', result)
        
        # Verify arrays
        arrays = result['arrays']
        self.assertIn('objective_0', arrays)
        self.assertIn('objective_1', arrays)
        
        # Verify sample size
        self.assertEqual(result['sample_size'], 2)
        
        # Verify metric names
        self.assertEqual(set(result['metric_names']), {'objective_0', 'objective_1'})
    
    @patch('numpy.array')
    def test_sklearn_adapter(self, mock_array):
        """Test scikit-learn adapter."""
        # Mock numpy arrays
        mock_features = Mock()
        mock_targets = Mock()
        mock_array.side_effect = [mock_features, mock_targets]
        
        result = sklearn_adapter(self.analysis_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('X', result)
        self.assertIn('y', result)
        self.assertIn('feature_names', result)
        self.assertIn('target_names', result)
        
        # Verify feature and target names
        self.assertEqual(set(result['feature_names']), {'pe', 'simd'})
        self.assertEqual(result['target_names'], ['objective_0', 'objective_1'])
        
        # Verify numpy.array was called twice (features and targets)
        self.assertEqual(mock_array.call_count, 2)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration test scenarios for analysis hooks."""
    
    def setUp(self):
        """Set up test fixtures."""
        if expose_analysis_data is None:
            self.skipTest("Analysis hooks not available")
    
    def test_pandas_workflow(self):
        """Test complete pandas analysis workflow."""
        # Create DSE results
        dse_results = [
            Mock(
                design_parameters={'pe': 4, 'simd': 2, 'freq': 100},
                objective_values=[150.0, 45.0, 0.75],
                constraint_violations=[],
                metadata={'run': 1}
            ),
            Mock(
                design_parameters={'pe': 8, 'simd': 4, 'freq': 125},
                objective_values=[250.0, 30.0, 0.85],
                constraint_violations=[],
                metadata={'run': 2}
            )
        ]
        
        # Expose data
        analysis_data = expose_analysis_data(dse_results)
        
        # Verify data can be converted to pandas format
        with patch('pandas.DataFrame') as mock_df:
            result = pandas_adapter(analysis_data)
            if result is not None:  # If pandas available
                mock_df.assert_called_once()
    
    def test_scipy_workflow(self):
        """Test complete scipy analysis workflow."""
        # Create DSE results
        dse_results = [Mock(objective_values=[np.random.random(), np.random.random()]) for _ in range(10)]
        
        # Expose data
        analysis_data = expose_analysis_data(dse_results)
        
        # Convert to scipy format
        scipy_data = scipy_adapter(analysis_data)
        
        # Verify scipy-compatible format
        self.assertIn('arrays', scipy_data)
        self.assertEqual(scipy_data['sample_size'], 10)
        
        # Arrays should be suitable for scipy functions
        arrays = scipy_data['arrays']
        for metric_name, values in arrays.items():
            self.assertIsInstance(values, np.ndarray)
            self.assertEqual(len(values), 10)


if __name__ == '__main__':
    unittest.main(verbosity=2)