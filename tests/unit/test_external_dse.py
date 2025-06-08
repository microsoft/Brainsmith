"""
Unit tests for ExternalDSEAdapter implementation.

Tests external framework integration and fallback mechanisms.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add brainsmith to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from brainsmith.dse.external import ExternalDSEAdapter, check_framework_availability
from brainsmith.dse.interface import DSEConfiguration, DSEObjective, OptimizationObjective
from brainsmith.core.design_space import DesignSpace, DesignPoint, ParameterDefinition, ParameterType
from brainsmith.core.result import BrainsmithResult


class TestExternalDSEAdapter(unittest.TestCase):
    """Test ExternalDSEAdapter functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = DSEConfiguration(max_evaluations=20)
        
        # Create test design space
        self.design_space = DesignSpace("test_space")
        
        int_param = ParameterDefinition("int_param", ParameterType.INTEGER, range_values=[1, 10])
        float_param = ParameterDefinition("float_param", ParameterType.CONTINUOUS, range_values=[0.1, 1.0])
        cat_param = ParameterDefinition("cat_param", ParameterType.CATEGORICAL, values=["A", "B", "C"])
        
        self.design_space.add_parameter(int_param)
        self.design_space.add_parameter(float_param)
        self.design_space.add_parameter(cat_param)
    
    def create_mock_result(self, throughput=100.0):
        """Create a mock BrainsmithResult."""
        result = Mock(spec=BrainsmithResult)
        result.metrics = Mock()
        result.metrics.performance = Mock()
        result.metrics.performance.throughput_ops_sec = throughput
        return result
    
    def test_framework_availability_detection(self):
        """Test framework availability detection."""
        availability = check_framework_availability()
        self.assertIsInstance(availability, dict)
        
        expected_frameworks = ["scikit-optimize", "optuna", "deap", "hyperopt"]
        for framework in expected_frameworks:
            self.assertIn(framework, availability)
            self.assertIsInstance(availability[framework], bool)
    
    def test_adapter_creation_unavailable_framework(self):
        """Test adapter creation when framework is unavailable."""
        # Test with unavailable framework
        adapter = ExternalDSEAdapter("nonexistent_framework", self.config)
        adapter.initialize(self.design_space)
        
        self.assertFalse(adapter.framework_available)
        self.assertTrue(hasattr(adapter, 'fallback_engine'))
    
    @patch('brainsmith.dse.external.ExternalDSEAdapter._check_framework_availability')
    def test_scikit_optimize_setup(self, mock_check):
        """Test scikit-optimize adapter setup."""
        mock_check.return_value = True
        
        with patch('builtins.__import__') as mock_import:
            # Mock skopt imports
            mock_skopt = MagicMock()
            mock_skopt.space.Real = MagicMock()
            mock_skopt.space.Integer = MagicMock()
            mock_skopt.space.Categorical = MagicMock()
            mock_skopt.gp_minimize = MagicMock()
            
            def import_side_effect(name, *args, **kwargs):
                if name == 'skopt':
                    return mock_skopt
                elif name == 'skopt.space':
                    return mock_skopt.space
                else:
                    return MagicMock()
            
            mock_import.side_effect = import_side_effect
            
            adapter = ExternalDSEAdapter("bayesian", self.config)
            adapter.initialize(self.design_space)
            
            self.assertTrue(adapter.framework_available)
            self.assertIsNotNone(adapter.external_optimizer)
    
    @patch('brainsmith.dse.external.ExternalDSEAdapter._check_framework_availability')
    def test_optuna_setup(self, mock_check):
        """Test Optuna adapter setup."""
        mock_check.return_value = True
        
        with patch('builtins.__import__') as mock_import:
            # Mock optuna imports
            mock_optuna = MagicMock()
            mock_study = MagicMock()
            mock_optuna.create_study.return_value = mock_study
            
            def import_side_effect(name, *args, **kwargs):
                if name == 'optuna':
                    return mock_optuna
                else:
                    return MagicMock()
            
            mock_import.side_effect = import_side_effect
            
            adapter = ExternalDSEAdapter("optuna", self.config)
            adapter.initialize(self.design_space)
            
            self.assertTrue(adapter.framework_available)
            self.assertEqual(adapter.external_optimizer, mock_study)
    
    def test_fallback_mechanism(self):
        """Test fallback to SimpleDSE when external framework unavailable."""
        adapter = ExternalDSEAdapter("unavailable_framework", self.config)
        adapter.initialize(self.design_space)
        
        # Should fallback to SimpleDSE
        self.assertFalse(adapter.framework_available)
        self.assertTrue(hasattr(adapter, 'fallback_engine'))
        
        # Should still be able to suggest points
        points = adapter.suggest_next_points(3)
        self.assertEqual(len(points), 3)
    
    def test_parameter_mapping(self):
        """Test parameter mapping creation."""
        adapter = ExternalDSEAdapter("bayesian", self.config)
        adapter.initialize(self.design_space)
        
        # Check parameter mapping
        self.assertEqual(len(adapter.param_mapping), 3)
        self.assertEqual(len(adapter.reverse_mapping), 3)
        
        param_names = list(self.design_space.parameters.keys())
        for i, param_name in enumerate(param_names):
            self.assertEqual(adapter.param_mapping[param_name], i)
            self.assertEqual(adapter.reverse_mapping[i], param_name)
    
    def test_objective_value_extraction(self):
        """Test objective value extraction for external optimizers."""
        adapter = ExternalDSEAdapter("bayesian", self.config)
        adapter.initialize(self.design_space)
        
        # Test single objective
        result = self.create_mock_result(throughput=150.0)
        objective_value = adapter._get_objective_value(result)
        self.assertEqual(objective_value, 150.0)
        
        # Test multi-objective with weights
        objectives = [
            DSEObjective("metric1", OptimizationObjective.MAXIMIZE, weight=1.0),
            DSEObjective("metric2", OptimizationObjective.MINIMIZE, weight=0.5)
        ]
        multi_config = DSEConfiguration(objectives=objectives)
        adapter = ExternalDSEAdapter("bayesian", multi_config)
        adapter.initialize(self.design_space)
        
        result = Mock(spec=BrainsmithResult)
        result.metrics = Mock()
        result.metrics.metric1 = 100.0
        result.metrics.metric2 = 50.0
        
        # Should compute weighted sum: 1.0 * 100.0 + 0.5 * (-50.0) = 75.0
        objective_value = adapter._get_objective_value(result)
        self.assertEqual(objective_value, 75.0)
    
    def test_framework_info(self):
        """Test framework information retrieval."""
        adapter = ExternalDSEAdapter("bayesian", self.config)
        adapter.initialize(self.design_space)
        
        info = adapter.get_framework_info()
        self.assertIsInstance(info, dict)
        self.assertEqual(info["framework"], "bayesian")
        self.assertIn("available", info)
        self.assertIn("fallback_active", info)
        self.assertIn("evaluations_completed", info)
    
    @patch('brainsmith.dse.external.ExternalDSEAdapter._check_framework_availability')
    def test_mock_skopt_suggestions(self, mock_check):
        """Test point suggestions with mocked scikit-optimize."""
        mock_check.return_value = True
        
        with patch('builtins.__import__') as mock_import:
            # Mock skopt completely
            mock_skopt = MagicMock()
            
            # Mock space classes
            mock_skopt.space.Real = MagicMock()
            mock_skopt.space.Integer = MagicMock()
            mock_skopt.space.Categorical = MagicMock()
            
            # Mock sampler
            mock_sampler = MagicMock()
            mock_sampler.generate.return_value = [[0.5, 5, 0]]  # Mock sample
            mock_skopt.sampler.Lhs = MagicMock(return_value=mock_sampler)
            
            def import_side_effect(name, *args, **kwargs):
                if 'skopt' in name:
                    return mock_skopt
                else:
                    return MagicMock()
            
            mock_import.side_effect = import_side_effect
            
            adapter = ExternalDSEAdapter("bayesian", self.config)
            adapter.initialize(self.design_space)
            
            # Mock the external optimizer setup
            adapter.external_optimizer = {
                "space": [mock_skopt.space.Real(), mock_skopt.space.Integer(), mock_skopt.space.Categorical()],
                "n_initial_points": 5
            }
            
            # Should be able to suggest points (even if mocked)
            points = adapter.suggest_next_points(2)
            self.assertIsInstance(points, list)
    
    def test_result_update_with_fallback(self):
        """Test result updating with fallback engine."""
        adapter = ExternalDSEAdapter("unavailable", self.config)
        adapter.initialize(self.design_space)
        
        # Should use fallback engine
        point = DesignPoint()
        point.set_parameter("int_param", 5)
        point.set_parameter("float_param", 0.5)
        point.set_parameter("cat_param", "A")
        
        result = self.create_mock_result()
        adapter.update_with_result(point, result)
        
        # Should update evaluation history
        self.assertEqual(len(adapter.evaluation_history), 1)
        
        # Should also update fallback engine
        if hasattr(adapter, 'fallback_engine'):
            self.assertEqual(len(adapter.fallback_engine.evaluation_history), 1)


class TestFrameworkAvailabilityChecking(unittest.TestCase):
    """Test framework availability checking functionality."""
    
    def test_check_all_frameworks(self):
        """Test checking availability of all supported frameworks."""
        availability = check_framework_availability()
        
        expected_frameworks = ["scikit-optimize", "optuna", "deap", "hyperopt"]
        for framework in expected_frameworks:
            self.assertIn(framework, availability)
            self.assertIsInstance(availability[framework], bool)
    
    @patch('builtins.__import__')
    def test_framework_import_success(self, mock_import):
        """Test successful framework import detection."""
        # Mock successful imports
        mock_import.return_value = MagicMock()
        
        availability = check_framework_availability()
        
        # All should be available with successful imports
        for framework, available in availability.items():
            self.assertTrue(available)
    
    @patch('builtins.__import__')
    def test_framework_import_failure(self, mock_import):
        """Test framework import failure detection."""
        # Mock import failures
        mock_import.side_effect = ImportError("Module not found")
        
        availability = check_framework_availability()
        
        # All should be unavailable with import failures
        for framework, available in availability.items():
            self.assertFalse(available)


class TestExternalFrameworkMocking(unittest.TestCase):
    """Test external framework integration with comprehensive mocking."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = DSEConfiguration(max_evaluations=10)
        self.design_space = DesignSpace("mock_test")
        
        param = ParameterDefinition("x", ParameterType.CONTINUOUS, range_values=[0.0, 1.0])
        self.design_space.add_parameter(param)
    
    @patch('brainsmith.dse.external.ExternalDSEAdapter._check_framework_availability')
    def test_mock_bayesian_optimization_workflow(self, mock_check):
        """Test complete Bayesian optimization workflow with mocking."""
        mock_check.return_value = True
        
        with patch('builtins.__import__') as mock_import:
            # Create comprehensive skopt mock
            mock_skopt = MagicMock()
            mock_optimizer = MagicMock()
            
            # Mock space
            mock_skopt.space.Real.return_value = "real_space"
            
            # Mock optimizer
            mock_optimizer.tell = MagicMock()
            mock_optimizer.ask.return_value = [0.5]
            mock_skopt.Optimizer.return_value = mock_optimizer
            
            def import_side_effect(name, *args, **kwargs):
                if 'skopt' in name:
                    return mock_skopt
                return MagicMock()
            
            mock_import.side_effect = import_side_effect
            
            # Create adapter
            adapter = ExternalDSEAdapter("bayesian", self.config)
            adapter.initialize(self.design_space)
            
            # Mock successful setup
            adapter.framework_available = True
            adapter.external_optimizer = {
                "space": ["real_space"],
                "n_initial_points": 2,
                "acquisition_function": "EI"
            }
            
            # Test initial suggestions (should work without evaluation history)
            points = adapter.suggest_next_points(2)
            self.assertIsInstance(points, list)
            
            # Test with some evaluation history
            point = DesignPoint()
            point.set_parameter("x", 0.3)
            result = Mock(spec=BrainsmithResult)
            result.metrics = Mock()
            result.metrics.performance = Mock()
            result.metrics.performance.throughput_ops_sec = 95.0
            
            adapter.update_with_result(point, result)
            
            # Should still be able to suggest points
            more_points = adapter.suggest_next_points(1)
            self.assertIsInstance(more_points, list)


if __name__ == '__main__':
    unittest.main()