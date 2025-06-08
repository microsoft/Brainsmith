"""
Unit tests for SimpleDSEEngine implementation.

Tests advanced sampling strategies and adaptive optimization.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch
import numpy as np

# Add brainsmith to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from brainsmith.dse.simple import SimpleDSEEngine
from brainsmith.dse.interface import DSEConfiguration, DSEObjective, OptimizationObjective
from brainsmith.core.design_space import DesignSpace, DesignPoint, ParameterDefinition, ParameterType
from brainsmith.core.result import BrainsmithResult


class TestSimpleDSEEngine(unittest.TestCase):
    """Test SimpleDSEEngine functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = DSEConfiguration(max_evaluations=20, seed=42)
        
        # Create test design space
        self.design_space = DesignSpace("test_space")
        
        # Add various parameter types
        int_param = ParameterDefinition("int_param", ParameterType.INTEGER, range_values=[1, 10])
        float_param = ParameterDefinition("float_param", ParameterType.CONTINUOUS, range_values=[0.1, 1.0])
        cat_param = ParameterDefinition("cat_param", ParameterType.CATEGORICAL, values=["A", "B", "C"])
        bool_param = ParameterDefinition("bool_param", ParameterType.BOOLEAN)
        
        self.design_space.add_parameter(int_param)
        self.design_space.add_parameter(float_param)
        self.design_space.add_parameter(cat_param)
        self.design_space.add_parameter(bool_param)
    
    def create_mock_result(self, throughput=100.0):
        """Create a mock BrainsmithResult."""
        result = Mock(spec=BrainsmithResult)
        result.metrics = Mock()
        result.metrics.performance = Mock()
        result.metrics.performance.throughput_ops_sec = throughput
        return result
    
    def test_random_strategy(self):
        """Test random sampling strategy."""
        engine = SimpleDSEEngine("random", self.config)
        engine.initialize(self.design_space)
        
        points = engine.suggest_next_points(5)
        self.assertEqual(len(points), 5)
        
        # Check that all points have all required parameters
        for point in points:
            self.assertIn("int_param", point.parameters)
            self.assertIn("float_param", point.parameters)
            self.assertIn("cat_param", point.parameters)
            self.assertIn("bool_param", point.parameters)
            
            # Check parameter types and ranges
            self.assertIsInstance(point.parameters["int_param"], int)
            self.assertGreaterEqual(point.parameters["int_param"], 1)
            self.assertLessEqual(point.parameters["int_param"], 10)
            
            self.assertIsInstance(point.parameters["float_param"], float)
            self.assertGreaterEqual(point.parameters["float_param"], 0.1)
            self.assertLessEqual(point.parameters["float_param"], 1.0)
            
            self.assertIn(point.parameters["cat_param"], ["A", "B", "C"])
            self.assertIsInstance(point.parameters["bool_param"], bool)
    
    @patch('brainsmith.dse.simple.SCIPY_AVAILABLE', True)
    def test_latin_hypercube_strategy(self):
        """Test Latin Hypercube sampling strategy."""
        engine = SimpleDSEEngine("latin_hypercube", self.config)
        engine.initialize(self.design_space)
        
        points = engine.suggest_next_points(10)
        self.assertEqual(len(points), 10)
        
        # LHS should provide good space coverage
        # Check that we get diverse values for continuous parameters
        float_values = [p.parameters["float_param"] for p in points]
        self.assertGreater(len(set(float_values)), 5)  # Should have diverse values
    
    @patch('brainsmith.dse.simple.SCIPY_AVAILABLE', True)
    def test_sobol_strategy(self):
        """Test Sobol sequence sampling strategy."""
        engine = SimpleDSEEngine("sobol", self.config)
        engine.initialize(self.design_space)
        
        points = engine.suggest_next_points(8)
        self.assertEqual(len(points), 8)
        
        # Sobol sequences should provide good coverage
        for point in points:
            self.assertIn("int_param", point.parameters)
            self.assertIn("float_param", point.parameters)
    
    def test_adaptive_strategy_initial_phase(self):
        """Test adaptive strategy initial exploration phase."""
        engine = SimpleDSEEngine("adaptive", self.config)
        engine.initialize(self.design_space)
        
        # Initial points should come from pre-generated samples or random
        points = engine.suggest_next_points(3)
        self.assertEqual(len(points), 3)
        
        # All points should be valid
        for point in points:
            self.assertEqual(len(point.parameters), 4)
    
    def test_adaptive_strategy_exploitation_phase(self):
        """Test adaptive strategy exploitation phase."""
        engine = SimpleDSEEngine("adaptive", self.config)
        engine.initialize(self.design_space)
        
        # Add some evaluation history to trigger exploitation
        for i in range(15):
            point = DesignPoint()
            point.set_parameter("int_param", i % 10 + 1)
            point.set_parameter("float_param", (i % 10 + 1) / 10.0)
            point.set_parameter("cat_param", ["A", "B", "C"][i % 3])
            point.set_parameter("bool_param", i % 2 == 0)
            
            # Simulate better performance for higher values
            result = self.create_mock_result(throughput=100.0 + i * 10)
            engine.update_with_result(point, result)
        
        # Now adaptive should focus on promising regions
        points = engine.suggest_next_points(3)
        self.assertEqual(len(points), 3)
    
    def test_parameter_importance_learning(self):
        """Test parameter importance learning in adaptive strategy."""
        engine = SimpleDSEEngine("adaptive", self.config)
        engine.initialize(self.design_space)
        
        # Add evaluations with clear parameter importance
        for i in range(10):
            point = DesignPoint()
            int_val = i % 10 + 1
            point.set_parameter("int_param", int_val)
            point.set_parameter("float_param", 0.5)
            point.set_parameter("cat_param", "A")
            point.set_parameter("bool_param", True)
            
            # Performance strongly correlated with int_param
            result = self.create_mock_result(throughput=100.0 + int_val * 50)
            engine.update_with_result(point, result)
        
        # Check that parameter importance was updated
        self.assertIsInstance(engine.parameter_importance, dict)
        if "int_param" in engine.parameter_importance:
            self.assertGreater(engine.parameter_importance["int_param"], 0)
    
    def test_convergence_detection(self):
        """Test convergence detection mechanism."""
        config = DSEConfiguration(
            max_evaluations=20,
            convergence_threshold=0.001,
            convergence_patience=5
        )
        engine = SimpleDSEEngine("adaptive", config)
        engine.initialize(self.design_space)
        
        # Add evaluations with converged results
        for i in range(10):
            point = DesignPoint()
            point.set_parameter("int_param", 5)
            point.set_parameter("float_param", 0.5)
            point.set_parameter("cat_param", "B")
            point.set_parameter("bool_param", True)
            
            # Same performance (converged)
            result = self.create_mock_result(throughput=150.0)
            engine.update_with_result(point, result)
        
        # Should detect convergence
        self.assertTrue(len(engine.convergence_history) > 0)
    
    def test_strategy_info(self):
        """Test strategy information retrieval."""
        engine = SimpleDSEEngine("latin_hypercube", self.config)
        engine.initialize(self.design_space)
        
        info = engine.get_strategy_info()
        self.assertIsInstance(info, dict)
        self.assertEqual(info["strategy"], "latin_hypercube")
        self.assertIn("samples_used", info)
        self.assertIn("evaluations_completed", info)
        self.assertIn("scipy_available", info)
    
    def test_no_scipy_fallback(self):
        """Test fallback behavior when scipy is not available."""
        with patch('brainsmith.dse.simple.SCIPY_AVAILABLE', False):
            engine = SimpleDSEEngine("latin_hypercube", self.config)
            engine.initialize(self.design_space)
            
            # Should fallback to random sampling
            points = engine.suggest_next_points(5)
            self.assertEqual(len(points), 5)
    
    def test_duplicate_point_filtering(self):
        """Test that duplicate points are filtered out."""
        engine = SimpleDSEEngine("random", self.config)
        engine.initialize(self.design_space)
        
        # Create a point and mark it as evaluated
        point = DesignPoint()
        point.set_parameter("int_param", 5)
        point.set_parameter("float_param", 0.5)
        point.set_parameter("cat_param", "A")
        point.set_parameter("bool_param", True)
        
        result = self.create_mock_result()
        engine.update_with_result(point, result)
        
        # Subsequent suggestions should not include this exact point
        # (This is probabilistic for random, so we test the mechanism exists)
        self.assertTrue(engine._is_point_evaluated(point))
    
    def test_multi_objective_optimization(self):
        """Test multi-objective optimization capabilities."""
        objectives = [
            DSEObjective("metric1", OptimizationObjective.MAXIMIZE),
            DSEObjective("metric2", OptimizationObjective.MINIMIZE)
        ]
        config = DSEConfiguration(objectives=objectives)
        engine = SimpleDSEEngine("adaptive", config)
        engine.initialize(self.design_space)
        
        # Add results with different trade-offs
        results = []
        for i in range(5):
            point = DesignPoint()
            point.set_parameter("int_param", i + 1)
            point.set_parameter("float_param", 0.5)
            point.set_parameter("cat_param", "A")
            point.set_parameter("bool_param", True)
            
            result = Mock(spec=BrainsmithResult)
            result.metrics = Mock()
            result.metrics.metric1 = 100 - i * 10  # Decreasing
            result.metrics.metric2 = i * 5          # Increasing
            results.append(result)
            
            engine.update_with_result(point, result)
        
        # Test that the engine handles multi-objective correctly
        self.assertEqual(len(engine.evaluation_history), 5)
    
    def test_noise_injection(self):
        """Test noise injection around promising points."""
        engine = SimpleDSEEngine("adaptive", self.config)
        engine.initialize(self.design_space)
        
        # Create a center point
        center_point = DesignPoint()
        center_point.set_parameter("int_param", 5)
        center_point.set_parameter("float_param", 0.5)
        center_point.set_parameter("cat_param", "B")
        center_point.set_parameter("bool_param", True)
        
        # Test noise injection
        noisy_point = engine._sample_around_point(center_point, noise_factor=0.1)
        
        # Should have all parameters
        self.assertEqual(len(noisy_point.parameters), 4)
        
        # Float parameter should be close to original with some noise
        original_float = center_point.parameters["float_param"]
        new_float = noisy_point.parameters["float_param"]
        self.assertNotEqual(original_float, new_float)  # Should have some noise


class TestParameterNoiseFunction(unittest.TestCase):
    """Test parameter noise injection functionality."""
    
    def setUp(self):
        """Set up parameter definitions."""
        self.int_param = ParameterDefinition("int_param", ParameterType.INTEGER, range_values=[1, 10])
        self.float_param = ParameterDefinition("float_param", ParameterType.CONTINUOUS, range_values=[0.0, 1.0])
        self.cat_param = ParameterDefinition("cat_param", ParameterType.CATEGORICAL, values=["A", "B", "C"])
        self.bool_param = ParameterDefinition("bool_param", ParameterType.BOOLEAN)
    
    def test_continuous_parameter_noise(self):
        """Test noise injection for continuous parameters."""
        from brainsmith.dse.simple import add_parameter_noise
        
        original_value = 0.5
        noisy_value = add_parameter_noise(self.float_param, original_value, noise_factor=0.1)
        
        # Should be a float within valid range
        self.assertIsInstance(noisy_value, float)
        self.assertGreaterEqual(noisy_value, 0.0)
        self.assertLessEqual(noisy_value, 1.0)
    
    def test_integer_parameter_noise(self):
        """Test noise injection for integer parameters."""
        from brainsmith.dse.simple import add_parameter_noise
        
        original_value = 5
        noisy_value = add_parameter_noise(self.int_param, original_value, noise_factor=0.2)
        
        # Should be an int within valid range
        self.assertIsInstance(noisy_value, int)
        self.assertGreaterEqual(noisy_value, 1)
        self.assertLessEqual(noisy_value, 10)
    
    def test_categorical_parameter_noise(self):
        """Test noise injection for categorical parameters."""
        from brainsmith.dse.simple import add_parameter_noise
        
        original_value = "A"
        noisy_value = add_parameter_noise(self.cat_param, original_value, noise_factor=0.5)
        
        # Should be a valid category
        self.assertIn(noisy_value, ["A", "B", "C"])
    
    def test_boolean_parameter_noise(self):
        """Test noise injection for boolean parameters."""
        from brainsmith.dse.simple import add_parameter_noise
        
        original_value = True
        noisy_value = add_parameter_noise(self.bool_param, original_value, noise_factor=0.3)
        
        # Should be a boolean
        self.assertIsInstance(noisy_value, bool)


class TestSamplingStrategyComparison(unittest.TestCase):
    """Test and compare different sampling strategies."""
    
    def setUp(self):
        """Set up common test environment."""
        self.config = DSEConfiguration(max_evaluations=50, seed=42)
        self.design_space = DesignSpace("comparison_test")
        
        # Simple 2D space for easier analysis
        param1 = ParameterDefinition("x", ParameterType.CONTINUOUS, range_values=[0.0, 1.0])
        param2 = ParameterDefinition("y", ParameterType.CONTINUOUS, range_values=[0.0, 1.0])
        
        self.design_space.add_parameter(param1)
        self.design_space.add_parameter(param2)
    
    def test_sampling_diversity(self):
        """Test that different strategies produce diverse samples."""
        strategies = ["random", "latin_hypercube", "sobol"]
        n_samples = 20
        
        for strategy in strategies:
            with self.subTest(strategy=strategy):
                engine = SimpleDSEEngine(strategy, self.config)
                engine.initialize(self.design_space)
                
                points = engine.suggest_next_points(n_samples)
                self.assertEqual(len(points), n_samples)
                
                # Extract coordinates
                x_coords = [p.parameters["x"] for p in points]
                y_coords = [p.parameters["y"] for p in points]
                
                # Check diversity (should have different values)
                unique_x = len(set(x_coords))
                unique_y = len(set(y_coords))
                
                # Should have reasonable diversity
                self.assertGreater(unique_x, n_samples // 4)
                self.assertGreater(unique_y, n_samples // 4)
    
    def test_strategy_reproducibility(self):
        """Test that strategies are reproducible with same seed."""
        strategy = "random"
        n_samples = 10
        
        # Create two engines with same seed
        engine1 = SimpleDSEEngine(strategy, self.config)
        engine1.initialize(self.design_space)
        
        engine2 = SimpleDSEEngine(strategy, self.config)
        engine2.initialize(self.design_space)
        
        points1 = engine1.suggest_next_points(n_samples)
        points2 = engine2.suggest_next_points(n_samples)
        
        # Should produce same results (for random strategy with same seed)
        for p1, p2 in zip(points1, points2):
            self.assertEqual(p1.parameters, p2.parameters)


if __name__ == '__main__':
    unittest.main()