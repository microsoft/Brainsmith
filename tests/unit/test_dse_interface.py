"""
Unit tests for DSE interface system.

Tests the core DSE interface, configuration, and engine management.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch

# Add brainsmith to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from brainsmith.dse.interface import (
    DSEInterface, DSEEngine, DSEConfiguration, DSEObjective, 
    OptimizationObjective, DSEProgress, create_dse_engine
)
from brainsmith.core.design_space import DesignSpace, DesignPoint, ParameterDefinition, ParameterType
from brainsmith.core.result import BrainsmithResult, DSEResult
from brainsmith.core.metrics import BrainsmithMetrics


class TestDSEObjective(unittest.TestCase):
    """Test DSEObjective functionality."""
    
    def test_objective_creation(self):
        """Test objective creation and properties."""
        obj = DSEObjective("performance.throughput", OptimizationObjective.MAXIMIZE, 1.5)
        self.assertEqual(obj.name, "performance.throughput")
        self.assertEqual(obj.direction, OptimizationObjective.MAXIMIZE)
        self.assertEqual(obj.weight, 1.5)
    
    def test_objective_evaluation(self):
        """Test objective evaluation from metrics."""
        # Create mock metrics with nested attributes
        metrics = Mock()
        metrics.performance = Mock()
        metrics.performance.throughput_ops_sec = 1000.0
        
        obj = DSEObjective("performance.throughput_ops_sec", OptimizationObjective.MAXIMIZE)
        value = obj.evaluate(metrics)
        self.assertEqual(value, 1000.0)
    
    def test_objective_evaluation_missing_attribute(self):
        """Test handling of missing metric attributes."""
        metrics = Mock()
        obj = DSEObjective("nonexistent.metric", OptimizationObjective.MAXIMIZE)
        
        with self.assertRaises(ValueError):
            obj.evaluate(metrics)
    
    def test_objective_evaluation_non_numeric(self):
        """Test handling of non-numeric metric values."""
        metrics = Mock()
        metrics.performance = Mock()
        metrics.performance.status = "complete"  # Non-numeric
        
        obj = DSEObjective("performance.status", OptimizationObjective.MAXIMIZE)
        
        with self.assertRaises(ValueError):
            obj.evaluate(metrics)


class TestDSEConfiguration(unittest.TestCase):
    """Test DSE configuration functionality."""
    
    def test_default_configuration(self):
        """Test default configuration values."""
        config = DSEConfiguration()
        self.assertEqual(config.max_evaluations, 50)
        self.assertIsNone(config.max_time_seconds)
        self.assertEqual(len(config.objectives), 1)
        self.assertEqual(config.objectives[0].name, "performance.throughput_ops_sec")
        self.assertEqual(config.strategy, "random")
    
    def test_custom_configuration(self):
        """Test custom configuration creation."""
        objectives = [
            DSEObjective("metric1", OptimizationObjective.MAXIMIZE),
            DSEObjective("metric2", OptimizationObjective.MINIMIZE)
        ]
        
        config = DSEConfiguration(
            max_evaluations=100,
            max_time_seconds=3600,
            objectives=objectives,
            strategy="bayesian",
            convergence_threshold=0.001,
            seed=42
        )
        
        self.assertEqual(config.max_evaluations, 100)
        self.assertEqual(config.max_time_seconds, 3600)
        self.assertEqual(len(config.objectives), 2)
        self.assertEqual(config.strategy, "bayesian")
        self.assertEqual(config.convergence_threshold, 0.001)
        self.assertEqual(config.seed, 42)


class TestDSEProgress(unittest.TestCase):
    """Test DSE progress tracking."""
    
    def test_progress_creation(self):
        """Test progress object creation."""
        progress = DSEProgress()
        self.assertEqual(progress.evaluations_completed, 0)
        self.assertEqual(progress.evaluations_total, 0)
        self.assertEqual(progress.time_elapsed, 0.0)
        self.assertEqual(progress.completion_percentage, 0.0)
    
    def test_completion_percentage(self):
        """Test completion percentage calculation."""
        progress = DSEProgress(evaluations_completed=25, evaluations_total=100)
        self.assertEqual(progress.completion_percentage, 25.0)
        
        # Test division by zero protection
        progress = DSEProgress(evaluations_completed=10, evaluations_total=0)
        self.assertEqual(progress.completion_percentage, 0.0)


class MockDSEEngine(DSEEngine):
    """Mock DSE engine for testing."""
    
    def __init__(self, name, config):
        super().__init__(name, config)
        self.suggested_count = 0
    
    def suggest_next_points(self, n_points=1):
        """Mock point suggestion."""
        points = []
        for i in range(n_points):
            point = DesignPoint()
            point.set_parameter("param1", self.suggested_count + i)
            points.append(point)
        self.suggested_count += n_points
        return points
    
    def update_with_result(self, design_point, result):
        """Mock result update."""
        super().update_with_result(design_point, result)


class TestDSEEngine(unittest.TestCase):
    """Test DSE engine base functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = DSEConfiguration(max_evaluations=10)
        self.engine = MockDSEEngine("test_engine", self.config)
        
        # Create a simple design space
        self.design_space = DesignSpace("test_space")
        param = ParameterDefinition("param1", ParameterType.INTEGER, range_values=[1, 10])
        self.design_space.add_parameter(param)
    
    def test_engine_creation(self):
        """Test engine creation."""
        self.assertEqual(self.engine.name, "test_engine")
        self.assertEqual(self.engine.config, self.config)
        self.assertEqual(len(self.engine.evaluation_history), 0)
    
    def test_suggest_points(self):
        """Test point suggestion."""
        points = self.engine.suggest_next_points(3)
        self.assertEqual(len(points), 3)
        self.assertEqual(points[0].parameters["param1"], 0)
        self.assertEqual(points[1].parameters["param1"], 1)
        self.assertEqual(points[2].parameters["param1"], 2)
    
    def test_update_with_result(self):
        """Test result updating."""
        point = DesignPoint()
        point.set_parameter("param1", 5)
        
        # Create mock result
        result = Mock(spec=BrainsmithResult)
        result.metrics = Mock()
        result.metrics.performance = Mock()
        result.metrics.performance.throughput_ops_sec = 100.0
        
        self.engine.update_with_result(point, result)
        
        self.assertEqual(len(self.engine.evaluation_history), 1)
        self.assertEqual(self.engine.evaluation_history[0][0], point)
        self.assertEqual(self.engine.evaluation_history[0][1], result)
    
    def test_progress_tracking(self):
        """Test progress tracking during optimization."""
        progress = self.engine.get_progress()
        self.assertEqual(progress.evaluations_completed, 0)
        
        # Simulate some evaluations
        for i in range(3):
            point = DesignPoint()
            result = Mock(spec=BrainsmithResult)
            self.engine.update_with_result(point, result)
        
        progress = self.engine.get_progress()
        self.assertEqual(progress.evaluations_completed, 3)
    
    def test_reset(self):
        """Test engine reset functionality."""
        # Add some evaluation history
        point = DesignPoint()
        result = Mock(spec=BrainsmithResult)
        self.engine.update_with_result(point, result)
        
        self.assertEqual(len(self.engine.evaluation_history), 1)
        
        # Reset and verify
        self.engine.reset()
        self.assertEqual(len(self.engine.evaluation_history), 0)
        self.assertIsNone(self.engine.start_time)
    
    def test_pareto_frontier_computation(self):
        """Test Pareto frontier computation for multi-objective."""
        # Create multi-objective configuration
        objectives = [
            DSEObjective("metric1", OptimizationObjective.MAXIMIZE),
            DSEObjective("metric2", OptimizationObjective.MAXIMIZE)
        ]
        config = DSEConfiguration(objectives=objectives)
        engine = MockDSEEngine("multi_obj", config)
        
        # Create mock results with different trade-offs
        results = []
        for i in range(5):
            result = Mock(spec=BrainsmithResult)
            result.metrics = Mock()
            result.metrics.metric1 = 10 - i  # Decreasing
            result.metrics.metric2 = i * 2   # Increasing
            results.append(result)
        
        # Test Pareto frontier computation
        pareto_results = engine._compute_pareto_frontier(results)
        
        # With these values, we expect multiple non-dominated solutions
        self.assertGreater(len(pareto_results), 1)
        self.assertLessEqual(len(pareto_results), len(results))
    
    def test_dominance_check(self):
        """Test dominance checking for multi-objective optimization."""
        engine = MockDSEEngine("test", self.config)
        
        # Test cases for dominance
        a = [10, 5]  # High on first, low on second
        b = [5, 10]  # Low on first, high on second
        c = [12, 6]  # Better than a on both
        
        # a and b should not dominate each other
        self.assertFalse(engine._dominates(a, b))
        self.assertFalse(engine._dominates(b, a))
        
        # c should dominate a
        self.assertTrue(engine._dominates(c, a))
        self.assertFalse(engine._dominates(a, c))


class TestDSEEngineFactory(unittest.TestCase):
    """Test DSE engine factory functionality."""
    
    def test_create_simple_engine(self):
        """Test creation of simple DSE engines."""
        config = DSEConfiguration(strategy="random")
        
        for strategy in ["simple", "random", "latin_hypercube", "adaptive"]:
            with self.subTest(strategy=strategy):
                engine = create_dse_engine(strategy, config)
                self.assertIsNotNone(engine)
                self.assertTrue(strategy in engine.name or "SimpleDSE" in engine.name)
    
    def test_create_external_engine(self):
        """Test creation of external DSE engines."""
        config = DSEConfiguration(strategy="bayesian")
        
        # Test external strategies (may fail if libraries not available)
        for strategy in ["bayesian", "genetic", "optuna"]:
            with self.subTest(strategy=strategy):
                try:
                    engine = create_dse_engine(strategy, config)
                    self.assertIsNotNone(engine)
                except (ImportError, ValueError):
                    # Expected if external libraries not available
                    pass
    
    def test_unknown_strategy(self):
        """Test handling of unknown strategies."""
        config = DSEConfiguration(strategy="unknown")
        
        with self.assertRaises(ValueError):
            create_dse_engine("unknown_strategy", config)


class TestDSEObjectiveEvaluation(unittest.TestCase):
    """Test objective evaluation with real metric structures."""
    
    def test_nested_metric_access(self):
        """Test accessing nested metrics."""
        # Create realistic metric structure
        metrics = Mock()
        metrics.performance = Mock()
        metrics.performance.throughput_ops_sec = 1000.0
        metrics.hardware = Mock()
        metrics.hardware.lut_utilization = 0.75
        
        # Test throughput objective
        throughput_obj = DSEObjective("performance.throughput_ops_sec", OptimizationObjective.MAXIMIZE)
        throughput_value = throughput_obj.evaluate(metrics)
        self.assertEqual(throughput_value, 1000.0)
        
        # Test LUT utilization objective
        lut_obj = DSEObjective("hardware.lut_utilization", OptimizationObjective.MINIMIZE)
        lut_value = lut_obj.evaluate(metrics)
        self.assertEqual(lut_value, 0.75)
    
    def test_deep_nested_access(self):
        """Test accessing deeply nested metrics."""
        metrics = Mock()
        metrics.detailed = Mock()
        metrics.detailed.performance = Mock()
        metrics.detailed.performance.latency = Mock()
        metrics.detailed.performance.latency.avg_ms = 5.2
        
        obj = DSEObjective("detailed.performance.latency.avg_ms", OptimizationObjective.MINIMIZE)
        value = obj.evaluate(metrics)
        self.assertEqual(value, 5.2)


if __name__ == '__main__':
    unittest.main()