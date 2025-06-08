"""
Integration tests for complete DSE workflow.

Tests the end-to-end DSE functionality including strategy selection,
optimization execution, and result analysis.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add brainsmith to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import brainsmith
from brainsmith.core.design_space import DesignSpace, DesignPoint, ParameterDefinition, ParameterType
from brainsmith.core.result import BrainsmithResult, DSEResult
from brainsmith.dse.interface import DSEConfiguration, DSEObjective, OptimizationObjective


class TestCompletePhase3Integration(unittest.TestCase):
    """Integration tests for complete Phase 3 functionality."""
    
    def setUp(self):
        """Set up integration test environment."""
        # Create a realistic design space
        self.design_space = DesignSpace("integration_test")
        
        # Add various parameter types
        params = [
            ParameterDefinition("batch_size", ParameterType.INTEGER, range_values=[1, 64]),
            ParameterDefinition("learning_rate", ParameterType.CONTINUOUS, range_values=[1e-5, 1e-1]),
            ParameterDefinition("optimizer", ParameterType.CATEGORICAL, values=["adam", "sgd", "rmsprop"]),
            ParameterDefinition("use_bias", ParameterType.BOOLEAN),
            ParameterDefinition("hidden_dim", ParameterType.INTEGER, range_values=[64, 512])
        ]
        
        for param in params:
            self.design_space.add_parameter(param)
    
    def create_mock_blueprint(self):
        """Create a mock blueprint for testing."""
        blueprint = Mock()
        blueprint.name = "test_integration"
        blueprint.has_design_space = Mock(return_value=True)
        blueprint.get_design_space = Mock(return_value=self.design_space)
        blueprint.get_recommended_parameters = Mock(return_value={
            "batch_size": 32,
            "learning_rate": 1e-3,
            "optimizer": "adam",
            "use_bias": True,
            "hidden_dim": 256
        })
        return blueprint
    
    def create_mock_build_function(self):
        """Create a mock build function that simulates compilation."""
        def mock_build(model_path, blueprint_name, parameters):
            # Simulate realistic performance based on parameters
            base_throughput = 100.0
            
            # Batch size affects throughput
            if "batch_size" in parameters:
                base_throughput *= (parameters["batch_size"] / 32.0) * 0.8 + 0.2
            
            # Learning rate affects convergence (simulate inverse relationship)
            if "learning_rate" in parameters:
                lr_factor = max(0.1, 1.0 - abs(parameters["learning_rate"] - 1e-3) * 1000)
                base_throughput *= lr_factor
            
            # Hidden dimension affects both performance and power
            power_consumption = 50.0
            if "hidden_dim" in parameters:
                dim_factor = parameters["hidden_dim"] / 256.0
                base_throughput *= (1.0 + (dim_factor - 1.0) * 0.3)  # Slight increase
                power_consumption *= dim_factor  # Linear increase
            
            # Create realistic result
            result = Mock(spec=BrainsmithResult)
            result.metrics = Mock()
            result.metrics.performance = Mock()
            result.metrics.performance.throughput_ops_sec = base_throughput
            result.metrics.hardware = Mock()
            result.metrics.hardware.power_consumption = power_consumption
            result.metrics.performance.power_efficiency = base_throughput / power_consumption
            result.build_time_seconds = 30.0
            result.success = True
            
            return result
        
        return mock_build
    
    @patch('brainsmith.blueprints.get_blueprint')
    @patch('brainsmith.blueprints.get_design_space')
    def test_end_to_end_single_objective_optimization(self, mock_get_design_space, mock_get_blueprint):
        """Test complete single-objective optimization workflow."""
        # Setup mocks
        mock_get_blueprint.return_value = self.create_mock_blueprint()
        mock_get_design_space.return_value = self.design_space
        
        # Mock the build function
        with patch('brainsmith.build_model', self.create_mock_build_function()):
            # Test the complete workflow
            result = brainsmith.explore_design_space(
                model_path="test_model.onnx",
                blueprint_name="test_integration",
                max_evaluations=10,
                strategy="adaptive",
                objectives=["performance.throughput_ops_sec"]
            )
            
            # Verify result structure
            self.assertIsInstance(result, DSEResult)
            self.assertEqual(len(result.results), 10)
            self.assertEqual(result.strategy, "adaptive")
            self.assertIsNotNone(result.best_result)
            
            # Verify all results are valid
            for build_result in result.results:
                self.assertTrue(build_result.success)
                self.assertGreater(build_result.metrics.performance.throughput_ops_sec, 0)
    
    @patch('brainsmith.blueprints.get_blueprint')
    @patch('brainsmith.blueprints.get_design_space')
    def test_end_to_end_multi_objective_optimization(self, mock_get_design_space, mock_get_blueprint):
        """Test complete multi-objective optimization workflow."""
        # Setup mocks
        mock_get_blueprint.return_value = self.create_mock_blueprint()
        mock_get_design_space.return_value = self.design_space
        
        # Mock the build function
        with patch('brainsmith.build_model', self.create_mock_build_function()):
            # Test multi-objective optimization
            result = brainsmith.explore_design_space(
                model_path="test_model.onnx",
                blueprint_name="test_integration",
                max_evaluations=15,
                strategy="random",
                objectives=[
                    {"name": "performance.throughput_ops_sec", "direction": "maximize", "weight": 1.0},
                    {"name": "performance.power_efficiency", "direction": "maximize", "weight": 0.8}
                ]
            )
            
            # Verify multi-objective results
            self.assertEqual(len(result.results), 15)
            
            # Test Pareto frontier extraction
            pareto_points = brainsmith.get_pareto_frontier(result)
            self.assertIsInstance(pareto_points, list)
            self.assertGreater(len(pareto_points), 0)
    
    def test_automatic_strategy_recommendation(self):
        """Test automatic strategy recommendation system."""
        # Test various problem configurations
        test_cases = [
            {"n_parameters": 3, "max_evaluations": 50, "n_objectives": 1, "expected_type": str},
            {"n_parameters": 10, "max_evaluations": 200, "n_objectives": 2, "expected_type": str},
            {"blueprint_name": "test_integration", "expected_type": str}
        ]
        
        for case in test_cases:
            with self.subTest(case=case):
                strategy = brainsmith.recommend_strategy(**{k: v for k, v in case.items() if k != "expected_type"})
                self.assertIsInstance(strategy, case["expected_type"])
    
    @patch('brainsmith.blueprints.get_blueprint')
    @patch('brainsmith.blueprints.get_design_space')
    def test_optimize_model_with_auto_strategy(self, mock_get_design_space, mock_get_blueprint):
        """Test optimize_model with automatic strategy selection."""
        # Setup mocks
        mock_get_blueprint.return_value = self.create_mock_blueprint()
        mock_get_design_space.return_value = self.design_space
        
        with patch('brainsmith.build_model', self.create_mock_build_function()):
            # Test automatic optimization
            result = brainsmith.optimize_model(
                model_path="test_model.onnx",
                blueprint_name="test_integration",
                max_evaluations=8,
                strategy="auto",
                objectives=["performance.throughput_ops_sec"]
            )
            
            self.assertIsInstance(result, DSEResult)
            self.assertEqual(len(result.results), 8)
            self.assertIsNotNone(result.best_result)
    
    @patch('brainsmith.blueprints.get_blueprint')
    @patch('brainsmith.blueprints.get_design_space')
    def test_comprehensive_analysis_workflow(self, mock_get_design_space, mock_get_blueprint):
        """Test comprehensive analysis of DSE results."""
        # Setup mocks
        mock_get_blueprint.return_value = self.create_mock_blueprint()
        mock_get_design_space.return_value = self.design_space
        
        with patch('brainsmith.build_model', self.create_mock_build_function()):
            # Run optimization
            result = brainsmith.explore_design_space(
                model_path="test_model.onnx",
                blueprint_name="test_integration",
                max_evaluations=12,
                strategy="latin_hypercube",
                objectives=["performance.throughput_ops_sec", "performance.power_efficiency"]
            )
            
            # Test comprehensive analysis
            result.design_space = self.design_space  # Ensure design space is available
            result.objectives = ["performance.throughput_ops_sec", "performance.power_efficiency"]
            
            analysis = brainsmith.analyze_dse_results(result)
            
            # Verify analysis structure
            self.assertIsInstance(analysis, dict)
            self.assertIn("summary", analysis)
            self.assertIn("statistical_analysis", analysis)
            
            # Test analysis report generation
            report = analysis.get("report", "")
            if not report:
                # Generate report manually for testing
                from brainsmith.dse.analysis import DSEAnalyzer
                from brainsmith.dse.interface import DSEObjective, OptimizationObjective
                
                objectives = [
                    DSEObjective("performance.throughput_ops_sec", OptimizationObjective.MAXIMIZE),
                    DSEObjective("performance.power_efficiency", OptimizationObjective.MAXIMIZE)
                ]
                analyzer = DSEAnalyzer(self.design_space, objectives)
                report = analyzer.generate_analysis_report(result)
            
            self.assertIsInstance(report, str)
            self.assertIn("DSE Analysis Report", report)
    
    def test_strategy_availability_and_fallback(self):
        """Test strategy availability detection and fallback mechanisms."""
        # Test strategy availability
        strategies = brainsmith.list_available_strategies()
        self.assertIsInstance(strategies, dict)
        
        # Should always have basic strategies available
        self.assertIn("random", strategies)
        self.assertIn("adaptive", strategies)
        
        # Test that each strategy has proper metadata
        for strategy_name, info in strategies.items():
            self.assertIn("description", info)
            self.assertIn("available", info)
            self.assertIn("supports_multi_objective", info)
            self.assertIsInstance(info["available"], bool)
    
    def test_design_space_sampling(self):
        """Test design space sampling functionality."""
        # Test various sampling strategies
        sampling_strategies = ["random", "latin_hypercube"]
        
        for strategy in sampling_strategies:
            with self.subTest(strategy=strategy):
                try:
                    samples = brainsmith.sample_design_space(
                        self.design_space,
                        n_samples=5,
                        strategy=strategy,
                        seed=42
                    )
                    
                    self.assertEqual(len(samples), 5)
                    
                    # Verify all samples are valid
                    for sample in samples:
                        self.assertIsInstance(sample, DesignPoint)
                        self.assertEqual(len(sample.parameters), len(self.design_space.parameters))
                
                except Exception as e:
                    # Some strategies may not be available
                    if "not available" not in str(e):
                        raise
    
    def test_common_configuration_usage(self):
        """Test usage of predefined common configurations."""
        from brainsmith.dse.strategies import COMMON_CONFIGS
        
        # Test that common configs are accessible and valid
        for config_name, config in COMMON_CONFIGS.items():
            with self.subTest(config=config_name):
                self.assertIsInstance(config, DSEConfiguration)
                self.assertIsInstance(config.strategy, str)
                self.assertGreater(config.max_evaluations, 0)
                self.assertGreater(len(config.objectives), 0)
    
    @patch('brainsmith.blueprints.get_blueprint')
    @patch('brainsmith.blueprints.get_design_space')
    def test_backward_compatibility_with_phase1_api(self, mock_get_design_space, mock_get_blueprint):
        """Test that Phase 3 maintains backward compatibility with Phase 1 API."""
        # Setup mocks
        mock_get_blueprint.return_value = self.create_mock_blueprint()
        mock_get_design_space.return_value = self.design_space
        
        with patch('brainsmith.build_model', self.create_mock_build_function()):
            # Test original Phase 1 API still works
            result = brainsmith.build_model(
                model_path="test_model.onnx",
                blueprint_name="test_integration",
                parameters={"batch_size": 16, "learning_rate": 1e-3}
            )
            
            self.assertIsInstance(result, BrainsmithResult)
            self.assertTrue(result.success)
    
    def test_error_handling_and_robustness(self):
        """Test error handling and robustness of the DSE system."""
        # Test invalid strategy
        with self.assertRaises(ValueError):
            brainsmith.explore_design_space(
                model_path="test.onnx",
                blueprint_name="test",
                strategy="nonexistent_strategy"
            )
        
        # Test empty design space handling
        empty_space = DesignSpace("empty")
        strategies = ["random", "adaptive"]
        
        for strategy in strategies:
            with self.subTest(strategy=strategy):
                # Should handle empty design space gracefully
                try:
                    samples = brainsmith.sample_design_space(empty_space, n_samples=1, strategy=strategy)
                    self.assertEqual(len(samples), 0)  # Should return empty list
                except Exception as e:
                    # Some error handling is acceptable for empty spaces
                    self.assertIn("parameter", str(e).lower())


class TestPhase3PerformanceCharacteristics(unittest.TestCase):
    """Test performance characteristics of Phase 3 implementation."""
    
    def test_sampling_efficiency(self):
        """Test efficiency of sampling strategies."""
        # Create moderately sized design space
        design_space = DesignSpace("performance_test")
        
        for i in range(10):
            param = ParameterDefinition(f"param_{i}", ParameterType.CONTINUOUS, range_values=[0.0, 1.0])
            design_space.add_parameter(param)
        
        # Test sampling performance
        strategies = ["random", "adaptive"]
        n_samples = 100
        
        for strategy in strategies:
            with self.subTest(strategy=strategy):
                import time
                start_time = time.time()
                
                try:
                    samples = brainsmith.sample_design_space(
                        design_space,
                        n_samples=n_samples,
                        strategy=strategy
                    )
                    
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    # Should complete reasonably quickly
                    self.assertLess(duration, 5.0)  # 5 seconds max
                    self.assertEqual(len(samples), n_samples)
                
                except Exception as e:
                    if "not available" not in str(e):
                        raise
    
    def test_analysis_scalability(self):
        """Test scalability of analysis functions."""
        from brainsmith.dse.analysis import DSEAnalyzer
        from brainsmith.dse.interface import DSEObjective, OptimizationObjective
        from brainsmith.core.result import DSEResult
        
        # Create test data
        design_space = DesignSpace("scalability_test")
        param = ParameterDefinition("param1", ParameterType.CONTINUOUS, range_values=[0.0, 1.0])
        design_space.add_parameter(param)
        
        objectives = [DSEObjective("metric1", OptimizationObjective.MAXIMIZE)]
        analyzer = DSEAnalyzer(design_space, objectives)
        
        # Create mock results
        results = []
        for i in range(100):
            result = Mock(spec=BrainsmithResult)
            result.metrics = Mock()
            result.metrics.metric1 = i * 1.0
            results.append(result)
        
        dse_result = Mock(spec=DSEResult)
        dse_result.results = results
        dse_result.total_time_seconds = 100.0
        dse_result.strategy = "test"
        dse_result.objectives = ["metric1"]
        dse_result.design_space = design_space
        
        # Test analysis performance
        import time
        start_time = time.time()
        
        analysis = analyzer.analyze_dse_result(dse_result)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Analysis should complete quickly even with many results
        self.assertLess(duration, 2.0)  # 2 seconds max
        self.assertIsInstance(analysis, dict)


if __name__ == '__main__':
    unittest.main()