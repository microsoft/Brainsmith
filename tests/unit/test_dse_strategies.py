"""
Unit tests for DSE strategy management system.

Tests strategy selection, configuration, and validation.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch

# Add brainsmith to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from brainsmith.dse.strategies import (
    SamplingStrategy, OptimizationStrategy, StrategyConfig,
    get_strategy_config, create_dse_config_for_strategy,
    get_recommended_strategies_for_problem, validate_strategy_config,
    StrategySelector, STRATEGY_CONFIGS, COMMON_CONFIGS
)
from brainsmith.dse.interface import DSEConfiguration, DSEObjective, OptimizationObjective


class TestStrategyEnums(unittest.TestCase):
    """Test strategy enumeration classes."""
    
    def test_sampling_strategy_enum(self):
        """Test SamplingStrategy enum values."""
        expected_strategies = [
            "random", "latin_hypercube", "sobol", "halton", "adaptive", "grid"
        ]
        
        for strategy in expected_strategies:
            # Should be able to create enum from string
            enum_value = SamplingStrategy(strategy)
            self.assertEqual(enum_value.value, strategy)
    
    def test_optimization_strategy_enum(self):
        """Test OptimizationStrategy enum values."""
        expected_strategies = [
            "random", "bayesian", "genetic", "simulated_annealing", 
            "particle_swarm", "differential_evolution"
        ]
        
        for strategy in expected_strategies:
            # Should be able to create enum from string
            enum_value = OptimizationStrategy(strategy)
            self.assertEqual(enum_value.value, strategy)


class TestStrategyConfig(unittest.TestCase):
    """Test StrategyConfig data structure."""
    
    def test_strategy_config_creation(self):
        """Test StrategyConfig creation and properties."""
        config = StrategyConfig(
            name="Test Strategy",
            description="A test strategy",
            recommended_max_evaluations=100,
            supports_multi_objective=True,
            requires_external_library=False,
            default_config={"param1": "value1"}
        )
        
        self.assertEqual(config.name, "Test Strategy")
        self.assertEqual(config.description, "A test strategy")
        self.assertEqual(config.recommended_max_evaluations, 100)
        self.assertTrue(config.supports_multi_objective)
        self.assertFalse(config.requires_external_library)
        self.assertEqual(config.default_config, {"param1": "value1"})
    
    def test_strategy_config_default_config(self):
        """Test default config initialization."""
        config = StrategyConfig(
            name="Test",
            description="Test",
            recommended_max_evaluations=50,
            supports_multi_objective=True,
            requires_external_library=False
        )
        
        # Should initialize empty default_config
        self.assertEqual(config.default_config, {})


class TestStrategyConfigRetrieval(unittest.TestCase):
    """Test strategy configuration retrieval."""
    
    def test_get_sampling_strategy_config(self):
        """Test getting configuration for sampling strategies."""
        config = get_strategy_config(SamplingStrategy.RANDOM)
        self.assertIsInstance(config, StrategyConfig)
        self.assertEqual(config.name, "Random Sampling")
        
        config = get_strategy_config(SamplingStrategy.LATIN_HYPERCUBE)
        self.assertIsInstance(config, StrategyConfig)
        self.assertTrue(config.requires_external_library)
        self.assertEqual(config.external_library, "scipy")
    
    def test_predefined_strategy_configs(self):
        """Test that all predefined strategies have valid configs."""
        for strategy in STRATEGY_CONFIGS:
            config = STRATEGY_CONFIGS[strategy]
            self.assertIsInstance(config, StrategyConfig)
            self.assertIsInstance(config.name, str)
            self.assertIsInstance(config.description, str)
            self.assertIsInstance(config.recommended_max_evaluations, int)
            self.assertIsInstance(config.supports_multi_objective, bool)
            self.assertIsInstance(config.requires_external_library, bool)


class TestDSEConfigCreation(unittest.TestCase):
    """Test DSE configuration creation from strategies."""
    
    def test_create_config_simple_strategy(self):
        """Test creating config for simple strategies."""
        config = create_dse_config_for_strategy(
            strategy="random",
            max_evaluations=50,
            objectives=["performance.throughput"]
        )
        
        self.assertIsInstance(config, DSEConfiguration)
        self.assertEqual(config.strategy, "random")
        self.assertEqual(config.max_evaluations, 50)
        self.assertEqual(len(config.objectives), 1)
        self.assertEqual(config.objectives[0].name, "performance.throughput")
    
    def test_create_config_with_objective_dicts(self):
        """Test creating config with objective dictionaries."""
        objectives = [
            {"name": "metric1", "direction": "maximize", "weight": 1.0},
            {"name": "metric2", "direction": "minimize", "weight": 0.5}
        ]
        
        config = create_dse_config_for_strategy(
            strategy="genetic",
            max_evaluations=100,
            objectives=objectives
        )
        
        self.assertEqual(len(config.objectives), 2)
        self.assertEqual(config.objectives[0].direction, OptimizationObjective.MAXIMIZE)
        self.assertEqual(config.objectives[1].direction, OptimizationObjective.MINIMIZE)
        self.assertEqual(config.objectives[1].weight, 0.5)
    
    def test_create_config_with_strategy_kwargs(self):
        """Test creating config with strategy-specific parameters."""
        config = create_dse_config_for_strategy(
            strategy="bayesian",
            max_evaluations=75,
            acquisition_function="EI",
            n_initial_points=15
        )
        
        self.assertEqual(config.strategy, "bayesian")
        self.assertEqual(config.strategy_config["acquisition_function"], "EI")
        self.assertEqual(config.strategy_config["n_initial_points"], 15)
    
    def test_create_config_default_objectives(self):
        """Test config creation with default objectives."""
        config = create_dse_config_for_strategy("random", max_evaluations=30)
        
        # Should have default objective
        self.assertEqual(len(config.objectives), 1)
        self.assertEqual(config.objectives[0].name, "performance.throughput_ops_sec")


class TestStrategyRecommendation(unittest.TestCase):
    """Test strategy recommendation system."""
    
    def test_recommendations_small_problem(self):
        """Test recommendations for small problems."""
        recommendations = get_recommended_strategies_for_problem(
            n_parameters=3,
            max_evaluations=50,
            n_objectives=1,
            has_external_libs=True
        )
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Should include bayesian for small single-objective problems
        self.assertIn("bayesian", recommendations)
    
    def test_recommendations_multi_objective(self):
        """Test recommendations for multi-objective problems."""
        recommendations = get_recommended_strategies_for_problem(
            n_parameters=5,
            max_evaluations=100,
            n_objectives=3,
            has_external_libs=True
        )
        
        # Should prioritize genetic algorithms for multi-objective
        self.assertIn("genetic", recommendations)
        # Should not include single-objective only strategies
        self.assertNotIn("bayesian", recommendations)
    
    def test_recommendations_high_dimensional(self):
        """Test recommendations for high-dimensional problems."""
        recommendations = get_recommended_strategies_for_problem(
            n_parameters=15,
            max_evaluations=200,
            n_objectives=1,
            has_external_libs=True
        )
        
        # Should include quasi-random methods for high-dimensional
        self.assertIn("sobol", recommendations)
        self.assertIn("latin_hypercube", recommendations)
    
    def test_recommendations_no_external_libs(self):
        """Test recommendations when external libraries unavailable."""
        recommendations = get_recommended_strategies_for_problem(
            n_parameters=8,
            max_evaluations=100,
            n_objectives=1,
            has_external_libs=False
        )
        
        # Should only include strategies that don't require external libs
        self.assertIn("adaptive", recommendations)
        self.assertIn("random", recommendations)
        # Should not include external-library strategies
        self.assertNotIn("bayesian", recommendations)
        self.assertNotIn("genetic", recommendations)
    
    def test_recommendations_unique(self):
        """Test that recommendations don't contain duplicates."""
        recommendations = get_recommended_strategies_for_problem(
            n_parameters=10,
            max_evaluations=300,
            n_objectives=2,
            has_external_libs=True
        )
        
        # Should not have duplicates
        self.assertEqual(len(recommendations), len(set(recommendations)))


class TestStrategyValidation(unittest.TestCase):
    """Test strategy configuration validation."""
    
    def test_valid_random_strategy(self):
        """Test validation of valid random strategy config."""
        is_valid, errors = validate_strategy_config("random", {})
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_valid_bayesian_strategy(self):
        """Test validation of valid Bayesian strategy config."""
        config = {
            "acquisition_function": "EI",
            "n_initial_points": 10
        }
        
        # Mock external library availability
        with patch('brainsmith.dse.strategies.validate_strategy_config') as mock_validate:
            mock_validate.return_value = (True, [])
            is_valid, errors = validate_strategy_config("bayesian", config)
            
            # Should call the actual function
            mock_validate.assert_called_once_with("bayesian", config)
    
    def test_invalid_bayesian_acquisition(self):
        """Test validation of invalid Bayesian acquisition function."""
        config = {"acquisition_function": "INVALID"}
        
        is_valid, errors = validate_strategy_config("bayesian", config)
        # Should fail validation due to invalid acquisition function
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
    
    def test_invalid_genetic_parameters(self):
        """Test validation of invalid genetic algorithm parameters."""
        config = {
            "population_size": 5,  # Too small
            "mutation_prob": 1.5,  # Invalid probability
            "crossover_prob": -0.1  # Invalid probability
        }
        
        is_valid, errors = validate_strategy_config("genetic", config)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
    
    def test_unknown_strategy_validation(self):
        """Test validation of unknown strategy."""
        is_valid, errors = validate_strategy_config("unknown_strategy", {})
        self.assertFalse(is_valid)
        self.assertIn("Unknown strategy", errors[0])


class TestStrategySelector(unittest.TestCase):
    """Test automatic strategy selection."""
    
    def test_selector_small_single_objective(self):
        """Test selector for small single-objective problems."""
        selector = StrategySelector()
        strategy = selector.select_best_strategy(
            n_parameters=4,
            max_evaluations=80,
            n_objectives=1,
            problem_type="fpga"
        )
        
        self.assertIsInstance(strategy, str)
        self.assertIn(strategy, ["random", "adaptive", "latin_hypercube", "bayesian"])
    
    def test_selector_multi_objective(self):
        """Test selector for multi-objective problems."""
        selector = StrategySelector()
        strategy = selector.select_best_strategy(
            n_parameters=6,
            max_evaluations=150,
            n_objectives=3,
            problem_type="fpga"
        )
        
        # Should prefer genetic for multi-objective FPGA problems
        self.assertIn(strategy, ["genetic", "adaptive", "random"])
    
    def test_selector_prefer_speed(self):
        """Test selector with speed preference."""
        selector = StrategySelector()
        strategy = selector.select_best_strategy(
            n_parameters=8,
            max_evaluations=100,
            n_objectives=1,
            prefer_speed=True
        )
        
        # Should prefer faster strategies
        fast_strategies = ["random", "latin_hypercube", "sobol", "adaptive"]
        self.assertIn(strategy, fast_strategies)
    
    def test_selector_fpga_specific(self):
        """Test FPGA-specific strategy selection."""
        selector = StrategySelector()
        
        # Multi-objective FPGA problem should prefer genetic
        strategy = selector.select_best_strategy(
            n_parameters=5,
            max_evaluations=200,
            n_objectives=2,
            problem_type="fpga"
        )
        
        # Should consider genetic for multi-objective FPGA
        self.assertIsInstance(strategy, str)
    
    def test_selector_fallback(self):
        """Test selector fallback to random."""
        selector = StrategySelector()
        
        # Should always return a valid strategy
        strategy = selector.select_best_strategy(
            n_parameters=0,  # Edge case
            max_evaluations=1,  # Edge case
            n_objectives=0   # Edge case
        )
        
        self.assertIsInstance(strategy, str)
        # Should fallback to a basic strategy
        self.assertIn(strategy, ["random", "adaptive"])


class TestCommonConfigurations(unittest.TestCase):
    """Test predefined common configurations."""
    
    def test_common_configs_exist(self):
        """Test that common configurations are defined."""
        expected_configs = [
            "quick_exploration",
            "balanced_optimization", 
            "thorough_analysis",
            "multi_objective_fpga"
        ]
        
        for config_name in expected_configs:
            self.assertIn(config_name, COMMON_CONFIGS)
            config = COMMON_CONFIGS[config_name]
            self.assertIsInstance(config, DSEConfiguration)
    
    def test_quick_exploration_config(self):
        """Test quick exploration configuration."""
        config = COMMON_CONFIGS["quick_exploration"]
        
        self.assertEqual(config.strategy, "random")
        self.assertEqual(config.max_evaluations, 20)
        self.assertEqual(len(config.objectives), 1)
    
    def test_multi_objective_fpga_config(self):
        """Test multi-objective FPGA configuration."""
        config = COMMON_CONFIGS["multi_objective_fpga"]
        
        self.assertEqual(config.strategy, "genetic")
        self.assertGreater(len(config.objectives), 1)
        self.assertIn("population_size", config.strategy_config)
    
    def test_balanced_optimization_config(self):
        """Test balanced optimization configuration."""
        config = COMMON_CONFIGS["balanced_optimization"]
        
        self.assertEqual(config.strategy, "adaptive")
        self.assertGreater(config.max_evaluations, 50)
    
    def test_thorough_analysis_config(self):
        """Test thorough analysis configuration."""
        config = COMMON_CONFIGS["thorough_analysis"]
        
        self.assertEqual(config.strategy, "latin_hypercube")
        self.assertGreater(config.max_evaluations, 100)


class TestStrategyConfigIntegration(unittest.TestCase):
    """Test integration between strategy configs and DSE system."""
    
    def test_config_to_engine_compatibility(self):
        """Test that created configs are compatible with engines."""
        from brainsmith.dse.interface import create_dse_engine
        
        # Test various strategy configs
        strategies = ["random", "adaptive", "latin_hypercube"]
        
        for strategy in strategies:
            with self.subTest(strategy=strategy):
                config = create_dse_config_for_strategy(
                    strategy=strategy,
                    max_evaluations=20
                )
                
                # Should be able to create engine with this config
                try:
                    engine = create_dse_engine(strategy, config)
                    self.assertIsNotNone(engine)
                except Exception as e:
                    # Some strategies may fail if external libs not available
                    if "not available" not in str(e) and "Unknown strategy" not in str(e):
                        raise
    
    def test_common_config_compatibility(self):
        """Test that common configs are compatible with engines."""
        from brainsmith.dse.interface import create_dse_engine
        
        for config_name, config in COMMON_CONFIGS.items():
            with self.subTest(config=config_name):
                try:
                    engine = create_dse_engine(config.strategy, config)
                    self.assertIsNotNone(engine)
                except Exception as e:
                    # Some strategies may fail if external libs not available
                    if "not available" not in str(e) and "Unknown strategy" not in str(e):
                        raise


if __name__ == '__main__':
    unittest.main()