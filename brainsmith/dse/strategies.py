"""
Strategy definitions and configurations for DSE engines.

This module provides standard strategy configurations and parameter definitions
for different optimization approaches.
"""

from enum import Enum
from typing import Dict, Any
from dataclasses import dataclass

from .interface import DSEConfiguration, DSEObjective, OptimizationObjective


class SamplingStrategy(Enum):
    """Available sampling strategies."""
    RANDOM = "random"
    LATIN_HYPERCUBE = "latin_hypercube"
    SOBOL = "sobol"
    HALTON = "halton"
    ADAPTIVE = "adaptive"
    GRID = "grid"


class OptimizationStrategy(Enum):
    """Available optimization strategies."""
    RANDOM_SEARCH = "random"
    BAYESIAN_OPTIMIZATION = "bayesian"
    GENETIC_ALGORITHM = "genetic"
    SIMULATED_ANNEALING = "simulated_annealing"
    PARTICLE_SWARM = "particle_swarm"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"


@dataclass
class StrategyConfig:
    """Configuration template for optimization strategies."""
    name: str
    description: str
    recommended_max_evaluations: int
    supports_multi_objective: bool
    requires_external_library: bool
    external_library: str = ""
    default_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.default_config is None:
            self.default_config = {}


# Predefined strategy configurations
STRATEGY_CONFIGS = {
    SamplingStrategy.RANDOM: StrategyConfig(
        name="Random Sampling",
        description="Pure random sampling from the design space",
        recommended_max_evaluations=100,
        supports_multi_objective=True,
        requires_external_library=False,
        default_config={}
    ),
    
    SamplingStrategy.LATIN_HYPERCUBE: StrategyConfig(
        name="Latin Hypercube Sampling",
        description="Quasi-random sampling ensuring good space coverage",
        recommended_max_evaluations=200,
        supports_multi_objective=True,
        requires_external_library=True,
        external_library="scipy",
        default_config={
            "criterion": "maximin",
            "iterations": 1000
        }
    ),
    
    SamplingStrategy.SOBOL: StrategyConfig(
        name="Sobol Sequence Sampling",
        description="Low-discrepancy sequence for uniform space exploration",
        recommended_max_evaluations=500,
        supports_multi_objective=True,
        requires_external_library=True,
        external_library="scipy",
        default_config={
            "scramble": True
        }
    ),
    
    SamplingStrategy.ADAPTIVE: StrategyConfig(
        name="Adaptive Sampling",
        description="Learning-based sampling that adapts to promising regions",
        recommended_max_evaluations=300,
        supports_multi_objective=True,
        requires_external_library=False,
        default_config={
            "initial_strategy": "latin_hypercube",
            "exploration_ratio": 0.3,
            "learning_rate": 0.1
        }
    ),
    
    OptimizationStrategy.BAYESIAN_OPTIMIZATION: StrategyConfig(
        name="Bayesian Optimization",
        description="Gaussian Process-based optimization with acquisition functions",
        recommended_max_evaluations=150,
        supports_multi_objective=False,
        requires_external_library=True,
        external_library="scikit-optimize",
        default_config={
            "acquisition_function": "EI",
            "n_initial_points": 10,
            "base_estimator": "gp",
            "alpha": 1e-10
        }
    ),
    
    OptimizationStrategy.GENETIC_ALGORITHM: StrategyConfig(
        name="Genetic Algorithm",
        description="Evolutionary optimization inspired by natural selection",
        recommended_max_evaluations=500,
        supports_multi_objective=True,
        requires_external_library=True,
        external_library="deap",
        default_config={
            "population_size": 50,
            "mutation_prob": 0.2,
            "crossover_prob": 0.8,
            "tournament_size": 3,
            "elite_size": 2
        }
    )
}


def get_strategy_config(strategy: SamplingStrategy) -> StrategyConfig:
    """Get configuration for a sampling strategy."""
    return STRATEGY_CONFIGS.get(strategy, STRATEGY_CONFIGS[SamplingStrategy.RANDOM])


def get_optimization_config(strategy: OptimizationStrategy) -> StrategyConfig:
    """Get configuration for an optimization strategy."""
    return STRATEGY_CONFIGS.get(strategy, STRATEGY_CONFIGS[SamplingStrategy.RANDOM])


def create_dse_config_for_strategy(
    strategy: str,
    max_evaluations: int = 100,
    objectives: list = None,
    **strategy_kwargs
) -> DSEConfiguration:
    """
    Create DSE configuration for a given strategy.
    
    Args:
        strategy: Strategy name
        max_evaluations: Maximum number of evaluations
        objectives: List of objective names or DSEObjective objects
        **strategy_kwargs: Additional strategy-specific parameters
        
    Returns:
        Configured DSEConfiguration object
    """
    # Handle objectives
    if objectives is None:
        objectives = [DSEObjective("performance.throughput_ops_sec", OptimizationObjective.MAXIMIZE)]
    elif isinstance(objectives, list) and objectives:
        processed_objectives = []
        for obj in objectives:
            if isinstance(obj, str):
                # Default to maximization for string objectives
                processed_objectives.append(DSEObjective(obj, OptimizationObjective.MAXIMIZE))
            elif isinstance(obj, DSEObjective):
                processed_objectives.append(obj)
            elif isinstance(obj, dict):
                # Create from dictionary
                name = obj.get("name", "performance.throughput_ops_sec")
                direction = OptimizationObjective(obj.get("direction", "maximize"))
                weight = obj.get("weight", 1.0)
                processed_objectives.append(DSEObjective(name, direction, weight))
        objectives = processed_objectives
    
    # Get strategy config
    strategy_config = {}
    if strategy in [s.value for s in SamplingStrategy]:
        strategy_enum = SamplingStrategy(strategy)
        config_template = get_strategy_config(strategy_enum)
        strategy_config.update(config_template.default_config)
    elif strategy in [s.value for s in OptimizationStrategy]:
        strategy_enum = OptimizationStrategy(strategy)
        config_template = get_optimization_config(strategy_enum)
        strategy_config.update(config_template.default_config)
    
    # Override with user-provided parameters
    strategy_config.update(strategy_kwargs)
    
    return DSEConfiguration(
        max_evaluations=max_evaluations,
        objectives=objectives,
        strategy=strategy,
        strategy_config=strategy_config
    )


def get_recommended_strategies_for_problem(
    n_parameters: int,
    max_evaluations: int,
    n_objectives: int = 1,
    has_external_libs: bool = True
) -> list:
    """
    Recommend strategies based on problem characteristics.
    
    Args:
        n_parameters: Number of design parameters
        max_evaluations: Available evaluation budget
        n_objectives: Number of objectives (1 for single, >1 for multi)
        has_external_libs: Whether external libraries are available
        
    Returns:
        List of recommended strategy names, ordered by preference
    """
    recommendations = []
    
    # High-dimensional problems
    if n_parameters > 10:
        if has_external_libs:
            recommendations.extend(["sobol", "latin_hypercube"])
        recommendations.append("adaptive")
    
    # Medium-dimensional problems
    elif n_parameters > 5:
        if max_evaluations > 100:
            if has_external_libs and n_objectives == 1:
                recommendations.append("bayesian")
            if has_external_libs:
                recommendations.extend(["latin_hypercube", "genetic"])
        recommendations.extend(["adaptive", "random"])
    
    # Low-dimensional problems
    else:
        if max_evaluations > 50 and has_external_libs and n_objectives == 1:
            recommendations.append("bayesian")
        if has_external_libs:
            recommendations.append("latin_hypercube")
        recommendations.extend(["adaptive", "random"])
    
    # Multi-objective specific recommendations
    if n_objectives > 1:
        if has_external_libs:
            recommendations.insert(0, "genetic")  # NSGA-II style
        # Filter out single-objective only strategies
        recommendations = [s for s in recommendations if s != "bayesian"]
    
    # Large evaluation budget
    if max_evaluations > 300:
        if has_external_libs:
            recommendations.insert(0, "genetic")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_recommendations = []
    for item in recommendations:
        if item not in seen:
            seen.add(item)
            unique_recommendations.append(item)
    
    return unique_recommendations


def validate_strategy_config(strategy: str, config: Dict[str, Any]) -> tuple:
    """
    Validate strategy configuration.
    
    Args:
        strategy: Strategy name
        config: Configuration dictionary
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Get strategy template
    strategy_template = None
    if strategy in [s.value for s in SamplingStrategy]:
        strategy_enum = SamplingStrategy(strategy)
        strategy_template = get_strategy_config(strategy_enum)
    elif strategy in [s.value for s in OptimizationStrategy]:
        strategy_enum = OptimizationStrategy(strategy)
        strategy_template = get_optimization_config(strategy_enum)
    
    if not strategy_template:
        errors.append(f"Unknown strategy: {strategy}")
        return False, errors
    
    # Check external library requirement
    if strategy_template.requires_external_library:
        try:
            if strategy_template.external_library == "scipy":
                import scipy
            elif strategy_template.external_library == "scikit-optimize":
                import skopt
            elif strategy_template.external_library == "deap":
                import deap
            elif strategy_template.external_library == "optuna":
                import optuna
            elif strategy_template.external_library == "hyperopt":
                import hyperopt
        except ImportError:
            errors.append(f"Required library '{strategy_template.external_library}' not available for strategy '{strategy}'")
    
    # Strategy-specific validation
    if strategy == "bayesian" and "acquisition_function" in config:
        valid_acq = ["EI", "PI", "UCB", "LCB"]
        if config["acquisition_function"] not in valid_acq:
            errors.append(f"Invalid acquisition function: {config['acquisition_function']}. Must be one of {valid_acq}")
    
    if strategy == "genetic":
        if "population_size" in config and config["population_size"] < 10:
            errors.append("Genetic algorithm population size should be at least 10")
        if "mutation_prob" in config and not 0 <= config["mutation_prob"] <= 1:
            errors.append("Mutation probability must be between 0 and 1")
        if "crossover_prob" in config and not 0 <= config["crossover_prob"] <= 1:
            errors.append("Crossover probability must be between 0 and 1")
    
    if strategy == "adaptive":
        if "exploration_ratio" in config and not 0 <= config["exploration_ratio"] <= 1:
            errors.append("Exploration ratio must be between 0 and 1")
    
    return len(errors) == 0, errors


# Strategy selection helper
class StrategySelector:
    """Helper class for automatic strategy selection."""
    
    @staticmethod
    def select_best_strategy(
        n_parameters: int,
        max_evaluations: int,
        n_objectives: int = 1,
        problem_type: str = "general",
        prefer_speed: bool = False
    ) -> str:
        """
        Select the best strategy for given problem characteristics.
        
        Args:
            n_parameters: Number of design parameters
            max_evaluations: Evaluation budget
            n_objectives: Number of objectives
            problem_type: Problem type hint ("fpga", "ml", "general")
            prefer_speed: Whether to prefer faster strategies
            
        Returns:
            Recommended strategy name
        """
        # Check external library availability
        has_external_libs = True
        try:
            import scipy
            import skopt
        except ImportError:
            has_external_libs = False
        
        recommendations = get_recommended_strategies_for_problem(
            n_parameters, max_evaluations, n_objectives, has_external_libs
        )
        
        if not recommendations:
            return "random"
        
        # Apply preferences
        if prefer_speed:
            # Prefer strategies that don't require expensive computations
            speed_order = ["random", "latin_hypercube", "sobol", "adaptive", "genetic", "bayesian"]
            for strategy in speed_order:
                if strategy in recommendations:
                    return strategy
        
        # FPGA-specific heuristics
        if problem_type == "fpga":
            # FPGA optimization often has discrete parameters and multiple objectives
            if n_objectives > 1 and "genetic" in recommendations:
                return "genetic"
            elif "adaptive" in recommendations:
                return "adaptive"
        
        # Return top recommendation
        return recommendations[0]


# Export commonly used configurations
COMMON_CONFIGS = {
    "quick_exploration": create_dse_config_for_strategy(
        "random",
        max_evaluations=20,
        objectives=["performance.throughput_ops_sec"]
    ),
    
    "balanced_optimization": create_dse_config_for_strategy(
        "adaptive",
        max_evaluations=100,
        objectives=["performance.throughput_ops_sec"]
    ),
    
    "thorough_analysis": create_dse_config_for_strategy(
        "latin_hypercube",
        max_evaluations=200,
        objectives=["performance.throughput_ops_sec"]
    ),
    
    "multi_objective_fpga": create_dse_config_for_strategy(
        "genetic",
        max_evaluations=300,
        objectives=[
            {"name": "performance.throughput_ops_sec", "direction": "maximize", "weight": 1.0},
            {"name": "performance.power_efficiency", "direction": "maximize", "weight": 0.8}
        ],
        population_size=50
    )
}