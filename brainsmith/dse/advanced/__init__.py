"""
Advanced Design Space Exploration (DSE) Framework

This module provides comprehensive multi-objective optimization, learning-based search,
and intelligent analysis capabilities for FPGA design optimization.

Key Features:
- Multi-objective Pareto optimization (NSGA-II, SPEA2, MOEA/D)
- FPGA-specific genetic algorithms and hybrid optimization
- Metrics-driven objective functions with constraint satisfaction
- Learning-based search with historical pattern recognition
- Comprehensive solution space analysis and visualization
- Full integration with Week 1 FINN workflows and Week 2 metrics

Main Components:
1. Multi-Objective Optimization: ParetoArchive, NSGA2, SPEA2, MOEAD
2. FPGA Algorithms: FPGAGeneticAlgorithm, AdaptiveSimulatedAnnealing, PSO
3. Objective Functions: MetricsObjectiveFunction, ConstraintSatisfactionEngine
4. Learning Systems: LearningBasedSearch, AdaptiveStrategySelector
5. Analysis Tools: DesignSpaceAnalyzer, ParetoFrontierAnalyzer
6. Integration: MetricsIntegratedDSE, FINNIntegratedDSE

Example Usage:
    # Quick DSE with default configuration
    from brainsmith.dse.advanced import run_quick_dse
    
    results = run_quick_dse(
        model_path="path/to/model.onnx",
        objectives=['maximize_throughput_ops', 'minimize_power_mw'],
        device_target='xczu7ev'
    )
    
    # Advanced DSE with full configuration
    from brainsmith.dse.advanced import create_integrated_dse_system, OptimizationConfiguration
    
    config = OptimizationConfiguration(
        algorithm='adaptive',
        population_size=100,
        learning_enabled=True
    )
    
    dse_system = create_integrated_dse_system(
        finn_interface, workflow_engine, metrics_manager
    )
    
    results = dse_system.optimize_finn_design(
        model_path="path/to/model.onnx",
        optimization_objectives=['maximize_throughput_ops', 'minimize_latency_cycles'],
        config=config
    )
    
    # Access results
    best_solution = results.best_single_objective
    pareto_solutions = results.pareto_solutions
    analysis = results.frontier_analysis
"""

import logging
from typing import List, Dict, Any, Optional

# Core multi-objective optimization
from .multi_objective import (
    # Data structures
    ParetoSolution,
    ParetoArchive,
    
    # Base classes
    MultiObjectiveOptimizer,
    
    # Algorithms
    NSGA2,
    SPEA2,
    MOEAD,
    
    # Utilities
    HypervolumeCalculator,
    ParetoRanking,
    CrowdingDistance,
    
    # Operators
    UniformCrossover,
    PolynomialMutation,
    TournamentSelection,
    BinaryTournamentSelection
)

# FPGA-specific algorithms
from .algorithms import (
    # FPGA Design Candidate
    FPGADesignCandidate,
    
    # Genetic operators
    FPGAGeneticOperators,
    
    # Algorithms
    FPGAGeneticAlgorithm,
    AdaptiveSimulatedAnnealing,
    ParticleSwarmOptimizer,
    HybridDSEFramework
)

# Objective functions and constraints
from .objectives import (
    # Data structures
    ObjectiveDefinition,
    ConstraintDefinition,
    OptimizationContext,
    
    # Main classes
    MetricsObjectiveFunction,
    ConstraintSatisfactionEngine,
    ObjectiveRegistry,
    ConstraintHandler
)

# Learning and adaptive strategies
from .learning import (
    # Data structures
    SearchPattern,
    StrategyPerformance,
    
    # Main classes
    SearchMemory,
    LearningBasedSearch,
    AdaptiveStrategySelector,
    SearchSpacePruner
)

# Analysis and visualization
from .analysis import (
    # Data structures
    DesignSpaceCharacteristics,
    ParetoFrontierAnalysis,
    SolutionCluster,
    
    # Main classes
    DesignSpaceAnalyzer,
    ParetoFrontierAnalyzer,
    SolutionClusterer,
    SensitivityAnalyzer,
    DesignSpaceNavigator
)

# Integration framework
from .integration import (
    # Data structures
    DSEResults,
    DesignProblem,
    OptimizationConfiguration,
    
    # Main classes
    MetricsIntegratedDSE,
    FINNIntegratedDSE,
    
    # Factory functions
    create_integrated_dse_system,
    run_quick_dse
)

# Setup logging
logger = logging.getLogger(__name__)

# Version information
__version__ = "1.0.0"
__author__ = "BrainSmith Development Team"

# API convenience functions
def create_dse_configuration(algorithm: str = 'adaptive',
                           population_size: int = 100,
                           max_generations: int = 100,
                           learning_enabled: bool = True,
                           parallel_evaluations: int = 4) -> OptimizationConfiguration:
    """
    Create DSE optimization configuration with reasonable defaults.
    
    Args:
        algorithm: Optimization algorithm ('adaptive', 'nsga2', 'genetic_algorithm', etc.)
        population_size: Population size for evolutionary algorithms
        max_generations: Maximum number of generations
        learning_enabled: Enable learning-based search enhancements
        parallel_evaluations: Number of parallel objective evaluations
    
    Returns:
        OptimizationConfiguration: Configuration object for DSE
    """
    return OptimizationConfiguration(
        algorithm=algorithm,
        population_size=population_size,
        max_generations=max_generations,
        learning_enabled=learning_enabled,
        parallel_evaluations=parallel_evaluations,
        convergence_tolerance=1e-6,
        diversity_threshold=0.1,
        checkpoint_interval=10,
        visualization_enabled=True
    )


def create_design_problem(model_path: str,
                         objectives: List[str],
                         device_target: str = 'xczu7ev',
                         constraints: List[str] = None,
                         time_budget: float = 3600.0,
                         custom_design_space: Dict[str, Any] = None) -> DesignProblem:
    """
    Create design problem specification for FPGA optimization.
    
    Args:
        model_path: Path to ONNX model
        objectives: List of optimization objectives
        device_target: Target FPGA device
        constraints: List of constraints to enforce
        time_budget: Time budget in seconds
        custom_design_space: Custom design space parameters
    
    Returns:
        DesignProblem: Complete design problem specification
    """
    if constraints is None:
        constraints = ['lut_budget', 'dsp_budget', 'power_budget', 'timing_closure']
    
    # Use provided design space or create default
    if custom_design_space is None:
        design_space = {
            'pe_parallelism': (1, 32),
            'simd_parallelism': (1, 16),
            'weight_precision': [4, 8, 16],
            'activation_precision': [4, 8, 16],
            'clock_frequency_mhz': (50.0, 200.0),
            'memory_mode': ['internal', 'external'],
            'folding_strategy': ['minimal', 'balanced', 'aggressive']
        }
    else:
        design_space = custom_design_space
    
    return DesignProblem(
        model_path=model_path,
        design_space=design_space,
        objectives=objectives,
        constraints=constraints,
        device_target=device_target,
        optimization_config={},
        time_budget=time_budget
    )


def analyze_pareto_frontier(solutions: List[ParetoSolution]) -> ParetoFrontierAnalysis:
    """
    Analyze Pareto frontier characteristics.
    
    Args:
        solutions: List of Pareto solutions
    
    Returns:
        ParetoFrontierAnalysis: Comprehensive frontier analysis
    """
    analyzer = ParetoFrontierAnalyzer()
    return analyzer.analyze_frontier(solutions)


def cluster_solutions(solutions: List[ParetoSolution], 
                     n_clusters: int = 5) -> List[SolutionCluster]:
    """
    Cluster solutions for pattern analysis.
    
    Args:
        solutions: List of solutions to cluster
        n_clusters: Number of clusters to create
    
    Returns:
        List[SolutionCluster]: List of solution clusters
    """
    clusterer = SolutionClusterer(n_clusters=n_clusters)
    return clusterer.cluster_solutions(solutions)


def analyze_parameter_sensitivity(solutions: List[ParetoSolution]) -> Dict[str, Dict[str, float]]:
    """
    Analyze parameter sensitivity for solutions.
    
    Args:
        solutions: List of solutions for sensitivity analysis
    
    Returns:
        Dict[str, Dict[str, float]]: Parameter sensitivity results
    """
    analyzer = SensitivityAnalyzer()
    return analyzer.analyze_sensitivity(solutions)


# Pre-defined optimization configurations
QUICK_DSE_CONFIG = OptimizationConfiguration(
    algorithm='adaptive',
    population_size=50,
    max_generations=50,
    learning_enabled=True,
    parallel_evaluations=2,
    convergence_tolerance=1e-4
)

THOROUGH_DSE_CONFIG = OptimizationConfiguration(
    algorithm='adaptive',
    population_size=200,
    max_generations=200,
    learning_enabled=True,
    parallel_evaluations=8,
    convergence_tolerance=1e-6
)

FAST_DSE_CONFIG = OptimizationConfiguration(
    algorithm='particle_swarm',
    population_size=30,
    max_generations=30,
    learning_enabled=False,
    parallel_evaluations=1,
    convergence_tolerance=1e-3
)

# Common objective sets
PERFORMANCE_OBJECTIVES = [
    'maximize_throughput_ops',
    'minimize_latency_cycles'
]

EFFICIENCY_OBJECTIVES = [
    'maximize_throughput_ops',
    'minimize_power_mw',
    'maximize_efficiency_ops_per_lut'
]

RESOURCE_OBJECTIVES = [
    'minimize_resources',
    'minimize_power_mw',
    'maximize_accuracy'
]

BALANCED_OBJECTIVES = [
    'maximize_throughput_ops',
    'minimize_power_mw',
    'minimize_latency_cycles',
    'maximize_accuracy'
]

# Common constraint sets
STANDARD_CONSTRAINTS = [
    'lut_budget',
    'dsp_budget',
    'bram_budget',
    'power_budget',
    'timing_closure'
]

STRICT_CONSTRAINTS = [
    'lut_budget',
    'dsp_budget',
    'bram_budget',
    'uram_budget',
    'power_budget',
    'timing_closure',
    'min_accuracy',
    'min_throughput'
]

RELAXED_CONSTRAINTS = [
    'lut_budget',
    'power_budget',
    'timing_closure'
]

# Export all public components
__all__ = [
    # Core multi-objective
    'ParetoSolution',
    'ParetoArchive',
    'MultiObjectiveOptimizer',
    'NSGA2',
    'SPEA2',
    'MOEAD',
    'HypervolumeCalculator',
    
    # FPGA algorithms
    'FPGADesignCandidate',
    'FPGAGeneticOperators',
    'FPGAGeneticAlgorithm',
    'AdaptiveSimulatedAnnealing',
    'ParticleSwarmOptimizer',
    'HybridDSEFramework',
    
    # Objectives and constraints
    'ObjectiveDefinition',
    'ConstraintDefinition',
    'OptimizationContext',
    'MetricsObjectiveFunction',
    'ConstraintSatisfactionEngine',
    'ObjectiveRegistry',
    'ConstraintHandler',
    
    # Learning and adaptation
    'SearchPattern',
    'StrategyPerformance',
    'SearchMemory',
    'LearningBasedSearch',
    'AdaptiveStrategySelector',
    'SearchSpacePruner',
    
    # Analysis
    'DesignSpaceCharacteristics',
    'ParetoFrontierAnalysis',
    'SolutionCluster',
    'DesignSpaceAnalyzer',
    'ParetoFrontierAnalyzer',
    'SolutionClusterer',
    'SensitivityAnalyzer',
    'DesignSpaceNavigator',
    
    # Integration
    'DSEResults',
    'DesignProblem',
    'OptimizationConfiguration',
    'MetricsIntegratedDSE',
    'FINNIntegratedDSE',
    'create_integrated_dse_system',
    'run_quick_dse',
    
    # Convenience functions
    'create_dse_configuration',
    'create_design_problem',
    'analyze_pareto_frontier',
    'cluster_solutions',
    'analyze_parameter_sensitivity',
    
    # Pre-defined configurations
    'QUICK_DSE_CONFIG',
    'THOROUGH_DSE_CONFIG',
    'FAST_DSE_CONFIG',
    
    # Pre-defined objective sets
    'PERFORMANCE_OBJECTIVES',
    'EFFICIENCY_OBJECTIVES',
    'RESOURCE_OBJECTIVES',
    'BALANCED_OBJECTIVES',
    
    # Pre-defined constraint sets
    'STANDARD_CONSTRAINTS',
    'STRICT_CONSTRAINTS',
    'RELAXED_CONSTRAINTS'
]

# Module initialization
logger.info(f"Advanced DSE Framework v{__version__} initialized")
logger.info(f"Available algorithms: NSGA-II, SPEA2, MOEA/D, GA, SA, PSO, Hybrid")
logger.info(f"Learning-based search and metrics integration enabled")