"""
Advanced DSE Integration Framework
Complete integration of multi-objective optimization, learning, and FINN workflow.
"""

import os
import sys
import time
import logging
import json
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Week 1 FINN Integration
from ...core.finn_interface import FINNInterface
from ...core.workflow import WorkflowEngine
from ...core.design_space_orchestrator import DesignSpaceOrchestrator

# Week 2 Metrics Integration
from ...metrics import MetricsManager, MetricsConfiguration, HistoricalAnalysisEngine

# Week 3 Advanced DSE Components
from .multi_objective import (
    MultiObjectiveOptimizer, NSGA2, SPEA2, MOEAD,
    ParetoArchive, ParetoSolution, HypervolumeCalculator
)
from .algorithms import (
    FPGAGeneticAlgorithm, AdaptiveSimulatedAnnealing,
    ParticleSwarmOptimizer, HybridDSEFramework, FPGADesignCandidate
)
from .objectives import (
    MetricsObjectiveFunction, ConstraintSatisfactionEngine,
    ObjectiveRegistry, ConstraintHandler, OptimizationContext
)
from .learning import (
    LearningBasedSearch, AdaptiveStrategySelector,
    SearchSpacePruner, SearchMemory
)
from .analysis import (
    DesignSpaceAnalyzer, ParetoFrontierAnalyzer,
    SolutionClusterer, SensitivityAnalyzer, DesignSpaceNavigator
)

logger = logging.getLogger(__name__)


@dataclass
class DSEResults:
    """Comprehensive DSE optimization results."""
    pareto_solutions: List[ParetoSolution]
    best_single_objective: Optional[ParetoSolution]
    optimization_history: List[Dict[str, Any]]
    frontier_analysis: Optional[Any]  # ParetoFrontierAnalysis
    design_space_analysis: Optional[Any]  # DesignSpaceCharacteristics
    learning_insights: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    execution_time: float
    total_evaluations: int
    convergence_achieved: bool


@dataclass
class DesignProblem:
    """Complete design problem specification."""
    model_path: str
    design_space: Dict[str, Any]
    objectives: List[str]
    constraints: List[str]
    device_target: str
    optimization_config: Dict[str, Any]
    time_budget: float = 3600.0  # 1 hour default
    quality_targets: Dict[str, float] = field(default_factory=dict)


@dataclass
class OptimizationConfiguration:
    """Configuration for DSE optimization."""
    algorithm: str = 'adaptive'
    population_size: int = 100
    max_generations: int = 100
    convergence_tolerance: float = 1e-6
    diversity_threshold: float = 0.1
    learning_enabled: bool = True
    parallel_evaluations: int = 4
    checkpoint_interval: int = 10
    visualization_enabled: bool = True


class MetricsIntegratedDSE:
    """Main DSE framework with Week 2 metrics and comprehensive optimization."""
    
    def __init__(self, 
                 metrics_manager: MetricsManager,
                 finn_interface: FINNInterface = None,
                 historical_engine: HistoricalAnalysisEngine = None):
        
        self.metrics_manager = metrics_manager
        self.finn_interface = finn_interface
        self.historical_engine = historical_engine
        
        # Initialize core components
        self.objective_registry = ObjectiveRegistry()
        self.constraint_engine = ConstraintSatisfactionEngine()
        self.space_analyzer = DesignSpaceAnalyzer(metrics_manager)
        self.frontier_analyzer = ParetoFrontierAnalyzer()
        self.solution_clusterer = SolutionClusterer()
        self.sensitivity_analyzer = SensitivityAnalyzer()
        
        # Learning components
        self.learning_search = LearningBasedSearch(historical_engine) if historical_engine else None
        self.strategy_selector = AdaptiveStrategySelector()
        self.search_pruner = SearchSpacePruner()
        self.search_memory = SearchMemory()
        
        # Optimization state
        self.current_archive = ParetoArchive()
        self.optimization_history = []
        self.evaluation_count = 0
        self.start_time = None
        
        # Thread safety
        self.lock = threading.Lock()
    
    def run_intelligent_dse(self, design_problem: DesignProblem,
                           config: OptimizationConfiguration = None) -> DSEResults:
        """Run comprehensive intelligent DSE optimization."""
        
        if config is None:
            config = OptimizationConfiguration()
        
        logger.info(f"Starting intelligent DSE for {design_problem.model_path}")
        self.start_time = time.time()
        
        try:
            # Phase 1: Problem Analysis and Preparation
            problem_analysis = self._analyze_design_problem(design_problem, config)
            
            # Phase 2: Learning from Historical Data
            if self.learning_search and config.learning_enabled:
                self._learn_from_history(design_problem, problem_analysis)
            
            # Phase 3: Adaptive Strategy Selection
            optimization_strategy = self._select_optimization_strategy(problem_analysis, config)
            
            # Phase 4: Design Space Pruning and Preparation
            pruned_space = self._prepare_design_space(design_problem, problem_analysis)
            
            # Phase 5: Multi-Objective Optimization
            pareto_solutions = self._run_multi_objective_optimization(
                design_problem, pruned_space, optimization_strategy, config
            )
            
            # Phase 6: Results Analysis and Learning Update
            results = self._analyze_and_finalize_results(
                design_problem, pareto_solutions, problem_analysis, config
            )
            
            logger.info(f"Intelligent DSE completed: {len(pareto_solutions)} Pareto solutions found")
            return results
            
        except Exception as e:
            logger.error(f"Intelligent DSE failed: {e}")
            # Return minimal results on failure
            execution_time = time.time() - self.start_time if self.start_time else 0.0
            return DSEResults(
                pareto_solutions=[],
                best_single_objective=None,
                optimization_history=self.optimization_history,
                frontier_analysis=None,
                design_space_analysis=None,
                learning_insights={},
                performance_metrics={'error': str(e)},
                execution_time=execution_time,
                total_evaluations=self.evaluation_count,
                convergence_achieved=False
            )
    
    def _analyze_design_problem(self, design_problem: DesignProblem,
                               config: OptimizationConfiguration) -> Dict[str, Any]:
        """Analyze design problem characteristics."""
        
        logger.info("Analyzing design problem characteristics")
        
        # Characterize design space
        space_characteristics = self.space_analyzer.characterize_space(
            design_problem.design_space,
            sample_size=min(500, config.population_size * 5)
        )
        
        # Estimate optimization effort
        optimization_effort = self.space_analyzer.estimate_optimization_effort(
            space_characteristics, 
            {'time_budget': design_problem.time_budget}
        )
        
        # Problem complexity assessment
        problem_complexity = {
            'num_parameters': len([p for p in design_problem.design_space.keys() if not p.startswith('_')]),
            'num_objectives': len(design_problem.objectives),
            'num_constraints': len(design_problem.constraints),
            'space_difficulty': space_characteristics.search_difficulty,
            'estimated_evaluations': optimization_effort['estimated_evaluations'],
            'estimated_time': optimization_effort['estimated_time_seconds']
        }
        
        return {
            'space_characteristics': space_characteristics,
            'optimization_effort': optimization_effort,
            'problem_complexity': problem_complexity,
            'recommended_algorithms': optimization_effort['recommended_algorithms'],
            'recommended_population': optimization_effort['recommended_population_size']
        }
    
    def _learn_from_history(self, design_problem: DesignProblem, 
                           problem_analysis: Dict[str, Any]):
        """Learn from historical optimization data."""
        
        logger.info("Learning from historical optimization data")
        
        try:
            # Learn patterns from historical data
            self.learning_search.learn_from_history(hours_lookback=168)  # 1 week
            
            # Update strategy performance based on historical data
            if hasattr(self.historical_engine, 'get_strategy_performance'):
                historical_performance = self.historical_engine.get_strategy_performance()
                for strategy_name, performance_data in historical_performance.items():
                    self.strategy_selector.update_strategy_performance(
                        strategy_name,
                        performance_data.get('improvement', 0.0),
                        performance_data.get('convergence_time', 0.0),
                        performance_data.get('success', False)
                    )
            
        except Exception as e:
            logger.warning(f"Historical learning failed: {e}")
    
    def _select_optimization_strategy(self, problem_analysis: Dict[str, Any],
                                    config: OptimizationConfiguration) -> Dict[str, Any]:
        """Select optimal optimization strategy."""
        
        problem_characteristics = {
            'num_parameters': problem_analysis['problem_complexity']['num_parameters'],
            'num_objectives': problem_analysis['problem_complexity']['num_objectives'],
            'num_constraints': problem_analysis['problem_complexity']['num_constraints'],
            'time_budget': problem_analysis['optimization_effort']['estimated_time_seconds'],
            'difficulty': problem_analysis['space_characteristics'].search_difficulty
        }
        
        # Select strategy adaptively
        if config.algorithm == 'adaptive':
            selected_strategy = self.strategy_selector.select_strategy(
                problem_characteristics, {'generation': 0}
            )
        else:
            selected_strategy = config.algorithm
        
        # Get strategy recommendations
        strategy_recommendations = self.strategy_selector.get_strategy_recommendations(
            problem_characteristics
        )
        
        return {
            'primary_strategy': selected_strategy,
            'alternative_strategies': [name for name, _ in strategy_recommendations[1:4]],
            'strategy_scores': dict(strategy_recommendations),
            'hybrid_enabled': problem_characteristics['num_objectives'] > 1,
            'learning_enabled': config.learning_enabled
        }
    
    def _prepare_design_space(self, design_problem: DesignProblem,
                             problem_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare and prune design space."""
        
        logger.info("Preparing and pruning design space")
        
        # Setup constraints
        constraints = []
        for constraint_name in design_problem.constraints:
            constraint_def = self.objective_registry.get_constraint(constraint_name)
            if constraint_def:
                constraints.append(constraint_def)
        
        # Prune design space using constraints and historical data
        historical_data = []
        if self.learning_search and hasattr(self.learning_search, 'learning_history'):
            historical_data = self.learning_search.learning_history
        
        pruned_space = self.search_pruner.prune_design_space(
            design_problem.design_space,
            constraints,
            historical_data
        )
        
        # Get promising regions
        promising_regions = self.search_pruner.suggest_promising_regions(
            pruned_space, historical_data
        )
        
        return {
            'original_space': design_problem.design_space,
            'pruned_space': pruned_space,
            'promising_regions': promising_regions,
            'pruning_applied': True
        }
    
    def _run_multi_objective_optimization(self, design_problem: DesignProblem,
                                        space_info: Dict[str, Any],
                                        strategy_info: Dict[str, Any],
                                        config: OptimizationConfiguration) -> List[ParetoSolution]:
        """Run multi-objective optimization with selected strategy."""
        
        logger.info(f"Running optimization with strategy: {strategy_info['primary_strategy']}")
        
        # Setup objectives
        objectives = []
        for obj_name in design_problem.objectives:
            obj_def = self.objective_registry.get_objective(obj_name)
            if obj_def:
                objectives.append(obj_def)
        
        # Setup constraints
        constraints = []
        for constraint_name in design_problem.constraints:
            constraint_def = self.objective_registry.get_constraint(constraint_name)
            if constraint_def:
                constraints.append(constraint_def)
        
        # Create metrics-based objective function
        metrics_objective = MetricsObjectiveFunction(
            self.metrics_manager, objectives, constraints
        )
        
        # Setup objective functions for optimization
        def single_objective_func(design_params: Dict[str, Any]) -> float:
            context = OptimizationContext(
                design_parameters=design_params,
                finn_model_path=design_problem.model_path,
                device_constraints={'device': design_problem.device_target}
            )
            return metrics_objective.evaluate_single_objective(design_params)
        
        def multi_objective_func(design_params: Dict[str, Any]) -> List[float]:
            context = OptimizationContext(
                design_parameters=design_params,
                finn_model_path=design_problem.model_path,
                device_constraints={'device': design_problem.device_target}
            )
            obj_values, _ = metrics_objective.evaluate(context)
            return obj_values
        
        # Select and run optimizer
        design_space = space_info['pruned_space']
        strategy = strategy_info['primary_strategy']
        
        if strategy == 'multi_objective' or len(design_problem.objectives) > 1:
            return self._run_multi_objective_optimizer(
                multi_objective_func, design_space, config, strategy_info
            )
        else:
            return self._run_single_objective_optimizer(
                single_objective_func, design_space, config, strategy_info
            )
    
    def _run_multi_objective_optimizer(self, objective_func: Callable,
                                     design_space: Dict[str, Any],
                                     config: OptimizationConfiguration,
                                     strategy_info: Dict[str, Any]) -> List[ParetoSolution]:
        """Run multi-objective optimizer."""
        
        # Setup multi-objective optimizer
        if strategy_info['primary_strategy'] == 'nsga2' or strategy_info['primary_strategy'] == 'multi_objective':
            optimizer = NSGA2(
                population_size=config.population_size,
                max_generations=config.max_generations,
                minimize_objectives=[True] * len(design_space.get('_objectives', ['obj1']))
            )
        elif strategy_info['primary_strategy'] == 'spea2':
            optimizer = SPEA2(
                population_size=config.population_size,
                archive_size=config.population_size,
                max_generations=config.max_generations
            )
        elif strategy_info['primary_strategy'] == 'moead':
            optimizer = MOEAD(
                population_size=config.population_size,
                max_generations=config.max_generations
            )
        else:
            # Default to NSGA-II
            optimizer = NSGA2(
                population_size=config.population_size,
                max_generations=config.max_generations
            )
        
        # Create wrapped objective functions
        objective_functions = [lambda params: objective_func(params)]
        
        # Run optimization
        pareto_solutions = optimizer.optimize(
            objective_functions, design_space
        )
        
        # Update optimization history
        self.optimization_history.extend(optimizer.optimization_history)
        self.evaluation_count += optimizer.evaluation_count
        
        return pareto_solutions
    
    def _run_single_objective_optimizer(self, objective_func: Callable,
                                      design_space: Dict[str, Any],
                                      config: OptimizationConfiguration,
                                      strategy_info: Dict[str, Any]) -> List[ParetoSolution]:
        """Run single-objective optimizer."""
        
        strategy = strategy_info['primary_strategy']
        
        if strategy == 'genetic_algorithm':
            optimizer = FPGAGeneticAlgorithm(
                population_size=config.population_size,
                max_generations=config.max_generations
            )
            
            def fitness_func(candidate: FPGADesignCandidate) -> float:
                return objective_func(candidate.parameters)
            
            best_individual = optimizer.evolve(fitness_func, design_space)
            best_solution = best_individual.to_pareto_solution([fitness_func(best_individual)])
            
        elif strategy == 'simulated_annealing':
            optimizer = AdaptiveSimulatedAnnealing(
                max_iterations=config.max_generations * config.population_size
            )
            
            best_params = optimizer.optimize(objective_func, design_space)
            best_obj_value = objective_func(best_params)
            best_solution = ParetoSolution(
                design_parameters=best_params,
                objective_values=[best_obj_value]
            )
            
        elif strategy == 'particle_swarm':
            optimizer = ParticleSwarmOptimizer(
                swarm_size=config.population_size,
                max_iterations=config.max_generations
            )
            
            best_params = optimizer.optimize(objective_func, design_space)
            best_obj_value = objective_func(best_params)
            best_solution = ParetoSolution(
                design_parameters=best_params,
                objective_values=[best_obj_value]
            )
            
        else:  # hybrid or adaptive
            hybrid_optimizer = HybridDSEFramework()
            best_params = hybrid_optimizer.optimize(
                objective_func, design_space, strategy='adaptive',
                max_time=config.convergence_tolerance * 3600  # Convert to seconds
            )
            best_obj_value = objective_func(best_params)
            best_solution = ParetoSolution(
                design_parameters=best_params,
                objective_values=[best_obj_value]
            )
        
        self.evaluation_count += getattr(optimizer, 'evaluation_count', config.population_size * config.max_generations)
        
        return [best_solution]
    
    def _analyze_and_finalize_results(self, design_problem: DesignProblem,
                                    pareto_solutions: List[ParetoSolution],
                                    problem_analysis: Dict[str, Any],
                                    config: OptimizationConfiguration) -> DSEResults:
        """Analyze results and create comprehensive output."""
        
        logger.info("Analyzing optimization results")
        
        execution_time = time.time() - self.start_time
        
        # Analyze Pareto frontier
        frontier_analysis = None
        if pareto_solutions:
            frontier_analysis = self.frontier_analyzer.analyze_frontier(pareto_solutions)
        
        # Find best single-objective solution
        best_single_objective = None
        if pareto_solutions:
            # Use first objective as primary for single best
            best_single_objective = max(pareto_solutions, 
                                      key=lambda sol: sol.objective_values[0] if sol.objective_values else -float('inf'))
        
        # Cluster solutions for pattern analysis
        clusters = []
        if pareto_solutions and len(pareto_solutions) >= 3:
            clusters = self.solution_clusterer.cluster_solutions(pareto_solutions)
        
        # Sensitivity analysis
        sensitivity_results = {}
        if pareto_solutions and len(pareto_solutions) >= 5:
            sensitivity_results = self.sensitivity_analyzer.analyze_sensitivity(pareto_solutions)
        
        # Learning insights
        learning_insights = {}
        if self.learning_search:
            learning_insights = {
                'patterns_learned': len(self.learning_search.learned_patterns),
                'parameter_correlations': self.learning_search.parameter_correlations,
                'exploration_rate': self.learning_search.exploration_rate,
                'memory_size': self.search_memory.size()
            }
            
            # Update learning with results
            for solution in pareto_solutions:
                success = solution.is_feasible and len(solution.objective_values) > 0
                self.learning_search.update_learning(
                    solution.design_parameters,
                    solution.objective_values,
                    success
                )
        
        # Performance metrics
        performance_metrics = {
            'total_evaluations': self.evaluation_count,
            'execution_time_seconds': execution_time,
            'evaluations_per_second': self.evaluation_count / max(1, execution_time),
            'convergence_rate': len(pareto_solutions) / max(1, self.evaluation_count),
            'pareto_efficiency': len(pareto_solutions) / max(1, config.population_size),
            'diversity_score': frontier_analysis.diversity_score if frontier_analysis else 0.0,
            'hypervolume': frontier_analysis.hypervolume if frontier_analysis else 0.0
        }
        
        # Check convergence
        convergence_achieved = (
            len(pareto_solutions) > 0 and
            execution_time < design_problem.time_budget and
            (frontier_analysis.convergence_score > 0.8 if frontier_analysis else False)
        )
        
        return DSEResults(
            pareto_solutions=pareto_solutions,
            best_single_objective=best_single_objective,
            optimization_history=self.optimization_history,
            frontier_analysis=frontier_analysis,
            design_space_analysis=problem_analysis['space_characteristics'],
            learning_insights=learning_insights,
            performance_metrics=performance_metrics,
            execution_time=execution_time,
            total_evaluations=self.evaluation_count,
            convergence_achieved=convergence_achieved
        )


class FINNIntegratedDSE:
    """DSE framework integrated with Week 1 FINN workflow system."""
    
    def __init__(self, 
                 finn_interface: FINNInterface,
                 workflow_engine: WorkflowEngine,
                 metrics_manager: MetricsManager,
                 design_space_orchestrator: DesignSpaceOrchestrator = None):
        
        self.finn_interface = finn_interface
        self.workflow_engine = workflow_engine
        self.metrics_manager = metrics_manager
        self.design_space_orchestrator = design_space_orchestrator
        
        # Initialize metrics-integrated DSE
        historical_engine = getattr(metrics_manager, 'historical_engine', None)
        self.metrics_dse = MetricsIntegratedDSE(
            metrics_manager, finn_interface, historical_engine
        )
        
        # FINN-specific configuration
        self.finn_transformations = self._get_finn_transformations()
        self.device_targets = self._get_supported_devices()
    
    def optimize_finn_design(self, model_path: str, 
                           optimization_objectives: List[str],
                           device_target: str = 'xczu7ev',
                           config: OptimizationConfiguration = None) -> DSEResults:
        """Optimize FINN design with full workflow integration."""
        
        logger.info(f"Starting FINN-integrated DSE for {model_path}")
        
        # Create design problem with FINN-specific parameters
        design_problem = self._create_finn_design_problem(
            model_path, optimization_objectives, device_target, config
        )
        
        # Setup FINN-specific objective functions
        self._setup_finn_objectives(design_problem)
        
        # Run optimization
        results = self.metrics_dse.run_intelligent_dse(design_problem, config)
        
        # Post-process results with FINN builds
        enhanced_results = self._enhance_results_with_finn_builds(results, design_problem)
        
        logger.info(f"FINN-integrated DSE completed with {len(results.pareto_solutions)} solutions")
        return enhanced_results
    
    def _create_finn_design_problem(self, model_path: str,
                                   objectives: List[str],
                                   device_target: str,
                                   config: OptimizationConfiguration = None) -> DesignProblem:
        """Create FINN-specific design problem."""
        
        if config is None:
            config = OptimizationConfiguration()
        
        # Define FINN design space
        design_space = self._define_finn_design_space(device_target)
        
        # Define FINN constraints
        constraints = self._define_finn_constraints(device_target)
        
        return DesignProblem(
            model_path=model_path,
            design_space=design_space,
            objectives=objectives,
            constraints=constraints,
            device_target=device_target,
            optimization_config=config.__dict__,
            time_budget=config.convergence_tolerance * 3600 if hasattr(config, 'convergence_tolerance') else 3600.0
        )
    
    def _define_finn_design_space(self, device_target: str) -> Dict[str, Any]:
        """Define FINN-specific design space parameters."""
        
        base_space = {
            # Processing Element Configuration
            'pe_parallelism': (1, 64),
            'simd_parallelism': (1, 32),
            'memory_width': [32, 64, 128, 256],
            'pipeline_depth': (1, 8),
            
            # Quantization Parameters
            'weight_precision': [2, 4, 8, 16],
            'activation_precision': [2, 4, 8, 16],
            'accumulator_precision': [16, 24, 32],
            
            # Memory Configuration
            'buffer_depth': (32, 2048),
            'memory_mode': ['internal', 'external', 'hybrid'],
            'dma_width': [64, 128, 256, 512],
            
            # Clock and Timing
            'clock_frequency_mhz': (50.0, 300.0),
            'timing_constraints': ['relaxed', 'balanced', 'tight'],
            
            # FINN Transformation Sequence
            'transformation_sequence': self.finn_transformations,
            'folding_strategy': ['minimal', 'balanced', 'aggressive'],
            'resource_sharing': [True, False]
        }
        
        # Device-specific adjustments
        if 'zu' in device_target.lower():  # Zynq UltraScale+
            base_space['clock_frequency_mhz'] = (50.0, 250.0)
            base_space['pe_parallelism'] = (1, 32)
        elif 'kintex' in device_target.lower():
            base_space['clock_frequency_mhz'] = (50.0, 350.0)
            base_space['pe_parallelism'] = (1, 128)
        
        return base_space
    
    def _define_finn_constraints(self, device_target: str) -> List[str]:
        """Define FINN-specific constraints."""
        
        constraints = [
            'lut_budget',      # LUT utilization constraint
            'dsp_budget',      # DSP utilization constraint
            'bram_budget',     # BRAM utilization constraint
            'power_budget',    # Power consumption constraint
            'timing_closure',  # Timing constraints
            'min_accuracy'     # Minimum accuracy requirement
        ]
        
        return constraints
    
    def _setup_finn_objectives(self, design_problem: DesignProblem):
        """Setup FINN-specific optimization objectives."""
        
        # Register FINN-specific objectives if not already present
        finn_objectives = {
            'maximize_throughput_ops': self.metrics_dse.objective_registry.get_objective('maximize_throughput'),
            'minimize_latency_cycles': self.metrics_dse.objective_registry.get_objective('minimize_latency'),
            'minimize_power_mw': self.metrics_dse.objective_registry.get_objective('minimize_power'),
            'maximize_efficiency_ops_per_lut': self.metrics_dse.objective_registry.get_objective('maximize_efficiency'),
            'minimize_resources': self.metrics_dse.objective_registry.get_objective('minimize_area'),
            'maximize_accuracy': self.metrics_dse.objective_registry.get_objective('maximize_accuracy')
        }
        
        # Verify all objectives are available
        for obj_name in design_problem.objectives:
            if obj_name not in finn_objectives or finn_objectives[obj_name] is None:
                logger.warning(f"Objective {obj_name} not found in registry")
    
    def _enhance_results_with_finn_builds(self, results: DSEResults, 
                                         design_problem: DesignProblem) -> DSEResults:
        """Enhance results with actual FINN builds for top solutions."""
        
        if not results.pareto_solutions:
            return results
        
        logger.info("Enhancing results with FINN builds")
        
        # Select top solutions for actual FINN builds
        top_solutions = results.pareto_solutions[:min(5, len(results.pareto_solutions))]
        
        enhanced_solutions = []
        
        for solution in top_solutions:
            try:
                # Create FINN build configuration
                build_config = self._create_finn_build_config(
                    solution.design_parameters, design_problem
                )
                
                # Run FINN workflow
                finn_result = self._run_finn_workflow(
                    design_problem.model_path, build_config
                )
                
                # Update solution with actual FINN results
                if finn_result['success']:
                    enhanced_solution = self._update_solution_with_finn_result(
                        solution, finn_result
                    )
                    enhanced_solutions.append(enhanced_solution)
                else:
                    enhanced_solutions.append(solution)
                    
            except Exception as e:
                logger.error(f"FINN build failed for solution: {e}")
                enhanced_solutions.append(solution)
        
        # Update results with enhanced solutions
        results.pareto_solutions = enhanced_solutions + results.pareto_solutions[len(enhanced_solutions):]
        
        return results
    
    def _create_finn_build_config(self, design_parameters: Dict[str, Any],
                                 design_problem: DesignProblem) -> Dict[str, Any]:
        """Create FINN build configuration from design parameters."""
        
        return {
            'model_path': design_problem.model_path,
            'device_target': design_problem.device_target,
            'pe_parallelism': design_parameters.get('pe_parallelism', 4),
            'simd_parallelism': design_parameters.get('simd_parallelism', 4),
            'weight_precision': design_parameters.get('weight_precision', 8),
            'activation_precision': design_parameters.get('activation_precision', 8),
            'clock_frequency_mhz': design_parameters.get('clock_frequency_mhz', 100.0),
            'folding_strategy': design_parameters.get('folding_strategy', 'balanced'),
            'memory_mode': design_parameters.get('memory_mode', 'internal'),
            'transformation_sequence': design_parameters.get('transformation_sequence', []),
            'build_options': {
                'enable_profiling': True,
                'generate_reports': True,
                'verify_correctness': True
            }
        }
    
    def _run_finn_workflow(self, model_path: str, build_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run FINN workflow with given configuration."""
        
        try:
            # Use workflow engine to run FINN transformations
            workflow_result = self.workflow_engine.run_workflow(
                model_path=model_path,
                transformations=build_config.get('transformation_sequence', []),
                build_config=build_config
            )
            
            return {
                'success': workflow_result.success,
                'build_result': workflow_result.result_data,
                'metrics': workflow_result.metrics,
                'artifacts': workflow_result.artifacts
            }
            
        except Exception as e:
            logger.error(f"FINN workflow execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'build_result': None,
                'metrics': {},
                'artifacts': []
            }
    
    def _update_solution_with_finn_result(self, solution: ParetoSolution,
                                        finn_result: Dict[str, Any]) -> ParetoSolution:
        """Update solution with actual FINN build results."""
        
        # Extract actual metrics from FINN build
        build_metrics = finn_result.get('metrics', {})
        
        # Update objective values with actual measurements
        if 'performance' in build_metrics:
            perf_metrics = build_metrics['performance']
            if 'throughput_ops_per_sec' in perf_metrics:
                solution.objective_values[0] = perf_metrics['throughput_ops_per_sec']
        
        # Add FINN-specific metadata
        solution.metadata.update({
            'finn_build_success': True,
            'actual_finn_metrics': build_metrics,
            'build_artifacts': finn_result.get('artifacts', []),
            'verification_passed': build_metrics.get('verification', {}).get('passed', False)
        })
        
        return solution
    
    def _get_finn_transformations(self) -> List[str]:
        """Get available FINN transformation sequence."""
        
        return [
            'ConvertONNXToFINN',
            'CreateDataflowPartition',
            'GiveUniqueNodeNames',
            'GiveReadableTensorNames',
            'InferShapes',
            'FoldConstants',
            'InsertTopK',
            'InsertIODMA',
            'AnnotateCycles',
            'CodeGen_cppsim',
            'Compile_cppsim',
            'Set_exec_mode',
            'Execute_cppsim',
            'CreateStitchedIP'
        ]
    
    def _get_supported_devices(self) -> List[str]:
        """Get supported FPGA devices."""
        
        return [
            'xczu7ev',  # Zynq UltraScale+ ZCU104
            'xczu9eg',  # Zynq UltraScale+ ZCU102
            'xczu3eg',  # Zynq UltraScale+ ZCU106
            'xc7z020',  # Zynq-7000 PYNQ-Z1
            'xc7z045'   # Zynq-7000 ZC706
        ]


def create_integrated_dse_system(finn_interface: FINNInterface,
                                workflow_engine: WorkflowEngine,
                                metrics_manager: MetricsManager,
                                config_path: Optional[str] = None) -> FINNIntegratedDSE:
    """Factory function to create complete integrated DSE system."""
    
    logger.info("Creating integrated DSE system")
    
    # Setup design space orchestrator if available
    design_space_orchestrator = None
    try:
        from ...core.design_space_orchestrator import DesignSpaceOrchestrator
        design_space_orchestrator = DesignSpaceOrchestrator(
            finn_interface, metrics_manager
        )
    except ImportError:
        logger.warning("Design space orchestrator not available")
    
    # Create integrated DSE system
    dse_system = FINNIntegratedDSE(
        finn_interface=finn_interface,
        workflow_engine=workflow_engine,
        metrics_manager=metrics_manager,
        design_space_orchestrator=design_space_orchestrator
    )
    
    logger.info("Integrated DSE system created successfully")
    return dse_system


# Convenience function for quick DSE execution
def run_quick_dse(model_path: str,
                 objectives: List[str] = None,
                 device_target: str = 'xczu7ev',
                 time_budget: float = 1800.0) -> DSEResults:
    """Run quick DSE optimization with default configuration."""
    
    if objectives is None:
        objectives = ['maximize_throughput_ops', 'minimize_power_mw']
    
    # Create minimal configuration
    config = OptimizationConfiguration(
        algorithm='adaptive',
        population_size=50,
        max_generations=50,
        learning_enabled=True,
        parallel_evaluations=2
    )
    
    try:
        # Initialize minimal system components
        from ...core.finn_interface import FINNInterface
        from ...core.workflow import WorkflowEngine
        from ...metrics import MetricsManager, MetricsConfiguration
        
        finn_interface = FINNInterface()
        workflow_engine = WorkflowEngine(finn_interface)
        metrics_config = MetricsConfiguration()
        metrics_manager = MetricsManager(metrics_config)
        
        # Create DSE system
        dse_system = create_integrated_dse_system(
            finn_interface, workflow_engine, metrics_manager
        )
        
        # Run optimization
        results = dse_system.optimize_finn_design(
            model_path=model_path,
            optimization_objectives=objectives,
            device_target=device_target,
            config=config
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Quick DSE failed: {e}")
        return DSEResults(
            pareto_solutions=[],
            best_single_objective=None,
            optimization_history=[],
            frontier_analysis=None,
            design_space_analysis=None,
            learning_insights={},
            performance_metrics={'error': str(e)},
            execution_time=0.0,
            total_evaluations=0,
            convergence_achieved=False
        )