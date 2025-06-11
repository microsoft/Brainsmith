"""
DSE Interface - Main Design Space Exploration Interface

Provides the primary interface for design space exploration functionality,
integrating the engine, helpers, and types into a cohesive API.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from .engine import parameter_sweep, batch_evaluate, find_best_result, compare_results, sample_design_space
from .helpers import (
    generate_parameter_grid, create_parameter_samples, estimate_runtime,
    validate_parameter_space, optimize_parameter_selection
)
from .types import (
    DSEConfiguration, DSEResult, DSEResults, DSEObjective, OptimizationObjective,
    SamplingStrategy, ParameterSpace, ExplorationStatistics
)
from .design_space import DesignSpace

logger = logging.getLogger(__name__)


class DSEInterface:
    """
    Main interface for design space exploration.
    
    Provides high-level methods for running DSE with different strategies
    and managing exploration results.
    """
    
    def __init__(self, config: DSEConfiguration):
        """
        Initialize DSE interface with configuration.
        
        Args:
            config: DSE configuration specifying objectives, constraints, etc.
        """
        self.config = config
        self.statistics = ExplorationStatistics()
        self.results: List[DSEResult] = []
        
        logger.info(f"DSE Interface initialized with {len(config.objectives)} objectives")
    
    def explore_design_space(
        self,
        model_path: str,
        stages: List[str] = None
    ) -> DSEResults:
        """
        Execute complete design space exploration.
        
        Args:
            model_path: Path to ONNX model
            stages: List of stages to execute (default: all stages)
            
        Returns:
            DSEResults containing all exploration results
        """
        if stages is None:
            stages = ['analysis', 'transformation', 'kernel_mapping', 'hw_optimization']
        
        logger.info(f"Starting design space exploration with stages: {stages}")
        
        # Validate parameter space
        is_valid, errors = validate_parameter_space(self.config.parameter_space)
        if not is_valid:
            raise ValueError(f"Invalid parameter space: {errors}")
        
        # Optimize sampling strategy if needed
        if self.config.sampling_strategy == SamplingStrategy.RANDOM and not self.config.parameter_space:
            strategy, n_samples = optimize_parameter_selection(
                self.config.parameter_space,
                self.config.max_evaluations,
                'auto'
            )
            logger.info(f"Auto-selected strategy: {strategy} with {n_samples} samples")
        
        # Execute parameter sweep
        results = parameter_sweep(
            model_path=model_path,
            blueprint_path=self.config.blueprint_path,
            parameters=self.config.parameter_space,
            config=self.config
        )
        
        # Store results
        self.results = results
        
        # Update statistics
        for result in results:
            objective_name = self.config.objectives[0].name if self.config.objectives else None
            self.statistics.update(result, objective_name)
        
        # Find best results
        best_result = None
        if self.config.objectives and results:
            best_result = self._find_best_result(results)
        
        # Create DSE results object
        dse_results = DSEResults(
            results=results,
            configuration=self.config,
            total_evaluations=len(results),
            successful_evaluations=sum(1 for r in results if r.build_success),
            total_time=self.statistics.total_time,
            best_result=best_result,
            convergence={'statistics': self.statistics}
        )
        
        logger.info(f"DSE complete: {len(results)} evaluations, {dse_results.get_success_rate():.1%} success rate")
        
        return dse_results
    
    def optimize_dataflow_graph(
        self,
        dataflow_graph: Any,
        stages: List[str] = None
    ) -> DSEResults:
        """
        Optimize existing dataflow graph with hardware-specific DSE.
        
        Args:
            dataflow_graph: Pre-existing dataflow graph
            stages: List of stages to execute (default: hw_optimization only)
            
        Returns:
            DSEResults containing optimization results
        """
        if stages is None:
            stages = ['hw_optimization']
        
        logger.info("Starting dataflow graph optimization")
        
        # For now, treat as regular DSE but with different stages
        # In future, this could have specialized logic for dataflow graphs
        
        # Create temporary model path (in practice, would handle dataflow graph directly)
        temp_model_path = "dataflow_graph_input"
        
        return self.explore_design_space(temp_model_path, stages)
    
    def evaluate_single_point(
        self,
        model_path: str,
        parameters: Dict[str, Any]
    ) -> DSEResult:
        """
        Evaluate a single design point.
        
        Args:
            model_path: Path to ONNX model
            parameters: Parameter values for this design point
            
        Returns:
            DSEResult for this single evaluation
        """
        logger.info(f"Evaluating single design point: {parameters}")
        
        # Use batch_evaluate with single model
        results = batch_evaluate(
            model_list=[model_path],
            blueprint_path=self.config.blueprint_path,
            parameters=parameters,
            config=self.config
        )
        
        result = results[model_path]
        
        # Update statistics
        objective_name = self.config.objectives[0].name if self.config.objectives else None
        self.statistics.update(result, objective_name)
        
        return result
    
    def get_pareto_frontier(self, results: List[DSEResult] = None) -> List[DSEResult]:
        """
        Get Pareto optimal solutions from results.
        
        Args:
            results: Results to analyze (default: use stored results)
            
        Returns:
            List of Pareto optimal results
        """
        if results is None:
            results = self.results
        
        if len(self.config.objectives) < 2:
            # Single objective - return best result
            best = self._find_best_result(results)
            return [best] if best else []
        
        # Multi-objective Pareto analysis
        pareto_points = []
        successful_results = [r for r in results if r.build_success]
        
        for candidate in successful_results:
            is_dominated = False
            
            for other in successful_results:
                if candidate == other:
                    continue
                
                # Check if other dominates candidate
                dominates = True
                better_in_any = False
                
                for objective in self.config.objectives:
                    candidate_val = candidate.get_objective_value(objective.name)
                    other_val = other.get_objective_value(objective.name)
                    
                    if candidate_val is None or other_val is None:
                        dominates = False
                        break
                    
                    if objective.direction == OptimizationObjective.MAXIMIZE:
                        if other_val > candidate_val:
                            better_in_any = True
                        elif other_val < candidate_val:
                            dominates = False
                            break
                    else:  # MINIMIZE
                        if other_val < candidate_val:
                            better_in_any = True
                        elif other_val > candidate_val:
                            dominates = False
                            break
                
                if dominates and better_in_any:
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_points.append(candidate)
        
        logger.info(f"Found {len(pareto_points)} Pareto optimal solutions from {len(successful_results)} successful results")
        return pareto_points
    
    def get_statistics(self) -> ExplorationStatistics:
        """Get exploration statistics."""
        return self.statistics
    
    def _find_best_result(self, results: List[DSEResult]) -> Optional[DSEResult]:
        """Find best result based on primary objective."""
        if not self.config.objectives or not results:
            return None
        
        primary_objective = self.config.objectives[0]
        return find_best_result(
            results,
            primary_objective.name,
            primary_objective.direction.value
        )


def create_dse_config_for_strategy(
    strategy: str,
    parameter_space: ParameterSpace,
    objectives: List[DSEObjective],
    max_evaluations: int = 50,
    **kwargs
) -> DSEConfiguration:
    """
    Create DSE configuration for a specific strategy.
    
    Args:
        strategy: Strategy name ('random', 'grid', 'lhs', 'sobol')
        parameter_space: Parameter space definition
        objectives: List of optimization objectives
        max_evaluations: Maximum number of evaluations
        **kwargs: Additional configuration options
        
    Returns:
        Configured DSEConfiguration
    """
    try:
        sampling_strategy = SamplingStrategy(strategy)
    except ValueError:
        logger.warning(f"Unknown strategy '{strategy}', using random")
        sampling_strategy = SamplingStrategy.RANDOM
    
    config = DSEConfiguration(
        sampling_strategy=sampling_strategy,
        parameter_space=parameter_space,
        objectives=objectives,
        max_evaluations=max_evaluations,
        **kwargs
    )
    
    return config


def run_simple_dse(
    model_path: str,
    blueprint_path: str,
    parameter_space: ParameterSpace,
    objectives: List[DSEObjective],
    strategy: str = 'random',
    max_evaluations: int = 50
) -> DSEResults:
    """
    Simple function interface for running DSE.
    
    Args:
        model_path: Path to ONNX model
        blueprint_path: Path to blueprint YAML
        parameter_space: Parameter space definition
        objectives: List of optimization objectives
        strategy: Sampling strategy
        max_evaluations: Maximum number of evaluations
        
    Returns:
        DSEResults containing exploration results
    """
    config = create_dse_config_for_strategy(
        strategy=strategy,
        parameter_space=parameter_space,
        objectives=objectives,
        max_evaluations=max_evaluations,
        blueprint_path=blueprint_path
    )
    
    dse_interface = DSEInterface(config)
    return dse_interface.explore_design_space(model_path)


# Utility functions for common DSE patterns

def quick_parameter_sweep(
    model_path: str,
    blueprint_path: str,
    parameters: ParameterSpace
) -> List[DSEResult]:
    """
    Quick parameter sweep with default settings.
    
    Args:
        model_path: Path to ONNX model
        blueprint_path: Path to blueprint YAML
        parameters: Parameter space to sweep
        
    Returns:
        List of DSE results
    """
    return parameter_sweep(
        model_path=model_path,
        blueprint_path=blueprint_path,
        parameters=parameters,
        config=DSEConfiguration()
    )


def find_best_throughput(results: List[DSEResult]) -> Optional[DSEResult]:
    """Find result with best throughput."""
    return find_best_result(results, 'performance.throughput_ops_sec', 'maximize')


def find_best_efficiency(results: List[DSEResult]) -> Optional[DSEResult]:
    """Find result with best resource efficiency."""
    return find_best_result(results, 'resources.lut_utilization_percent', 'minimize')