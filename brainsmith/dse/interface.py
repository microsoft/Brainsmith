"""
Core DSE interfaces and engine framework for Brainsmith platform.

This module defines the abstract interfaces for DSE engines and provides
the foundation for implementing various optimization strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time

from ..core.design_space import DesignSpace, DesignPoint
from ..core.result import DSEResult, BrainsmithResult
from ..core.metrics import BrainsmithMetrics
from ..blueprints.base import Blueprint


class OptimizationObjective(Enum):
    """Optimization objective types."""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass
class DSEObjective:
    """Definition of a DSE optimization objective."""
    name: str
    direction: OptimizationObjective
    weight: float = 1.0
    description: Optional[str] = None
    
    def evaluate(self, metrics: BrainsmithMetrics) -> float:
        """Extract objective value from metrics."""
        # Handle nested metric access (e.g., "performance.throughput_ops_sec")
        value = metrics
        for attr in self.name.split('.'):
            if hasattr(value, attr):
                value = getattr(value, attr)
            else:
                raise ValueError(f"Metric '{self.name}' not found in metrics")
        
        if not isinstance(value, (int, float)):
            raise ValueError(f"Metric '{self.name}' is not numeric: {type(value)}")
        
        return float(value)


@dataclass
class DSEConfiguration:
    """Configuration for DSE optimization."""
    max_evaluations: int = 50
    max_time_seconds: Optional[int] = None
    objectives: List[DSEObjective] = field(default_factory=list)
    strategy: str = "random"
    strategy_config: Dict[str, Any] = field(default_factory=dict)
    convergence_threshold: float = 0.001
    convergence_patience: int = 10
    seed: Optional[int] = None
    
    def __post_init__(self):
        """Validate and set defaults."""
        if not self.objectives:
            # Default to throughput maximization
            self.objectives = [
                DSEObjective("performance.throughput_ops_sec", OptimizationObjective.MAXIMIZE)
            ]


@dataclass
class DSEProgress:
    """Progress information for ongoing DSE."""
    evaluations_completed: int = 0
    evaluations_total: int = 0
    time_elapsed: float = 0.0
    time_remaining_estimate: Optional[float] = None
    best_objectives: List[float] = field(default_factory=list)
    current_pareto_size: int = 0
    convergence_score: float = 0.0
    
    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.evaluations_total == 0:
            return 0.0
        return (self.evaluations_completed / self.evaluations_total) * 100.0


class DSEInterface(ABC):
    """Abstract interface for Design Space Exploration engines."""
    
    def __init__(self, name: str, config: DSEConfiguration):
        self.name = name
        self.config = config
        self.evaluation_history: List[Tuple[DesignPoint, BrainsmithResult]] = []
        self.start_time: Optional[float] = None
        self.progress = DSEProgress()
    
    @abstractmethod
    def suggest_next_points(self, n_points: int = 1) -> List[DesignPoint]:
        """
        Suggest next design points to evaluate.
        
        Args:
            n_points: Number of points to suggest
            
        Returns:
            List of suggested design points
        """
        pass
    
    @abstractmethod
    def update_with_result(self, design_point: DesignPoint, result: BrainsmithResult):
        """
        Update engine with evaluation result.
        
        Args:
            design_point: Design point that was evaluated
            result: Result of the evaluation
        """
        pass
    
    def optimize(self, model_path: str, blueprint: Blueprint, 
                design_space: DesignSpace, build_function) -> DSEResult:
        """
        Run complete DSE optimization.
        
        Args:
            model_path: Path to model to optimize
            blueprint: Blueprint to use for builds
            design_space: Design space to explore
            build_function: Function to call for evaluating design points
            
        Returns:
            Complete DSE results
        """
        self.start_time = time.time()
        self.progress = DSEProgress(evaluations_total=self.config.max_evaluations)
        
        # Initialize result collection
        all_results = []
        best_results = []
        
        try:
            while not self._should_stop():
                # Get next points to evaluate
                next_points = self.suggest_next_points(
                    min(5, self.config.max_evaluations - len(self.evaluation_history))
                )
                
                if not next_points:
                    break
                
                # Evaluate points
                for point in next_points:
                    if self._should_stop():
                        break
                    
                    # Build and evaluate
                    result = build_function(model_path, blueprint.name, point.parameters)
                    
                    # Update engine with result
                    self.update_with_result(point, result)
                    self.evaluation_history.append((point, result))
                    all_results.append(result)
                    
                    # Update progress
                    self._update_progress()
                    
                    # Track best results
                    if self._is_better_result(result, best_results):
                        best_results.append(result)
        
        except KeyboardInterrupt:
            print("DSE interrupted by user")
        except Exception as e:
            print(f"DSE failed with error: {e}")
            raise
        
        # Create final DSE result
        end_time = time.time()
        total_time = end_time - self.start_time if self.start_time else 0.0
        
        return DSEResult(
            design_space=design_space,
            results=all_results,
            total_evaluations=len(all_results),
            total_time_seconds=total_time,
            best_result=self._get_best_single_result(all_results),
            strategy=self.name,
            strategy_config=self.config.strategy_config,
            objectives=[obj.name for obj in self.config.objectives]
        )
    
    def _should_stop(self) -> bool:
        """Check if optimization should stop."""
        # Max evaluations reached
        if len(self.evaluation_history) >= self.config.max_evaluations:
            return True
        
        # Max time reached
        if (self.config.max_time_seconds and self.start_time and 
            time.time() - self.start_time >= self.config.max_time_seconds):
            return True
        
        # Convergence check (implement in subclasses)
        return False
    
    def _update_progress(self):
        """Update progress tracking."""
        current_time = time.time()
        self.progress.evaluations_completed = len(self.evaluation_history)
        self.progress.time_elapsed = current_time - self.start_time if self.start_time else 0.0
        
        # Estimate remaining time
        if self.progress.evaluations_completed > 0:
            avg_time_per_eval = self.progress.time_elapsed / self.progress.evaluations_completed
            remaining_evals = self.config.max_evaluations - self.progress.evaluations_completed
            self.progress.time_remaining_estimate = avg_time_per_eval * remaining_evals
    
    def _is_better_result(self, result: BrainsmithResult, 
                         best_results: List[BrainsmithResult]) -> bool:
        """Check if result is better than current best (implement in subclasses)."""
        return True  # Default: keep all results
    
    def _get_best_single_result(self, results: List[BrainsmithResult]) -> BrainsmithResult:
        """Get single best result according to objectives."""
        if not results:
            return None
        
        if len(self.config.objectives) == 1:
            # Single objective optimization
            objective = self.config.objectives[0]
            best_result = results[0]
            best_value = objective.evaluate(best_result.metrics)
            
            for result in results[1:]:
                value = objective.evaluate(result.metrics)
                if ((objective.direction == OptimizationObjective.MAXIMIZE and value > best_value) or
                    (objective.direction == OptimizationObjective.MINIMIZE and value < best_value)):
                    best_result = result
                    best_value = value
            
            return best_result
        else:
            # Multi-objective: return first non-dominated result
            pareto_results = self._compute_pareto_frontier(results)
            return pareto_results[0] if pareto_results else results[0]
    
    def _compute_pareto_frontier(self, results: List[BrainsmithResult]) -> List[BrainsmithResult]:
        """Compute Pareto frontier for multi-objective optimization."""
        if not results:
            return []
        
        # Extract objective values for all results
        objective_values = []
        for result in results:
            values = []
            for objective in self.config.objectives:
                value = objective.evaluate(result.metrics)
                # Convert to maximization problem
                if objective.direction == OptimizationObjective.MINIMIZE:
                    value = -value
                values.append(value)
            objective_values.append(values)
        
        # Find non-dominated solutions
        pareto_indices = []
        for i, values_i in enumerate(objective_values):
            is_dominated = False
            for j, values_j in enumerate(objective_values):
                if i != j and self._dominates(values_j, values_i):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_indices.append(i)
        
        return [results[i] for i in pareto_indices]
    
    def _dominates(self, a: List[float], b: List[float]) -> bool:
        """Check if solution a dominates solution b (assuming maximization)."""
        return all(a_i >= b_i for a_i, b_i in zip(a, b)) and any(a_i > b_i for a_i, b_i in zip(a, b))
    
    def get_progress(self) -> DSEProgress:
        """Get current optimization progress."""
        return self.progress
    
    def reset(self):
        """Reset engine state for new optimization."""
        self.evaluation_history.clear()
        self.start_time = None
        self.progress = DSEProgress()


class DSEEngine(DSEInterface):
    """
    Base implementation of DSE engine with common functionality.
    
    This class provides a concrete base for implementing specific DSE strategies
    while handling common tasks like progress tracking and result management.
    """
    
    def __init__(self, name: str, config: DSEConfiguration):
        super().__init__(name, config)
        self.suggested_points: List[DesignPoint] = []
        self.evaluated_points: set = set()
    
    def suggest_next_points(self, n_points: int = 1) -> List[DesignPoint]:
        """Base implementation - should be overridden by subclasses."""
        return []
    
    def update_with_result(self, design_point: DesignPoint, result: BrainsmithResult):
        """Update engine with evaluation result."""
        point_hash = design_point.get_hash()
        self.evaluated_points.add(point_hash)
        
        # Remove from suggested points if present
        self.suggested_points = [p for p in self.suggested_points 
                               if p.get_hash() != point_hash]
    
    def _is_point_evaluated(self, point: DesignPoint) -> bool:
        """Check if point has already been evaluated."""
        return point.get_hash() in self.evaluated_points
    
    def _filter_unevaluated_points(self, points: List[DesignPoint]) -> List[DesignPoint]:
        """Filter out already evaluated points."""
        return [p for p in points if not self._is_point_evaluated(p)]


# Factory function for creating DSE engines
def create_dse_engine(strategy: str, config: DSEConfiguration) -> DSEInterface:
    """
    Factory function to create DSE engines.
    
    Args:
        strategy: Strategy name ("simple", "random", "bayesian", etc.)
        config: DSE configuration
        
    Returns:
        DSE engine instance
    """
    if strategy in ["simple", "random", "latin_hypercube", "adaptive"]:
        from .simple import SimpleDSEEngine
        return SimpleDSEEngine(strategy, config)
    elif strategy in ["bayesian", "genetic", "optuna", "skopt"]:
        from .external import ExternalDSEAdapter
        return ExternalDSEAdapter(strategy, config)
    else:
        raise ValueError(f"Unknown DSE strategy: {strategy}")