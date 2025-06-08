"""
SimpleDSEEngine implementation with advanced sampling and optimization strategies.

This module provides built-in DSE capabilities including random sampling,
Latin Hypercube sampling, and adaptive exploration strategies.
"""

import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

from .interface import DSEEngine, DSEConfiguration, OptimizationObjective
from ..core.design_space import DesignSpace, DesignPoint
from ..core.result import BrainsmithResult

try:
    from scipy.stats import qmc
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy not available - falling back to basic sampling")


class SimpleDSEEngine(DSEEngine):
    """
    Advanced DSE engine with multiple sampling and optimization strategies.
    
    Supports:
    - Random sampling
    - Latin Hypercube sampling
    - Sobol sequences
    - Adaptive sampling based on previous results
    - Multi-objective optimization with Pareto analysis
    """
    
    def __init__(self, strategy: str, config: DSEConfiguration):
        super().__init__(f"SimpleDSE_{strategy}", config)
        self.strategy = strategy
        self.design_space: Optional[DesignSpace] = None
        
        # Sampling state
        self.sampler = None
        self.sample_index = 0
        self.pregenerated_samples: List[DesignPoint] = []
        
        # Adaptive learning
        self.parameter_correlations: Dict[str, float] = {}
        self.parameter_importance: Dict[str, float] = {}
        self.convergence_history: List[float] = []
        
        # Set random seed
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)
    
    def initialize(self, design_space: DesignSpace):
        """Initialize engine with design space."""
        self.design_space = design_space
        self._setup_sampler()
    
    def _setup_sampler(self):
        """Setup sampling strategy based on configuration."""
        if not self.design_space:
            return
        
        n_dimensions = len(self.design_space.parameters)
        
        if self.strategy == "latin_hypercube" and SCIPY_AVAILABLE:
            self.sampler = qmc.LatinHypercube(d=n_dimensions, seed=self.config.seed)
            # Pre-generate samples for efficiency
            unit_samples = self.sampler.random(n=self.config.max_evaluations)
            self.pregenerated_samples = self._convert_unit_samples_to_points(unit_samples)
        
        elif self.strategy == "sobol" and SCIPY_AVAILABLE:
            self.sampler = qmc.Sobol(d=n_dimensions, seed=self.config.seed)
            # Pre-generate samples
            unit_samples = self.sampler.random(n=self.config.max_evaluations)
            self.pregenerated_samples = self._convert_unit_samples_to_points(unit_samples)
        
        elif self.strategy == "adaptive":
            # Adaptive sampling starts with LHS if available, otherwise random
            if SCIPY_AVAILABLE:
                self.sampler = qmc.LatinHypercube(d=n_dimensions, seed=self.config.seed)
                # Start with smaller initial sample
                initial_samples = min(10, self.config.max_evaluations // 2)
                unit_samples = self.sampler.random(n=initial_samples)
                self.pregenerated_samples = self._convert_unit_samples_to_points(unit_samples)
            else:
                # Fallback to random for initial exploration
                pass
    
    def _convert_unit_samples_to_points(self, unit_samples: np.ndarray) -> List[DesignPoint]:
        """Convert unit hypercube samples to design points."""
        if not self.design_space:
            return []
        
        design_points = []
        param_names = list(self.design_space.parameters.keys())
        
        for sample in unit_samples:
            point = DesignPoint()
            for i, param_name in enumerate(param_names):
                param_def = self.design_space.parameters[param_name]
                value = param_def.sample_from_unit(sample[i])
                point.set_parameter(param_name, value)
            design_points.append(point)
        
        return design_points
    
    def suggest_next_points(self, n_points: int = 1) -> List[DesignPoint]:
        """Suggest next design points based on strategy."""
        if not self.design_space:
            return []
        
        suggested_points = []
        
        for _ in range(n_points):
            if self.strategy == "random":
                point = self._suggest_random_point()
            elif self.strategy in ["latin_hypercube", "sobol"]:
                point = self._suggest_from_pregenerated()
            elif self.strategy == "adaptive":
                point = self._suggest_adaptive_point()
            else:
                # Fallback to random
                point = self._suggest_random_point()
            
            if point and not self._is_point_evaluated(point):
                suggested_points.append(point)
        
        # Filter out duplicates and already evaluated points
        unique_points = []
        seen_hashes = set()
        for point in suggested_points:
            point_hash = point.get_hash()
            if point_hash not in seen_hashes and not self._is_point_evaluated(point):
                unique_points.append(point)
                seen_hashes.add(point_hash)
        
        return unique_points[:n_points]
    
    def _suggest_random_point(self) -> DesignPoint:
        """Generate random design point."""
        return self.design_space.sample_random_point()
    
    def _suggest_from_pregenerated(self) -> Optional[DesignPoint]:
        """Get next point from pre-generated samples."""
        if self.sample_index < len(self.pregenerated_samples):
            point = self.pregenerated_samples[self.sample_index]
            self.sample_index += 1
            return point
        else:
            # Fallback to random if we run out of pre-generated samples
            return self._suggest_random_point()
    
    def _suggest_adaptive_point(self) -> DesignPoint:
        """
        Suggest point using adaptive strategy based on evaluation history.
        
        This implements a simple adaptive strategy that:
        1. Uses pre-generated samples for initial exploration
        2. Focuses on promising regions based on results
        3. Balances exploration vs exploitation
        """
        # If we have few evaluations, use pre-generated samples
        if len(self.evaluation_history) < 10:
            return self._suggest_from_pregenerated() or self._suggest_random_point()
        
        # Adaptive strategy based on results
        if len(self.evaluation_history) >= 5:
            # Analyze results to find promising regions
            best_points = self._get_best_points(n=3)
            
            if best_points and random.random() < 0.7:  # 70% exploitation
                # Sample around best points
                return self._sample_around_point(random.choice(best_points))
            else:
                # 30% exploration - random sampling
                return self._suggest_random_point()
        
        return self._suggest_random_point()
    
    def _get_best_points(self, n: int = 5) -> List[DesignPoint]:
        """Get the best design points evaluated so far."""
        if not self.evaluation_history or not self.config.objectives:
            return []
        
        # Score points based on objectives
        scored_points = []
        for point, result in self.evaluation_history:
            score = self._score_result(result)
            scored_points.append((score, point))
        
        # Sort by score (higher is better)
        scored_points.sort(key=lambda x: x[0], reverse=True)
        
        return [point for _, point in scored_points[:n]]
    
    def _score_result(self, result: BrainsmithResult) -> float:
        """Score a result based on objectives."""
        total_score = 0.0
        
        for objective in self.config.objectives:
            try:
                value = objective.evaluate(result.metrics)
                # Normalize and weight
                weighted_value = value * objective.weight
                
                # Convert minimization to maximization
                if objective.direction == OptimizationObjective.MINIMIZE:
                    weighted_value = -weighted_value
                
                total_score += weighted_value
            except (AttributeError, ValueError):
                # Skip objectives that can't be evaluated
                continue
        
        return total_score
    
    def _sample_around_point(self, center_point: DesignPoint, 
                           noise_factor: float = 0.1) -> DesignPoint:
        """
        Sample a new point around a given center point.
        
        Args:
            center_point: Point to sample around
            noise_factor: Amount of noise to add (0.1 = 10% of parameter range)
        """
        new_point = DesignPoint()
        
        for param_name, param_def in self.design_space.parameters.items():
            center_value = center_point.parameters.get(param_name)
            if center_value is None:
                # Use random value if not in center point
                new_value = param_def.sample_random()
            else:
                # Add noise around center value
                new_value = param_def.add_noise(center_value, noise_factor)
            
            new_point.set_parameter(param_name, new_value)
        
        return new_point
    
    def update_with_result(self, design_point: DesignPoint, result: BrainsmithResult):
        """Update engine with evaluation result and learn from it."""
        super().update_with_result(design_point, result)
        
        # Update learning for adaptive strategy
        if self.strategy == "adaptive":
            self._update_parameter_learning(design_point, result)
            self._update_convergence_tracking(result)
    
    def _update_parameter_learning(self, design_point: DesignPoint, result: BrainsmithResult):
        """Update parameter importance and correlation learning."""
        if len(self.evaluation_history) < 5:
            return  # Not enough data for learning
        
        # Simple parameter importance based on correlation with objectives
        result_score = self._score_result(result)
        
        for param_name, param_value in design_point.parameters.items():
            # Update importance based on how this parameter correlates with good results
            if isinstance(param_value, (int, float)):
                # For numeric parameters, track correlation
                if param_name not in self.parameter_importance:
                    self.parameter_importance[param_name] = 0.0
                
                # Simple learning rate
                learning_rate = 0.1
                current_importance = self.parameter_importance[param_name]
                
                # Update importance (simplified - in real implementation would use proper correlation)
                self.parameter_importance[param_name] = (
                    (1 - learning_rate) * current_importance + 
                    learning_rate * abs(result_score)
                )
    
    def _update_convergence_tracking(self, result: BrainsmithResult):
        """Track convergence progress."""
        result_score = self._score_result(result)
        self.convergence_history.append(result_score)
        
        # Keep only recent history
        if len(self.convergence_history) > 20:
            self.convergence_history = self.convergence_history[-20:]
    
    def _should_stop(self) -> bool:
        """Check if optimization should stop based on convergence."""
        if super()._should_stop():
            return True
        
        # Convergence check for adaptive strategy
        if (self.strategy == "adaptive" and 
            len(self.convergence_history) >= self.config.convergence_patience):
            
            # Check if improvement has stagnated
            recent_scores = self.convergence_history[-self.config.convergence_patience:]
            if len(set(recent_scores)) == 1:  # All scores identical
                return True
            
            # Check if variance is below threshold
            if len(recent_scores) > 1:
                variance = np.var(recent_scores)
                if variance < self.config.convergence_threshold:
                    return True
        
        return False
    
    def _is_better_result(self, result: BrainsmithResult, 
                         best_results: List[BrainsmithResult]) -> bool:
        """Check if result should be added to best results."""
        if not best_results:
            return True
        
        if len(self.config.objectives) == 1:
            # Single objective comparison
            current_score = self._score_result(result)
            best_score = max(self._score_result(r) for r in best_results)
            return current_score > best_score
        else:
            # Multi-objective: add if not dominated
            current_objectives = []
            for obj in self.config.objectives:
                try:
                    value = obj.evaluate(result.metrics)
                    if obj.direction == OptimizationObjective.MINIMIZE:
                        value = -value
                    current_objectives.append(value)
                except:
                    return False
            
            # Check if dominated by any existing result
            for best_result in best_results:
                best_objectives = []
                for obj in self.config.objectives:
                    try:
                        value = obj.evaluate(best_result.metrics)
                        if obj.direction == OptimizationObjective.MINIMIZE:
                            value = -value
                        best_objectives.append(value)
                    except:
                        continue
                
                if (len(best_objectives) == len(current_objectives) and
                    self._dominates(best_objectives, current_objectives)):
                    return False
            
            return True
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about current strategy state."""
        info = {
            "strategy": self.strategy,
            "samples_used": self.sample_index,
            "total_pregenerated": len(self.pregenerated_samples),
            "evaluations_completed": len(self.evaluation_history),
            "scipy_available": SCIPY_AVAILABLE
        }
        
        if self.strategy == "adaptive":
            info.update({
                "parameter_importance": self.parameter_importance.copy(),
                "convergence_history_length": len(self.convergence_history),
                "recent_convergence": (
                    self.convergence_history[-5:] if len(self.convergence_history) >= 5 else []
                )
            })
        
        return info


# Helper functions for parameter manipulation
def add_parameter_noise(param_def, center_value: Any, noise_factor: float = 0.1) -> Any:
    """Add noise to a parameter value."""
    param_type = param_def.type.value
    
    if param_type == "continuous":
        # Add Gaussian noise
        range_size = param_def.range[1] - param_def.range[0]
        noise_std = range_size * noise_factor
        new_value = center_value + np.random.normal(0, noise_std)
        # Clamp to valid range
        return max(param_def.range[0], min(param_def.range[1], new_value))
    
    elif param_type == "integer":
        # Add integer noise
        range_size = param_def.range[1] - param_def.range[0]
        noise_range = max(1, int(range_size * noise_factor))
        noise = np.random.randint(-noise_range, noise_range + 1)
        new_value = center_value + noise
        # Clamp to valid range
        return max(param_def.range[0], min(param_def.range[1], new_value))
    
    elif param_type == "categorical":
        # Random chance to change to different category
        if random.random() < noise_factor:
            return random.choice(param_def.values)
        else:
            return center_value
    
    elif param_type == "boolean":
        # Random chance to flip
        if random.random() < noise_factor:
            return not center_value
        else:
            return center_value
    
    return center_value


# Extend ParameterDefinition with noise method (monkey patch)
def _add_noise_method():
    """Add noise method to ParameterDefinition."""
    from ..core.design_space import ParameterDefinition
    
    def add_noise(self, center_value, noise_factor=0.1):
        return add_parameter_noise(self, center_value, noise_factor)
    
    if not hasattr(ParameterDefinition, 'add_noise'):
        ParameterDefinition.add_noise = add_noise

# Apply the monkey patch
_add_noise_method()