"""
Solution Space Analysis and Visualization
Comprehensive analysis tools for design space characterization and Pareto frontier analysis.
"""

import os
import sys
import time
import logging
import math
import json
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logger.warning("Matplotlib/Seaborn not available - plotting disabled")

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available - advanced analysis disabled")

from .multi_objective import ParetoSolution, ParetoArchive

logger = logging.getLogger(__name__)


@dataclass
class DesignSpaceCharacteristics:
    """Characteristics of a design space."""
    dimensionality: int
    parameter_types: Dict[str, str]
    parameter_ranges: Dict[str, Tuple[float, float]]
    estimated_size: float
    complexity_score: float
    landscape_smoothness: float
    constraint_density: float
    objective_correlations: Dict[str, float]
    feasible_region_ratio: float
    search_difficulty: str  # 'easy', 'medium', 'hard'


@dataclass 
class ParetoFrontierAnalysis:
    """Analysis results for a Pareto frontier."""
    num_solutions: int
    frontier_shape: str  # 'convex', 'concave', 'mixed'
    diversity_score: float
    convergence_score: float
    extreme_points: List[ParetoSolution]
    knee_points: List[ParetoSolution]
    trade_off_analysis: Dict[str, Any]
    dominated_space_volume: float
    hypervolume: float


@dataclass
class SolutionCluster:
    """Represents a cluster of similar solutions."""
    cluster_id: int
    solutions: List[ParetoSolution]
    centroid: Dict[str, float]
    characteristics: Dict[str, Any]
    cluster_quality: float


class DesignSpaceAnalyzer:
    """Comprehensive analysis of FPGA design space characteristics."""
    
    def __init__(self, metrics_manager: Any = None):
        self.metrics_manager = metrics_manager
        self.analysis_cache = {}
        self.sample_evaluations = []
        self.space_model = None
    
    def characterize_space(self, design_parameters: Dict[str, Any], 
                          objective_functions: List[Callable] = None,
                          constraints: List[Callable] = None,
                          sample_size: int = 1000) -> DesignSpaceCharacteristics:
        """Characterize design space through sampling and analysis."""
        
        logger.info(f"Characterizing design space with {sample_size} samples")
        
        # Generate representative samples
        samples = self._generate_samples(design_parameters, sample_size)
        
        # Evaluate samples
        evaluations = []
        if objective_functions:
            evaluations = self._evaluate_samples(samples, objective_functions, constraints)
        
        # Analyze characteristics
        characteristics = self._analyze_characteristics(design_parameters, samples, evaluations)
        
        # Cache results
        cache_key = self._create_cache_key(design_parameters, sample_size)
        self.analysis_cache[cache_key] = characteristics
        
        logger.info(f"Design space characterization complete: {characteristics.search_difficulty} difficulty")
        return characteristics
    
    def estimate_optimization_effort(self, characteristics: DesignSpaceCharacteristics,
                                   optimization_goals: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate optimization effort required."""
        
        # Base effort estimation
        base_effort = characteristics.complexity_score * 100
        
        # Adjust for dimensionality
        dimensionality_factor = math.log(characteristics.dimensionality + 1)
        base_effort *= dimensionality_factor
        
        # Adjust for landscape smoothness
        smoothness_factor = 2.0 - characteristics.landscape_smoothness
        base_effort *= smoothness_factor
        
        # Adjust for constraint density
        constraint_factor = 1.0 + characteristics.constraint_density
        base_effort *= constraint_factor
        
        # Estimate convergence time
        estimated_evaluations = int(base_effort)
        estimated_time_seconds = estimated_evaluations * 10  # Assume 10s per evaluation
        
        # Difficulty-based recommendations
        if characteristics.search_difficulty == 'easy':
            recommended_algorithms = ['particle_swarm', 'simulated_annealing']
            recommended_population = 50
        elif characteristics.search_difficulty == 'medium':
            recommended_algorithms = ['genetic_algorithm', 'hybrid']
            recommended_population = 100
        else:  # hard
            recommended_algorithms = ['genetic_algorithm', 'multi_objective', 'hybrid']
            recommended_population = 200
        
        return {
            'estimated_evaluations': estimated_evaluations,
            'estimated_time_seconds': estimated_time_seconds,
            'recommended_algorithms': recommended_algorithms,
            'recommended_population_size': recommended_population,
            'difficulty_factors': {
                'dimensionality': dimensionality_factor,
                'smoothness': smoothness_factor,
                'constraints': constraint_factor
            }
        }
    
    def _generate_samples(self, design_parameters: Dict[str, Any], sample_size: int) -> List[Dict[str, Any]]:
        """Generate representative samples from design space."""
        
        samples = []
        
        # Use Latin Hypercube Sampling for better coverage
        continuous_params = []
        discrete_params = []
        
        for param_name, param_range in design_parameters.items():
            if param_name.startswith('_'):
                continue
            
            if isinstance(param_range, tuple) and len(param_range) == 2:
                min_val, max_val = param_range
                if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                    continuous_params.append((param_name, min_val, max_val, isinstance(min_val, int)))
            elif isinstance(param_range, (list, tuple)):
                discrete_params.append((param_name, param_range))
        
        # Generate Latin Hypercube samples for continuous parameters
        if continuous_params:
            lhs_samples = self._latin_hypercube_sampling(len(continuous_params), sample_size)
        else:
            lhs_samples = []
        
        for i in range(sample_size):
            sample = {}
            
            # Continuous parameters
            for j, (param_name, min_val, max_val, is_int) in enumerate(continuous_params):
                if lhs_samples:
                    normalized_value = lhs_samples[i][j]
                    value = min_val + normalized_value * (max_val - min_val)
                    if is_int:
                        value = int(round(value))
                    sample[param_name] = value
            
            # Discrete parameters
            for param_name, choices in discrete_params:
                sample[param_name] = np.random.choice(choices)
            
            samples.append(sample)
        
        return samples
    
    def _latin_hypercube_sampling(self, dimensions: int, samples: int) -> np.ndarray:
        """Generate Latin Hypercube samples."""
        
        # Simple LHS implementation
        result = np.zeros((samples, dimensions))
        
        for dim in range(dimensions):
            # Generate random permutation
            perm = np.random.permutation(samples)
            
            # Generate uniform random values within each interval
            for i in range(samples):
                interval_start = perm[i] / samples
                interval_end = (perm[i] + 1) / samples
                result[i, dim] = np.random.uniform(interval_start, interval_end)
        
        return result
    
    def _evaluate_samples(self, samples: List[Dict[str, Any]], 
                         objective_functions: List[Callable],
                         constraints: List[Callable] = None) -> List[List[float]]:
        """Evaluate objective functions for samples."""
        
        evaluations = []
        
        for sample in samples:
            try:
                # Evaluate objectives
                obj_values = []
                for obj_func in objective_functions:
                    value = obj_func(sample)
                    obj_values.append(float(value))
                
                # Check constraints
                feasible = True
                if constraints:
                    for constraint in constraints:
                        violation = constraint(sample)
                        if violation > 0:
                            feasible = False
                            break
                
                # Store evaluation (mark infeasible solutions)
                if not feasible:
                    obj_values = [float('-inf')] * len(obj_values)
                
                evaluations.append(obj_values)
                
            except Exception as e:
                logger.error(f"Sample evaluation failed: {e}")
                evaluations.append([float('-inf')] * len(objective_functions))
        
        self.sample_evaluations = evaluations
        return evaluations
    
    def _analyze_characteristics(self, design_parameters: Dict[str, Any],
                               samples: List[Dict[str, Any]], 
                               evaluations: List[List[float]]) -> DesignSpaceCharacteristics:
        """Analyze design space characteristics."""
        
        # Calculate basic characteristics
        dimensionality = len([p for p in design_parameters.keys() if not p.startswith('_')])
        
        # Parameter types
        parameter_types = {}
        parameter_ranges = {}
        
        for param_name, param_range in design_parameters.items():
            if param_name.startswith('_'):
                continue
            
            if isinstance(param_range, tuple) and len(param_range) == 2:
                min_val, max_val = param_range
                if isinstance(min_val, int) and isinstance(max_val, int):
                    parameter_types[param_name] = 'integer'
                elif isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                    parameter_types[param_name] = 'continuous'
                    parameter_ranges[param_name] = (float(min_val), float(max_val))
            elif isinstance(param_range, (list, tuple)):
                parameter_types[param_name] = 'discrete'
        
        # Estimate space size
        estimated_size = 1.0
        for param_name, param_range in design_parameters.items():
            if param_name.startswith('_'):
                continue
            
            if isinstance(param_range, tuple) and len(param_range) == 2:
                min_val, max_val = param_range
                if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                    estimated_size *= (max_val - min_val + 1)
            elif isinstance(param_range, (list, tuple)):
                estimated_size *= len(param_range)
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(dimensionality, parameter_types, estimated_size)
        
        # Analyze landscape smoothness
        landscape_smoothness = self._analyze_landscape_smoothness(samples, evaluations)
        
        # Calculate constraint density
        constraint_density = self._calculate_constraint_density(evaluations)
        
        # Analyze objective correlations
        objective_correlations = self._analyze_objective_correlations(evaluations)
        
        # Calculate feasible region ratio
        feasible_region_ratio = self._calculate_feasible_ratio(evaluations)
        
        # Determine search difficulty
        search_difficulty = self._determine_search_difficulty(
            complexity_score, landscape_smoothness, constraint_density, feasible_region_ratio
        )
        
        return DesignSpaceCharacteristics(
            dimensionality=dimensionality,
            parameter_types=parameter_types,
            parameter_ranges=parameter_ranges,
            estimated_size=estimated_size,
            complexity_score=complexity_score,
            landscape_smoothness=landscape_smoothness,
            constraint_density=constraint_density,
            objective_correlations=objective_correlations,
            feasible_region_ratio=feasible_region_ratio,
            search_difficulty=search_difficulty
        )
    
    def _calculate_complexity_score(self, dimensionality: int, 
                                   parameter_types: Dict[str, str],
                                   estimated_size: float) -> float:
        """Calculate complexity score for design space."""
        
        score = 0.0
        
        # Dimensionality contribution
        score += math.log(dimensionality + 1) / math.log(2)
        
        # Parameter type contribution
        type_complexity = {
            'continuous': 1.0,
            'integer': 0.8,
            'discrete': 0.6
        }
        
        avg_type_complexity = np.mean([type_complexity.get(ptype, 1.0) for ptype in parameter_types.values()])
        score += avg_type_complexity
        
        # Size contribution
        score += math.log(estimated_size + 1) / math.log(10) * 0.5
        
        return score
    
    def _analyze_landscape_smoothness(self, samples: List[Dict[str, Any]], 
                                    evaluations: List[List[float]]) -> float:
        """Analyze smoothness of objective landscape."""
        
        if not evaluations or len(evaluations) < 10:
            return 0.5  # Default smoothness
        
        # Calculate local smoothness by examining neighboring points
        smoothness_scores = []
        
        for i in range(len(evaluations)):
            if all(v != float('-inf') for v in evaluations[i]):
                # Find nearest neighbors
                neighbors = self._find_nearest_neighbors(samples, i, k=5)
                
                if neighbors:
                    # Calculate objective variance among neighbors
                    neighbor_objectives = [evaluations[j] for j in neighbors if all(v != float('-inf') for v in evaluations[j])]
                    
                    if len(neighbor_objectives) >= 2:
                        for obj_idx in range(len(evaluations[i])):
                            neighbor_values = [obj[obj_idx] for obj in neighbor_objectives]
                            variance = np.var(neighbor_values)
                            # Convert variance to smoothness (lower variance = higher smoothness)
                            smoothness = 1.0 / (1.0 + variance)
                            smoothness_scores.append(smoothness)
        
        return float(np.mean(smoothness_scores)) if smoothness_scores else 0.5
    
    def _find_nearest_neighbors(self, samples: List[Dict[str, Any]], 
                               index: int, k: int = 5) -> List[int]:
        """Find k nearest neighbors to a sample."""
        
        target_sample = samples[index]
        distances = []
        
        for i, sample in enumerate(samples):
            if i != index:
                distance = self._calculate_sample_distance(target_sample, sample)
                distances.append((distance, i))
        
        # Sort by distance and return k nearest
        distances.sort(key=lambda x: x[0])
        return [idx for _, idx in distances[:k]]
    
    def _calculate_sample_distance(self, sample1: Dict[str, Any], sample2: Dict[str, Any]) -> float:
        """Calculate normalized distance between samples."""
        
        common_params = set(sample1.keys()) & set(sample2.keys())
        
        if not common_params:
            return float('inf')
        
        distance = 0.0
        param_count = 0
        
        for param in common_params:
            val1, val2 = sample1[param], sample2[param]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Normalize by parameter range (simplified)
                normalized_distance = abs(val1 - val2) / max(abs(val1), abs(val2), 1.0)
                distance += normalized_distance ** 2
                param_count += 1
        
        return math.sqrt(distance / max(1, param_count))
    
    def _calculate_constraint_density(self, evaluations: List[List[float]]) -> float:
        """Calculate density of constraint violations."""
        
        if not evaluations:
            return 0.0
        
        infeasible_count = sum(1 for eval_list in evaluations if any(v == float('-inf') for v in eval_list))
        return infeasible_count / len(evaluations)
    
    def _analyze_objective_correlations(self, evaluations: List[List[float]]) -> Dict[str, float]:
        """Analyze correlations between objectives."""
        
        correlations = {}
        
        if not evaluations or len(evaluations[0]) < 2:
            return correlations
        
        # Filter feasible evaluations
        feasible_evals = [eval_list for eval_list in evaluations if all(v != float('-inf') for v in eval_list)]
        
        if len(feasible_evals) < 10:
            return correlations
        
        objective_matrix = np.array(feasible_evals)
        
        for i in range(objective_matrix.shape[1]):
            for j in range(i + 1, objective_matrix.shape[1]):
                corr = np.corrcoef(objective_matrix[:, i], objective_matrix[:, j])[0, 1]
                if not np.isnan(corr):
                    correlations[f'obj_{i}_obj_{j}'] = float(corr)
        
        return correlations
    
    def _calculate_feasible_ratio(self, evaluations: List[List[float]]) -> float:
        """Calculate ratio of feasible solutions."""
        
        if not evaluations:
            return 0.0
        
        feasible_count = sum(1 for eval_list in evaluations if all(v != float('-inf') for v in eval_list))
        return feasible_count / len(evaluations)
    
    def _determine_search_difficulty(self, complexity_score: float, landscape_smoothness: float,
                                   constraint_density: float, feasible_region_ratio: float) -> str:
        """Determine overall search difficulty."""
        
        difficulty_score = 0.0
        
        # Complexity contribution
        if complexity_score > 5:
            difficulty_score += 3
        elif complexity_score > 3:
            difficulty_score += 2
        else:
            difficulty_score += 1
        
        # Smoothness contribution (lower smoothness = higher difficulty)
        if landscape_smoothness < 0.3:
            difficulty_score += 3
        elif landscape_smoothness < 0.6:
            difficulty_score += 2
        else:
            difficulty_score += 1
        
        # Constraint density contribution
        if constraint_density > 0.5:
            difficulty_score += 3
        elif constraint_density > 0.2:
            difficulty_score += 2
        else:
            difficulty_score += 1
        
        # Feasible region contribution
        if feasible_region_ratio < 0.3:
            difficulty_score += 3
        elif feasible_region_ratio < 0.6:
            difficulty_score += 2
        else:
            difficulty_score += 1
        
        # Map to difficulty levels
        if difficulty_score <= 5:
            return 'easy'
        elif difficulty_score <= 8:
            return 'medium'
        else:
            return 'hard'
    
    def _create_cache_key(self, design_parameters: Dict[str, Any], sample_size: int) -> str:
        """Create cache key for analysis results."""
        
        param_signature = str(sorted(design_parameters.items()))
        return f"{hash(param_signature)}_{sample_size}"


class ParetoFrontierAnalyzer:
    """Advanced analysis and visualization of Pareto frontiers."""
    
    def __init__(self):
        self.analysis_cache = {}
        self.visualization_cache = {}
    
    def analyze_frontier(self, pareto_solutions: List[ParetoSolution]) -> ParetoFrontierAnalysis:
        """Comprehensive analysis of Pareto frontier."""
        
        if not pareto_solutions:
            return ParetoFrontierAnalysis(
                num_solutions=0,
                frontier_shape='empty',
                diversity_score=0.0,
                convergence_score=0.0,
                extreme_points=[],
                knee_points=[],
                trade_off_analysis={},
                dominated_space_volume=0.0,
                hypervolume=0.0
            )
        
        logger.info(f"Analyzing Pareto frontier with {len(pareto_solutions)} solutions")
        
        # Analyze frontier shape
        frontier_shape = self._analyze_frontier_shape(pareto_solutions)
        
        # Calculate diversity score
        diversity_score = self._calculate_diversity_score(pareto_solutions)
        
        # Calculate convergence score
        convergence_score = self._calculate_convergence_score(pareto_solutions)
        
        # Find extreme points
        extreme_points = self._find_extreme_points(pareto_solutions)
        
        # Find knee points
        knee_points = self._find_knee_points(pareto_solutions)
        
        # Analyze trade-offs
        trade_off_analysis = self._analyze_trade_offs(pareto_solutions)
        
        # Calculate dominated space volume
        dominated_space_volume = self._calculate_dominated_space_volume(pareto_solutions)
        
        # Calculate hypervolume (requires reference point)
        hypervolume = self._calculate_hypervolume_with_reference(pareto_solutions)
        
        analysis = ParetoFrontierAnalysis(
            num_solutions=len(pareto_solutions),
            frontier_shape=frontier_shape,
            diversity_score=diversity_score,
            convergence_score=convergence_score,
            extreme_points=extreme_points,
            knee_points=knee_points,
            trade_off_analysis=trade_off_analysis,
            dominated_space_volume=dominated_space_volume,
            hypervolume=hypervolume
        )
        
        logger.info(f"Frontier analysis complete: {frontier_shape} shape, diversity={diversity_score:.3f}")
        return analysis
    
    def _analyze_frontier_shape(self, pareto_solutions: List[ParetoSolution]) -> str:
        """Analyze the shape of Pareto frontier."""
        
        if len(pareto_solutions) < 3:
            return 'linear'
        
        obj_values = np.array([sol.objective_values for sol in pareto_solutions])
        
        if obj_values.shape[1] < 2:
            return 'single_objective'
        
        # Analyze curvature for 2D case
        if obj_values.shape[1] == 2:
            # Sort by first objective
            sorted_indices = np.argsort(obj_values[:, 0])
            sorted_objectives = obj_values[sorted_indices]
            
            # Calculate second derivatives to determine curvature
            if len(sorted_objectives) >= 3:
                second_derivatives = []
                for i in range(1, len(sorted_objectives) - 1):
                    x_prev, y_prev = sorted_objectives[i-1]
                    x_curr, y_curr = sorted_objectives[i]
                    x_next, y_next = sorted_objectives[i+1]
                    
                    # Approximate second derivative
                    if x_next != x_prev:
                        second_deriv = 2 * (y_next - 2*y_curr + y_prev) / ((x_next - x_prev) ** 2)
                        second_derivatives.append(second_deriv)
                
                if second_derivatives:
                    avg_curvature = np.mean(second_derivatives)
                    if avg_curvature > 0.1:
                        return 'convex'
                    elif avg_curvature < -0.1:
                        return 'concave'
                    else:
                        return 'linear'
        
        return 'complex'
    
    def _calculate_diversity_score(self, pareto_solutions: List[ParetoSolution]) -> float:
        """Calculate diversity score for Pareto frontier."""
        
        if len(pareto_solutions) < 2:
            return 0.0
        
        obj_values = np.array([sol.objective_values for sol in pareto_solutions])
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(obj_values)):
            for j in range(i + 1, len(obj_values)):
                dist = np.linalg.norm(obj_values[i] - obj_values[j])
                distances.append(dist)
        
        if not distances:
            return 0.0
        
        # Diversity is related to the distribution of distances
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        # Higher standard deviation relative to mean indicates better diversity
        if mean_distance > 0:
            diversity_score = std_distance / mean_distance
        else:
            diversity_score = 0.0
        
        # Normalize to [0, 1] range
        return min(1.0, diversity_score)
    
    def _calculate_convergence_score(self, pareto_solutions: List[ParetoSolution]) -> float:
        """Calculate convergence score for Pareto frontier."""
        
        if len(pareto_solutions) < 2:
            return 1.0
        
        obj_values = np.array([sol.objective_values for sol in pareto_solutions])
        
        # Calculate distance to ideal point
        ideal_point = np.min(obj_values, axis=0)
        
        distances_to_ideal = []
        for obj_val in obj_values:
            dist = np.linalg.norm(obj_val - ideal_point)
            distances_to_ideal.append(dist)
        
        # Convergence score based on proximity to ideal point
        max_distance = np.max(distances_to_ideal)
        if max_distance > 0:
            # Normalize distances
            normalized_distances = np.array(distances_to_ideal) / max_distance
            # Convergence score is inverse of mean distance
            convergence_score = 1.0 - np.mean(normalized_distances)
        else:
            convergence_score = 1.0
        
        return max(0.0, min(1.0, convergence_score))
    
    def _find_extreme_points(self, pareto_solutions: List[ParetoSolution]) -> List[ParetoSolution]:
        """Find extreme points in Pareto frontier."""
        
        if not pareto_solutions:
            return []
        
        obj_values = np.array([sol.objective_values for sol in pareto_solutions])
        extreme_points = []
        
        # For each objective, find the solution that optimizes it best
        for obj_idx in range(obj_values.shape[1]):
            # Assuming minimization objectives
            best_idx = np.argmin(obj_values[:, obj_idx])
            extreme_points.append(pareto_solutions[best_idx])
        
        # Remove duplicates
        unique_extremes = []
        for point in extreme_points:
            if point not in unique_extremes:
                unique_extremes.append(point)
        
        return unique_extremes
    
    def _find_knee_points(self, pareto_solutions: List[ParetoSolution]) -> List[ParetoSolution]:
        """Find knee points (points with best trade-offs) in Pareto frontier."""
        
        if len(pareto_solutions) < 3:
            return []
        
        obj_values = np.array([sol.objective_values for sol in pareto_solutions])
        
        if obj_values.shape[1] != 2:
            # For higher dimensions, use a simplified approach
            return self._find_knee_points_nd(pareto_solutions)
        
        # For 2D case, find points with maximum curvature
        sorted_indices = np.argsort(obj_values[:, 0])
        sorted_solutions = [pareto_solutions[i] for i in sorted_indices]
        sorted_objectives = obj_values[sorted_indices]
        
        knee_points = []
        
        # Calculate curvature for each point
        for i in range(1, len(sorted_objectives) - 1):
            prev_point = sorted_objectives[i-1]
            curr_point = sorted_objectives[i]
            next_point = sorted_objectives[i+1]
            
            # Calculate angle at current point
            vec1 = prev_point - curr_point
            vec2 = next_point - curr_point
            
            # Calculate angle between vectors
            cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            
            # Knee points have angles close to 90 degrees
            if abs(angle - np.pi/2) < np.pi/4:  # Within 45 degrees of 90 degrees
                knee_points.append(sorted_solutions[i])
        
        return knee_points[:3]  # Return at most 3 knee points
    
    def _find_knee_points_nd(self, pareto_solutions: List[ParetoSolution]) -> List[ParetoSolution]:
        """Find knee points for n-dimensional objectives."""
        
        obj_values = np.array([sol.objective_values for sol in pareto_solutions])
        
        # Use distance from origin as a simple knee point criterion
        distances_from_origin = [np.linalg.norm(obj_val) for obj_val in obj_values]
        
        # Find solutions with minimum distance (best compromise)
        min_distance_idx = np.argmin(distances_from_origin)
        
        return [pareto_solutions[min_distance_idx]]
    
    def _analyze_trade_offs(self, pareto_solutions: List[ParetoSolution]) -> Dict[str, Any]:
        """Analyze trade-offs between objectives."""
        
        if len(pareto_solutions) < 2:
            return {}
        
        obj_values = np.array([sol.objective_values for sol in pareto_solutions])
        num_objectives = obj_values.shape[1]
        
        trade_off_analysis = {}
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(obj_values.T)
        trade_off_analysis['objective_correlations'] = correlation_matrix.tolist()
        
        # Calculate trade-off intensities
        trade_off_intensities = {}
        for i in range(num_objectives):
            for j in range(i + 1, num_objectives):
                # Calculate how much one objective changes when the other changes
                obj_i_range = np.max(obj_values[:, i]) - np.min(obj_values[:, i])
                obj_j_range = np.max(obj_values[:, j]) - np.min(obj_values[:, j])
                
                if obj_i_range > 0 and obj_j_range > 0:
                    # Trade-off intensity is related to the correlation and ranges
                    intensity = abs(correlation_matrix[i, j]) * min(obj_i_range, obj_j_range) / max(obj_i_range, obj_j_range)
                    trade_off_intensities[f'obj_{i}_obj_{j}'] = float(intensity)
        
        trade_off_analysis['trade_off_intensities'] = trade_off_intensities
        
        return trade_off_analysis
    
    def _calculate_dominated_space_volume(self, pareto_solutions: List[ParetoSolution]) -> float:
        """Calculate volume of space dominated by Pareto frontier."""
        
        if not pareto_solutions:
            return 0.0
        
        # Simplified calculation - use bounding box approach
        obj_values = np.array([sol.objective_values for sol in pareto_solutions])
        
        # Calculate ranges in each dimension
        min_values = np.min(obj_values, axis=0)
        max_values = np.max(obj_values, axis=0)
        
        # Calculate volume of bounding box
        ranges = max_values - min_values
        volume = np.prod(ranges)
        
        return float(volume)
    
    def _calculate_hypervolume_with_reference(self, pareto_solutions: List[ParetoSolution]) -> float:
        """Calculate hypervolume with automatic reference point."""
        
        if not pareto_solutions:
            return 0.0
        
        obj_values = np.array([sol.objective_values for sol in pareto_solutions])
        
        # Create reference point (worst case in each objective)
        reference_point = np.max(obj_values, axis=0) + np.ones(obj_values.shape[1])
        
        # Use simplified hypervolume calculation
        return self._calculate_hypervolume_simple(obj_values, reference_point)
    
    def _calculate_hypervolume_simple(self, points: np.ndarray, reference: np.ndarray) -> float:
        """Simplified hypervolume calculation."""
        
        if len(points) == 0:
            return 0.0
        
        # For 2D case, use exact calculation
        if points.shape[1] == 2:
            # Sort points by first objective
            sorted_indices = np.argsort(points[:, 0])
            sorted_points = points[sorted_indices]
            
            total_volume = 0.0
            prev_x = reference[0]
            
            for point in sorted_points:
                if point[0] < prev_x and point[1] < reference[1]:
                    width = prev_x - point[0]
                    height = reference[1] - point[1]
                    total_volume += width * height
                    prev_x = point[0]
            
            return total_volume
        
        else:
            # For higher dimensions, use approximation
            volume = 1.0
            for dim in range(points.shape[1]):
                best_in_dim = np.min(points[:, dim])
                volume *= max(0, reference[dim] - best_in_dim)
            
            return volume * 0.5  # Approximation factor


class SolutionClusterer:
    """Cluster and classify solutions for pattern recognition."""
    
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.clusters = []
        self.cluster_model = None
    
    def cluster_solutions(self, solutions: List[ParetoSolution]) -> List[SolutionCluster]:
        """Cluster solutions based on parameter similarity."""
        
        if not solutions or not SKLEARN_AVAILABLE:
            logger.warning("Clustering unavailable - returning single cluster")
            return [SolutionCluster(
                cluster_id=0,
                solutions=solutions,
                centroid={},
                characteristics={},
                cluster_quality=1.0
            )]
        
        # Extract parameter vectors
        param_matrix, param_names = self._extract_parameter_matrix(solutions)
        
        if param_matrix is None:
            return []
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(param_matrix)
        
        # Perform clustering
        n_clusters = min(self.n_clusters, len(solutions))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        self.cluster_model = kmeans
        
        # Create cluster objects
        clusters = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_solutions = [solutions[i] for i in cluster_indices]
            
            # Calculate centroid
            centroid = {}
            for j, param_name in enumerate(param_names):
                param_values = [param_matrix[i, j] for i in cluster_indices]
                centroid[param_name] = float(np.mean(param_values))
            
            # Calculate cluster characteristics
            characteristics = self._analyze_cluster_characteristics(cluster_solutions)
            
            # Calculate cluster quality (silhouette-like measure)
            cluster_quality = self._calculate_cluster_quality(cluster_indices, scaled_features, cluster_labels)
            
            cluster = SolutionCluster(
                cluster_id=cluster_id,
                solutions=cluster_solutions,
                centroid=centroid,
                characteristics=characteristics,
                cluster_quality=cluster_quality
            )
            
            clusters.append(cluster)
        
        self.clusters = clusters
        logger.info(f"Created {len(clusters)} solution clusters")
        return clusters
    
    def _extract_parameter_matrix(self, solutions: List[ParetoSolution]) -> Tuple[Optional[np.ndarray], List[str]]:
        """Extract parameter matrix from solutions."""
        
        if not solutions:
            return None, []
        
        # Get all parameter names
        all_params = set()
        for solution in solutions:
            all_params.update(solution.design_parameters.keys())
        
        # Filter to numeric parameters only
        numeric_params = []
        for param_name in all_params:
            for solution in solutions:
                if param_name in solution.design_parameters:
                    value = solution.design_parameters[param_name]
                    if isinstance(value, (int, float)):
                        numeric_params.append(param_name)
                        break
        
        if not numeric_params:
            return None, []
        
        # Create parameter matrix
        param_matrix = np.zeros((len(solutions), len(numeric_params)))
        
        for i, solution in enumerate(solutions):
            for j, param_name in enumerate(numeric_params):
                if param_name in solution.design_parameters:
                    param_matrix[i, j] = float(solution.design_parameters[param_name])
        
        return param_matrix, numeric_params
    
    def _analyze_cluster_characteristics(self, cluster_solutions: List[ParetoSolution]) -> Dict[str, Any]:
        """Analyze characteristics of a solution cluster."""
        
        if not cluster_solutions:
            return {}
        
        characteristics = {}
        
        # Objective statistics
        obj_values = np.array([sol.objective_values for sol in cluster_solutions])
        characteristics['objective_means'] = np.mean(obj_values, axis=0).tolist()
        characteristics['objective_stds'] = np.std(obj_values, axis=0).tolist()
        characteristics['objective_ranges'] = (np.max(obj_values, axis=0) - np.min(obj_values, axis=0)).tolist()
        
        # Parameter statistics
        all_params = set()
        for solution in cluster_solutions:
            all_params.update(solution.design_parameters.keys())
        
        param_stats = {}
        for param_name in all_params:
            param_values = []
            for solution in cluster_solutions:
                if param_name in solution.design_parameters:
                    value = solution.design_parameters[param_name]
                    if isinstance(value, (int, float)):
                        param_values.append(value)
            
            if param_values:
                param_stats[param_name] = {
                    'mean': float(np.mean(param_values)),
                    'std': float(np.std(param_values)),
                    'min': float(np.min(param_values)),
                    'max': float(np.max(param_values))
                }
        
        characteristics['parameter_statistics'] = param_stats
        characteristics['cluster_size'] = len(cluster_solutions)
        
        return characteristics
    
    def _calculate_cluster_quality(self, cluster_indices: np.ndarray, 
                                  features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate cluster quality score."""
        
        if len(cluster_indices) <= 1:
            return 1.0
        
        cluster_points = features[cluster_indices]
        cluster_center = np.mean(cluster_points, axis=0)
        
        # Intra-cluster distance (lower is better)
        intra_distances = [np.linalg.norm(point - cluster_center) for point in cluster_points]
        avg_intra_distance = np.mean(intra_distances)
        
        # Inter-cluster distance (higher is better)
        other_indices = np.where(labels != labels[cluster_indices[0]])[0]
        if len(other_indices) > 0:
            other_points = features[other_indices]
            inter_distances = [np.linalg.norm(point - cluster_center) for point in other_points]
            avg_inter_distance = np.mean(inter_distances)
        else:
            avg_inter_distance = 1.0
        
        # Quality score (silhouette-like)
        if avg_intra_distance > 0:
            quality = (avg_inter_distance - avg_intra_distance) / max(avg_inter_distance, avg_intra_distance)
        else:
            quality = 1.0
        
        return max(0.0, min(1.0, (quality + 1.0) / 2.0))  # Normalize to [0, 1]


class SensitivityAnalyzer:
    """Analyze parameter sensitivity and importance."""
    
    def __init__(self):
        self.sensitivity_results = {}
        self.parameter_rankings = {}
    
    def analyze_sensitivity(self, solutions: List[ParetoSolution], 
                          perturbation_ratio: float = 0.1) -> Dict[str, Dict[str, float]]:
        """Analyze parameter sensitivity using local perturbation."""
        
        if not solutions:
            return {}
        
        logger.info(f"Analyzing parameter sensitivity for {len(solutions)} solutions")
        
        # Extract parameter matrix and objective matrix
        param_matrix, param_names = self._extract_data_matrices(solutions)
        
        if param_matrix is None:
            return {}
        
        obj_matrix = np.array([sol.objective_values for sol in solutions])
        
        # Calculate sensitivity for each parameter-objective pair
        sensitivities = {}
        
        for i, param_name in enumerate(param_names):
            param_sensitivities = {}
            
            for j in range(obj_matrix.shape[1]):
                sensitivity = self._calculate_parameter_sensitivity(
                    param_matrix[:, i], obj_matrix[:, j], param_name, perturbation_ratio
                )
                param_sensitivities[f'objective_{j}'] = sensitivity
            
            # Calculate overall sensitivity (across all objectives)
            overall_sensitivity = np.mean(list(param_sensitivities.values()))
            param_sensitivities['overall'] = overall_sensitivity
            
            sensitivities[param_name] = param_sensitivities
        
        # Rank parameters by overall sensitivity
        self.parameter_rankings = self._rank_parameters(sensitivities)
        self.sensitivity_results = sensitivities
        
        logger.info(f"Sensitivity analysis complete. Most sensitive: {self.parameter_rankings['most_sensitive'][:3]}")
        return sensitivities
    
    def get_parameter_importance_ranking(self) -> List[Tuple[str, float]]:
        """Get parameters ranked by importance."""
        
        if not self.sensitivity_results:
            return []
        
        importance_scores = []
        for param_name, sensitivities in self.sensitivity_results.items():
            overall_sensitivity = sensitivities.get('overall', 0.0)
            importance_scores.append((param_name, overall_sensitivity))
        
        # Sort by importance descending
        importance_scores.sort(key=lambda x: x[1], reverse=True)
        return importance_scores
    
    def _extract_data_matrices(self, solutions: List[ParetoSolution]) -> Tuple[Optional[np.ndarray], List[str]]:
        """Extract parameter and objective matrices."""
        
        if not solutions:
            return None, []
        
        # Get numeric parameters
        all_params = set()
        for solution in solutions:
            all_params.update(solution.design_parameters.keys())
        
        numeric_params = []
        for param_name in all_params:
            for solution in solutions:
                if param_name in solution.design_parameters:
                    value = solution.design_parameters[param_name]
                    if isinstance(value, (int, float)):
                        numeric_params.append(param_name)
                        break
        
        if not numeric_params:
            return None, []
        
        # Create parameter matrix
        param_matrix = np.zeros((len(solutions), len(numeric_params)))
        
        for i, solution in enumerate(solutions):
            for j, param_name in enumerate(numeric_params):
                if param_name in solution.design_parameters:
                    param_matrix[i, j] = float(solution.design_parameters[param_name])
        
        return param_matrix, numeric_params
    
    def _calculate_parameter_sensitivity(self, param_values: np.ndarray, 
                                       objective_values: np.ndarray,
                                       param_name: str, perturbation_ratio: float) -> float:
        """Calculate sensitivity of objective to parameter changes."""
        
        if len(param_values) != len(objective_values) or len(param_values) < 3:
            return 0.0
        
        # Calculate numerical gradient using finite differences
        gradients = []
        
        for i in range(1, len(param_values) - 1):
            param_delta = param_values[i+1] - param_values[i-1]
            obj_delta = objective_values[i+1] - objective_values[i-1]
            
            if abs(param_delta) > 1e-8:
                gradient = abs(obj_delta / param_delta)
                gradients.append(gradient)
        
        if not gradients:
            return 0.0
        
        # Average gradient magnitude as sensitivity measure
        avg_gradient = np.mean(gradients)
        
        # Normalize by parameter and objective ranges
        param_range = np.max(param_values) - np.min(param_values)
        obj_range = np.max(objective_values) - np.min(objective_values)
        
        if param_range > 0 and obj_range > 0:
            normalized_sensitivity = avg_gradient * param_range / obj_range
        else:
            normalized_sensitivity = 0.0
        
        return float(normalized_sensitivity)
    
    def _rank_parameters(self, sensitivities: Dict[str, Dict[str, float]]) -> Dict[str, List[str]]:
        """Rank parameters by different criteria."""
        
        rankings = {}
        
        # Most sensitive overall
        overall_scores = [(param, data['overall']) for param, data in sensitivities.items()]
        overall_scores.sort(key=lambda x: x[1], reverse=True)
        rankings['most_sensitive'] = [param for param, _ in overall_scores]
        
        # Least sensitive overall
        rankings['least_sensitive'] = [param for param, _ in reversed(overall_scores)]
        
        return rankings


class DesignSpaceNavigator:
    """Intelligent guidance for design space exploration."""
    
    def __init__(self, space_analyzer: DesignSpaceAnalyzer):
        self.space_analyzer = space_analyzer
        self.navigation_history = []
        self.promising_regions = []
    
    def suggest_exploration_direction(self, current_solutions: List[ParetoSolution],
                                    space_characteristics: DesignSpaceCharacteristics) -> Dict[str, Any]:
        """Suggest direction for continued exploration."""
        
        if not current_solutions:
            return self._suggest_initial_exploration(space_characteristics)
        
        # Analyze current solution distribution
        distribution_analysis = self._analyze_solution_distribution(current_solutions)
        
        # Identify gaps in coverage
        coverage_gaps = self._identify_coverage_gaps(current_solutions, space_characteristics)
        
        # Suggest exploration strategy
        exploration_strategy = self._determine_exploration_strategy(
            distribution_analysis, coverage_gaps, space_characteristics
        )
        
        # Generate specific suggestions
        suggestions = self._generate_exploration_suggestions(
            current_solutions, exploration_strategy, space_characteristics
        )
        
        return {
            'distribution_analysis': distribution_analysis,
            'coverage_gaps': coverage_gaps,
            'exploration_strategy': exploration_strategy,
            'suggestions': suggestions
        }
    
    def _suggest_initial_exploration(self, space_characteristics: DesignSpaceCharacteristics) -> Dict[str, Any]:
        """Suggest initial exploration strategy."""
        
        if space_characteristics.search_difficulty == 'easy':
            strategy = 'broad_sampling'
            sample_count = 100
        elif space_characteristics.search_difficulty == 'medium':
            strategy = 'focused_sampling'
            sample_count = 200
        else:
            strategy = 'adaptive_sampling'
            sample_count = 500
        
        return {
            'strategy': strategy,
            'sample_count': sample_count,
            'focus_areas': ['center', 'extremes'],
            'recommended_algorithms': ['latin_hypercube', 'random_sampling']
        }
    
    def _analyze_solution_distribution(self, solutions: List[ParetoSolution]) -> Dict[str, Any]:
        """Analyze distribution of current solutions."""
        
        if not solutions:
            return {}
        
        obj_matrix = np.array([sol.objective_values for sol in solutions])
        
        # Calculate distribution statistics
        distribution_stats = {
            'mean': np.mean(obj_matrix, axis=0).tolist(),
            'std': np.std(obj_matrix, axis=0).tolist(),
            'min': np.min(obj_matrix, axis=0).tolist(),
            'max': np.max(obj_matrix, axis=0).tolist(),
            'coverage_volume': self._calculate_coverage_volume(obj_matrix)
        }
        
        # Analyze clustering
        clustering_info = self._analyze_solution_clustering(solutions)
        
        return {
            'statistics': distribution_stats,
            'clustering': clustering_info,
            'density_map': self._create_density_map(obj_matrix)
        }
    
    def _identify_coverage_gaps(self, solutions: List[ParetoSolution], 
                               space_characteristics: DesignSpaceCharacteristics) -> List[Dict[str, Any]]:
        """Identify gaps in solution space coverage."""
        
        gaps = []
        
        if not solutions:
            return gaps
        
        obj_matrix = np.array([sol.objective_values for sol in solutions])
        
        # Grid-based gap detection
        for dim in range(obj_matrix.shape[1]):
            dim_values = obj_matrix[:, dim]
            dim_min, dim_max = np.min(dim_values), np.max(dim_values)
            
            # Divide dimension into bins and find empty ones
            n_bins = 10
            bin_edges = np.linspace(dim_min, dim_max, n_bins + 1)
            hist, _ = np.histogram(dim_values, bins=bin_edges)
            
            # Find empty or sparse bins
            sparse_threshold = len(solutions) / (n_bins * 3)  # Less than 1/3 of average
            for i, count in enumerate(hist):
                if count < sparse_threshold:
                    gap_center = (bin_edges[i] + bin_edges[i+1]) / 2
                    gap_info = {
                        'dimension': dim,
                        'center': float(gap_center),
                        'range': (float(bin_edges[i]), float(bin_edges[i+1])),
                        'severity': 1.0 - (count / max(1, np.max(hist)))
                    }
                    gaps.append(gap_info)
        
        # Sort gaps by severity
        gaps.sort(key=lambda x: x['severity'], reverse=True)
        
        return gaps[:5]  # Return top 5 gaps
    
    def _determine_exploration_strategy(self, distribution_analysis: Dict[str, Any],
                                      coverage_gaps: List[Dict[str, Any]],
                                      space_characteristics: DesignSpaceCharacteristics) -> str:
        """Determine appropriate exploration strategy."""
        
        # Analyze current exploration state
        coverage_volume = distribution_analysis.get('statistics', {}).get('coverage_volume', 0)
        num_clusters = len(distribution_analysis.get('clustering', {}).get('clusters', []))
        num_gaps = len(coverage_gaps)
        
        # Decision logic
        if coverage_volume < 0.3:  # Low coverage
            if space_characteristics.search_difficulty == 'hard':
                return 'intensive_local_search'
            else:
                return 'broad_exploration'
        
        elif num_gaps > 3:  # Many coverage gaps
            return 'gap_filling'
        
        elif num_clusters > 5:  # Many clusters found
            return 'cluster_refinement'
        
        else:
            return 'balanced_exploration'
    
    def _generate_exploration_suggestions(self, current_solutions: List[ParetoSolution],
                                        strategy: str,
                                        space_characteristics: DesignSpaceCharacteristics) -> List[Dict[str, Any]]:
        """Generate specific exploration suggestions."""
        
        suggestions = []
        
        if strategy == 'broad_exploration':
            suggestions.append({
                'type': 'parameter_space_expansion',
                'description': 'Explore wider parameter ranges',
                'action': 'increase_search_bounds',
                'priority': 'high'
            })
        
        elif strategy == 'gap_filling':
            suggestions.append({
                'type': 'targeted_sampling',
                'description': 'Focus sampling on identified coverage gaps',
                'action': 'sample_gap_regions',
                'priority': 'high'
            })
        
        elif strategy == 'cluster_refinement':
            suggestions.append({
                'type': 'local_optimization',
                'description': 'Refine solutions within promising clusters',
                'action': 'local_search_clusters',
                'priority': 'medium'
            })
        
        elif strategy == 'intensive_local_search':
            suggestions.append({
                'type': 'exploitation',
                'description': 'Intensively search around best solutions',
                'action': 'local_search_best',
                'priority': 'high'
            })
        
        else:  # balanced_exploration
            suggestions.extend([
                {
                    'type': 'diversification',
                    'description': 'Add diversity to solution set',
                    'action': 'diverse_sampling',
                    'priority': 'medium'
                },
                {
                    'type': 'intensification',
                    'description': 'Improve best solutions',
                    'action': 'local_optimization',
                    'priority': 'medium'
                }
            ])
        
        return suggestions
    
    def _calculate_coverage_volume(self, obj_matrix: np.ndarray) -> float:
        """Calculate volume covered by solutions in objective space."""
        
        if obj_matrix.shape[0] < 2:
            return 0.0
        
        # Use convex hull volume as coverage measure
        ranges = np.max(obj_matrix, axis=0) - np.min(obj_matrix, axis=0)
        volume = np.prod(ranges + 1e-8)  # Add small constant to avoid zero
        
        # Normalize by theoretical maximum (rough approximation)
        max_theoretical_volume = np.prod(np.max(obj_matrix, axis=0) + 1e-8)
        
        if max_theoretical_volume > 0:
            normalized_volume = volume / max_theoretical_volume
        else:
            normalized_volume = 0.0
        
        return float(min(1.0, normalized_volume))
    
    def _analyze_solution_clustering(self, solutions: List[ParetoSolution]) -> Dict[str, Any]:
        """Analyze clustering in solution space."""
        
        if not SKLEARN_AVAILABLE or len(solutions) < 3:
            return {'clusters': [], 'num_clusters': 0}
        
        # Use solution clusterer
        clusterer = SolutionClusterer(n_clusters=min(5, len(solutions) // 2))
        clusters = clusterer.cluster_solutions(solutions)
        
        return {
            'clusters': [{'id': c.cluster_id, 'size': len(c.solutions), 'quality': c.cluster_quality} 
                        for c in clusters],
            'num_clusters': len(clusters)
        }
    
    def _create_density_map(self, obj_matrix: np.ndarray) -> Dict[str, Any]:
        """Create density map of solutions in objective space."""
        
        if obj_matrix.shape[0] < 5:
            return {'type': 'insufficient_data'}
        
        # Simple 2D density map for first two objectives
        if obj_matrix.shape[1] >= 2:
            # Create grid
            x_min, x_max = np.min(obj_matrix[:, 0]), np.max(obj_matrix[:, 0])
            y_min, y_max = np.min(obj_matrix[:, 1]), np.max(obj_matrix[:, 1])
            
            # Count solutions in grid cells
            n_bins = 5
            x_edges = np.linspace(x_min, x_max, n_bins + 1)
            y_edges = np.linspace(y_min, y_max, n_bins + 1)
            
            density_grid = np.zeros((n_bins, n_bins))
            
            for point in obj_matrix:
                x_bin = min(n_bins - 1, int((point[0] - x_min) / (x_max - x_min + 1e-8) * n_bins))
                y_bin = min(n_bins - 1, int((point[1] - y_min) / (y_max - y_min + 1e-8) * n_bins))
                density_grid[y_bin, x_bin] += 1
            
            return {
                'type': '2d_grid',
                'density_grid': density_grid.tolist(),
                'x_edges': x_edges.tolist(),
                'y_edges': y_edges.tolist()
            }
        
        else:
            # 1D density for single objective
            return {
                'type': '1d_histogram',
                'counts': np.histogram(obj_matrix[:, 0], bins=10)[0].tolist()
            }