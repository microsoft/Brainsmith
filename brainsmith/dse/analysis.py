"""
Advanced analysis tools for DSE results.

This module provides comprehensive analysis capabilities including Pareto frontier
analysis, sensitivity analysis, and statistical evaluation of DSE results.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import logging
import json

from ..core.result import DSEResult, BrainsmithResult
from ..core.design_space import DesignSpace, DesignPoint
from ..core.metrics import BrainsmithMetrics
from .interface import OptimizationObjective, DSEObjective


@dataclass
class ParetoPoint:
    """A point on the Pareto frontier."""
    design_point: DesignPoint
    result: BrainsmithResult
    objective_values: List[float]
    rank: int = 0
    crowding_distance: float = 0.0


@dataclass
class SensitivityAnalysis:
    """Results of parameter sensitivity analysis."""
    parameter_importance: Dict[str, float]
    parameter_correlations: Dict[str, Dict[str, float]]
    sobol_indices: Optional[Dict[str, Dict[str, float]]] = None
    morris_effects: Optional[Dict[str, Dict[str, float]]] = None


@dataclass
class ConvergenceAnalysis:
    """Analysis of optimization convergence."""
    best_values_over_time: List[float]
    pareto_size_over_time: List[int]
    improvement_rate: float
    stagnation_periods: List[Tuple[int, int]]
    convergence_score: float
    estimated_convergence: bool


class ParetoAnalyzer:
    """
    Advanced Pareto frontier analysis for multi-objective optimization.
    
    Provides comprehensive analysis of trade-offs between objectives.
    """
    
    def __init__(self, objectives: List[DSEObjective]):
        self.objectives = objectives
        self.is_multi_objective = len(objectives) > 1
    
    def compute_pareto_frontier(self, results: List[BrainsmithResult]) -> List[ParetoPoint]:
        """
        Compute Pareto frontier from results.
        
        Args:
            results: List of optimization results
            
        Returns:
            List of non-dominated points (Pareto frontier)
        """
        if not results:
            return []
        
        # Extract objective values
        pareto_points = []
        for result in results:
            objective_values = []
            for obj in self.objectives:
                try:
                    value = obj.evaluate(result.metrics)
                    # Convert to maximization problem for consistency
                    if obj.direction == OptimizationObjective.MINIMIZE:
                        value = -value
                    objective_values.append(value)
                except (AttributeError, ValueError) as e:
                    logging.warning(f"Could not evaluate objective {obj.name}: {e}")
                    continue
            
            if len(objective_values) == len(self.objectives):
                pareto_points.append(ParetoPoint(
                    design_point=DesignPoint(),  # Would need to be passed in
                    result=result,
                    objective_values=objective_values
                ))
        
        if not self.is_multi_objective:
            # Single objective: sort by objective value
            pareto_points.sort(key=lambda p: p.objective_values[0], reverse=True)
            return pareto_points[:min(10, len(pareto_points))]  # Top 10
        
        # Multi-objective: find non-dominated solutions
        non_dominated = []
        for i, point_i in enumerate(pareto_points):
            is_dominated = False
            for j, point_j in enumerate(pareto_points):
                if i != j and self._dominates(point_j.objective_values, point_i.objective_values):
                    is_dominated = True
                    break
            
            if not is_dominated:
                non_dominated.append(point_i)
        
        # Rank and compute crowding distance
        self._rank_and_distance(non_dominated)
        
        return non_dominated
    
    def _dominates(self, a: List[float], b: List[float]) -> bool:
        """Check if solution a dominates solution b (assuming maximization)."""
        return (all(a_i >= b_i for a_i, b_i in zip(a, b)) and 
                any(a_i > b_i for a_i, b_i in zip(a, b)))
    
    def _rank_and_distance(self, pareto_points: List[ParetoPoint]):
        """Compute NSGA-II style ranking and crowding distance."""
        # All points in frontier have rank 0
        for point in pareto_points:
            point.rank = 0
        
        # Compute crowding distance
        if len(pareto_points) <= 2:
            for point in pareto_points:
                point.crowding_distance = float('inf')
            return
        
        # Initialize distances
        for point in pareto_points:
            point.crowding_distance = 0.0
        
        # For each objective
        for obj_idx in range(len(self.objectives)):
            # Sort by this objective
            pareto_points.sort(key=lambda p: p.objective_values[obj_idx])
            
            # Boundary points get infinite distance
            pareto_points[0].crowding_distance = float('inf')
            pareto_points[-1].crowding_distance = float('inf')
            
            # Compute distances for interior points
            obj_range = (pareto_points[-1].objective_values[obj_idx] - 
                        pareto_points[0].objective_values[obj_idx])
            
            if obj_range > 0:
                for i in range(1, len(pareto_points) - 1):
                    distance = (pareto_points[i + 1].objective_values[obj_idx] - 
                               pareto_points[i - 1].objective_values[obj_idx]) / obj_range
                    pareto_points[i].crowding_distance += distance
    
    def analyze_trade_offs(self, pareto_points: List[ParetoPoint]) -> Dict[str, Any]:
        """
        Analyze trade-offs between objectives on the Pareto frontier.
        
        Returns:
            Analysis of objective trade-offs and correlations
        """
        if len(pareto_points) < 2 or not self.is_multi_objective:
            return {"trade_offs": "Insufficient data for trade-off analysis"}
        
        # Extract objective values matrix
        objective_matrix = np.array([p.objective_values for p in pareto_points])
        
        # Compute correlations between objectives
        correlations = {}
        for i, obj_i in enumerate(self.objectives):
            correlations[obj_i.name] = {}
            for j, obj_j in enumerate(self.objectives):
                if i != j:
                    corr = np.corrcoef(objective_matrix[:, i], objective_matrix[:, j])[0, 1]
                    correlations[obj_i.name][obj_j.name] = float(corr)
        
        # Analyze spread and diversity
        ranges = {}
        for i, obj in enumerate(self.objectives):
            obj_values = objective_matrix[:, i]
            ranges[obj.name] = {
                "min": float(np.min(obj_values)),
                "max": float(np.max(obj_values)),
                "range": float(np.max(obj_values) - np.min(obj_values)),
                "std": float(np.std(obj_values))
            }
        
        # Compute hypervolume (if possible)
        hypervolume = self._compute_hypervolume(objective_matrix) if len(self.objectives) <= 3 else None
        
        return {
            "correlations": correlations,
            "objective_ranges": ranges,
            "pareto_size": len(pareto_points),
            "hypervolume": hypervolume,
            "diversity_metrics": {
                "mean_crowding_distance": float(np.mean([p.crowding_distance for p in pareto_points 
                                                       if p.crowding_distance != float('inf')])),
                "max_crowding_distance": float(max(p.crowding_distance for p in pareto_points 
                                                 if p.crowding_distance != float('inf')))
            }
        }
    
    def _compute_hypervolume(self, objective_matrix: np.ndarray, 
                           reference_point: Optional[np.ndarray] = None) -> float:
        """Compute hypervolume indicator (simplified implementation)."""
        if reference_point is None:
            # Use worst point as reference
            reference_point = np.min(objective_matrix, axis=0) - 1.0
        
        # For 2D case, compute area under curve
        if objective_matrix.shape[1] == 2:
            # Sort by first objective
            sorted_indices = np.argsort(objective_matrix[:, 0])
            sorted_points = objective_matrix[sorted_indices]
            
            area = 0.0
            prev_x = reference_point[0]
            
            for point in sorted_points:
                x, y = point
                area += (x - prev_x) * (y - reference_point[1])
                prev_x = x
            
            return float(area)
        
        # For higher dimensions, use Monte Carlo approximation
        elif objective_matrix.shape[1] == 3:
            return self._hypervolume_3d(objective_matrix, reference_point)
        
        return 0.0  # Not implemented for >3D
    
    def _hypervolume_3d(self, points: np.ndarray, reference: np.ndarray, 
                       n_samples: int = 10000) -> float:
        """Approximate 3D hypervolume using Monte Carlo sampling."""
        # Find bounding box
        max_point = np.max(points, axis=0)
        
        # Generate random points in the box
        random_points = np.random.uniform(
            low=reference, 
            high=max_point, 
            size=(n_samples, 3)
        )
        
        # Count points dominated by at least one Pareto point
        dominated_count = 0
        for random_point in random_points:
            for pareto_point in points:
                if all(pareto_point >= random_point):
                    dominated_count += 1
                    break
        
        # Estimate volume
        box_volume = np.prod(max_point - reference)
        return float(box_volume * dominated_count / n_samples)


class DSEAnalyzer:
    """
    Comprehensive analyzer for DSE results.
    
    Provides sensitivity analysis, convergence analysis, and statistical insights.
    """
    
    def __init__(self, design_space: DesignSpace, objectives: List[DSEObjective]):
        self.design_space = design_space
        self.objectives = objectives
        self.pareto_analyzer = ParetoAnalyzer(objectives)
    
    def analyze_dse_result(self, dse_result: DSEResult) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of DSE results.
        
        Args:
            dse_result: Complete DSE result to analyze
            
        Returns:
            Comprehensive analysis dictionary
        """
        analysis = {
            "summary": self._analyze_summary(dse_result),
            "pareto_analysis": None,
            "sensitivity_analysis": None,
            "convergence_analysis": None,
            "statistical_analysis": self._analyze_statistics(dse_result)
        }
        
        # Pareto frontier analysis
        if len(dse_result.results) > 1:
            pareto_points = self.pareto_analyzer.compute_pareto_frontier(dse_result.results)
            analysis["pareto_analysis"] = {
                "pareto_frontier": self._serialize_pareto_points(pareto_points),
                "trade_offs": self.pareto_analyzer.analyze_trade_offs(pareto_points)
            }
        
        # Sensitivity analysis
        if len(dse_result.results) >= 10:  # Need sufficient data
            analysis["sensitivity_analysis"] = self._analyze_sensitivity(dse_result)
        
        # Convergence analysis
        if len(dse_result.results) >= 5:
            analysis["convergence_analysis"] = self._analyze_convergence(dse_result)
        
        return analysis
    
    def _analyze_summary(self, dse_result: DSEResult) -> Dict[str, Any]:
        """Generate summary statistics."""
        return {
            "total_evaluations": len(dse_result.results),
            "total_time_seconds": dse_result.total_time_seconds,
            "evaluations_per_second": len(dse_result.results) / max(dse_result.total_time_seconds, 1),
            "strategy": dse_result.strategy,
            "objectives": [obj.name for obj in self.objectives],
            "design_space_size": self.design_space.estimate_space_size(),
            "coverage_percentage": min(100.0, (len(dse_result.results) / max(self.design_space.estimate_space_size(), 1)) * 100)
        }
    
    def _analyze_statistics(self, dse_result: DSEResult) -> Dict[str, Any]:
        """Analyze statistical properties of results."""
        if not dse_result.results:
            return {}
        
        stats = {}
        
        # For each objective
        for obj in self.objectives:
            obj_values = []
            for result in dse_result.results:
                try:
                    value = obj.evaluate(result.metrics)
                    obj_values.append(value)
                except:
                    continue
            
            if obj_values:
                stats[obj.name] = {
                    "count": len(obj_values),
                    "mean": float(np.mean(obj_values)),
                    "std": float(np.std(obj_values)),
                    "min": float(np.min(obj_values)),
                    "max": float(np.max(obj_values)),
                    "median": float(np.median(obj_values)),
                    "q25": float(np.percentile(obj_values, 25)),
                    "q75": float(np.percentile(obj_values, 75))
                }
        
        return stats
    
    def _analyze_sensitivity(self, dse_result: DSEResult) -> SensitivityAnalysis:
        """Analyze parameter sensitivity."""
        # Extract parameter values and objective values
        param_values = {}
        objective_values = {}
        
        # Initialize containers
        for param_name in self.design_space.parameters.keys():
            param_values[param_name] = []
        for obj in self.objectives:
            objective_values[obj.name] = []
        
        # Extract values from results
        for result in dse_result.results:
            # Get design point (need to store this in DSEResult)
            # For now, skip this analysis
            pass
        
        # Placeholder implementation
        return SensitivityAnalysis(
            parameter_importance={},
            parameter_correlations={}
        )
    
    def _analyze_convergence(self, dse_result: DSEResult) -> ConvergenceAnalysis:
        """Analyze optimization convergence."""
        if not dse_result.results:
            return ConvergenceAnalysis([], [], 0.0, [], 0.0, False)
        
        # Track best objective value over time
        best_values = []
        current_best = float('-inf') if self.objectives[0].direction == OptimizationObjective.MAXIMIZE else float('inf')
        
        for result in dse_result.results:
            try:
                value = self.objectives[0].evaluate(result.metrics)
                
                if self.objectives[0].direction == OptimizationObjective.MAXIMIZE:
                    current_best = max(current_best, value)
                else:
                    current_best = min(current_best, value)
                
                best_values.append(current_best)
            except:
                best_values.append(current_best)
        
        # Analyze improvement rate
        if len(best_values) > 1:
            improvements = [abs(best_values[i] - best_values[i-1]) for i in range(1, len(best_values))]
            improvement_rate = np.mean(improvements)
        else:
            improvement_rate = 0.0
        
        # Detect stagnation periods
        stagnation_periods = []
        stagnation_start = None
        stagnation_threshold = improvement_rate * 0.1  # 10% of average improvement
        
        for i in range(1, len(best_values)):
            improvement = abs(best_values[i] - best_values[i-1])
            
            if improvement < stagnation_threshold:
                if stagnation_start is None:
                    stagnation_start = i - 1
            else:
                if stagnation_start is not None:
                    stagnation_periods.append((stagnation_start, i - 1))
                    stagnation_start = None
        
        # Close final stagnation period if needed
        if stagnation_start is not None:
            stagnation_periods.append((stagnation_start, len(best_values) - 1))
        
        # Compute convergence score
        recent_window = min(10, len(best_values) // 4)
        if recent_window > 1:
            recent_values = best_values[-recent_window:]
            convergence_score = 1.0 - (np.std(recent_values) / (np.mean(recent_values) + 1e-8))
        else:
            convergence_score = 0.0
        
        # Estimate if converged
        estimated_convergence = convergence_score > 0.95 or (
            len(stagnation_periods) > 0 and 
            stagnation_periods[-1][1] - stagnation_periods[-1][0] > len(best_values) * 0.3
        )
        
        return ConvergenceAnalysis(
            best_values_over_time=best_values,
            pareto_size_over_time=[],  # Would need tracking during optimization
            improvement_rate=float(improvement_rate),
            stagnation_periods=stagnation_periods,
            convergence_score=float(convergence_score),
            estimated_convergence=estimated_convergence
        )
    
    def _serialize_pareto_points(self, pareto_points: List[ParetoPoint]) -> List[Dict[str, Any]]:
        """Serialize Pareto points for JSON export."""
        serialized = []
        for point in pareto_points:
            serialized.append({
                "objective_values": point.objective_values,
                "rank": point.rank,
                "crowding_distance": point.crowding_distance if point.crowding_distance != float('inf') else "infinity",
                "metrics": point.result.metrics.to_dict() if hasattr(point.result.metrics, 'to_dict') else str(point.result.metrics)
            })
        return serialized
    
    def export_analysis(self, analysis: Dict[str, Any], filepath: str):
        """Export analysis to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
    
    def generate_analysis_report(self, dse_result: DSEResult) -> str:
        """Generate human-readable analysis report."""
        analysis = self.analyze_dse_result(dse_result)
        
        report = []
        report.append("# DSE Analysis Report")
        report.append("=" * 50)
        
        # Summary
        summary = analysis["summary"]
        report.append(f"\n## Summary")
        report.append(f"- Total Evaluations: {summary['total_evaluations']}")
        report.append(f"- Total Time: {summary['total_time_seconds']:.2f} seconds")
        report.append(f"- Evaluation Rate: {summary['evaluations_per_second']:.2f} eval/sec")
        report.append(f"- Strategy: {summary['strategy']}")
        report.append(f"- Design Space Coverage: {summary['coverage_percentage']:.4f}%")
        
        # Statistical Analysis
        if "statistical_analysis" in analysis and analysis["statistical_analysis"]:
            report.append(f"\n## Statistical Analysis")
            for obj_name, stats in analysis["statistical_analysis"].items():
                report.append(f"\n### {obj_name}")
                report.append(f"- Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
                report.append(f"- Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                report.append(f"- Median: {stats['median']:.4f}")
        
        # Pareto Analysis
        if analysis["pareto_analysis"]:
            pareto_info = analysis["pareto_analysis"]["trade_offs"]
            report.append(f"\n## Pareto Analysis")
            report.append(f"- Pareto Frontier Size: {pareto_info['pareto_size']}")
            if "hypervolume" in pareto_info and pareto_info["hypervolume"]:
                report.append(f"- Hypervolume: {pareto_info['hypervolume']:.4f}")
        
        # Convergence Analysis
        if analysis["convergence_analysis"]:
            conv = analysis["convergence_analysis"]
            report.append(f"\n## Convergence Analysis")
            report.append(f"- Improvement Rate: {conv['improvement_rate']:.6f}")
            report.append(f"- Convergence Score: {conv['convergence_score']:.4f}")
            report.append(f"- Estimated Converged: {conv['estimated_convergence']}")
            report.append(f"- Stagnation Periods: {len(conv['stagnation_periods'])}")
        
        return "\n".join(report)


# Utility functions for external analysis tools
def export_for_external_analysis(dse_result: DSEResult, format: str = "csv") -> str:
    """Export DSE results in format suitable for external analysis tools."""
    if format == "csv":
        # Would implement CSV export
        return "CSV export not implemented yet"
    elif format == "json":
        # Export as JSON
        data = {
            "results": [
                {
                    "metrics": result.metrics.to_dict() if hasattr(result.metrics, 'to_dict') else str(result.metrics),
                    "build_time": result.build_time_seconds
                }
                for result in dse_result.results
            ],
            "metadata": {
                "total_evaluations": len(dse_result.results),
                "total_time": dse_result.total_time_seconds,
                "strategy": dse_result.strategy
            }
        }
        return json.dumps(data, indent=2, default=str)
    else:
        raise ValueError(f"Unsupported export format: {format}")