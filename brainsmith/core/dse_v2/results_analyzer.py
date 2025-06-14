"""
Results Analysis for DSE V2

Provides analysis capabilities for design space exploration results,
including multi-objective analysis, Pareto frontier calculation, and
performance trend analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
import numpy as np
from collections import defaultdict
import logging

from .combination_generator import ComponentCombination

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    throughput: float = 0.0
    latency: float = 0.0
    resource_efficiency: float = 0.0
    power_consumption: float = 0.0
    accuracy: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'throughput': self.throughput,
            'latency': self.latency,
            'resource_efficiency': self.resource_efficiency,
            'power_consumption': self.power_consumption,
            'accuracy': self.accuracy
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetrics':
        """Create from dictionary."""
        return cls(
            throughput=data.get('throughput', 0.0),
            latency=data.get('latency', 0.0),
            resource_efficiency=data.get('resource_efficiency', 0.0),
            power_consumption=data.get('power_consumption', 0.0),
            accuracy=data.get('accuracy', 0.0)
        )


@dataclass
class DSEResultsV2:
    """Enhanced results container for DSE V2."""
    combination: ComponentCombination
    metrics: PerformanceMetrics
    evaluation_time: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def dominates(self, other: 'DSEResultsV2', objectives: List[str], 
                 directions: List[str]) -> bool:
        """Check if this result dominates another in Pareto sense."""
        if not (self.success and other.success):
            return False
        
        better_in_any = False
        
        for obj, direction in zip(objectives, directions):
            self_value = getattr(self.metrics, obj, 0.0)
            other_value = getattr(other.metrics, obj, 0.0)
            
            if direction.lower() == 'maximize':
                if self_value < other_value:
                    return False
                elif self_value > other_value:
                    better_in_any = True
            else:  # minimize
                if self_value > other_value:
                    return False
                elif self_value < other_value:
                    better_in_any = True
        
        return better_in_any


class ResultsAnalyzer:
    """Analyzes design space exploration results."""
    
    def __init__(self):
        """Initialize results analyzer."""
        self.analysis_cache: Dict[str, Any] = {}
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Comprehensive analysis of exploration results.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Analysis summary
        """
        if not results:
            return {'error': 'No results to analyze'}
        
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            return {'error': 'No successful evaluations to analyze'}
        
        analysis = {
            'summary': self._generate_summary_statistics(successful_results),
            'performance_trends': self._analyze_performance_trends(successful_results),
            'component_analysis': self._analyze_component_impact(successful_results),
            'correlation_analysis': self._analyze_metric_correlations(successful_results),
            'outlier_analysis': self._identify_outliers(successful_results),
            'recommendations': self._generate_recommendations(successful_results)
        }
        
        return analysis
    
    def _generate_summary_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics for results."""
        metrics_data = defaultdict(list)
        
        # Collect all metrics
        for result in results:
            if 'metrics' in result:
                for metric, value in result['metrics'].items():
                    if isinstance(value, (int, float)):
                        metrics_data[metric].append(value)
            
            # Also check top-level metrics
            for key, value in result.items():
                if key not in ['combination', 'success', 'metrics'] and isinstance(value, (int, float)):
                    metrics_data[key].append(value)
        
        summary = {}
        for metric, values in metrics_data.items():
            if values:
                summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'count': len(values)
                }
        
        return summary
    
    def _analyze_performance_trends(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance trends over evaluation sequence."""
        if len(results) < 10:
            return {'message': 'Insufficient data for trend analysis'}
        
        primary_metrics = [r.get('primary_metric', 0) for r in results]
        
        # Calculate moving averages
        window_size = min(10, len(primary_metrics) // 4)
        moving_avg = []
        
        for i in range(window_size, len(primary_metrics) + 1):
            window = primary_metrics[i-window_size:i]
            moving_avg.append(sum(window) / len(window))
        
        # Trend analysis
        if len(moving_avg) >= 2:
            trend_slope = (moving_avg[-1] - moving_avg[0]) / len(moving_avg)
            trend_direction = 'improving' if trend_slope > 0.01 else 'declining' if trend_slope < -0.01 else 'stable'
        else:
            trend_slope = 0
            trend_direction = 'unknown'
        
        return {
            'trend_direction': trend_direction,
            'trend_slope': trend_slope,
            'best_so_far_progression': self._calculate_best_so_far(primary_metrics),
            'convergence_rate': self._calculate_convergence_rate(moving_avg),
            'plateau_detection': self._detect_performance_plateau(primary_metrics)
        }
    
    def _analyze_component_impact(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze impact of different components on performance."""
        component_performance = defaultdict(list)
        
        for result in results:
            if 'combination' not in result:
                continue
                
            combination = result['combination']
            primary_metric = result.get('primary_metric', 0)
            
            # Analyze canonical ops impact
            for op in combination.canonical_ops:
                component_performance[f"canonical_op_{op}"].append(primary_metric)
            
            # Analyze hw kernels impact
            for kernel, option in combination.hw_kernels.items():
                component_performance[f"hw_kernel_{kernel}_{option}"].append(primary_metric)
            
            # Analyze transforms impact
            for transform in combination.model_topology:
                component_performance[f"transform_{transform}"].append(primary_metric)
        
        # Calculate statistics for each component
        component_stats = {}
        for component, performances in component_performance.items():
            if len(performances) >= 3:  # Minimum samples for meaningful statistics
                component_stats[component] = {
                    'mean_performance': np.mean(performances),
                    'std_performance': np.std(performances),
                    'sample_count': len(performances),
                    'performance_range': (np.min(performances), np.max(performances))
                }
        
        # Rank components by impact
        ranked_components = sorted(
            component_stats.items(),
            key=lambda x: x[1]['mean_performance'],
            reverse=True
        )
        
        return {
            'component_statistics': component_stats,
            'top_performing_components': ranked_components[:10],
            'component_frequency': {comp: len(perfs) for comp, perfs in component_performance.items()},
            'impact_analysis': self._calculate_component_impact_scores(component_stats)
        }
    
    def _analyze_metric_correlations(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze correlations between different metrics."""
        metrics_data = defaultdict(list)
        
        # Collect metric data
        for result in results:
            if 'metrics' in result:
                for metric, value in result['metrics'].items():
                    if isinstance(value, (int, float)):
                        metrics_data[metric].append(value)
        
        # Calculate correlations
        correlations = {}
        metric_names = list(metrics_data.keys())
        
        for i, metric1 in enumerate(metric_names):
            for metric2 in metric_names[i+1:]:
                if len(metrics_data[metric1]) == len(metrics_data[metric2]) and len(metrics_data[metric1]) > 1:
                    try:
                        corr = np.corrcoef(metrics_data[metric1], metrics_data[metric2])[0, 1]
                        if not np.isnan(corr):
                            correlations[f"{metric1}_vs_{metric2}"] = corr
                    except:
                        pass
        
        # Identify strong correlations
        strong_correlations = {
            k: v for k, v in correlations.items() 
            if abs(v) > 0.7
        }
        
        return {
            'all_correlations': correlations,
            'strong_correlations': strong_correlations,
            'metric_data_summary': {
                metric: {'count': len(values), 'range': (min(values), max(values))}
                for metric, values in metrics_data.items()
            }
        }
    
    def _identify_outliers(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify outlier results."""
        primary_metrics = [r.get('primary_metric', 0) for r in results]
        
        if len(primary_metrics) < 10:
            return {'message': 'Insufficient data for outlier detection'}
        
        # Calculate quartiles and IQR
        q1 = np.percentile(primary_metrics, 25)
        q3 = np.percentile(primary_metrics, 75)
        iqr = q3 - q1
        
        # Define outlier bounds
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Identify outliers
        outliers = []
        for i, metric in enumerate(primary_metrics):
            if metric < lower_bound or metric > upper_bound:
                outliers.append({
                    'index': i,
                    'value': metric,
                    'type': 'low' if metric < lower_bound else 'high',
                    'combination_id': results[i].get('combination', {}).get('combination_id', 'unknown')
                })
        
        return {
            'outlier_bounds': {'lower': lower_bound, 'upper': upper_bound},
            'outliers_found': len(outliers),
            'outlier_details': outliers[:10],  # Return top 10 outliers
            'outlier_percentage': len(outliers) / len(primary_metrics) * 100
        }
    
    def _generate_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Analyze best performing combinations
        best_results = sorted(results, key=lambda x: x.get('primary_metric', 0), reverse=True)[:5]
        
        if best_results:
            # Component recommendations
            best_components = set()
            for result in best_results:
                if 'combination' in result:
                    combo = result['combination']
                    best_components.update(f"canonical_op_{op}" for op in combo.canonical_ops)
                    best_components.update(f"hw_kernel_{k}_{v}" for k, v in combo.hw_kernels.items())
                    best_components.update(f"transform_{t}" for t in combo.model_topology)
            
            recommendations.append(f"Consider using components found in top performers: {', '.join(list(best_components)[:5])}")
        
        # Performance trend recommendations
        primary_metrics = [r.get('primary_metric', 0) for r in results]
        if len(primary_metrics) > 10:
            recent_avg = np.mean(primary_metrics[-10:])
            early_avg = np.mean(primary_metrics[:10])
            
            if recent_avg > early_avg * 1.1:
                recommendations.append("Performance is improving over time - continue current exploration strategy")
            elif recent_avg < early_avg * 0.9:
                recommendations.append("Performance is declining - consider adjusting exploration strategy")
            else:
                recommendations.append("Performance has plateaued - consider expanding search space or changing strategy")
        
        # Evaluation efficiency recommendations
        avg_eval_time = np.mean([r.get('evaluation_time', 0) for r in results if r.get('evaluation_time')])
        if avg_eval_time > 60:  # More than 1 minute per evaluation
            recommendations.append("Evaluation time is high - consider using faster evaluation methods or more aggressive caching")
        
        return recommendations
    
    def _calculate_best_so_far(self, metrics: List[float]) -> List[float]:
        """Calculate best-so-far progression."""
        best_so_far = []
        current_best = float('-inf')
        
        for metric in metrics:
            current_best = max(current_best, metric)
            best_so_far.append(current_best)
        
        return best_so_far
    
    def _calculate_convergence_rate(self, moving_averages: List[float]) -> float:
        """Calculate convergence rate from moving averages."""
        if len(moving_averages) < 5:
            return 0.0
        
        # Calculate rate of change in moving averages
        changes = [abs(moving_averages[i] - moving_averages[i-1]) for i in range(1, len(moving_averages))]
        recent_changes = changes[-5:]  # Last 5 changes
        
        return np.mean(recent_changes) if recent_changes else 0.0
    
    def _detect_performance_plateau(self, metrics: List[float], window_size: int = 10) -> Dict[str, Any]:
        """Detect if performance has plateaued."""
        if len(metrics) < window_size * 2:
            return {'plateau_detected': False, 'reason': 'insufficient_data'}
        
        recent_window = metrics[-window_size:]
        variance = np.var(recent_window)
        mean_recent = np.mean(recent_window)
        
        # Check if variance is low (indicating plateau)
        coefficient_of_variation = np.sqrt(variance) / mean_recent if mean_recent > 0 else 0
        
        plateau_detected = coefficient_of_variation < 0.05  # 5% coefficient of variation threshold
        
        return {
            'plateau_detected': plateau_detected,
            'coefficient_of_variation': coefficient_of_variation,
            'recent_variance': variance,
            'window_mean': mean_recent
        }
    
    def _calculate_component_impact_scores(self, component_stats: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate impact scores for components."""
        impact_scores = {}
        
        # Calculate overall performance baseline
        all_means = [stats['mean_performance'] for stats in component_stats.values()]
        if not all_means:
            return impact_scores
        
        baseline = np.mean(all_means)
        
        for component, stats in component_stats.items():
            # Impact score based on performance deviation from baseline
            # weighted by sample count (confidence)
            performance_deviation = stats['mean_performance'] - baseline
            confidence_weight = min(1.0, stats['sample_count'] / 10.0)  # Cap at 10 samples
            
            impact_scores[component] = performance_deviation * confidence_weight
        
        return impact_scores


class ParetoFrontierAnalyzer:
    """Analyzes and maintains Pareto frontiers for multi-objective optimization."""
    
    def __init__(self):
        """Initialize Pareto frontier analyzer."""
        pass
    
    def update_frontier(self, results: List[Dict[str, Any]], 
                       objectives: List[str],
                       directions: Optional[List[str]] = None) -> List[ComponentCombination]:
        """
        Update Pareto frontier with new results.
        
        Args:
            results: List of evaluation results
            objectives: List of objective names to optimize
            directions: List of optimization directions ('maximize' or 'minimize')
            
        Returns:
            List of combinations on Pareto frontier
        """
        if not results or not objectives:
            return []
        
        if directions is None:
            directions = ['maximize'] * len(objectives)
        
        if len(directions) != len(objectives):
            raise ValueError("Number of directions must match number of objectives")
        
        # Convert results to DSEResultsV2 format
        dse_results = []
        for result in results:
            if not result.get('success', False) or 'combination' not in result:
                continue
            
            metrics_data = result.get('metrics', {})
            metrics_data.update({k: v for k, v in result.items() 
                               if k not in ['combination', 'success', 'metrics'] and isinstance(v, (int, float))})
            
            metrics = PerformanceMetrics.from_dict(metrics_data)
            
            dse_result = DSEResultsV2(
                combination=result['combination'],
                metrics=metrics,
                evaluation_time=result.get('evaluation_time', 0),
                success=True
            )
            dse_results.append(dse_result)
        
        # Find Pareto frontier
        frontier = self._find_pareto_frontier(dse_results, objectives, directions)
        
        return [result.combination for result in frontier]
    
    def _find_pareto_frontier(self, results: List[DSEResultsV2], 
                            objectives: List[str], 
                            directions: List[str]) -> List[DSEResultsV2]:
        """Find Pareto frontier from results."""
        if not results:
            return []
        
        frontier = []
        
        for candidate in results:
            is_dominated = False
            
            # Check if candidate is dominated by any other result
            for other in results:
                if other != candidate and other.dominates(candidate, objectives, directions):
                    is_dominated = True
                    break
            
            if not is_dominated:
                frontier.append(candidate)
        
        return frontier
    
    def analyze_frontier_diversity(self, frontier: List[ComponentCombination]) -> Dict[str, Any]:
        """Analyze diversity of Pareto frontier."""
        if len(frontier) < 2:
            return {'diversity_score': 0.0, 'message': 'Insufficient frontier points'}
        
        # Analyze component diversity
        all_components = set()
        for combo in frontier:
            all_components.update(combo.canonical_ops)
            all_components.update(combo.hw_kernels.keys())
            all_components.update(combo.model_topology)
        
        # Calculate diversity metrics
        unique_components_per_combo = [
            len(set(combo.canonical_ops + list(combo.hw_kernels.keys()) + combo.model_topology))
            for combo in frontier
        ]
        
        diversity_score = len(all_components) / len(frontier) if frontier else 0
        
        return {
            'diversity_score': diversity_score,
            'total_unique_components': len(all_components),
            'frontier_size': len(frontier),
            'avg_components_per_solution': np.mean(unique_components_per_combo),
            'component_distribution': self._analyze_component_distribution(frontier)
        }
    
    def _analyze_component_distribution(self, frontier: List[ComponentCombination]) -> Dict[str, int]:
        """Analyze distribution of components across frontier."""
        component_counts = defaultdict(int)
        
        for combo in frontier:
            for op in combo.canonical_ops:
                component_counts[f"canonical_op_{op}"] += 1
            for kernel, option in combo.hw_kernels.items():
                component_counts[f"hw_kernel_{kernel}_{option}"] += 1
            for transform in combo.model_topology:
                component_counts[f"transform_{transform}"] += 1
        
        return dict(component_counts)