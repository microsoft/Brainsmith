"""
Automation utilities for result aggregation and analysis.
"""

import json
import itertools
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def generate_parameter_combinations(parameter_ranges: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Generate all combinations of parameters for parameter sweep.
    
    Args:
        parameter_ranges: Dict mapping parameter names to lists of values
        
    Returns:
        List of parameter dictionaries
        
    Example:
        combinations = generate_parameter_combinations({
            'pe_count': [4, 8, 16],
            'simd_width': [2, 4]
        })
        # Returns: [
        #   {'pe_count': 4, 'simd_width': 2},
        #   {'pe_count': 4, 'simd_width': 4},
        #   {'pe_count': 8, 'simd_width': 2},
        #   ...
        # ]
    """
    if not parameter_ranges:
        return [{}]
    
    keys = list(parameter_ranges.keys())
    values = list(parameter_ranges.values())
    
    combinations = []
    for combination in itertools.product(*values):
        param_dict = dict(zip(keys, combination))
        combinations.append(param_dict)
    
    return combinations


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate results from multiple forge() runs.
    
    Args:
        results: List of forge() results
        
    Returns:
        Aggregated analysis with statistics and best results
    """
    if not results:
        return {'error': 'No results to aggregate'}
    
    # Filter successful results
    successful_results = [r for r in results if r.get('success', True) and 'error' not in r]
    
    if not successful_results:
        return {
            'total_runs': len(results),
            'successful_runs': 0,
            'success_rate': 0.0,
            'error': 'No successful results found'
        }
    
    # Extract metrics from successful results
    all_metrics = []
    for result in successful_results:
        metrics = result.get('metrics', {})
        performance = metrics.get('performance', {})
        if performance:
            all_metrics.append(performance)
    
    # Calculate aggregate statistics
    aggregated_stats = {}
    if all_metrics:
        # Get all metric names
        all_metric_names = set()
        for metrics in all_metrics:
            all_metric_names.update(metrics.keys())
        
        # Calculate statistics for each metric
        for metric_name in all_metric_names:
            values = [m.get(metric_name, 0) for m in all_metrics if metric_name in m]
            if values:
                aggregated_stats[metric_name] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
                
                if len(values) > 1:
                    variance = sum((x - aggregated_stats[metric_name]['mean']) ** 2 for x in values) / (len(values) - 1)
                    aggregated_stats[metric_name]['std'] = variance ** 0.5
                else:
                    aggregated_stats[metric_name]['std'] = 0.0
    
    return {
        'total_runs': len(results),
        'successful_runs': len(successful_results),
        'success_rate': len(successful_results) / len(results),
        'aggregated_metrics': aggregated_stats,
        'best_results': find_top_results(successful_results, n=3)
    }


def find_best_result(results: List[Dict[str, Any]], 
                    metric: str = 'throughput',
                    maximize: bool = True) -> Optional[Dict[str, Any]]:
    """
    Find best result based on specified metric.
    
    Args:
        results: List of forge() results
        metric: Metric name to optimize
        maximize: Whether to maximize (True) or minimize (False) the metric
        
    Returns:
        Best result or None if no valid results found
    """
    if not results:
        return None
    
    # Filter successful results
    successful_results = [r for r in results if r.get('success', True) and 'error' not in r]
    
    if not successful_results:
        return None
    
    def get_metric_value(result: Dict[str, Any]) -> float:
        """Extract metric value from result."""
        metrics = result.get('metrics', {})
        performance = metrics.get('performance', {})
        return performance.get(metric, 0.0)
    
    # Find best result
    if maximize:
        best_result = max(successful_results, key=get_metric_value)
    else:
        best_result = min(successful_results, key=get_metric_value)
    
    # Add optimization metadata
    best_result['optimization_info'] = {
        'optimized_metric': metric,
        'maximize': maximize,
        'metric_value': get_metric_value(best_result),
        'total_candidates': len(results),
        'successful_candidates': len(successful_results)
    }
    
    return best_result


def find_top_results(results: List[Dict[str, Any]], 
                    n: int = 5,
                    metric: str = 'throughput') -> List[Dict[str, Any]]:
    """
    Find top N results based on metric.
    
    Args:
        results: List of forge() results
        n: Number of top results to return
        metric: Metric to rank by
        
    Returns:
        List of top N results
    """
    if not results:
        return []
    
    # Filter successful results
    successful_results = [r for r in results if r.get('success', True) and 'error' not in r]
    
    def get_metric_value(result: Dict[str, Any]) -> float:
        metrics = result.get('metrics', {})
        performance = metrics.get('performance', {})
        return performance.get(metric, 0.0)
    
    # Sort by metric (descending)
    sorted_results = sorted(successful_results, key=get_metric_value, reverse=True)
    
    # Return top N
    top_results = sorted_results[:n]
    
    # Add ranking metadata
    for i, result in enumerate(top_results):
        result['ranking_info'] = {
            'rank': i + 1,
            'ranked_by': metric,
            'metric_value': get_metric_value(result)
        }
    
    return top_results


def save_automation_results(results: List[Dict[str, Any]], 
                           output_path: str,
                           include_analysis: bool = True) -> None:
    """
    Save automation results to file.
    
    Args:
        results: List of forge() results
        output_path: Path to save results
        include_analysis: Whether to include aggregated analysis
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for saving
    save_data = {
        'automation_results': results,
        'summary': {
            'total_runs': len(results),
            'successful_runs': sum(1 for r in results if r.get('success', True) and 'error' not in r),
            'timestamp': str(Path().resolve()),
        }
    }
    
    # Add aggregated analysis if requested
    if include_analysis:
        save_data['aggregated_analysis'] = aggregate_results(results)
    
    # Save as JSON
    try:
        with open(output_path, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        logger.info(f"Automation results saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save automation results: {e}")
        raise


def load_automation_results(file_path: str) -> Dict[str, Any]:
    """
    Load automation results from file.
    
    Args:
        file_path: Path to results file
        
    Returns:
        Loaded automation results
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded automation results from {file_path}")
        return data
        
    except Exception as e:
        logger.error(f"Failed to load automation results: {e}")
        raise


def compare_automation_runs(results1: List[Dict[str, Any]], 
                           results2: List[Dict[str, Any]],
                           metric: str = 'throughput') -> Dict[str, Any]:
    """
    Compare results from two automation runs.
    
    Args:
        results1: First set of results
        results2: Second set of results
        metric: Metric to compare
        
    Returns:
        Comparison analysis
    """
    def get_metric_values(results):
        values = []
        for result in results:
            if result.get('success', True) and 'error' not in result:
                metrics = result.get('metrics', {})
                performance = metrics.get('performance', {})
                if metric in performance:
                    values.append(performance[metric])
        return values
    
    values1 = get_metric_values(results1)
    values2 = get_metric_values(results2)
    
    if not values1 or not values2:
        return {'error': 'Insufficient data for comparison'}
    
    mean1 = sum(values1) / len(values1)
    mean2 = sum(values2) / len(values2)
    
    improvement = ((mean2 - mean1) / mean1 * 100) if mean1 != 0 else 0
    
    return {
        'metric': metric,
        'run1': {
            'mean': mean1,
            'best': max(values1),
            'count': len(values1)
        },
        'run2': {
            'mean': mean2,
            'best': max(values2),
            'count': len(values2)
        },
        'improvement_percent': improvement,
        'better_run': 'run2' if improvement > 0 else 'run1'
    }