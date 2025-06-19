"""
Parameter Sweep and Result Analysis

Simple utilities for exploring parameter spaces and analyzing results
by running forge() multiple times with different configurations.
"""

import itertools
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)


def parameter_sweep(
    model_path: str,
    blueprint_path: str, 
    param_ranges: Dict[str, List[Any]],
    max_workers: int = 4,
    progress_callback: Optional[callable] = None
) -> List[Dict[str, Any]]:
    """
    Run forge() with different parameter combinations.
    
    Args:
        model_path: Path to ONNX model
        blueprint_path: Path to blueprint YAML
        param_ranges: Dict mapping parameter names to lists of values
        max_workers: Number of parallel workers
        progress_callback: Optional callback for progress updates
        
    Returns:
        List of forge() results with sweep metadata
        
    Example:
        results = parameter_sweep(
            "model.onnx", 
            "blueprint.yaml",
            {
                'pe_count': [4, 8, 16, 32],
                'simd_width': [2, 4, 8, 16]
            }
        )
    """
    from ...core.api import forge
    from ..hooks import track_parameter
    
    # Generate parameter combinations
    combinations = _generate_combinations(param_ranges)
    total = len(combinations)
    
    logger.info(f"Starting parameter sweep: {total} combinations")
    
    def run_combination(params: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Run single parameter combination."""
        try:
            # Track parameters in hooks system
            for param_name, param_value in params.items():
                track_parameter(param_name, param_value)
            
            # Run forge with parameters as constraints
            result = forge(
                model_path=model_path,
                blueprint_path=blueprint_path,
                constraints=params
            )
            
            # Add sweep metadata
            result['sweep_info'] = {
                'parameters': params,
                'index': index,
                'success': True
            }
            
            if progress_callback:
                progress_callback(index + 1, total, params)
            
            return result
            
        except Exception as e:
            logger.error(f"Parameter combination {index} failed: {e}")
            return {
                'sweep_info': {
                    'parameters': params,
                    'index': index,
                    'success': False,
                    'error': str(e)
                }
            }
    
    # Execute with optional parallelization
    results = []
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_combination, params, i): i
                for i, params in enumerate(combinations)
            }
            
            for future in as_completed(futures):
                results.append(future.result())
    else:
        for i, params in enumerate(combinations):
            results.append(run_combination(params, i))
    
    # Sort by index
    results.sort(key=lambda x: x.get('sweep_info', {}).get('index', 0))
    
    successful = sum(1 for r in results if r.get('sweep_info', {}).get('success', False))
    logger.info(f"Parameter sweep completed: {successful}/{total} successful")
    
    return results


def find_best(
    results: List[Dict[str, Any]], 
    metric: str = 'throughput',
    maximize: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Find optimal result by specified metric.
    
    Args:
        results: List of forge() results
        metric: Metric name to optimize ('throughput', 'latency', 'power', etc.)
        maximize: Whether to maximize (True) or minimize (False) the metric
        
    Returns:
        Best result or None if no valid results found
        
    Example:
        best = find_best(results, metric='throughput', maximize=True)
    """
    if not results:
        return None
    
    # Filter successful results
    successful = [r for r in results if _is_successful_result(r)]
    
    if not successful:
        logger.warning("No successful results found for optimization")
        return None
    
    def get_metric_value(result: Dict[str, Any]) -> float:
        """Extract metric value from result."""
        metrics = result.get('metrics', {})
        performance = metrics.get('performance', {})
        return performance.get(metric, 0.0)
    
    # Find best result
    if maximize:
        best_result = max(successful, key=get_metric_value)
    else:
        best_result = min(successful, key=get_metric_value)
    
    # Add optimization metadata
    best_result['optimization_info'] = {
        'optimized_metric': metric,
        'maximize': maximize,
        'metric_value': get_metric_value(best_result),
        'total_candidates': len(results),
        'successful_candidates': len(successful)
    }
    
    logger.info(f"Best result: {metric}={get_metric_value(best_result):.2f}")
    return best_result


def aggregate_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate statistical summary of results.
    
    Args:
        results: List of forge() results
        
    Returns:
        Statistical summary with aggregated metrics
        
    Example:
        stats = aggregate_stats(results)
        print(f"Success rate: {stats['success_rate']:.1%}")
    """
    if not results:
        return {'error': 'No results to aggregate'}
    
    # Filter successful results
    successful = [r for r in results if _is_successful_result(r)]
    
    if not successful:
        return {
            'total_runs': len(results),
            'successful_runs': 0,
            'success_rate': 0.0,
            'error': 'No successful results found'
        }
    
    # Extract metrics from successful results
    all_metrics = []
    for result in successful:
        metrics = result.get('metrics', {})
        performance = metrics.get('performance', {})
        if performance:
            all_metrics.append(performance)
    
    # Calculate statistics for each metric
    aggregated_metrics = {}
    if all_metrics:
        # Get all metric names
        metric_names = set()
        for metrics in all_metrics:
            metric_names.update(metrics.keys())
        
        # Calculate stats for each metric
        for metric_name in metric_names:
            values = [m.get(metric_name, 0) for m in all_metrics if metric_name in m]
            if values:
                aggregated_metrics[metric_name] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
                
                if len(values) > 1:
                    variance = sum((x - aggregated_metrics[metric_name]['mean']) ** 2 for x in values) / (len(values) - 1)
                    aggregated_metrics[metric_name]['std'] = variance ** 0.5
                else:
                    aggregated_metrics[metric_name]['std'] = 0.0
    
    return {
        'total_runs': len(results),
        'successful_runs': len(successful),
        'success_rate': len(successful) / len(results),
        'aggregated_metrics': aggregated_metrics
    }


def _generate_combinations(param_ranges: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Generate all combinations of parameters."""
    if not param_ranges:
        return [{}]
    
    keys = list(param_ranges.keys())
    values = list(param_ranges.values())
    
    combinations = []
    for combination in itertools.product(*values):
        param_dict = dict(zip(keys, combination))
        combinations.append(param_dict)
    
    return combinations


def _is_successful_result(result: Dict[str, Any]) -> bool:
    """Check if result represents a successful forge() run."""
    # Check sweep_info for success
    if 'sweep_info' in result:
        return result['sweep_info'].get('success', False)
    
    # Check for error indicators
    if 'error' in result or result.get('success') is False:
        return False
    
    # Check if we have metrics (indicates successful run)
    return 'metrics' in result and bool(result['metrics'])