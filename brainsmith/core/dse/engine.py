"""
Core DSE Functions - North Star Aligned

Simple functions for FPGA design space exploration.
Integrates seamlessly with streamlined BrainSmith modules.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .types import DSEResult, ParameterSet, ComparisonResult, DSEConfiguration, ParameterSpace
from .helpers import generate_parameter_grid, estimate_runtime

logger = logging.getLogger(__name__)

# Direct imports - fail immediately if missing
from ...core.api import forge
from ...core.metrics import create_metrics
from .blueprint_functions import load_blueprint_yaml, get_build_steps, get_objectives
from ..hooks import log_dse_event, log_optimization_event
from ..finn import build_accelerator


def parameter_sweep(
    model_path: str,
    blueprint_path: str, 
    parameters: ParameterSpace,
    config: Optional[DSEConfiguration] = None
) -> List[DSEResult]:
    """
    Core BrainSmith DSE function: Run parameter combinations efficiently.
    
    Args:
        model_path: Path to ONNX model
        blueprint_path: Path to blueprint YAML
        parameters: Dictionary of parameter names to value lists
        config: Optional DSE configuration
        
    Returns:
        List of DSE results for all parameter combinations
        
    Example:
        parameters = {
            'pe_count': [1, 2, 4, 8],
            'simd_factor': [1, 2, 4],
            'precision': [8, 16]
        }
        results = parameter_sweep('model.onnx', 'blueprint.yaml', parameters)
    """
    if config is None:
        config = DSEConfiguration()
    
    # Load blueprint using simplified functions
    blueprint_data = load_blueprint_yaml(blueprint_path)
    
    # Generate parameter combinations
    param_combinations = generate_parameter_grid(parameters)
    total_combinations = len(param_combinations)
    
    logger.info(f"Starting parameter sweep: {total_combinations} combinations")
    
    # Log DSE start event
    log_optimization_event('dse_start', {
        'model': model_path,
        'blueprint': blueprint_path,
        'parameter_count': len(parameters),
        'total_combinations': total_combinations,
        'parallel_workers': config.max_parallel
    })
    
    # Estimate runtime
    estimated_time = estimate_runtime(param_combinations, benchmark_time=30.0)
    logger.info(f"Estimated runtime: {estimated_time:.1f} seconds")
    
    results = []
    start_time = time.time()
    
    if config.max_parallel > 1:
        # Parallel execution
        results = _run_parallel_sweep(
            model_path, blueprint_data, param_combinations, config
        )
    else:
        # Sequential execution
        results = _run_sequential_sweep(
            model_path, blueprint_data, param_combinations, config
        )
    
    # Log completion
    total_time = time.time() - start_time
    success_rate = sum(1 for r in results if r.build_success) / len(results) if results else 0
    
    log_optimization_event('dse_complete', {
        'results_count': len(results),
        'success_rate': success_rate,
        'total_time': total_time,
        'avg_time_per_combination': total_time / len(results) if results else 0
    })
    
    logger.info(f"Parameter sweep complete: {len(results)} results, {success_rate:.1%} success rate")
    
    return results


def batch_evaluate(
    model_list: List[str],
    blueprint_path: str,
    parameters: Dict[str, Any],
    config: Optional[DSEConfiguration] = None
) -> Dict[str, DSEResult]:
    """
    Evaluate multiple models with same parameters.
    
    Args:
        model_list: List of model paths
        blueprint_path: Path to blueprint YAML
        parameters: Single parameter combination
        config: Optional DSE configuration
        
    Returns:
        Dictionary mapping model paths to DSE results
    """
    if config is None:
        config = DSEConfiguration()
    
    logger.info(f"Batch evaluating {len(model_list)} models")
    
    log_optimization_event('batch_evaluation_start', {
        'model_count': len(model_list),
        'parameters': parameters
    })
    
    results = {}
    
    for model_path in model_list:
        log_dse_event('model_evaluation_start', {
            'model': model_path,
            'parameters': parameters
        })
        
        try:
            result = _evaluate_single_configuration(
                model_path, blueprint_path, parameters
            )
            results[model_path] = result
            
            log_dse_event('model_evaluation_complete', {
                'model': model_path,
                'success': result.build_success,
                'build_time': result.build_time
            })
            
        except Exception as e:
            logger.error(f"Model evaluation failed for {model_path}: {e}")
            if config.continue_on_failure:
                # Create failed result
                results[model_path] = DSEResult(
                    parameters=parameters,
                    metrics=create_metrics(),
                    build_success=False,
                    build_time=0.0,
                    metadata={'error': str(e)}
                )
            else:
                raise
    
    log_optimization_event('batch_evaluation_complete', {
        'results_count': len(results),
        'success_count': sum(1 for r in results.values() if r.build_success)
    })
    
    return results


def find_best_result(
    results: List[DSEResult],
    metric: str,
    direction: str = 'maximize'
) -> Optional[DSEResult]:
    """
    Find best result based on a single metric.
    
    Args:
        results: List of DSE results
        metric: Metric name (e.g., 'throughput_ops_sec', 'latency_ms')
        direction: 'maximize' or 'minimize'
        
    Returns:
        Best DSE result or None if no successful results
        
    Example:
        best = find_best_result(results, 'throughput_ops_sec', 'maximize')
    """
    if not results:
        return None
    
    # Filter to successful results only
    successful_results = [r for r in results if r.build_success]
    if not successful_results:
        logger.warning("No successful results found")
        return None
    
    def get_metric_value(result: DSEResult) -> float:
        """Extract metric value from result."""
        try:
            # Navigate nested metric structure
            value = result.metrics
            for part in metric.split('.'):
                if hasattr(value, part):
                    value = getattr(value, part)
                elif isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return float('-inf') if direction == 'maximize' else float('inf')
            
            return float(value) if value is not None else (
                float('-inf') if direction == 'maximize' else float('inf')
            )
        except (AttributeError, ValueError, TypeError):
            return float('-inf') if direction == 'maximize' else float('inf')
    
    # Find best result
    if direction == 'maximize':
        best_result = max(successful_results, key=get_metric_value)
    else:
        best_result = min(successful_results, key=get_metric_value)
    
    best_value = get_metric_value(best_result)
    logger.info(f"Best result: {metric} = {best_value} ({direction})")
    
    return best_result


def compare_results(
    results: List[DSEResult],
    metrics: List[str],
    weights: Optional[List[float]] = None
) -> ComparisonResult:
    """
    Compare results across multiple metrics.
    
    Args:
        results: List of DSE results
        metrics: List of metric names to compare
        weights: Optional weights for each metric (default: equal weights)
        
    Returns:
        Comparison result with ranking and analysis
    """
    if not results:
        return ComparisonResult(
            best_result=None,
            ranking=[],
            comparison_metric='none',
            summary_stats={}
        )
    
    if weights is None:
        weights = [1.0] * len(metrics)
    
    if len(weights) != len(metrics):
        raise ValueError("Number of weights must match number of metrics")
    
    # Filter successful results
    successful_results = [r for r in results if r.build_success]
    
    if not successful_results:
        return ComparisonResult(
            best_result=None,
            ranking=[],
            comparison_metric='no_successful_results',
            summary_stats={'total_results': len(results), 'successful_results': 0}
        )
    
    # Calculate composite scores
    scored_results = []
    
    for result in successful_results:
        score = 0.0
        valid_metrics = 0
        
        for metric, weight in zip(metrics, weights):
            try:
                # Extract metric value
                value = result.metrics
                for part in metric.split('.'):
                    if hasattr(value, part):
                        value = getattr(value, part)
                    elif isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        value = None
                        break
                
                if value is not None:
                    score += weight * float(value)
                    valid_metrics += 1
                    
            except (AttributeError, ValueError, TypeError):
                continue
        
        if valid_metrics > 0:
            normalized_score = score / valid_metrics
            scored_results.append((normalized_score, result))
    
    # Sort by score (descending)
    scored_results.sort(key=lambda x: x[0], reverse=True)
    ranking = [result for _, result in scored_results]
    
    # Calculate summary statistics
    summary_stats = {
        'total_results': len(results),
        'successful_results': len(successful_results),
        'ranked_results': len(ranking),
        'metrics_compared': metrics,
        'weights_used': weights
    }
    
    if ranking:
        best_score = scored_results[0][0]
        summary_stats['best_score'] = best_score
        summary_stats['score_range'] = (
            scored_results[-1][0], scored_results[0][0]
        ) if len(scored_results) > 1 else (best_score, best_score)
    
    return ComparisonResult(
        best_result=ranking[0] if ranking else None,
        ranking=ranking,
        comparison_metric=f"weighted_composite_{len(metrics)}_metrics",
        summary_stats=summary_stats
    )


def sample_design_space(
    parameters: ParameterSpace,
    strategy: str = 'random',
    n_samples: int = 10,
    seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Sample parameter combinations from design space.
    
    Args:
        parameters: Parameter space definition
        strategy: Sampling strategy ('random', 'lhs', 'grid')
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of parameter combinations
    """
    from .helpers import create_parameter_samples
    
    logger.info(f"Sampling design space: {strategy} strategy, {n_samples} samples")
    
    samples = create_parameter_samples(parameters, strategy, n_samples, seed)
    
    log_dse_event('design_space_sampled', {
        'strategy': strategy,
        'n_samples': len(samples),
        'parameter_count': len(parameters)
    })
    
    return samples


def _run_sequential_sweep(
    model_path: str,
    blueprint_data: Dict[str, Any],
    param_combinations: List[Dict[str, Any]],
    config: DSEConfiguration
) -> List[DSEResult]:
    """Run parameter sweep sequentially."""
    results = []
    
    for i, params in enumerate(param_combinations):
        logger.info(f"Evaluating combination {i+1}/{len(param_combinations)}: {params}")
        
        try:
            result = _evaluate_single_configuration(model_path, blueprint_data, params)
            results.append(result)
            
            log_dse_event('parameter_evaluation_complete', {
                'combination_index': i,
                'parameters': params,
                'success': result.build_success,
                'build_time': result.build_time
            })
            
        except Exception as e:
            logger.error(f"Configuration evaluation failed: {e}")
            if config.continue_on_failure:
                # Add failed result
                failed_result = DSEResult(
                    parameters=params,
                    metrics=create_metrics(),
                    build_success=False,
                    build_time=0.0,
                    metadata={'error': str(e)}
                )
                results.append(failed_result)
            else:
                raise
    
    return results


def _run_parallel_sweep(
    model_path: str,
    blueprint_data: Dict[str, Any],
    param_combinations: List[Dict[str, Any]],
    config: DSEConfiguration
) -> List[DSEResult]:
    """Run parameter sweep in parallel."""
    results = []
    
    with ThreadPoolExecutor(max_workers=config.max_parallel) as executor:
        # Submit all evaluations
        future_to_params = {
            executor.submit(_evaluate_single_configuration, model_path, blueprint_data, params): params
            for params in param_combinations
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_params, timeout=config.timeout_seconds):
            params = future_to_params[future]
            
            try:
                result = future.result()
                results.append(result)
                
                log_dse_event('parameter_evaluation_complete', {
                    'parameters': params,
                    'success': result.build_success,
                    'build_time': result.build_time
                })
                
            except Exception as e:
                logger.error(f"Parallel evaluation failed for {params}: {e}")
                if config.continue_on_failure:
                    failed_result = DSEResult(
                        parameters=params,
                        metrics=create_metrics(),
                        build_success=False,
                        build_time=0.0,
                        metadata={'error': str(e)}
                    )
                    results.append(failed_result)
                else:
                    raise
    
    return results


def _evaluate_single_configuration(
    model_path: str,
    blueprint_data: Dict[str, Any],
    parameters: Dict[str, Any]
) -> DSEResult:
    """
    Evaluate single parameter configuration using streamlined modules.
    
    Integrates with:
    - brainsmith.core.api.forge() for builds
    - brainsmith.finn.build_accelerator() for FINN integration
    - brainsmith.core.metrics for result metrics
    """
    start_time = time.time()
    
    try:
        # Use simplified core API
        forge_result = forge(
            model_path=model_path,
            blueprint_path=None,  # Pass blueprint data directly
            **parameters
        )
        
        # Extract metrics from forge result
        if 'metrics' in forge_result:
            metrics = forge_result['metrics']
        else:
            metrics = create_metrics()
        
        build_success = forge_result.get('build_success', True)
        
    except Exception as e:
        logger.error(f"Configuration evaluation failed: {e}")
        metrics = create_metrics()
        build_success = False
    
    build_time = time.time() - start_time
    
    return DSEResult(
        parameters=parameters,
        metrics=metrics,
        build_success=build_success,
        build_time=build_time,
        metadata={'evaluation_timestamp': start_time}
    )