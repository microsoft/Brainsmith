"""
Parameter Sweep Automation

Simple utilities for exploring design parameter spaces by running forge()
with different parameter combinations.
"""

import itertools
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)


def parameter_sweep(
    model_path: str,
    blueprint_path: str,
    parameter_ranges: Dict[str, List[Any]],
    max_workers: int = 4,
    progress_callback: Optional[callable] = None
) -> List[Dict[str, Any]]:
    """
    Run parameter sweep by calling forge() with different parameter combinations.
    
    Args:
        model_path: Path to ONNX model
        blueprint_path: Path to blueprint YAML
        parameter_ranges: Dict mapping parameter names to lists of values
        max_workers: Number of parallel workers
        progress_callback: Optional callback for progress updates
        
    Returns:
        List of forge() results with parameter information
        
    Example:
        results = parameter_sweep(
            "model.onnx", 
            "blueprint.yaml",
            {
                'pe_count': [4, 8, 16, 32],
                'simd_width': [2, 4, 8, 16],
                'frequency': [100, 150, 200]
            }
        )
    """
    from ..core.api import forge
    
    # Generate all parameter combinations
    combinations = generate_parameter_combinations(parameter_ranges)
    total_combinations = len(combinations)
    
    logger.info(f"Starting parameter sweep with {total_combinations} combinations")
    
    results = []
    
    def run_single_combination(params: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Run forge() with single parameter combination."""
        try:
            # Create objectives/constraints from parameters
            objectives = _extract_objectives_from_params(params)
            constraints = _extract_constraints_from_params(params)
            
            # Run forge with parameters
            result = forge(
                model_path=model_path,
                blueprint_path=blueprint_path,
                objectives=objectives,
                constraints=constraints
            )
            
            # Add parameter information to result
            result['sweep_parameters'] = params
            result['sweep_index'] = index
            result['success'] = True
            
            if progress_callback:
                progress_callback(index + 1, total_combinations, params)
            
            return result
            
        except Exception as e:
            logger.error(f"Parameter combination {index} failed: {e}")
            return {
                'sweep_parameters': params,
                'sweep_index': index,
                'success': False,
                'error': str(e)
            }
    
    # Run parameter sweep with parallel execution
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_params = {
                executor.submit(run_single_combination, params, i): (params, i)
                for i, params in enumerate(combinations)
            }
            
            for future in as_completed(future_to_params):
                result = future.result()
                results.append(result)
    else:
        # Sequential execution
        for i, params in enumerate(combinations):
            result = run_single_combination(params, i)
            results.append(result)
    
    # Sort results by sweep index
    results.sort(key=lambda x: x.get('sweep_index', 0))
    
    successful_runs = sum(1 for r in results if r.get('success', False))
    logger.info(f"Parameter sweep completed: {successful_runs}/{total_combinations} successful")
    
    return results


def grid_search(
    model_path: str,
    blueprint_path: str,
    parameter_grid: Dict[str, List[Any]],
    metric: str = 'throughput',
    maximize: bool = True
) -> Dict[str, Any]:
    """
    Grid search to find best parameter combination.
    
    Args:
        model_path: Path to ONNX model
        blueprint_path: Path to blueprint YAML
        parameter_grid: Grid of parameters to search
        metric: Metric to optimize
        maximize: Whether to maximize (True) or minimize (False) metric
        
    Returns:
        Best result with parameters
    """
    # Run parameter sweep
    results = parameter_sweep(model_path, blueprint_path, parameter_grid)
    
    # Filter successful results
    successful_results = [r for r in results if r.get('success', False)]
    
    if not successful_results:
        raise ValueError("No successful parameter combinations found")
    
    # Find best result based on metric
    def get_metric_value(result):
        metrics = result.get('metrics', {})
        performance = metrics.get('performance', {})
        return performance.get(metric, 0.0)
    
    if maximize:
        best_result = max(successful_results, key=get_metric_value)
    else:
        best_result = min(successful_results, key=get_metric_value)
    
    # Add grid search metadata
    best_result['grid_search'] = {
        'total_combinations': len(results),
        'successful_combinations': len(successful_results),
        'optimization_metric': metric,
        'maximize': maximize,
        'best_metric_value': get_metric_value(best_result)
    }
    
    return best_result


def random_search(
    model_path: str,
    blueprint_path: str,
    parameter_distributions: Dict[str, Any],
    n_iterations: int = 20,
    random_seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Random search over parameter space.
    
    Args:
        model_path: Path to ONNX model
        blueprint_path: Path to blueprint YAML
        parameter_distributions: Parameter distributions (ranges or choices)
        n_iterations: Number of random samples
        random_seed: Random seed for reproducibility
        
    Returns:
        Best result from random search
    """
    import random
    
    if random_seed is not None:
        random.seed(random_seed)
    
    # Generate random parameter combinations
    random_combinations = []
    for _ in range(n_iterations):
        combination = {}
        for param_name, distribution in parameter_distributions.items():
            if isinstance(distribution, list):
                # Choose randomly from list
                combination[param_name] = random.choice(distribution)
            elif isinstance(distribution, tuple) and len(distribution) == 2:
                # Random value in range
                min_val, max_val = distribution
                if isinstance(min_val, int) and isinstance(max_val, int):
                    combination[param_name] = random.randint(min_val, max_val)
                else:
                    combination[param_name] = random.uniform(min_val, max_val)
            else:
                raise ValueError(f"Invalid distribution for parameter {param_name}")
        
        random_combinations.append(combination)
    
    # Run parameter sweep with generated combinations
    results = []
    for i, params in enumerate(random_combinations):
        try:
            from ..core.api import forge
            
            objectives = _extract_objectives_from_params(params)
            constraints = _extract_constraints_from_params(params)
            
            result = forge(
                model_path=model_path,
                blueprint_path=blueprint_path,
                objectives=objectives,
                constraints=constraints
            )
            
            result['random_parameters'] = params
            result['iteration'] = i
            result['success'] = True
            results.append(result)
            
        except Exception as e:
            results.append({
                'random_parameters': params,
                'iteration': i,
                'success': False,
                'error': str(e)
            })
    
    # Find best result
    successful_results = [r for r in results if r.get('success', False)]
    if not successful_results:
        raise ValueError("No successful random search iterations")
    
    # Use first metric found for optimization
    def get_first_metric_value(result):
        metrics = result.get('metrics', {})
        performance = metrics.get('performance', {})
        if performance:
            return list(performance.values())[0]
        return 0.0
    
    best_result = max(successful_results, key=get_first_metric_value)
    
    # Add random search metadata
    best_result['random_search'] = {
        'total_iterations': n_iterations,
        'successful_iterations': len(successful_results),
        'random_seed': random_seed
    }
    
    return best_result


def generate_parameter_combinations(parameter_ranges: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Generate all combinations of parameters."""
    keys = list(parameter_ranges.keys())
    values = list(parameter_ranges.values())
    
    combinations = []
    for combination in itertools.product(*values):
        param_dict = dict(zip(keys, combination))
        combinations.append(param_dict)
    
    return combinations


def _extract_objectives_from_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Extract objectives from parameters (if any)."""
    objectives = {}
    
    # Map common parameter names to objectives
    param_to_objective = {
        'target_throughput': 'throughput',
        'target_latency': 'latency', 
        'target_power': 'power'
    }
    
    for param_name, obj_name in param_to_objective.items():
        if param_name in params:
            objectives[obj_name] = {'direction': 'maximize' if obj_name == 'throughput' else 'minimize'}
    
    return objectives


def _extract_constraints_from_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Extract constraints from parameters (if any)."""
    constraints = {}
    
    # Map common parameter names to constraints
    param_to_constraint = {
        'max_luts': 'max_luts',
        'max_dsps': 'max_dsps',
        'max_power': 'max_power',
        'target_frequency': 'target_frequency'
    }
    
    for param_name, constraint_name in param_to_constraint.items():
        if param_name in params:
            constraints[constraint_name] = params[param_name]
    
    return constraints