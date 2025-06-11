"""
DSE Helper Functions

Utility functions for design space exploration including parameter generation,
sampling strategies, and runtime estimation.
"""

import itertools
import random
import logging
from typing import Dict, List, Any, Optional
import math

logger = logging.getLogger(__name__)

# Try to import advanced sampling libraries
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.debug("NumPy not available - using basic sampling")

try:
    from scipy.stats import qmc
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.debug("SciPy not available - using basic sampling")


def generate_parameter_grid(parameters: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Generate all combinations of parameters for grid search.
    
    Args:
        parameters: Dictionary mapping parameter names to lists of values
        
    Returns:
        List of parameter combination dictionaries
        
    Example:
        params = {'pe_count': [1, 2, 4], 'simd': [1, 2]}
        combinations = generate_parameter_grid(params)
        # Returns: [{'pe_count': 1, 'simd': 1}, {'pe_count': 1, 'simd': 2}, ...]
    """
    if not parameters:
        return [{}]
    
    param_names = list(parameters.keys())
    param_values = list(parameters.values())
    
    combinations = []
    for combination in itertools.product(*param_values):
        param_dict = dict(zip(param_names, combination))
        combinations.append(param_dict)
    
    logger.info(f"Generated {len(combinations)} parameter combinations")
    return combinations


def create_parameter_samples(
    parameters: Dict[str, List[Any]],
    strategy: str = 'random',
    n_samples: int = 10,
    seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Sample parameter combinations using different strategies.
    
    Args:
        parameters: Dictionary mapping parameter names to lists of values
        strategy: Sampling strategy ('random', 'lhs', 'grid', 'sobol')
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of sampled parameter combinations
    """
    if seed is not None:
        random.seed(seed)
        if NUMPY_AVAILABLE:
            np.random.seed(seed)
    
    param_names = list(parameters.keys())
    param_values = list(parameters.values())
    
    if strategy == 'grid':
        # For grid, return all combinations (ignore n_samples)
        return generate_parameter_grid(parameters)
    
    elif strategy == 'random':
        return _random_sampling(param_names, param_values, n_samples)
    
    elif strategy == 'lhs' and SCIPY_AVAILABLE:
        return _latin_hypercube_sampling(param_names, param_values, n_samples)
    
    elif strategy == 'sobol' and SCIPY_AVAILABLE:
        return _sobol_sampling(param_names, param_values, n_samples)
    
    else:
        logger.warning(f"Strategy '{strategy}' not available, using random sampling")
        return _random_sampling(param_names, param_values, n_samples)


def _random_sampling(
    param_names: List[str],
    param_values: List[List[Any]],
    n_samples: int
) -> List[Dict[str, Any]]:
    """Random sampling from parameter space."""
    samples = []
    
    for _ in range(n_samples):
        sample = {}
        for name, values in zip(param_names, param_values):
            sample[name] = random.choice(values)
        samples.append(sample)
    
    return samples


def _latin_hypercube_sampling(
    param_names: List[str],
    param_values: List[List[Any]],
    n_samples: int
) -> List[Dict[str, Any]]:
    """Latin Hypercube Sampling for better space coverage."""
    if not SCIPY_AVAILABLE:
        return _random_sampling(param_names, param_values, n_samples)
    
    n_dims = len(param_names)
    sampler = qmc.LatinHypercube(d=n_dims, seed=42)
    lhs_samples = sampler.random(n=n_samples)
    
    samples = []
    for lhs_sample in lhs_samples:
        sample = {}
        for i, (name, values) in enumerate(zip(param_names, param_values)):
            # Map LHS sample [0,1] to discrete parameter index
            idx = int(lhs_sample[i] * len(values))
            idx = min(idx, len(values) - 1)  # Ensure valid index
            sample[name] = values[idx]
        samples.append(sample)
    
    return samples


def _sobol_sampling(
    param_names: List[str],
    param_values: List[List[Any]],
    n_samples: int
) -> List[Dict[str, Any]]:
    """Sobol sequence sampling for low-discrepancy sampling."""
    if not SCIPY_AVAILABLE:
        return _random_sampling(param_names, param_values, n_samples)
    
    n_dims = len(param_names)
    sampler = qmc.Sobol(d=n_dims, seed=42)
    sobol_samples = sampler.random(n=n_samples)
    
    samples = []
    for sobol_sample in sobol_samples:
        sample = {}
        for i, (name, values) in enumerate(zip(param_names, param_values)):
            # Map Sobol sample [0,1] to discrete parameter index
            idx = int(sobol_sample[i] * len(values))
            idx = min(idx, len(values) - 1)  # Ensure valid index
            sample[name] = values[idx]
        samples.append(sample)
    
    return samples


def estimate_runtime(
    param_combinations: List[Dict[str, Any]],
    benchmark_time: float = 30.0
) -> float:
    """
    Estimate total runtime for parameter sweep.
    
    Args:
        param_combinations: List of parameter combinations to evaluate
        benchmark_time: Estimated time per evaluation in seconds
        
    Returns:
        Estimated total runtime in seconds
    """
    n_combinations = len(param_combinations)
    estimated_time = n_combinations * benchmark_time
    
    logger.debug(f"Runtime estimation: {n_combinations} combinations × {benchmark_time}s = {estimated_time}s")
    
    return estimated_time


def calculate_parameter_space_size(parameters: Dict[str, List[Any]]) -> int:
    """
    Calculate total size of parameter space.
    
    Args:
        parameters: Dictionary mapping parameter names to lists of values
        
    Returns:
        Total number of possible parameter combinations
    """
    if not parameters:
        return 0
    
    total_size = 1
    for param_values in parameters.values():
        total_size *= len(param_values)
    
    return total_size


def validate_parameter_space(parameters: Dict[str, List[Any]]) -> tuple[bool, List[str]]:
    """
    Validate parameter space definition.
    
    Args:
        parameters: Parameter space to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    if not parameters:
        errors.append("Parameter space is empty")
        return False, errors
    
    for param_name, param_values in parameters.items():
        if not isinstance(param_name, str):
            errors.append(f"Parameter name must be string, got {type(param_name)}")
        
        if not isinstance(param_values, list):
            errors.append(f"Parameter '{param_name}' values must be list, got {type(param_values)}")
        elif len(param_values) == 0:
            errors.append(f"Parameter '{param_name}' has no values")
    
    # Check for reasonable space size
    space_size = calculate_parameter_space_size(parameters)
    if space_size > 10000:
        errors.append(f"Parameter space very large ({space_size:,} combinations) - consider sampling")
    
    return len(errors) == 0, errors


def create_parameter_ranges(
    base_params: Dict[str, Any],
    variations: Dict[str, Dict[str, Any]]
) -> Dict[str, List[Any]]:
    """
    Create parameter ranges from base parameters and variation specifications.
    
    Args:
        base_params: Base parameter values
        variations: Specifications for how to vary each parameter
        
    Returns:
        Dictionary mapping parameter names to lists of values
        
    Example:
        base = {'pe_count': 4, 'simd': 2}
        variations = {
            'pe_count': {'type': 'range', 'min': 1, 'max': 8, 'step': 1},
            'simd': {'type': 'list', 'values': [1, 2, 4, 8]}
        }
        ranges = create_parameter_ranges(base, variations)
        # Returns: {'pe_count': [1, 2, 3, 4, 5, 6, 7, 8], 'simd': [1, 2, 4, 8]}
    """
    param_ranges = {}
    
    for param_name, base_value in base_params.items():
        if param_name in variations:
            variation = variations[param_name]
            
            if variation['type'] == 'range':
                # Create range of values
                min_val = variation.get('min', base_value)
                max_val = variation.get('max', base_value)
                step = variation.get('step', 1)
                
                if isinstance(base_value, int):
                    param_ranges[param_name] = list(range(int(min_val), int(max_val) + 1, int(step)))
                else:
                    # Float range
                    values = []
                    current = float(min_val)
                    while current <= float(max_val):
                        values.append(current)
                        current += float(step)
                    param_ranges[param_name] = values
            
            elif variation['type'] == 'list':
                # Use provided list of values
                param_ranges[param_name] = variation['values']
            
            elif variation['type'] == 'scale':
                # Scale base value by factors
                factors = variation.get('factors', [0.5, 1.0, 2.0])
                if isinstance(base_value, int):
                    param_ranges[param_name] = [int(base_value * f) for f in factors]
                else:
                    param_ranges[param_name] = [base_value * f for f in factors]
            
            else:
                # Unknown variation type, use base value
                param_ranges[param_name] = [base_value]
        else:
            # No variation specified, use base value
            param_ranges[param_name] = [base_value]
    
    return param_ranges


def optimize_parameter_selection(
    parameters: Dict[str, List[Any]],
    max_evaluations: int,
    strategy: str = 'auto'
) -> tuple[str, int]:
    """
    Optimize parameter selection strategy based on space size and budget.
    
    Args:
        parameters: Parameter space definition
        max_evaluations: Maximum number of evaluations allowed
        strategy: Strategy preference ('auto', 'grid', 'random', 'lhs')
        
    Returns:
        Tuple of (recommended_strategy, recommended_samples)
    """
    space_size = calculate_parameter_space_size(parameters)
    n_params = len(parameters)
    
    if strategy == 'auto':
        if space_size <= max_evaluations:
            # Can do full grid search
            return 'grid', space_size
        elif n_params <= 3 and max_evaluations >= 50:
            # Use LHS for good coverage with few parameters
            return 'lhs', min(max_evaluations, 100)
        elif max_evaluations >= 20:
            # Use Sobol for efficient sampling
            return 'sobol', max_evaluations
        else:
            # Use random for small budgets
            return 'random', max_evaluations
    else:
        # Use specified strategy
        if strategy == 'grid':
            return strategy, min(space_size, max_evaluations)
        else:
            return strategy, max_evaluations


def analyze_parameter_coverage(
    samples: List[Dict[str, Any]],
    parameters: Dict[str, List[Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze how well samples cover the parameter space.
    
    Args:
        samples: List of parameter samples
        parameters: Original parameter space definition
        
    Returns:
        Coverage analysis for each parameter
    """
    coverage = {}
    
    for param_name, param_values in parameters.items():
        sampled_values = [sample.get(param_name) for sample in samples if param_name in sample]
        unique_sampled = list(set(sampled_values))
        
        coverage[param_name] = {
            'total_possible': len(param_values),
            'sampled_unique': len(unique_sampled),
            'coverage_ratio': len(unique_sampled) / len(param_values) if param_values else 0,
            'missing_values': [v for v in param_values if v not in unique_sampled],
            'sample_distribution': {v: sampled_values.count(v) for v in unique_sampled}
        }
    
    return coverage