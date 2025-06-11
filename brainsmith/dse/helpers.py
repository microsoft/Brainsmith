"""
DSE Helper Functions - North Star Aligned

Utility functions for design space exploration.
Simple, practical functions that work with external analysis tools.
"""

import itertools
import random
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from .types import DSEResult, ParameterSpace

logger = logging.getLogger(__name__)


def generate_parameter_grid(parameters: ParameterSpace) -> List[Dict[str, Any]]:
    """
    Generate all parameter combinations from parameter space.
    
    Args:
        parameters: Dictionary mapping parameter names to value lists
        
    Returns:
        List of parameter combinations (Cartesian product)
        
    Example:
        parameters = {'pe_count': [1, 2, 4], 'precision': [8, 16]}
        grid = generate_parameter_grid(parameters)
        # Returns: [{'pe_count': 1, 'precision': 8}, {'pe_count': 1, 'precision': 16}, ...]
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
    parameters: ParameterSpace,
    strategy: str = 'random',
    n_samples: int = 10,
    seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Sample parameter combinations using different strategies.
    
    Args:
        parameters: Parameter space definition
        strategy: Sampling strategy ('random', 'lhs', 'grid')
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of sampled parameter combinations
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    if strategy == 'grid':
        # Return full grid (ignore n_samples)
        return generate_parameter_grid(parameters)
    
    elif strategy == 'random':
        return _random_sampling(parameters, n_samples)
    
    elif strategy == 'lhs':
        return _latin_hypercube_sampling(parameters, n_samples)
    
    else:
        logger.warning(f"Unknown strategy '{strategy}', using random sampling")
        return _random_sampling(parameters, n_samples)


def _random_sampling(parameters: ParameterSpace, n_samples: int) -> List[Dict[str, Any]]:
    """Generate random parameter samples."""
    samples = []
    
    for _ in range(n_samples):
        sample = {}
        for param_name, param_values in parameters.items():
            sample[param_name] = random.choice(param_values)
        samples.append(sample)
    
    return samples


def _latin_hypercube_sampling(parameters: ParameterSpace, n_samples: int) -> List[Dict[str, Any]]:
    """
    Generate Latin Hypercube samples.
    Provides better space coverage than random sampling.
    """
    # Separate numeric and categorical parameters
    numeric_params = {}
    categorical_params = {}
    
    for param_name, param_values in parameters.items():
        if all(isinstance(v, (int, float)) for v in param_values):
            # Numeric parameter - use min/max range
            numeric_params[param_name] = (min(param_values), max(param_values))
        else:
            # Categorical parameter
            categorical_params[param_name] = param_values
    
    samples = []
    
    # Generate LHS samples for numeric parameters
    if numeric_params:
        lhs_samples = _generate_lhs_matrix(len(numeric_params), n_samples)
        param_names = list(numeric_params.keys())
        
        for i in range(n_samples):
            sample = {}
            
            # Map LHS values to parameter ranges
            for j, param_name in enumerate(param_names):
                min_val, max_val = numeric_params[param_name]
                normalized_value = lhs_samples[i, j]
                
                # Scale to parameter range
                if isinstance(min_val, int) and isinstance(max_val, int):
                    # Integer parameter
                    param_value = int(min_val + normalized_value * (max_val - min_val))
                else:
                    # Float parameter
                    param_value = min_val + normalized_value * (max_val - min_val)
                
                sample[param_name] = param_value
            
            # Add random categorical parameters
            for param_name, param_values in categorical_params.items():
                sample[param_name] = random.choice(param_values)
            
            samples.append(sample)
    
    else:
        # Only categorical parameters - use random sampling
        samples = _random_sampling(parameters, n_samples)
    
    return samples


def _generate_lhs_matrix(n_dims: int, n_samples: int) -> np.ndarray:
    """Generate Latin Hypercube Sample matrix."""
    lhs_matrix = np.zeros((n_samples, n_dims))
    
    for dim in range(n_dims):
        # Generate permutation for this dimension
        perm = np.random.permutation(n_samples)
        
        # Generate random values within each interval
        for i in range(n_samples):
            interval_start = perm[i] / n_samples
            interval_end = (perm[i] + 1) / n_samples
            lhs_matrix[i, dim] = np.random.uniform(interval_start, interval_end)
    
    return lhs_matrix


def export_results(
    results: List[DSEResult], 
    format: str = 'pandas',
    filepath: Optional[str] = None
) -> Any:
    """
    Export DSE results for external analysis tools.
    
    Integrates with analysis hooks patterns for data exposure.
    
    Args:
        results: List of DSE results
        format: Export format ('pandas', 'csv', 'json')
        filepath: Optional file path for saving
        
    Returns:
        Exported data in requested format
        
    Example:
        # Export to pandas for analysis
        df = export_results(results, 'pandas')
        df.plot(x='pe_count', y='throughput', kind='scatter')
        
        # Export to CSV file
        export_results(results, 'csv', 'results.csv')
    """
    if not results:
        logger.warning("No results to export")
        return None
    
    if format == 'pandas':
        return _export_to_pandas(results, filepath)
    elif format == 'csv':
        return _export_to_csv(results, filepath)
    elif format == 'json':
        return _export_to_json(results, filepath)
    else:
        raise ValueError(f"Unsupported export format: {format}")


def _export_to_pandas(results: List[DSEResult], filepath: Optional[str] = None):
    """Export to pandas DataFrame."""
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas not available - cannot export to DataFrame")
        return None
    
    data = []
    for result in results:
        # Start with parameters
        row = result.parameters.copy()
        
        # Add metrics
        if result.metrics and hasattr(result.metrics, 'to_dict'):
            metrics_dict = result.metrics.to_dict()
            # Flatten nested metrics
            for key, value in metrics_dict.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        row[f"{key}_{subkey}"] = subvalue
                else:
                    row[key] = value
        
        # Add build info
        row['build_success'] = result.build_success
        row['build_time'] = result.build_time
        
        # Add metadata
        for key, value in result.metadata.items():
            row[f"meta_{key}"] = value
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    if filepath:
        df.to_csv(filepath, index=False)
        logger.info(f"Results exported to {filepath}")
    
    return df


def _export_to_csv(results: List[DSEResult], filepath: Optional[str] = None) -> str:
    """Export to CSV format."""
    df = _export_to_pandas(results)
    if df is None:
        # Fallback CSV generation without pandas
        return _manual_csv_export(results, filepath)
    
    csv_data = df.to_csv(index=False)
    
    if filepath:
        with open(filepath, 'w') as f:
            f.write(csv_data)
        logger.info(f"CSV exported to {filepath}")
    
    return csv_data


def _export_to_json(results: List[DSEResult], filepath: Optional[str] = None) -> str:
    """Export to JSON format."""
    import json
    
    data = [result.to_dict() for result in results]
    json_data = json.dumps(data, indent=2, default=str)
    
    if filepath:
        with open(filepath, 'w') as f:
            f.write(json_data)
        logger.info(f"JSON exported to {filepath}")
    
    return json_data


def _manual_csv_export(results: List[DSEResult], filepath: Optional[str] = None) -> str:
    """Manual CSV export without pandas dependency."""
    import csv
    import io
    
    if not results:
        return ""
    
    # Collect all possible columns
    columns = set()
    for result in results:
        columns.update(result.parameters.keys())
        columns.add('build_success')
        columns.add('build_time')
        columns.update(f"meta_{k}" for k in result.metadata.keys())
    
    columns = sorted(columns)
    
    # Generate CSV
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(columns)
    
    # Write data rows
    for result in results:
        row = []
        for col in columns:
            if col in result.parameters:
                row.append(result.parameters[col])
            elif col == 'build_success':
                row.append(result.build_success)
            elif col == 'build_time':
                row.append(result.build_time)
            elif col.startswith('meta_'):
                meta_key = col[5:]  # Remove 'meta_' prefix
                row.append(result.metadata.get(meta_key, ''))
            else:
                row.append('')
        writer.writerow(row)
    
    csv_data = output.getvalue()
    output.close()
    
    if filepath:
        with open(filepath, 'w') as f:
            f.write(csv_data)
        logger.info(f"CSV exported to {filepath}")
    
    return csv_data


def estimate_runtime(
    parameter_combinations: List[Dict[str, Any]], 
    benchmark_time: float = 30.0
) -> float:
    """
    Estimate total runtime for parameter sweep.
    
    Args:
        parameter_combinations: List of parameter combinations to evaluate
        benchmark_time: Estimated time per evaluation in seconds
        
    Returns:
        Estimated total runtime in seconds
    """
    return len(parameter_combinations) * benchmark_time


def count_parameter_combinations(parameters: ParameterSpace) -> int:
    """
    Count total number of parameter combinations.
    
    Args:
        parameters: Parameter space definition
        
    Returns:
        Total number of combinations
    """
    if not parameters:
        return 0
    
    total = 1
    for param_values in parameters.values():
        total *= len(param_values)
    
    return total


def validate_parameter_space(parameters: ParameterSpace) -> tuple[bool, List[str]]:
    """
    Validate parameter space definition.
    
    Args:
        parameters: Parameter space to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    if not parameters:
        errors.append("Parameter space is empty")
        return False, errors
    
    for param_name, param_values in parameters.items():
        if not param_name:
            errors.append("Parameter name cannot be empty")
            continue
        
        if not isinstance(param_values, (list, tuple)):
            errors.append(f"Parameter '{param_name}' values must be a list or tuple")
            continue
        
        if len(param_values) == 0:
            errors.append(f"Parameter '{param_name}' has no values")
            continue
        
        # Check for consistent types
        first_type = type(param_values[0])
        if not all(isinstance(v, first_type) for v in param_values):
            errors.append(f"Parameter '{param_name}' has inconsistent value types")
    
    return len(errors) == 0, errors


def create_parameter_subsets(
    parameters: ParameterSpace,
    max_combinations: int = 100
) -> List[ParameterSpace]:
    """
    Split large parameter space into manageable subsets.
    
    Args:
        parameters: Full parameter space
        max_combinations: Maximum combinations per subset
        
    Returns:
        List of parameter space subsets
    """
    total_combinations = count_parameter_combinations(parameters)
    
    if total_combinations <= max_combinations:
        return [parameters]
    
    # Simple strategy: reduce the parameter with the most values
    subsets = []
    remaining_params = parameters.copy()
    
    while count_parameter_combinations(remaining_params) > max_combinations:
        # Find parameter with most values
        max_param = max(remaining_params.keys(), 
                        key=lambda k: len(remaining_params[k]))
        
        max_values = remaining_params[max_param]
        
        # Split this parameter in half
        mid = len(max_values) // 2
        first_half = max_values[:mid]
        second_half = max_values[mid:]
        
        # Create subset with first half
        subset = remaining_params.copy()
        subset[max_param] = first_half
        subsets.append(subset)
        
        # Continue with second half
        remaining_params[max_param] = second_half
    
    # Add final subset
    if remaining_params:
        subsets.append(remaining_params)
    
    logger.info(f"Split parameter space into {len(subsets)} subsets")
    return subsets


def filter_results(
    results: List[DSEResult],
    filter_func: callable
) -> List[DSEResult]:
    """
    Filter results based on custom criteria.
    
    Args:
        results: List of DSE results
        filter_func: Function that takes DSEResult and returns bool
        
    Returns:
        Filtered list of results
        
    Example:
        # Filter for successful results with high throughput
        good_results = filter_results(
            results, 
            lambda r: r.build_success and r.metrics.performance.throughput_ops_sec > 1000
        )
    """
    filtered = [r for r in results if filter_func(r)]
    logger.info(f"Filtered {len(results)} results to {len(filtered)}")
    return filtered


def sort_results(
    results: List[DSEResult],
    key_func: callable,
    reverse: bool = True
) -> List[DSEResult]:
    """
    Sort results based on custom key function.
    
    Args:
        results: List of DSE results
        key_func: Function that takes DSEResult and returns sortable value
        reverse: Sort in descending order if True
        
    Returns:
        Sorted list of results
        
    Example:
        # Sort by throughput (highest first)
        sorted_results = sort_results(
            results,
            lambda r: r.metrics.performance.throughput_ops_sec if r.build_success else 0
        )
    """
    return sorted(results, key=key_func, reverse=reverse)