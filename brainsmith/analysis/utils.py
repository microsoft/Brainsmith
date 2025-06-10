"""
Utility functions for analysis hooks framework.

Simple utilities that complement the hooks-based approach.
Users should prefer external libraries (pandas, scipy) for complex analysis.
"""

import numpy as np
from typing import Dict, List, Any


def calculate_basic_statistics(values: np.ndarray) -> Dict[str, float]:
    """
    Calculate basic statistics for values.
    
    Note: For advanced statistics, use scipy.stats instead.
    This is a minimal implementation for simple use cases.
    """
    if len(values) == 0:
        return {}
    
    return {
        'count': len(values),
        'mean': float(np.mean(values)),
        'std': float(np.std(values, ddof=1) if len(values) > 1 else 0),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'median': float(np.median(values)),
    }


def extract_metric_arrays(solutions: List[Any]) -> Dict[str, np.ndarray]:
    """
    Extract metric arrays from solution objects.
    
    Args:
        solutions: List of solution objects with objective_values
        
    Returns:
        Dictionary mapping metric names to numpy arrays
    """
    metrics = {}
    
    if not solutions:
        return metrics
    
    # Determine number of objectives from first solution
    first_solution = solutions[0]
    if hasattr(first_solution, 'objective_values') and first_solution.objective_values:
        num_objectives = len(first_solution.objective_values)
        
        # Extract each objective as a separate metric
        for obj_idx in range(num_objectives):
            values = []
            for solution in solutions:
                if (hasattr(solution, 'objective_values') and 
                    solution.objective_values and 
                    obj_idx < len(solution.objective_values)):
                    values.append(solution.objective_values[obj_idx])
            
            if values:
                metrics[f'objective_{obj_idx}'] = np.array(values)
    
    return metrics


def format_for_external_tool(solutions: List[Any], tool: str = 'pandas') -> Any:
    """
    Format solution data for external analysis tools.
    
    Args:
        solutions: List of solution objects
        tool: Target tool ('pandas', 'scipy', 'sklearn')
        
    Returns:
        Data formatted for the specified tool
    """
    if tool == 'pandas':
        try:
            import pandas as pd
            
            # Create rows for DataFrame
            rows = []
            for i, sol in enumerate(solutions):
                row = {'solution_id': i}
                
                # Add parameters
                if hasattr(sol, 'design_parameters'):
                    for param, value in sol.design_parameters.items():
                        row[f'param_{param}'] = value
                
                # Add objectives
                if hasattr(sol, 'objective_values') and sol.objective_values:
                    for j, obj_val in enumerate(sol.objective_values):
                        row[f'objective_{j}'] = obj_val
                
                rows.append(row)
            
            return pd.DataFrame(rows)
            
        except ImportError:
            return None
    
    elif tool == 'scipy':
        metrics = extract_metric_arrays(solutions)
        return {
            'arrays': metrics,
            'sample_size': len(solutions),
            'metric_names': list(metrics.keys())
        }
    
    elif tool == 'sklearn':
        try:
            import numpy as np
            
            # Extract features (parameters) and targets (objectives)
            features = []
            targets = []
            
            for sol in solutions:
                if hasattr(sol, 'design_parameters'):
                    param_values = list(sol.design_parameters.values())
                    if param_values:
                        features.append(param_values)
                
                if hasattr(sol, 'objective_values') and sol.objective_values:
                    targets.append(sol.objective_values)
            
            if features and targets:
                return {
                    'X': np.array(features),
                    'y': np.array(targets)
                }
            
        except ImportError:
            pass
    
    return None