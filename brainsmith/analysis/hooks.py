"""
Core hooks for external analysis tool integration.

Provides data exposure functions that allow external analysis libraries
(pandas, scipy, scikit-learn, etc.) to process BrainSmith DSE results.
"""

import numpy as np
from typing import Dict, List, Any, Callable, Optional

# Registry for external analyzers
_analyzer_registry: Dict[str, Callable] = {}


def expose_analysis_data(dse_results) -> Dict[str, Any]:
    """
    Expose structured data for external analysis tools.
    
    Args:
        dse_results: DSE results from forge() function
        
    Returns:
        Structured data compatible with external analysis libraries
    """
    if not dse_results:
        return {'solutions': [], 'metrics': {}, 'pareto_frontier': []}
    
    # Extract solution data
    solutions = []
    for i, result in enumerate(dse_results):
        solution = {
            'id': i,
            'parameters': getattr(result, 'design_parameters', {}),
            'objectives': getattr(result, 'objective_values', []),
            'constraints': getattr(result, 'constraint_violations', []),
            'metadata': getattr(result, 'metadata', {})
        }
        solutions.append(solution)
    
    # Extract metric arrays
    metrics = {}
    if solutions:
        # Get all unique metric names
        metric_names = set()
        for sol in solutions:
            if 'objectives' in sol and sol['objectives']:
                for i, val in enumerate(sol['objectives']):
                    metric_names.add(f'objective_{i}')
        
        # Create metric arrays
        for metric in metric_names:
            values = []
            for sol in solutions:
                if 'objectives' in sol and sol['objectives']:
                    obj_idx = int(metric.split('_')[1])
                    if obj_idx < len(sol['objectives']):
                        values.append(sol['objectives'][obj_idx])
            if values:
                metrics[metric] = np.array(values)
    
    # Find Pareto frontier (simplified)
    pareto_indices = _find_pareto_frontier(solutions)
    
    return {
        'solutions': solutions,
        'metrics': metrics,
        'pareto_frontier': pareto_indices,
        'metadata': {
            'num_solutions': len(solutions),
            'num_metrics': len(metrics),
            'data_format': 'brainsmith_v1'
        }
    }


def register_analyzer(name: str, analyzer_func: Callable) -> None:
    """Register external analysis function."""
    _analyzer_registry[name] = analyzer_func


def get_registered_analyzers() -> Dict[str, Callable]:
    """Get all registered analyzers."""
    return _analyzer_registry.copy()


def get_raw_data(dse_results) -> Dict[str, np.ndarray]:
    """Get raw metric arrays for external processing."""
    analysis_data = expose_analysis_data(dse_results)
    return analysis_data['metrics']


def export_to_dataframe(dse_results) -> Optional['pd.DataFrame']:
    """Export to pandas-compatible format."""
    try:
        import pandas as pd
        analysis_data = expose_analysis_data(dse_results)
        
        # Flatten solution data for DataFrame
        flattened_data = []
        for sol in analysis_data['solutions']:
            row = {'solution_id': sol['id']}
            
            # Add parameters
            for param, value in sol.get('parameters', {}).items():
                row[f'param_{param}'] = value
            
            # Add objectives
            for i, obj_val in enumerate(sol.get('objectives', [])):
                row[f'objective_{i}'] = obj_val
            
            # Add constraints
            for i, const_val in enumerate(sol.get('constraints', [])):
                row[f'constraint_{i}'] = const_val
            
            flattened_data.append(row)
        
        return pd.DataFrame(flattened_data)
        
    except ImportError:
        return None


def _find_pareto_frontier(solutions: List[Dict[str, Any]]) -> List[int]:
    """Find Pareto frontier indices (simplified implementation)."""
    if not solutions:
        return []
    
    pareto_indices = []
    
    for i, sol_i in enumerate(solutions):
        is_pareto = True
        objectives_i = sol_i.get('objectives', [])
        
        if not objectives_i:
            continue
            
        for j, sol_j in enumerate(solutions):
            if i == j:
                continue
                
            objectives_j = sol_j.get('objectives', [])
            if not objectives_j or len(objectives_j) != len(objectives_i):
                continue
            
            # Check if sol_j dominates sol_i (assuming minimization)
            dominates = True
            for obj_i, obj_j in zip(objectives_i, objectives_j):
                if obj_j >= obj_i:  # Not better in this objective
                    dominates = False
                    break
            
            if dominates:
                is_pareto = False
                break
        
        if is_pareto:
            pareto_indices.append(i)
    
    return pareto_indices