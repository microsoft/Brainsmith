"""
Adapters for external analysis tools.

Provides helper functions to convert BrainSmith data into formats
compatible with popular analysis libraries.
"""

from typing import Dict, Any, Optional


def pandas_adapter(analysis_data: Dict[str, Any]) -> Optional['pd.DataFrame']:
    """
    Convert analysis data to pandas DataFrame.
    
    Args:
        analysis_data: Data from expose_analysis_data()
        
    Returns:
        pandas DataFrame or None if pandas not available
    """
    try:
        import pandas as pd
        
        solutions = analysis_data.get('solutions', [])
        if not solutions:
            return pd.DataFrame()
        
        # Flatten data for DataFrame
        rows = []
        for sol in solutions:
            row = {'solution_id': sol['id']}
            
            # Add parameters with prefix
            for param, value in sol.get('parameters', {}).items():
                row[f'param_{param}'] = value
            
            # Add objectives
            for i, obj_val in enumerate(sol.get('objectives', [])):
                row[f'objective_{i}'] = obj_val
            
            # Add constraints if present
            for i, const_val in enumerate(sol.get('constraints', [])):
                row[f'constraint_{i}'] = const_val
                
            rows.append(row)
        
        return pd.DataFrame(rows)
        
    except ImportError:
        return None


def scipy_adapter(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare data for scipy analysis.
    
    Args:
        analysis_data: Data from expose_analysis_data()
        
    Returns:
        Dictionary with scipy-friendly data format
    """
    metrics = analysis_data.get('metrics', {})
    
    # Return metrics in scipy-friendly format
    return {
        'arrays': metrics,
        'sample_size': len(next(iter(metrics.values()))) if metrics else 0,
        'metric_names': list(metrics.keys()),
        'metadata': analysis_data.get('metadata', {})
    }


def sklearn_adapter(analysis_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Prepare data for scikit-learn analysis.
    
    Args:
        analysis_data: Data from expose_analysis_data()
        
    Returns:
        Dictionary with feature matrix and target arrays
    """
    try:
        import numpy as np
        
        solutions = analysis_data.get('solutions', [])
        if not solutions:
            return None
        
        # Extract features (parameters) and targets (objectives)
        features = []
        targets = []
        
        for sol in solutions:
            # Features: design parameters
            param_values = list(sol.get('parameters', {}).values())
            if param_values:
                features.append(param_values)
            
            # Targets: objectives
            obj_values = sol.get('objectives', [])
            if obj_values:
                targets.append(obj_values)
        
        if not features or not targets:
            return None
        
        return {
            'X': np.array(features),  # Feature matrix
            'y': np.array(targets),   # Target matrix
            'feature_names': list(solutions[0].get('parameters', {}).keys()),
            'target_names': [f'objective_{i}' for i in range(len(targets[0]))]
        }
        
    except ImportError:
        return None