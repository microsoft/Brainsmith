"""
Utility functions for selection framework
"""

import numpy as np
from typing import Dict, List, Any
from .models import SelectionCriteria

def normalize_matrix(matrix: np.ndarray, method: str = 'vector') -> np.ndarray:
    """Normalize decision matrix using specified method."""
    if method == 'vector':
        norms = np.linalg.norm(matrix, axis=0)
        norms[norms == 0] = 1  # Avoid division by zero
        return matrix / norms
    elif method == 'minmax':
        min_vals = np.min(matrix, axis=0)
        max_vals = np.max(matrix, axis=0)
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1
        return (matrix - min_vals) / ranges
    else:
        return matrix

def calculate_weights(objectives: List[str], weight_dict: Dict[str, float]) -> np.ndarray:
    """Calculate weight array from objectives and weight dictionary."""
    return np.array([weight_dict.get(obj, 0.0) for obj in objectives])

def validate_preferences(criteria: SelectionCriteria) -> bool:
    """Validate selection criteria preferences."""
    return criteria.validate()

def create_selection_criteria(objectives: List[str], 
                            weights: Dict[str, float],
                            **kwargs) -> SelectionCriteria:
    """Create selection criteria with validation."""
    return SelectionCriteria(
        objectives=objectives,
        weights=weights,
        constraints=kwargs.get('constraints', []),
        preferences=kwargs.get('preferences', {}),
        maximize_objectives=kwargs.get('maximize_objectives', {})
    )