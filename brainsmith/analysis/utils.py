"""
Utility functions for performance analysis framework
"""

import numpy as np
from typing import Dict, List, Any, Optional
from .models import AnalysisContext, PerformanceData, AnalysisConfiguration

def calculate_statistics(values: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive statistics for values."""
    if len(values) == 0:
        return {}
    
    return {
        'count': len(values),
        'mean': float(np.mean(values)),
        'std': float(np.std(values, ddof=1) if len(values) > 1 else 0),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'median': float(np.median(values)),
        'q25': float(np.percentile(values, 25)),
        'q75': float(np.percentile(values, 75)),
        'range': float(np.ptp(values)),
        'iqr': float(np.percentile(values, 75) - np.percentile(values, 25))
    }

def fit_distributions(values: np.ndarray) -> Dict[str, Any]:
    """Fit statistical distributions to data."""
    # Simple implementation - fit normal distribution
    if len(values) < 2:
        return {'normal': {'mean': 0, 'std': 1, 'score': 0}}
    
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    
    return {
        'normal': {
            'mean': mean,
            'std': std,
            'score': 0.8  # Placeholder score
        }
    }

def detect_outliers(values: np.ndarray, threshold: float = 2.0) -> List[int]:
    """Detect outliers using Z-score method."""
    if len(values) < 3:
        return []
    
    z_scores = np.abs((values - np.mean(values)) / np.std(values))
    return np.where(z_scores > threshold)[0].tolist()

def normalize_performance_data(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Normalize performance data to [0, 1] range."""
    normalized = {}
    
    for metric, values in data.items():
        if len(values) == 0:
            normalized[metric] = values
            continue
        
        min_val = np.min(values)
        max_val = np.max(values)
        
        if max_val > min_val:
            normalized[metric] = (values - min_val) / (max_val - min_val)
        else:
            normalized[metric] = np.ones_like(values) * 0.5
    
    return normalized

def create_analysis_context(solutions: List[Any], 
                          metrics: Optional[Dict[str, np.ndarray]] = None) -> AnalysisContext:
    """Create analysis context from solutions and metrics."""
    
    # Convert metrics to PerformanceData objects
    performance_data = {}
    if metrics:
        for name, values in metrics.items():
            performance_data[name] = PerformanceData(
                metric_name=name,
                values=values
            )
    
    from .models import AnalysisType
    
    return AnalysisContext(
        solutions=solutions,
        performance_data=performance_data,
        analysis_types=[AnalysisType.DESCRIPTIVE, AnalysisType.STATISTICAL]
    )