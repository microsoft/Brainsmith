"""
Optimization strategies for hardware design space exploration.

Provides various optimization strategies and algorithms for FPGA
accelerator design optimization.
"""

from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod


class OptimizationStrategy(ABC):
    """Base class for optimization strategies."""
    
    @abstractmethod
    def optimize(self, design_space: Dict[str, Any], objectives: List[str], 
                max_evaluations: int) -> Dict[str, Any]:
        """Perform optimization."""
        pass


def get_available_strategies() -> Dict[str, str]:
    """Get available optimization strategies."""
    return {
        'genetic': 'Genetic Algorithm (NSGA-II)',
        'bayesian': 'Bayesian Optimization',  
        'grid': 'Grid Search',
        'random': 'Random Search',
        'pareto': 'Pareto-optimal exploration'
    }