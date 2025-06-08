"""
Hardware Optimization Library - Week 4 Implementation

Integrates existing dse/ functionality and provides advanced optimization
strategies for FPGA accelerator design space exploration.
"""

from .library import HwOptimLibrary
from .strategies import OptimizationStrategy, get_available_strategies
from .algorithms import GeneticOptimizer, BayesianOptimizer

# Convenience functions
def get_optimization_strategies():
    """Get available optimization strategies."""
    return get_available_strategies()

def optimize_design(design_space, objectives, strategy='genetic'):
    """Optimize design using specified strategy."""
    library = HwOptimLibrary()
    library.initialize()
    return library.execute("optimize_design", {
        'design_space': design_space,
        'objectives': objectives,
        'strategy': strategy
    })

def estimate_resources(design_config):
    """Estimate resource usage for design configuration."""
    library = HwOptimLibrary()
    library.initialize()
    return library.execute("estimate_resources", {'config': design_config})

__all__ = [
    'HwOptimLibrary',
    'OptimizationStrategy',
    'GeneticOptimizer',
    'BayesianOptimizer',
    'get_optimization_strategies',
    'optimize_design',
    'estimate_resources'
]

# Version info
__version__ = "1.0.0"  # Week 4 implementation