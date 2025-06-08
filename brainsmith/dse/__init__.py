"""
Design Space Exploration (DSE) interface library for Brainsmith platform.

This module provides advanced DSE interfaces, optimization algorithms, and
external framework integration for comprehensive design space exploration.
"""

from .interface import DSEInterface, DSEEngine
from .simple import SimpleDSEEngine
from .external import ExternalDSEAdapter
from .analysis import DSEAnalyzer, ParetoAnalyzer
from .strategies import SamplingStrategy, OptimizationStrategy

__all__ = [
    # Core interfaces
    'DSEInterface',
    'DSEEngine',
    
    # Engine implementations
    'SimpleDSEEngine',
    'ExternalDSEAdapter',
    
    # Analysis tools
    'DSEAnalyzer',
    'ParetoAnalyzer',
    
    # Strategy definitions
    'SamplingStrategy',
    'OptimizationStrategy'
]