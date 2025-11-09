"""Performance profiling and roofline modeling utilities for Brainsmith.

This module provides tools for analyzing the performance characteristics of
neural network models and hardware accelerators using roofline models.

Modules:
    model_profiling: Model profiling infrastructure and RooflineModel class
    roofline: Roofline analysis functions and utilities
    roofline_runner: Orchestrator for running roofline analysis
"""

__all__ = [
    'model_profiling',
    'roofline',
    'roofline_runner',
]
