"""
BrainSmith Plugin System

This module provides the unified plugin infrastructure for registering and managing
transforms, kernels, backends, and hardware transforms.
"""

# Import core decorators and registry
from .core import transform, kernel, backend, get_registry

__all__ = [
    # Core decorators
    'transform',
    'kernel', 
    'backend',
    
    # Registry access
    'get_registry',
]

__version__ = "2.0.0"