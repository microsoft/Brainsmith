"""
BrainSmith Plugin System

This module provides the core plugin infrastructure for registering and managing
transforms, kernels, backends, and hardware transforms.
"""

from .decorators import transform, kernel, backend, hw_transform
from .registry import PluginRegistry
from .discovery import PluginDiscovery
from .exceptions import (
    PluginError,
    PluginNotFoundError,
    PluginRegistrationError,
    PluginValidationError
)

__all__ = [
    # Decorators
    'transform',
    'kernel', 
    'backend',
    'hw_transform',
    
    # Core classes
    'PluginRegistry',
    'PluginDiscovery',
    
    # Exceptions
    'PluginError',
    'PluginNotFoundError',
    'PluginRegistrationError',
    'PluginValidationError'
]

__version__ = "1.0.0"