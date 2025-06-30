"""
BrainSmith Plugin System V2 - Complete Architectural Redesign

This is a BREAKING CHANGE implementation that replaces the entire plugin system.
NO COMPATIBILITY with v1 - this is intentional following Prime Directive 1.

New Features:
- Thread-safe operations with proper locking
- Plugin contracts and validation
- O(log n) indexed queries
- Plugin lifecycle management
- Dependency resolution
- Resource isolation and sandboxing
"""

# Core contracts and interfaces
from .contracts import (
    PluginContract,
    TransformContract,
    KernelContract, 
    BackendContract
)

# Registry and management
from .registry import PluginRegistry
from .query import QueryEngine, PluginQuery
from .lifecycle import LifecycleManager, PluginState

# Decorators (BREAKING: New API)
from .decorators import transform, kernel, backend

# Discovery and validation
from .discovery import EntryPointDiscovery
from .validation import PluginValidator, ValidationResult

# Configuration and dependency management
from .config import PluginConfig, ConfigSchema
from .dependencies import PluginDependency, DependencyResolver

# Get registry instance
from .registry import get_registry

__version__ = "2.0.0"
__all__ = [
    # Contracts
    'PluginContract',
    'TransformContract', 
    'KernelContract',
    'BackendContract',
    
    # Core components
    'PluginRegistry',
    'QueryEngine',
    'PluginQuery',
    'LifecycleManager',
    'PluginState',
    
    # Decorators (NEW API)
    'transform',
    'kernel', 
    'backend',
    
    # Discovery and validation
    'EntryPointDiscovery',
    'PluginValidator',
    'ValidationResult',
    
    # Configuration
    'PluginConfig',
    'ConfigSchema',
    'PluginDependency',
    'DependencyResolver',
    
    # Registry access
    'get_registry'
]