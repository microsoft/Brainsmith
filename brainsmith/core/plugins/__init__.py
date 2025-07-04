"""
BrainSmith Core Plugin System - Perfect Code Implementation

Zero-discovery plugin registry with O(1) lookups and decoration-time registration.
"""

# Core registry system
from .registry import BrainsmithPluginRegistry, get_registry

# Decoration-time registration
from .decorators import (
    plugin,
    transform, 
    kernel, 
    backend, 
    step, 
    kernel_inference
)

# Natural access collections
from .plugin_collections import create_collections

# Blueprint optimization
from .blueprint_loader import BlueprintPluginLoader, load_blueprint_plugins

# Framework initialization
from .framework_adapters import initialize_framework_integrations


def plugin_status():
    """Get status of the plugin system."""
    registry = get_registry()
    return registry.get_stats()


def reset_plugin_system():
    """Reset the plugin system (useful for testing)."""
    from .registry import reset_registry
    reset_registry()


__all__ = [
    # Core registry
    "BrainsmithPluginRegistry",
    "get_registry",
    
    # Decorators
    "plugin",
    "transform", 
    "kernel", 
    "backend", 
    "step", 
    "kernel_inference",
    
    # Collections
    "create_collections",
    
    # Blueprint optimization
    "BlueprintPluginLoader",
    "load_blueprint_plugins",
    
    # Framework integration
    "initialize_framework_integrations",
    
    # Utility functions
    "plugin_status",
    "reset_plugin_system",
]

__version__ = "1.0.0"