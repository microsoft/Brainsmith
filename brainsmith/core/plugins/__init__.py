"""
BrainSmith Core Plugin System - Perfect Code Implementation

A high-performance plugin registry system with direct lookups and pre-computed indexes.
Eliminates discovery overhead while maintaining zero-friction development experience.

Key Features:
- Direct dictionary lookups for maximum performance
- Pre-computed indexes for blueprint optimization
- Auto-registration at decoration time
- Framework integration for QONNX/FINN
- Natural access patterns preserved

Usage:
    # Using convenience decorators (recommended)
    from brainsmith.core.plugins import transform, kernel, backend
    
    @transform(name="MyPreProcessTransform", stage="pre_proc")
    class MyPreProcessTransform:
        def apply(self, model):
            return model, False
    
    @transform(name="MyTransform", stage="topology_opt")
    class MyTransform:
        def apply(self, model):
            return model, False
    
    # Or using generic decorator
    from brainsmith.core.plugins import plugin
    
    @plugin(type="transform", name="MyTransform", stage="topology_opt")
    class MyTransform:
        def apply(self, model):
            return model, False
    
    # Natural access through collections
    from brainsmith.core.plugins import create_collections, get_registry
    collections = create_collections(get_registry())
    model = model.transform(collections['transforms'].MyTransform())
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
from .collections import create_collections

# Blueprint optimization
from .blueprint_loader import BlueprintPluginLoader, load_blueprint_plugins

# Framework adapters
from .framework_adapters import ensure_frameworks_initialized


def plugin_status():
    """Get status of the plugin system."""
    registry = get_registry()
    return registry.get_stats()


def reset_plugin_system():
    """Reset the plugin system (useful for testing)."""
    from .registry import reset_registry
    reset_registry()


# Natural collection accessors for convenience
class _CollectionProvider:
    """Lazy collection provider for natural access patterns."""
    
    def __init__(self):
        self._collections = None
        # Ensure frameworks are initialized
        ensure_frameworks_initialized()
    
    @property
    def transforms(self):
        """Get transforms collection."""
        if self._collections is None:
            self._collections = create_collections(get_registry())
        return self._collections['transforms']
    
    @property 
    def kernels(self):
        """Get kernels collection."""
        if self._collections is None:
            self._collections = create_collections(get_registry())
        return self._collections['kernels']
    
    @property
    def backends(self):
        """Get backends collection."""
        if self._collections is None:
            self._collections = create_collections(get_registry())
        return self._collections['backends']
    
    @property
    def steps(self):
        """Get steps collection."""
        if self._collections is None:
            self._collections = create_collections(get_registry())
        return self._collections['steps']


# Create convenience accessors
_provider = _CollectionProvider()
transforms = _provider.transforms
kernels = _provider.kernels 
backends = _provider.backends
steps = _provider.steps


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
    "ensure_frameworks_initialized",
    
    # Utility functions
    "plugin_status",
    "reset_plugin_system",
    
    # Natural access
    "transforms",
    "kernels", 
    "backends",
    "steps"
]

__version__ = "1.0.0"
__description__ = "Perfect Code Plugin System - Direct Registry with Zero Discovery Overhead"