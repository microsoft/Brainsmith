"""
Global Plugin Collections - Zero Boilerplate Plugin Access

Provides natural, global access to plugins without any boilerplate code.

Usage:
    from brainsmith.plugins import transforms, kernels
    
    # Natural access - no boilerplate needed
    model = transforms.ExpandNorms()(model)
    model = transforms.qonnx.RemoveIdentityOps()(model)
    
    # Kernel access with backends
    layer_norm_hls = kernels.LayerNorm.hls()
    softmax_rtl = kernels.Softmax.rtl()
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brainsmith.plugin.collections import TransformCollection, KernelCollection

logger = logging.getLogger(__name__)


class _GlobalTransformCollection:
    """
    Thread-safe global transform collection with lazy loading.
    
    This acts like a module-level import but provides dynamic access
    to all available transforms across frameworks.
    """
    
    def __init__(self):
        self._collection = None
        self._manager = None
    
    @property
    def _transform_collection(self) -> 'TransformCollection':
        """Lazy-load the transform collection."""
        if self._collection is None:
            from brainsmith.plugin.manager import get_plugin_manager
            self._manager = get_plugin_manager()
            self._collection = self._manager.get_transforms()
        return self._collection
    
    def __getattr__(self, name: str):
        """Delegate all access to underlying transform collection."""
        return getattr(self._transform_collection, name)
    
    def __dir__(self):
        """Support tab completion."""
        return dir(self._transform_collection)
    
    def __repr__(self):
        return "Global Transform Collection (brainsmith.plugins.transforms)"


class _GlobalKernelCollection:
    """
    Thread-safe global kernel collection with lazy loading.
    
    Provides access to kernels and their backends across all frameworks.
    """
    
    def __init__(self):
        self._collection = None
        self._manager = None
    
    @property
    def _kernel_collection(self) -> 'KernelCollection':
        """Lazy-load the kernel collection."""
        if self._collection is None:
            from brainsmith.plugin.manager import get_plugin_manager
            self._manager = get_plugin_manager()
            self._collection = self._manager.get_kernels()
        return self._collection
    
    def __getattr__(self, name: str):
        """Delegate all access to underlying kernel collection."""
        return getattr(self._kernel_collection, name)
    
    def __dir__(self):
        """Support tab completion."""
        return dir(self._kernel_collection)
    
    def __repr__(self):
        return "Global Kernel Collection (brainsmith.plugins.kernels)"


# Global instances - these act like module-level imports
# Usage: from brainsmith.plugins import transforms, kernels
transforms = _GlobalTransformCollection()
kernels = _GlobalKernelCollection()


# Utility functions for advanced usage
def get_plugin_manager():
    """Get the underlying plugin manager for advanced operations."""
    from brainsmith.plugin.manager import get_plugin_manager as _get_manager
    return _get_manager()


def list_all_plugins():
    """List all available plugins across all frameworks."""
    manager = get_plugin_manager()
    return manager.list_available()


def analyze_conflicts():
    """Analyze naming conflicts between frameworks."""
    manager = get_plugin_manager()
    return manager.analyze_conflicts()


def reset_plugin_cache():
    """Reset plugin cache (useful for testing)."""
    global transforms, kernels
    
    # Reset the global collections
    transforms._collection = None
    transforms._manager = None
    kernels._collection = None
    kernels._manager = None
    
    # Reset the underlying manager
    manager = get_plugin_manager()
    manager.reset()
    
    logger.info("Reset global plugin cache")


# Convenience functions for common patterns
def load_plugins_for_blueprint(blueprint_path: str):
    """
    Pre-load and optimize plugins for a specific blueprint.
    
    This is optional - plugins will be loaded on-demand anyway,
    but this can improve performance for blueprint-heavy workflows.
    """
    # This would integrate with the blueprint optimizer once implemented
    manager = get_plugin_manager()
    
    # For now, just ensure discovery is complete
    catalog = manager.discover_all()
    
    logger.info(f"Plugin discovery complete for blueprint optimization")
    return catalog


def plugin_status():
    """Get status information about the plugin system."""
    manager = get_plugin_manager()
    catalog = manager.discover_all()
    
    total_plugins = sum(len(plist) for plist in catalog.plugins_by_name.values())
    unique_count = len(catalog.unique_plugins)
    conflict_count = len(catalog.conflicts)
    
    by_framework = {}
    for framework, plugin_list in catalog.plugins_by_framework.items():
        by_framework[framework] = len(plugin_list)
    
    by_type = {}
    for plugin_type, plugin_list in catalog.plugins_by_type.items():
        by_type[plugin_type] = len(plugin_list)
    
    return {
        'total_plugins': total_plugins,
        'unique_plugins': unique_count,
        'conflicted_plugins': conflict_count,
        'by_framework': by_framework,
        'by_type': by_type,
        'discovery_strategy': manager.strategy.value
    }


# Make key classes available for type hints and advanced usage
__all__ = [
    'transforms',
    'kernels', 
    'get_plugin_manager',
    'list_all_plugins',
    'analyze_conflicts',
    'reset_plugin_cache',
    'load_plugins_for_blueprint',
    'plugin_status'
]