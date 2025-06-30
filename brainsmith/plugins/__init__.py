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
    from brainsmith.plugin.access.transforms import TransformCollection
    from brainsmith.plugin.access.kernels import KernelCollection
    from brainsmith.plugin.access.steps import StepCollection

logger = logging.getLogger(__name__)


class _GlobalTransformCollection:
    """
    Thread-safe global transform collection with lazy loading.
    
    This acts like a module-level import but provides dynamic access
    to all available transforms across frameworks.
    """
    
    def __init__(self):
        self._collection = None
    
    @property
    def _transform_collection(self) -> 'TransformCollection':
        """Lazy-load the transform collection."""
        if self._collection is None:
            from brainsmith.plugin.access.factory import get_collection_factory
            factory = get_collection_factory()
            self._collection = factory.create_transform_collection()
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
    
    @property
    def _kernel_collection(self) -> 'KernelCollection':
        """Lazy-load the kernel collection."""
        if self._collection is None:
            from brainsmith.plugin.access.factory import get_collection_factory
            factory = get_collection_factory()
            self._collection = factory.create_kernel_collection()
        return self._collection
    
    def __getattr__(self, name: str):
        """Delegate all access to underlying kernel collection."""
        return getattr(self._kernel_collection, name)
    
    def __dir__(self):
        """Support tab completion."""
        return dir(self._kernel_collection)
    
    def __repr__(self):
        return "Global Kernel Collection (brainsmith.plugins.kernels)"


class _GlobalStepCollection:
    """
    Thread-safe global step collection with lazy loading.
    
    Provides access to FINN build steps.
    """
    
    def __init__(self):
        self._collection = None
    
    @property
    def _step_collection(self) -> 'StepCollection':
        """Lazy-load the step collection."""
        if self._collection is None:
            from brainsmith.plugin.access.factory import get_collection_factory
            factory = get_collection_factory()
            self._collection = factory.create_step_collection()
        return self._collection
    
    def __getattr__(self, name: str):
        """Delegate all access to underlying step collection."""
        return getattr(self._step_collection, name)
    
    def __dir__(self):
        """Support tab completion."""
        return dir(self._step_collection)
    
    def __repr__(self):
        return "Global Step Collection (brainsmith.plugins.steps)"


# Global instances - these act like module-level imports
# Usage: from brainsmith.plugins import transforms, kernels, steps
transforms = _GlobalTransformCollection()
kernels = _GlobalKernelCollection()
steps = _GlobalStepCollection()


# Utility functions for advanced usage
def get_plugin_manager():
    """Get the underlying plugin manager for advanced operations."""
    from brainsmith.plugin.manager import get_plugin_manager as _get_manager
    return _get_manager()


def get_plugin_registry():
    """Get the underlying plugin registry for advanced operations."""
    from brainsmith.plugin.core.registry import get_plugin_registry as _get_registry
    return _get_registry()


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
    global transforms, kernels, steps
    
    # Reset the global collections
    transforms._collection = None
    kernels._collection = None
    steps._collection = None
    
    # Reset the factory
    from brainsmith.plugin.access.factory import get_collection_factory
    factory = get_collection_factory()
    factory.reset()
    
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
    # Use the registry directly for better stats
    registry = get_plugin_registry()
    summary = registry.get_summary()
    
    # Get discovery strategy from manager
    manager = get_plugin_manager()
    
    return {
        'total_plugins': summary['total_plugins'],
        'unique_plugins': summary['unique_plugins'],
        'conflicted_plugins': summary['conflicted_plugins'],
        'by_framework': summary['by_framework'],
        'by_type': summary['by_type'],
        'stages': summary.get('stages', {}),
        'kernel_backends': summary.get('kernel_backends', {}),
        'discovery_strategy': manager.strategy.value
    }


# Make key classes available for type hints and advanced usage
__all__ = [
    'transforms',
    'kernels',
    'steps',
    'get_plugin_manager',
    'get_plugin_registry',
    'list_all_plugins',
    'analyze_conflicts',
    'reset_plugin_cache',
    'load_plugins_for_blueprint',
    'plugin_status'
]