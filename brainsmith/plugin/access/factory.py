"""
Collection Factory

Factory for creating plugin collections with proper initialization.
"""

import logging
from typing import Optional, TYPE_CHECKING

from .transforms import TransformCollection
from .kernels import KernelCollection
from .steps import StepCollection

if TYPE_CHECKING:
    from ..core.registry import PluginRegistry
    from ..core.loader import PluginLoader

logger = logging.getLogger(__name__)


class CollectionFactory:
    """
    Factory for creating plugin collections.
    
    This factory ensures collections are properly initialized with
    the registry and loader, and provides caching for singleton behavior.
    """
    
    def __init__(self, registry: Optional['PluginRegistry'] = None,
                 loader: Optional['PluginLoader'] = None):
        """
        Initialize factory with optional registry and loader.
        
        If not provided, will use the global instances.
        """
        self._registry = registry
        self._loader = loader
        
        # Cache created collections
        self._transform_collection: Optional[TransformCollection] = None
        self._kernel_collection: Optional[KernelCollection] = None
        self._step_collection: Optional[StepCollection] = None
    
    @property
    def registry(self) -> 'PluginRegistry':
        """Get registry, using global if not set."""
        if self._registry is None:
            from ..core.registry import get_plugin_registry
            self._registry = get_plugin_registry()
        return self._registry
    
    @property
    def loader(self) -> 'PluginLoader':
        """Get loader, using global if not set."""
        if self._loader is None:
            from ..core.loader import PluginLoader
            self._loader = PluginLoader(self.registry)
        return self._loader
    
    def create_transform_collection(self) -> TransformCollection:
        """
        Create or return cached transform collection.
        
        Returns the same instance on repeated calls (singleton behavior).
        """
        if self._transform_collection is None:
            self._transform_collection = TransformCollection(
                self.registry, self.loader
            )
            logger.debug("Created TransformCollection")
        return self._transform_collection
    
    def create_kernel_collection(self) -> KernelCollection:
        """
        Create or return cached kernel collection.
        
        Returns the same instance on repeated calls (singleton behavior).
        """
        if self._kernel_collection is None:
            self._kernel_collection = KernelCollection(
                self.registry, self.loader
            )
            logger.debug("Created KernelCollection")
        return self._kernel_collection
    
    def create_step_collection(self) -> StepCollection:
        """
        Create or return cached step collection.
        
        Returns the same instance on repeated calls (singleton behavior).
        """
        if self._step_collection is None:
            self._step_collection = StepCollection(
                self.registry, self.loader
            )
            logger.debug("Created StepCollection")
        return self._step_collection
    
    def reset(self) -> None:
        """
        Reset all cached collections.
        
        Useful for testing or when the registry changes.
        """
        self._transform_collection = None
        self._kernel_collection = None
        self._step_collection = None
        logger.debug("Reset all collections")


# Global factory instance
_global_factory: Optional[CollectionFactory] = None


def get_collection_factory() -> CollectionFactory:
    """Get the global collection factory instance."""
    global _global_factory
    
    if _global_factory is None:
        _global_factory = CollectionFactory()
    
    return _global_factory


def set_collection_factory(factory: CollectionFactory) -> None:
    """Set the global collection factory (useful for testing)."""
    global _global_factory
    _global_factory = factory