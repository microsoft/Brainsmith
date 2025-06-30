"""
Plugin Loader

Handles the loading and instantiation of plugins, separate from
discovery and registry concerns.
"""

import logging
from threading import RLock
from typing import Dict, Optional, Any, Set
from weakref import WeakValueDictionary

from .data_models import PluginInfo
from .registry import PluginRegistry, get_plugin_registry

logger = logging.getLogger(__name__)


class PluginLoader:
    """
    Handles plugin loading and instantiation.
    
    Features:
    - Lazy loading of plugin classes
    - Instance caching for stateless plugins
    - Thread-safe loading
    - Memory-efficient weak references
    """
    
    def __init__(self, registry: Optional[PluginRegistry] = None):
        self.registry = registry or get_plugin_registry()
        self._lock = RLock()
        
        # Cache loaded plugin classes
        self._loaded_plugins: Dict[str, PluginInfo] = {}
        
        # Cache plugin instances (weak references to allow GC)
        self._instance_cache: WeakValueDictionary = WeakValueDictionary()
        
        # Track which plugins are stateless (can be cached)
        self._stateless_plugins: Set[str] = set()
    
    def load_plugin(self, name: str, plugin_type: str = None, 
                   framework: str = None) -> Optional[PluginInfo]:
        """
        Load a plugin and mark it as loaded.
        
        This ensures the plugin class is imported and ready to use,
        but does not instantiate it.
        """
        with self._lock:
            # Create cache key
            cache_key = self._make_cache_key(name, plugin_type, framework)
            
            # Check if already loaded
            if cache_key in self._loaded_plugins:
                return self._loaded_plugins[cache_key]
            
            # Get plugin from registry
            plugin_info = self.registry.get_plugin(name, framework)
            if not plugin_info:
                return None
            
            # Validate type if specified
            if plugin_type and plugin_info.plugin_type != plugin_type:
                logger.warning(
                    f"Plugin '{name}' has type '{plugin_info.plugin_type}', "
                    f"expected '{plugin_type}'"
                )
                return None
            
            # Mark as loaded
            plugin_info.metadata['is_loaded'] = True
            self._loaded_plugins[cache_key] = plugin_info
            
            # Check if plugin is stateless (for caching)
            if self._is_stateless(plugin_info):
                self._stateless_plugins.add(cache_key)
            
            logger.debug(f"Loaded plugin: {plugin_info}")
            return plugin_info
    
    def instantiate(self, name: str, plugin_type: str = None,
                   framework: str = None, **kwargs) -> Optional[Any]:
        """
        Load and instantiate a plugin.
        
        For stateless plugins with no kwargs, returns cached instance.
        Otherwise creates a new instance.
        """
        # Load the plugin first
        plugin_info = self.load_plugin(name, plugin_type, framework)
        if not plugin_info:
            return None
        
        # Check if we can use cached instance
        cache_key = self._make_cache_key(name, plugin_type, framework)
        if (not kwargs and 
            cache_key in self._stateless_plugins and
            cache_key in self._instance_cache):
            logger.debug(f"Returning cached instance for: {name}")
            return self._instance_cache[cache_key]
        
        # Create new instance
        try:
            instance = plugin_info.instantiate(**kwargs)
            
            # Cache if appropriate
            if not kwargs and cache_key in self._stateless_plugins:
                self._instance_cache[cache_key] = instance
            
            return instance
            
        except Exception as e:
            logger.error(f"Failed to instantiate plugin '{name}': {e}")
            raise
    
    def get_transform(self, name: str, framework: str = None, **kwargs) -> Optional[Any]:
        """Convenience method to get a transform instance."""
        return self.instantiate(name, "transform", framework, **kwargs)
    
    def get_kernel(self, name: str, framework: str = None, **kwargs) -> Optional[Any]:
        """Convenience method to get a kernel instance."""
        return self.instantiate(name, "kernel", framework, **kwargs)
    
    def get_backend(self, name: str, kernel: str, backend_type: str, **kwargs) -> Optional[Any]:
        """
        Get a backend instance for a specific kernel.
        
        Args:
            name: Backend name
            kernel: Kernel this backend implements
            backend_type: Either "hls" or "rtl"
        """
        # Find backend in registry
        backends = self.registry.list_backends(kernel, backend_type)
        
        for backend in backends:
            if backend.name == name:
                return self.instantiate(
                    backend.name, 
                    "backend", 
                    backend.framework, 
                    **kwargs
                )
        
        logger.error(f"Backend '{name}' not found for kernel '{kernel}' ({backend_type})")
        return None
    
    def get_step(self, name: str, **kwargs) -> Optional[Any]:
        """Convenience method to get a step function."""
        return self.instantiate(name, "step", "brainsmith", **kwargs)
    
    def preload_plugins(self, plugin_names: List[str]) -> Dict[str, PluginInfo]:
        """
        Preload multiple plugins for better performance.
        
        Useful for blueprint optimization where we know which
        plugins will be needed.
        """
        loaded = {}
        errors = []
        
        for name in plugin_names:
            try:
                # Parse framework prefix if present
                if ":" in name:
                    framework, plugin_name = name.split(":", 1)
                    plugin_info = self.load_plugin(plugin_name, framework=framework)
                else:
                    plugin_info = self.load_plugin(name)
                
                if plugin_info:
                    loaded[name] = plugin_info
                else:
                    errors.append(f"Plugin '{name}' not found")
                    
            except Exception as e:
                errors.append(f"Failed to load '{name}': {e}")
        
        if errors:
            logger.warning(f"Plugin loading errors: {'; '.join(errors)}")
        
        return loaded
    
    def _is_stateless(self, plugin_info: PluginInfo) -> bool:
        """
        Determine if a plugin is stateless and can be cached.
        
        Most transforms are stateless, but some kernels and backends
        might maintain state.
        """
        # Transforms are typically stateless
        if plugin_info.plugin_type == "transform":
            # Check for known stateful transforms
            stateful_names = {'ConfigureDataType', 'SetExecMode'}
            return plugin_info.name not in stateful_names
        
        # Steps are functions, always stateless
        if plugin_info.plugin_type == "step":
            return True
        
        # Kernels and backends might have state
        if plugin_info.plugin_type in ["kernel", "backend"]:
            # Check class for state indicators
            cls = plugin_info.plugin_class
            
            # If it has __init__ with required args, it's stateful
            if hasattr(cls, '__init__'):
                import inspect
                sig = inspect.signature(cls.__init__)
                params = list(sig.parameters.values())
                # Skip 'self' parameter
                if len(params) > 1:
                    # Has required parameters, likely stateful
                    return False
            
            return True
        
        # Default to not caching
        return False
    
    def _make_cache_key(self, name: str, plugin_type: Optional[str], 
                       framework: Optional[str]) -> str:
        """Create a cache key for a plugin."""
        return f"{framework or 'any'}:{plugin_type or 'any'}:{name}"
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        with self._lock:
            self._loaded_plugins.clear()
            self._instance_cache.clear()
            self._stateless_plugins.clear()
            logger.info("Cleared plugin loader cache")
    
    def get_loaded_plugins(self) -> List[PluginInfo]:
        """Get list of all loaded plugins."""
        with self._lock:
            return list(self._loaded_plugins.values())