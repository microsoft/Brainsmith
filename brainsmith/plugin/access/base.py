"""
Base Classes for Plugin Collections

Provides common functionality for all collection types.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from threading import RLock

if TYPE_CHECKING:
    from ..core.registry import PluginRegistry
    from ..core.loader import PluginLoader
    from ..core.data_models import PluginInfo

logger = logging.getLogger(__name__)


class BaseCollection(ABC):
    """
    Abstract base class for all plugin collections.
    
    Provides common functionality like caching, error handling,
    and plugin access patterns.
    """
    
    def __init__(self, registry: 'PluginRegistry', loader: 'PluginLoader'):
        self.registry = registry
        self.loader = loader
        self._lock = RLock()
        self._wrapper_cache: Dict[str, Any] = {}
    
    @property
    @abstractmethod
    def plugin_type(self) -> str:
        """The type of plugins this collection manages."""
        pass
    
    @abstractmethod
    def _create_wrapper(self, plugin_info: 'PluginInfo') -> Any:
        """Create appropriate wrapper for the plugin."""
        pass
    
    def _get_wrapper(self, name: str, framework: Optional[str] = None) -> Any:
        """
        Get or create a wrapper for the named plugin.
        
        Handles caching and error reporting.
        """
        with self._lock:
            # Create cache key
            cache_key = f"{framework or 'any'}:{name}"
            
            # Check cache
            if cache_key in self._wrapper_cache:
                return self._wrapper_cache[cache_key]
            
            # Get plugin from registry
            try:
                plugin_info = self.registry.get_plugin(name, framework)
                if not plugin_info:
                    raise AttributeError(self._plugin_not_found_message(name, framework))
                
                # Validate plugin type
                if plugin_info.plugin_type != self.plugin_type:
                    raise AttributeError(
                        f"Plugin '{name}' is a {plugin_info.plugin_type}, "
                        f"not a {self.plugin_type}"
                    )
                
                # Create and cache wrapper
                wrapper = self._create_wrapper(plugin_info)
                self._wrapper_cache[cache_key] = wrapper
                return wrapper
                
            except ValueError as e:
                # Handle ambiguous names
                if "ambiguous" in str(e).lower():
                    raise AttributeError(self._ambiguous_plugin_message(name, e))
                raise
    
    def _plugin_not_found_message(self, name: str, framework: Optional[str]) -> str:
        """Generate helpful error message for missing plugins."""
        # Get available plugins
        available = self.registry.find_plugins(
            plugin_type=self.plugin_type,
            framework=framework
        )
        
        # Find similar names
        similar = [
            p.name for p in available 
            if name.lower() in p.name.lower() or p.name.lower() in name.lower()
        ][:3]
        
        msg = f"{self.plugin_type.title()} '{name}' not found"
        if framework:
            msg += f" in {framework} framework"
        
        if similar:
            msg += f". Similar {self.plugin_type}s: {similar}"
        
        return msg
    
    def _ambiguous_plugin_message(self, name: str, error: Exception) -> str:
        """Generate helpful error message for ambiguous plugins."""
        return (
            f"{self.plugin_type.title()} '{name}' is ambiguous. {str(error)} "
            f"Use framework-specific access (e.g., qonnx.{name} or finn.{name})."
        )
    
    def list_available(self, framework: Optional[str] = None) -> List[str]:
        """List all available plugins of this type."""
        plugins = self.registry.find_plugins(
            plugin_type=self.plugin_type,
            framework=framework
        )
        return sorted([p.name for p in plugins])
    
    def __dir__(self) -> List[str]:
        """Support tab completion."""
        return self.list_available()


class FrameworkCollection(BaseCollection):
    """
    Base class for framework-specific collections.
    
    Provides plugins from a specific framework (qonnx, finn, brainsmith).
    """
    
    def __init__(self, framework: str, registry: 'PluginRegistry', 
                 loader: 'PluginLoader'):
        super().__init__(registry, loader)
        self.framework = framework
    
    def __getattr__(self, name: str) -> Any:
        """Get plugin by name from this framework."""
        if name.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
        
        return self._get_wrapper(name, self.framework)
    
    def __dir__(self) -> List[str]:
        """Support tab completion with framework-specific plugins."""
        return self.list_available(self.framework)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.framework})"