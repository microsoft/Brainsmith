"""
Plugin Registry

Central registry for managing all plugins in the BrainSmith system.
"""

import logging
from typing import Dict, Type, Optional, List, Tuple, Any
from threading import Lock

from .exceptions import (
    PluginNotFoundError,
    PluginRegistrationError,
    PluginDependencyError
)

logger = logging.getLogger(__name__)


class PluginRegistry:
    """
    Singleton registry for all plugins.
    
    This registry manages transforms, kernels, backends, and hardware transforms,
    providing methods for registration, retrieval, and discovery.
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        """Thread-safe singleton implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the registry if not already initialized."""
        if self._initialized:
            return
            
        self._plugins = {
            "transform": {},
            "kernel": {},
            "backend": {},
            "hw_transform": {}
        }
        self._initialized = True
        logger.info("PluginRegistry initialized")
    
    @classmethod
    def register(cls, plugin_class: Type) -> None:
        """
        Register a plugin class.
        
        Args:
            plugin_class: Class with _plugin_metadata attribute
            
        Raises:
            PluginRegistrationError: If registration fails
        """
        instance = cls()
        
        # Validate plugin has metadata
        if not hasattr(plugin_class, '_plugin_metadata'):
            raise PluginRegistrationError(
                f"Plugin class {plugin_class.__name__} missing _plugin_metadata"
            )
        
        metadata = plugin_class._plugin_metadata
        plugin_type = metadata.get("type")
        name = metadata.get("name")
        
        # Validate metadata
        if not plugin_type:
            raise PluginRegistrationError("Plugin metadata missing 'type'")
        if not name:
            raise PluginRegistrationError("Plugin metadata missing 'name'")
        if plugin_type not in instance._plugins:
            raise PluginRegistrationError(f"Invalid plugin type: {plugin_type}")
        
        # Check for duplicates
        if name in instance._plugins[plugin_type]:
            existing_class = instance._plugins[plugin_type][name]
            if existing_class != plugin_class:
                raise PluginRegistrationError(
                    f"{plugin_type} '{name}' already registered by {existing_class.__name__}"
                )
            else:
                # Same class registered again, skip
                return
        
        # Validate dependencies if specified
        if "requires" in metadata:
            instance._validate_requires(metadata["requires"])
        
        # Register the plugin
        instance._plugins[plugin_type][name] = plugin_class
        logger.debug(f"Registered {plugin_type}: {name} ({plugin_class.__name__})")
    
    def get_transform(self, name: str) -> Optional[Type]:
        """
        Get a transform by name.
        
        Args:
            name: Transform name
            
        Returns:
            Transform class or None if not found
        """
        return self._plugins["transform"].get(name)
    
    def get_kernel(self, name: str) -> Optional[Type]:
        """
        Get a kernel by name.
        
        Args:
            name: Kernel name
            
        Returns:
            Kernel class or None if not found
        """
        return self._plugins["kernel"].get(name)
    
    def get_backend(self, name: str) -> Optional[Type]:
        """
        Get a backend by name.
        
        Args:
            name: Backend name
            
        Returns:
            Backend class or None if not found
        """
        return self._plugins["backend"].get(name)
    
    def get_hw_transform(self, name: str) -> Optional[Type]:
        """
        Get a hardware transform by name.
        
        Args:
            name: Hardware transform name
            
        Returns:
            Hardware transform class or None if not found
        """
        return self._plugins["hw_transform"].get(name)
    
    def list_transforms(self, stage: Optional[str] = None) -> List[Tuple[str, Type]]:
        """
        List all transforms, optionally filtered by stage.
        
        Args:
            stage: Optional stage to filter by
            
        Returns:
            List of (name, class) tuples
        """
        transforms = []
        for name, cls in self._plugins["transform"].items():
            metadata = cls._plugin_metadata
            if stage is None or metadata.get("stage") == stage:
                transforms.append((name, cls))
        return sorted(transforms, key=lambda x: x[0])
    
    def list_kernels(self) -> List[Tuple[str, Type]]:
        """
        List all registered kernels.
        
        Returns:
            List of (name, class) tuples
        """
        return sorted(self._plugins["kernel"].items(), key=lambda x: x[0])
    
    def list_backends(self) -> List[Tuple[str, Type]]:
        """
        List all registered backends.
        
        Returns:
            List of (name, class) tuples
        """
        return sorted(self._plugins["backend"].items(), key=lambda x: x[0])
    
    def list_hw_transforms(self) -> List[Tuple[str, Type]]:
        """
        List all registered hardware transforms.
        
        Returns:
            List of (name, class) tuples
        """
        return sorted(self._plugins["hw_transform"].items(), key=lambda x: x[0])
    
    def get_plugin_info(self, plugin_type: str, name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a plugin.
        
        Args:
            plugin_type: Type of plugin ("transform", "kernel", etc.)
            name: Plugin name
            
        Returns:
            Plugin metadata dictionary or None if not found
        """
        if plugin_type not in self._plugins:
            return None
            
        plugin_class = self._plugins[plugin_type].get(name)
        if not plugin_class:
            return None
            
        return plugin_class._plugin_metadata.copy()
    
    def search_plugins(self, query: str, plugin_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for plugins by name or description.
        
        Args:
            query: Search query (case-insensitive)
            plugin_type: Optional plugin type to search within
            
        Returns:
            List of plugin metadata dictionaries
        """
        results = []
        query_lower = query.lower()
        
        search_types = [plugin_type] if plugin_type else self._plugins.keys()
        
        for ptype in search_types:
            if ptype not in self._plugins:
                continue
                
            for name, cls in self._plugins[ptype].items():
                metadata = cls._plugin_metadata
                
                # Search in name and description
                if (query_lower in name.lower() or 
                    (metadata.get("description") and 
                     query_lower in metadata["description"].lower())):
                    results.append(metadata.copy())
        
        return results
    
    def _validate_requires(self, requires: List[str]) -> None:
        """
        Validate plugin requirements.
        
        Args:
            requires: List of requirement strings
            
        Raises:
            PluginDependencyError: If requirements cannot be satisfied
        """
        for req in requires:
            # Check if it's a kernel requirement
            if req.startswith("kernel:"):
                kernel_name = req[7:]  # Remove "kernel:" prefix
                if kernel_name not in self._plugins["kernel"]:
                    logger.warning(
                        f"Required kernel '{kernel_name}' not found. "
                        "It may be registered later."
                    )
            
            # Check if it's a transform requirement
            elif req.startswith("transform:"):
                transform_name = req[10:]  # Remove "transform:" prefix
                if transform_name not in self._plugins["transform"]:
                    logger.warning(
                        f"Required transform '{transform_name}' not found. "
                        "It may be registered later."
                    )
            
            # For library requirements, we just log them
            # Actual validation would happen at runtime
            else:
                logger.debug(f"Library requirement: {req}")
    
    def clear(self) -> None:
        """Clear all registered plugins. Useful for testing."""
        self._plugins = {
            "transform": {},
            "kernel": {},
            "backend": {},
            "hw_transform": {}
        }
        logger.info("PluginRegistry cleared")
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about registered plugins.
        
        Returns:
            Dictionary with counts for each plugin type
        """
        return {
            plugin_type: len(plugins)
            for plugin_type, plugins in self._plugins.items()
        }