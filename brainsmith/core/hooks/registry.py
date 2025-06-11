"""
Hooks Registry System

Auto-discovery and management of hooks plugins and event handlers.
Provides registration, caching, and lookup functionality for extending
the optimization event system with custom capabilities.
"""

import os
import inspect
import logging
from typing import Dict, List, Optional, Set, Type, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from brainsmith.core.registry import BaseRegistry, ComponentInfo
from .types import HooksPlugin, EventHandler
from .plugins import PluginManager

logger = logging.getLogger(__name__)


class PluginType(Enum):
    """Types of hooks plugins available."""
    EVENT_HANDLER = "event_handler"
    ANALYSIS = "analysis"
    MONITORING = "monitoring"
    STATISTICS = "statistics"
    MACHINE_LEARNING = "machine_learning"
    PERSISTENCE = "persistence"
    UTILITY = "utility"


@dataclass
class PluginInfo:
    """Information about a discovered plugin."""
    name: str
    plugin_type: PluginType
    plugin_class: Type[HooksPlugin]
    module_path: str
    description: str = ""
    version: str = "1.0.0"
    dependencies: List[str] = None
    capabilities: List[str] = None
    installed: bool = False
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.capabilities is None:
            self.capabilities = []


@dataclass
class HandlerInfo:
    """Information about a discovered event handler."""
    name: str
    handler_class: Type[EventHandler]
    module_path: str
    event_types: List[str] = None
    description: str = ""
    is_global: bool = False
    
    def __post_init__(self):
        if self.event_types is None:
            self.event_types = []


class HooksRegistry(BaseRegistry[PluginInfo]):
    """Registry for auto-discovery and management of hooks plugins and handlers."""
    
    def __init__(self, search_dirs: Optional[List[str]] = None, config_manager=None):
        """
        Initialize hooks registry.
        
        Args:
            search_dirs: List of directories to search for hooks.
                        If None, uses default hooks directories.
            config_manager: Optional configuration manager.
        """
        super().__init__(search_dirs, config_manager)
        self.handler_cache = {}
        self.plugin_manager = PluginManager()
    
    def discover_components(self, rescan: bool = False) -> Dict[str, PluginInfo]:
        """
        Discover all available plugin components.
        
        Args:
            rescan: Force rescan even if cache exists
            
        Returns:
            Dictionary mapping component names to PluginInfo objects
        """
        if self._cache and not rescan:
            return self._cache
        
        discovered = {}
        
        # Discover plugins in the plugins directory
        plugins_discovered = self._discover_plugins_directory()
        discovered.update(plugins_discovered)
        
        # Discover contrib plugins
        contrib_discovered = self._discover_contrib_plugins()
        discovered.update(contrib_discovered)
        
        # Check which plugins are currently installed
        installed_plugins = self.plugin_manager.list_plugins()
        for plugin_info in discovered.values():
            plugin_info.installed = plugin_info.name in installed_plugins
        
        # Cache the results
        self._cache = discovered
        
        self._log_debug(f"Discovered {len(discovered)} hook plugins")
        return discovered

    
    def discover_handlers(self, rescan: bool = False) -> Dict[str, HandlerInfo]:
        """
        Discover all available event handler classes.
        
        Args:
            rescan: Force rescan even if cache exists
            
        Returns:
            Dictionary mapping handler names to HandlerInfo objects
        """
        if self.handler_cache and not rescan:
            return self.handler_cache
        
        discovered = {}
        
        # Discover built-in handlers
        builtin_handlers = self._discover_builtin_handlers()
        discovered.update(builtin_handlers)
        
        # Discover handlers from plugins
        plugin_handlers = self._discover_plugin_handlers()
        discovered.update(plugin_handlers)
        
        # Cache the results
        self.handler_cache = discovered
        
        self._log_info(f"Discovered {len(discovered)} event handlers")
        return discovered
    
    def _discover_plugins_directory(self) -> Dict[str, PluginInfo]:
        """Discover plugins in the plugins directory."""
        plugins = {}
        
        for hooks_dir in self.search_dirs:
            plugins_dir = os.path.join(hooks_dir, "plugins")
            
            if not os.path.exists(plugins_dir):
                continue
            
            # Look for Python files in plugins directory
            for item in os.listdir(plugins_dir):
                if item.endswith('.py') and item not in ['__init__.py', 'examples.py']:
                    module_name = item[:-3]
                    
                    try:
                        # Import plugin module
                        import importlib.util
                        spec = importlib.util.spec_from_file_location(
                            f"plugins.{module_name}", 
                            os.path.join(plugins_dir, item)
                        )
                        
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            
                            # Look for HooksPlugin classes
                            for name, obj in inspect.getmembers(module):
                                if (inspect.isclass(obj) and 
                                    issubclass(obj, HooksPlugin) and 
                                    obj != HooksPlugin):
                                    
                                    plugin_info = PluginInfo(
                                        name=name.lower().replace('plugin', ''),
                                        plugin_type=self._classify_plugin_type(name, obj),
                                        plugin_class=obj,
                                        module_path=f'brainsmith.core.hooks.plugins.{module_name}',
                                        description=self._extract_description(obj),
                                        capabilities=self._extract_capabilities(obj)
                                    )
                                    plugins[plugin_info.name] = plugin_info
                                    
                    except Exception as e:
                        self._log_debug(f"Could not load plugin module {module_name}: {e}")
        
        return plugins
    
    def _discover_contrib_plugins(self) -> Dict[str, PluginInfo]:
        """Discover contrib plugins."""
        plugins = {}
        
        for hooks_dir in self.search_dirs:
            contrib_dir = os.path.join(hooks_dir, "contrib")
            
            if not os.path.exists(contrib_dir):
                continue
            
            # Look for Python files in contrib directory
            for item in os.listdir(contrib_dir):
                if item.endswith('.py') and item != '__init__.py':
                    module_name = item[:-3]
                    
                    try:
                        # Import contrib module
                        import importlib.util
                        spec = importlib.util.spec_from_file_location(
                            f"contrib.{module_name}", 
                            os.path.join(contrib_dir, item)
                        )
                        
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            
                            # Look for HooksPlugin classes
                            for name, obj in inspect.getmembers(module):
                                if (inspect.isclass(obj) and 
                                    issubclass(obj, HooksPlugin) and 
                                    obj != HooksPlugin):
                                    
                                    plugin_name = f"contrib_{name.lower().replace('plugin', '')}"
                                    plugin_info = PluginInfo(
                                        name=plugin_name,
                                        plugin_type=self._classify_plugin_type(name, obj),
                                        plugin_class=obj,
                                        module_path=f'brainsmith.core.hooks.contrib.{module_name}',
                                        description=self._extract_description(obj),
                                        capabilities=self._extract_capabilities(obj)
                                    )
                                    plugins[plugin_name] = plugin_info
                                    
                    except Exception as e:
                        self._log_debug(f"Could not load contrib plugin module {module_name}: {e}")
        
        return plugins
    
    def _discover_builtin_handlers(self) -> Dict[str, HandlerInfo]:
        """Discover built-in event handlers."""
        handlers = {}
        
        try:
            # Import events module to find built-in handlers
            from . import events
            
            for name, obj in inspect.getmembers(events):
                if (inspect.isclass(obj) and 
                    issubclass(obj, EventHandler) and 
                    obj != EventHandler):
                    
                    handler_info = HandlerInfo(
                        name=name.lower().replace('handler', ''),
                        handler_class=obj,
                        module_path='brainsmith.core.hooks.events',
                        description=self._extract_description(obj),
                        event_types=self._extract_event_types(obj)
                    )
                    handlers[handler_info.name] = handler_info
                    
        except ImportError:
            self._log_debug("Could not import events module")
        
        return handlers
    
    def _discover_plugin_handlers(self) -> Dict[str, HandlerInfo]:
        """Discover event handlers from installed plugins."""
        handlers = {}
        
        # Get handlers from installed plugins
        plugin_handlers = self.plugin_manager.get_all_handlers()
        
        for handler in plugin_handlers:
            handler_class = type(handler)
            handler_name = handler_class.__name__.lower().replace('handler', '')
            
            handler_info = HandlerInfo(
                name=f"plugin_{handler_name}",
                handler_class=handler_class,
                module_path=f"{handler_class.__module__}",
                description=self._extract_description(handler_class),
                event_types=self._extract_event_types(handler_class),
                is_global=True  # Plugin handlers are typically global
            )
            handlers[handler_info.name] = handler_info
        
        return handlers
    
    def get_plugin(self, plugin_name: str) -> Optional[PluginInfo]:
        """Get a specific plugin by name."""
        plugins = self.discover_components()
        return plugins.get(plugin_name)
    
    def get_handler(self, handler_name: str) -> Optional[HandlerInfo]:
        """Get a specific handler by name."""
        handlers = self.discover_handlers()
        return handlers.get(handler_name)
    

    def find_components_by_type(self, component_type: Any) -> List[PluginInfo]:
        """Find plugins by type."""
        components = self.discover_components()
        matches = []
        
        for component in components.values():
            # Check if matches PluginType enum value
            if isinstance(component_type, PluginType) and component.plugin_type == component_type:
                matches.append(component)
            # Check if matches string representation
            elif isinstance(component_type, str) and component.plugin_type.value == component_type:
                matches.append(component)
        
        return matches
    
    def _get_default_dirs(self) -> List[str]:
        """Get default search directories for hooks."""
        current_dir = Path(__file__).parent
        return [str(current_dir)]
    
    def _extract_info(self, component: PluginInfo) -> Dict[str, Any]:
        """Extract standardized info from plugin component."""
        return {
            'name': component.name,
            'type': component.plugin_type.value,
            'description': component.description,
            'module_path': component.module_path,
            'version': component.version,
            'dependencies': component.dependencies,
            'capabilities': component.capabilities,
            'installed': component.installed,
            'class_name': component.plugin_class.__name__ if component.plugin_class else None
        }
    
    def _validate_component_implementation(self, component: PluginInfo) -> tuple[bool, List[str]]:
        """Registry-specific validation logic for hooks plugins."""
        errors = []
        
        try:
            # Validate name
            if not component.name or not isinstance(component.name, str):
                errors.append("Component name must be a non-empty string")
            
            # Validate plugin class
            if not component.plugin_class or not inspect.isclass(component.plugin_class):
                errors.append("Component plugin_class must be a valid class")
            elif not issubclass(component.plugin_class, HooksPlugin):
                errors.append("Component plugin_class must inherit from HooksPlugin")
            
            # Validate module path
            if not component.module_path or not isinstance(component.module_path, str):
                errors.append("Component module_path must be a non-empty string")
            
            # Validate plugin type
            if not isinstance(component.plugin_type, PluginType):
                errors.append("Component plugin_type must be a valid PluginType")
            
            # Validate dependencies list
            if component.dependencies and not isinstance(component.dependencies, list):
                errors.append("Component dependencies must be a list")
            
            # Validate capabilities list
            if component.capabilities and not isinstance(component.capabilities, list):
                errors.append("Component capabilities must be a list")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            return False, [f"Validation error: {e}"]

    def find_plugins_by_type(self, plugin_type: PluginType) -> List[PluginInfo]:
        """Find plugins by type."""
        plugins = self.discover_components()
        matches = []
        
        for plugin in plugins.values():
            if plugin.plugin_type == plugin_type:
                matches.append(plugin)
        
        return matches
    
    def find_handlers_by_event_type(self, event_type: str) -> List[HandlerInfo]:
        """Find handlers that can handle a specific event type."""
        handlers = self.discover_handlers()
        matches = []
        
        for handler in handlers.values():
            if event_type in handler.event_types or handler.is_global:
                matches.append(handler)
        
        return matches
    
    def list_available_plugins(self) -> List[str]:
        """Get list of available plugin names."""
        plugins = self.discover_components()
        return list(plugins.keys())
    
    def list_installed_plugins(self) -> List[str]:
        """Get list of currently installed plugin names."""
        return self.plugin_manager.list_plugins()
    
    def list_available_handlers(self) -> List[str]:
        """Get list of available handler names."""
        handlers = self.discover_handlers()
        return list(handlers.keys())
    
    def install_plugin(self, plugin_name: str) -> bool:
        """
        Install a plugin by name.
        
        Args:
            plugin_name: Name of the plugin to install
            
        Returns:
            True if installation successful, False otherwise
        """
        plugin_info = self.get_plugin(plugin_name)
        if not plugin_info:
            logger.error(f"Plugin '{plugin_name}' not found")
            return False
        
        try:
            plugin_instance = plugin_info.plugin_class()
            self.plugin_manager.install_plugin(plugin_name, plugin_instance)
            plugin_info.installed = True
            return True
        except Exception as e:
            logger.error(f"Failed to install plugin '{plugin_name}': {e}")
            return False
    
    def uninstall_plugin(self, plugin_name: str) -> bool:
        """
        Uninstall a plugin by name.
        
        Args:
            plugin_name: Name of the plugin to uninstall
            
        Returns:
            True if uninstallation successful, False otherwise
        """
        try:
            self.plugin_manager.uninstall_plugin(plugin_name)
            plugin_info = self.get_plugin(plugin_name)
            if plugin_info:
                plugin_info.installed = False
            return True
        except Exception as e:
            logger.error(f"Failed to uninstall plugin '{plugin_name}': {e}")
            return False
    
    def refresh_cache(self):
        """Refresh the registry cache by clearing it."""
        super().refresh_cache()
        self.handler_cache.clear()
        self._log_info("Hooks registry cache refreshed")
    
    def _classify_plugin_type(self, name: str, plugin_class: Type[HooksPlugin]) -> PluginType:
        """Classify plugin type based on name and class."""
        name_lower = name.lower()
        
        if 'statistic' in name_lower or 'stats' in name_lower:
            return PluginType.STATISTICS
        elif 'ml' in name_lower or 'machine' in name_lower or 'learning' in name_lower:
            return PluginType.MACHINE_LEARNING
        elif 'monitor' in name_lower or 'tracking' in name_lower:
            return PluginType.MONITORING
        elif 'analysis' in name_lower or 'analyzer' in name_lower:
            return PluginType.ANALYSIS
        elif 'persist' in name_lower or 'storage' in name_lower or 'save' in name_lower:
            return PluginType.PERSISTENCE
        elif 'handler' in name_lower or 'event' in name_lower:
            return PluginType.EVENT_HANDLER
        else:
            return PluginType.UTILITY
    
    def _extract_description(self, cls: type) -> str:
        """Extract description from class docstring."""
        docstring = cls.__doc__ or ""
        
        # Get first line of docstring as description
        first_line = docstring.split('\n')[0].strip()
        return first_line if first_line else f"Hooks component: {cls.__name__}"
    
    def _extract_capabilities(self, plugin_class: Type[HooksPlugin]) -> List[str]:
        """Extract capabilities from plugin class."""
        capabilities = []
        
        # Check for common capability methods
        for method_name in dir(plugin_class):
            if method_name.startswith('can_') or method_name.startswith('supports_'):
                capabilities.append(method_name.replace('can_', '').replace('supports_', ''))
        
        return capabilities
    
    def _extract_event_types(self, handler_class: Type[EventHandler]) -> List[str]:
        """Extract supported event types from handler class."""
        event_types = []
        
        # Check for event_types attribute
        if hasattr(handler_class, 'event_types'):
            event_types.extend(handler_class.event_types)
        
        # Check for supported_events attribute
        if hasattr(handler_class, 'supported_events'):
            event_types.extend(handler_class.supported_events)
        
        return event_types


# Global registry instance
_global_registry = None


def get_hooks_registry() -> HooksRegistry:
    """Get the global hooks registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = HooksRegistry()
    return _global_registry


# Convenience functions for common operations
def discover_all_plugins(rescan: bool = False) -> Dict[str, PluginInfo]:
    """
    Discover all available hooks plugins.
    
    Args:
        rescan: Force rescan even if cache exists
        
    Returns:
        Dictionary mapping plugin names to PluginInfo objects
    """
    registry = get_hooks_registry()
    return registry.discover_components(rescan)


def discover_all_handlers(rescan: bool = False) -> Dict[str, HandlerInfo]:
    """
    Discover all available event handlers.
    
    Args:
        rescan: Force rescan even if cache exists
        
    Returns:
        Dictionary mapping handler names to HandlerInfo objects
    """
    registry = get_hooks_registry()
    return registry.discover_handlers(rescan)


def get_plugin_by_name(plugin_name: str) -> Optional[PluginInfo]:
    """
    Get a plugin by name.
    
    Args:
        plugin_name: Name of the plugin
        
    Returns:
        PluginInfo object or None if not found
    """
    registry = get_hooks_registry()
    return registry.get_plugin(plugin_name)


def install_hook_plugin(plugin_name: str) -> bool:
    """
    Install a hooks plugin by name.
    
    Args:
        plugin_name: Name of the plugin to install
        
    Returns:
        True if installation successful, False otherwise
    """
    registry = get_hooks_registry()
    return registry.install_plugin(plugin_name)


def list_available_hook_plugins() -> List[str]:
    """
    Get list of all available hooks plugin names.
    
    Returns:
        List of plugin names
    """
    registry = get_hooks_registry()
    return registry.list_available_plugins()


def refresh_hooks_registry():
    """Refresh the hooks registry cache."""
    registry = get_hooks_registry()
    registry.refresh_cache()