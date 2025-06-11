"""
Plugin System for Future Extensions

Foundation for adding back sophisticated capabilities without
affecting the simple core interface.
"""

from typing import List, Dict, Any, Optional
from ..types import EventHandler, HooksPlugin
import logging

logger = logging.getLogger(__name__)


class PluginManager:
    """Manage hooks plugins (future extension capabilities)."""
    
    def __init__(self):
        self.plugins: Dict[str, HooksPlugin] = {}
        self._handlers_cache: List[EventHandler] = []
    
    def install_plugin(self, name: str, plugin: HooksPlugin) -> None:
        """Install plugin for future capabilities."""
        try:
            plugin.install()
            self.plugins[name] = plugin
            self._refresh_handlers_cache()
            logger.info(f"Installed plugin: {name}")
        except Exception as e:
            logger.error(f"Failed to install plugin {name}: {e}")
            raise
    
    def uninstall_plugin(self, name: str) -> None:
        """Uninstall plugin."""
        if name in self.plugins:
            try:
                self.plugins[name].uninstall()
                del self.plugins[name]
                self._refresh_handlers_cache()
                logger.info(f"Uninstalled plugin: {name}")
            except Exception as e:
                logger.error(f"Failed to uninstall plugin {name}: {e}")
    
    def get_plugin(self, name: str) -> Optional[HooksPlugin]:
        """Get plugin by name."""
        return self.plugins.get(name)
    
    def list_plugins(self) -> List[str]:
        """List installed plugin names."""
        return list(self.plugins.keys())
    
    def get_all_handlers(self) -> List[EventHandler]:
        """Get handlers from all plugins."""
        return self._handlers_cache.copy()
    
    def _refresh_handlers_cache(self) -> None:
        """Refresh the handlers cache."""
        self._handlers_cache.clear()
        for plugin in self.plugins.values():
            try:
                handlers = plugin.get_handlers()
                self._handlers_cache.extend(handlers)
            except Exception as e:
                logger.error(f"Error getting handlers from plugin {plugin.get_name()}: {e}")


# Global plugin manager
_plugin_manager = PluginManager()


def install_plugin(name: str, plugin: HooksPlugin) -> None:
    """Install hooks plugin (extension point)."""
    _plugin_manager.install_plugin(name, plugin)


def uninstall_plugin(name: str) -> None:
    """Uninstall hooks plugin."""
    _plugin_manager.uninstall_plugin(name)


def get_plugin(name: str) -> Optional[HooksPlugin]:
    """Get installed plugin by name."""
    return _plugin_manager.get_plugin(name)


def list_plugins() -> List[str]:
    """List installed plugin names."""
    return _plugin_manager.list_plugins()


def get_plugin_handlers() -> List[EventHandler]:
    """Get all handlers from installed plugins."""
    return _plugin_manager.get_all_handlers()


__all__ = [
    'install_plugin', 
    'uninstall_plugin', 
    'get_plugin',
    'list_plugins',
    'get_plugin_handlers',
    'PluginManager'
]