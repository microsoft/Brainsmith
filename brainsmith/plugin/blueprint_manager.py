"""
Blueprint-Driven Plugin Manager

Efficiently loads only the plugins required by a specific blueprint,
rather than loading all 255+ plugins upfront.
"""

import logging
import yaml
from typing import Dict, List, Set, Optional, Union, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

# Note: Stevedore adapters are used for discovery but not directly imported here
# The stevedore_adapters module handles all framework-specific discovery

logger = logging.getLogger(__name__)


@dataclass
class PluginInfo:
    """
    Clean plugin metadata without ugly colon prefixes.
    Tracks framework attribution naturally.
    """
    name: str
    plugin_class: type
    framework: str  # "qonnx", "finn", "brainsmith"
    plugin_type: str  # "transform", "kernel", "backend", "kernel_inference"
    metadata: Dict[str, Any]
    
    @property
    def qualified_name(self) -> str:
        """Get the qualified name with framework prefix when needed."""
        return f"{self.framework}:{self.name}"
    
    def __str__(self) -> str:
        return f"{self.name} ({self.framework})"


class BlueprintPluginManager:
    """
    Load plugins selectively based on blueprint requirements.
    
    This replaces the problematic approach of loading all 255+ plugins
    upfront with efficient, blueprint-driven lazy loading.
    """
    
    def __init__(self):
        self._loaded_plugins: Dict[str, PluginInfo] = {}
        self._available_plugins: Optional[Dict[str, List[PluginInfo]]] = None
        self._framework_managers = {}
        
    def load_for_blueprint(self, blueprint_path: str) -> Dict[str, List[PluginInfo]]:
        """
        Load only the plugins required by the given blueprint.
        
        Args:
            blueprint_path: Path to blueprint YAML file
            
        Returns:
            Dict mapping plugin types to lists of loaded plugins
            
        Example:
            # Blueprint specifies: kernels: ["matmul", ("softmax", ["hls"])]
            # Only loads: matmul kernel + all its backends, softmax kernel + HLS backend
        """
        logger.info(f"Loading plugins for blueprint: {blueprint_path}")
        
        # Parse blueprint requirements
        requirements = self._parse_blueprint_requirements(blueprint_path)
        logger.debug(f"Blueprint requires: {requirements}")
        
        # Load only required plugins
        loaded = {}
        for plugin_type, names in requirements.items():
            loaded[plugin_type] = []
            for name in names:
                plugins = self._load_plugins_by_name(plugin_type, name)
                loaded[plugin_type].extend(plugins)
        
        logger.info(f"Loaded {sum(len(p) for p in loaded.values())} plugins total")
        return loaded
    
    def _parse_blueprint_requirements(self, blueprint_path: str) -> Dict[str, Set[str]]:
        """
        Extract plugin requirements from blueprint YAML.
        
        Returns:
            Dict mapping plugin types to sets of required plugin names
        """
        with open(blueprint_path, 'r') as f:
            blueprint = yaml.safe_load(f)
        
        requirements = {
            "kernels": set(),
            "transforms": set(), 
            "backends": set(),
            "kernel_inference": set()
        }
        
        # Parse kernel requirements
        hw_compiler = blueprint.get("hw_compiler", {})
        
        # Handle kernel specifications
        for kernel_spec in hw_compiler.get("kernels", []):
            if isinstance(kernel_spec, str):
                # Simple kernel name - need kernel + all its backends
                kernel_name = kernel_spec.lstrip("~")
                requirements["kernels"].add(kernel_name)
                # Also load backends for this kernel (will be resolved later)
                
            elif isinstance(kernel_spec, dict):
                # YAML-compatible dict format: {"kernel": "softmax", "backends": ["hls"]}
                kernel_name = kernel_spec.get("kernel", "").lstrip("~")
                if kernel_name:
                    requirements["kernels"].add(kernel_name)
                
            elif isinstance(kernel_spec, tuple) and len(kernel_spec) == 2:
                # Python tuple format: ("softmax", ["hls", "rtl"])
                # Note: This won't work in YAML but included for completeness
                kernel_name, backend_types = kernel_spec
                kernel_name = kernel_name.lstrip("~")
                requirements["kernels"].add(kernel_name)
                # Backend names will be resolved as kernel_name + backend_type
                
            elif isinstance(kernel_spec, list):
                # Mutually exclusive group - need all options available
                for item in kernel_spec:
                    if isinstance(item, str):
                        requirements["kernels"].add(item.lstrip("~"))
                    elif isinstance(item, dict):
                        kernel_name = item.get("kernel", "").lstrip("~")
                        if kernel_name:
                            requirements["kernels"].add(kernel_name)
                    elif isinstance(item, tuple):
                        kernel_name, _ = item
                        requirements["kernels"].add(kernel_name.lstrip("~"))
        
        # Parse transform requirements
        transforms = hw_compiler.get("transforms", [])
        if isinstance(transforms, list):
            # Flat list format
            for transform in transforms:
                requirements["transforms"].add(transform.lstrip("~"))
        elif isinstance(transforms, dict):
            # Phased format
            for phase, transform_list in transforms.items():
                for transform in transform_list:
                    requirements["transforms"].add(transform.lstrip("~"))
        
        return requirements
    
    def _load_plugins_by_name(self, plugin_type: str, name: str) -> List[PluginInfo]:
        """
        Load specific plugins by name using Stevedore.
        
        Handles framework conflicts intelligently - if name is unique,
        loads it directly. If conflicts exist, requires disambiguation.
        """
        if name in self._loaded_plugins:
            return [self._loaded_plugins[name]]
        
        # Find all available plugins with this name
        available = self._find_available_plugins(plugin_type, name)
        
        if len(available) == 0:
            logger.warning(f"No {plugin_type} found with name '{name}'")
            return []
        
        elif len(available) == 1:
            # Unique name - load it
            plugin = available[0]
            loaded_plugin = self._load_single_plugin(plugin)
            self._loaded_plugins[name] = loaded_plugin
            return [loaded_plugin]
        
        else:
            # Multiple plugins with same name - need all for blueprint evaluation
            logger.info(f"Multiple {plugin_type}s found for '{name}': {[p.framework for p in available]}")
            loaded = []
            for plugin in available:
                loaded_plugin = self._load_single_plugin(plugin)
                self._loaded_plugins[plugin.qualified_name] = loaded_plugin
                loaded.append(loaded_plugin)
            return loaded
    
    def _find_available_plugins(self, plugin_type: str, name: str) -> List[PluginInfo]:
        """
        Find all available plugins of given type and name across frameworks.
        """
        if self._available_plugins is None:
            self._discover_available_plugins()
        
        # Look for exact name matches
        matches = []
        for available_name, plugin_list in self._available_plugins.items():
            for plugin in plugin_list:
                if plugin.plugin_type == plugin_type and plugin.name == name:
                    matches.append(plugin)
        
        return matches
    
    def _discover_available_plugins(self):
        """
        Discover all available plugins using unified discovery adapters.
        
        This replaces the current approach of loading everything upfront.
        Instead, we just discover what's available for later lazy loading.
        """
        logger.info("Discovering available plugins...")
        self._available_plugins = {}
        
        # Use unified discovery system that handles QONNX/FINN/BrainSmith
        from brainsmith.plugin.stevedore_adapters import get_unified_discovery
        
        discovery = get_unified_discovery()
        all_plugins = discovery.discover_all_plugins()
        
        # Convert to our format
        for adapted_plugin in all_plugins:
            plugin_info = PluginInfo(
                name=adapted_plugin.name,
                plugin_class=adapted_plugin.plugin_class,
                framework=adapted_plugin.framework,
                plugin_type=adapted_plugin.plugin_type,
                metadata=adapted_plugin.metadata
            )
            
            if adapted_plugin.name not in self._available_plugins:
                self._available_plugins[adapted_plugin.name] = []
            self._available_plugins[adapted_plugin.name].append(plugin_info)
        
        total_discovered = sum(len(plugins) for plugins in self._available_plugins.values())
        logger.info(f"Discovered {total_discovered} available plugins")
    
    
    def _load_single_plugin(self, plugin_info: PluginInfo) -> PluginInfo:
        """
        Actually load a single plugin class.
        
        This is where the lazy loading happens - we only instantiate
        the class when it's actually needed by a blueprint.
        """
        logger.debug(f"Loading plugin: {plugin_info}")
        
        # The plugin class is already discovered, just return the info
        # The actual instantiation happens when the plugin is used
        return plugin_info
    
    def get_loaded_plugin(self, name: str, framework: Optional[str] = None) -> Optional[PluginInfo]:
        """
        Get a loaded plugin by name, optionally specifying framework.
        
        Args:
            name: Plugin name
            framework: Optional framework filter
            
        Returns:
            PluginInfo if found, None otherwise
        """
        # Try exact name first
        if name in self._loaded_plugins:
            plugin = self._loaded_plugins[name]
            if framework is None or plugin.framework == framework:
                return plugin
        
        # Try qualified name
        if framework and f"{framework}:{name}" in self._loaded_plugins:
            return self._loaded_plugins[f"{framework}:{name}"]
        
        # Search through all loaded plugins
        for plugin in self._loaded_plugins.values():
            if plugin.name == name and (framework is None or plugin.framework == framework):
                return plugin
        
        return None
    
    def list_loaded_plugins(self, plugin_type: Optional[str] = None) -> List[PluginInfo]:
        """
        List all currently loaded plugins, optionally filtered by type.
        """
        plugins = list(self._loaded_plugins.values())
        if plugin_type:
            plugins = [p for p in plugins if p.plugin_type == plugin_type]
        return plugins
    
    def get_plugin_conflicts(self) -> Dict[str, List[PluginInfo]]:
        """
        Analyze naming conflicts among loaded plugins.
        
        Returns:
            Dict mapping conflicted names to lists of conflicting plugins
        """
        conflicts = {}
        by_name = {}
        
        # Group plugins by name
        for plugin in self._loaded_plugins.values():
            if plugin.name not in by_name:
                by_name[plugin.name] = []
            by_name[plugin.name].append(plugin)
        
        # Find conflicts (multiple plugins with same name)
        for name, plugins in by_name.items():
            if len(plugins) > 1:
                conflicts[name] = plugins
        
        return conflicts


# Global instance for easy access
_blueprint_manager = None


def get_blueprint_manager() -> BlueprintPluginManager:
    """Get the global blueprint plugin manager instance."""
    global _blueprint_manager
    if _blueprint_manager is None:
        _blueprint_manager = BlueprintPluginManager()
    return _blueprint_manager


def load_plugins_for_blueprint(blueprint_path: str) -> Dict[str, List[PluginInfo]]:
    """
    Convenience function to load plugins for a blueprint.
    
    Args:
        blueprint_path: Path to blueprint YAML file
        
    Returns:
        Dict mapping plugin types to loaded plugins
    """
    manager = get_blueprint_manager()
    return manager.load_for_blueprint(blueprint_path)