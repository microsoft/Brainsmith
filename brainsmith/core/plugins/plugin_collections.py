"""
Natural Access Collections - Perfect Code Implementation

Direct registry delegation for zero-overhead plugin access.
Supports both attribute and dictionary-style access with framework qualification.
Single unified collection class for all plugin types.
"""

import logging
from typing import Dict, Any, Optional, Type, List

logger = logging.getLogger(__name__)


class PluginCollection:
    """
    Universal collection class for all plugin types.
    
    Provides:
    - Direct attribute access: plugins.MyPlugin
    - Dictionary-style access: plugins["MyPlugin"]
    - Framework-qualified access: plugins.framework.MyPlugin or plugins["framework:MyPlugin"]
    - Universal find() method for any search criteria
    """
    
    def __init__(self, plugin_type: str, registry):
        self.plugin_type = plugin_type
        self.registry = registry
    
    def __getattr__(self, name: str):
        """Handle attribute access - framework, category, or direct plugin."""
        # Check if it's a framework accessor
        frameworks = self._get_frameworks()
        if name in frameworks:
            return FrameworkAccessor(self.plugin_type, name, self.registry)
        
        # For steps, check if it's a category accessor
        if self.plugin_type == "step":
            categories = self._get_categories()
            if name in categories:
                return CategoryAccessor(name, self.registry)
        
        # Otherwise, direct plugin lookup
        plugin_cls = self._get_plugin(name)
        if plugin_cls:
            return plugin_cls
        
        # Helpful error message
        available = self._list_available()[:10]
        raise AttributeError(
            f"{self.plugin_type.title()} '{name}' not found. "
            f"Available: {available}{' ...' if len(self._list_available()) > 10 else ''}"
        )
    
    def __getitem__(self, key: str):
        """Dictionary-style access with optional framework qualification."""
        if ":" in key:
            framework, name = key.split(":", 1)
            plugin_cls = self._get_plugin(name, framework)
        else:
            plugin_cls = self._get_plugin(key)
        
        if plugin_cls:
            return plugin_cls
        
        raise KeyError(f"{self.plugin_type.title()} '{key}' not found")
    
    def __dir__(self):
        """Support tab completion."""
        names = set(self._list_available())
        names.update(self._get_frameworks())
        if self.plugin_type == "step":
            names.update(self._get_categories())
        return sorted(names)
    
    def find(self, **criteria) -> List[Type]:
        """
        Universal find method for all plugin types.
        
        Args:
            **criteria: Search criteria (stage, category, kernel, language, 
                       framework, optimization, etc.)
        
        Returns:
            List of plugin classes matching the criteria
        
        Examples:
            # Find transforms by stage
            transforms.find(stage="cleanup")
            
            # Find backends by kernel and language
            backends.find(kernel="MatMul", language="hls")
            
            # Find steps by category
            steps.find(category="testing")
            
            # Find plugins by framework
            kernels.find(framework="finn")
            
            # Multiple criteria
            backends.find(kernel="LayerNorm", optimization="area", language="hls")
        """
        if not criteria:
            # No criteria = return all plugins of this type
            return [self._get_plugin(name) for name in self._list_available()]
        
        # Get all plugins of this type with their metadata
        matching_plugins = []
        
        for plugin_name in self._list_available():
            plugin_cls = self._get_plugin(plugin_name)
            metadata = self.registry.get_plugin_metadata(plugin_name)
            
            # Check if plugin matches ALL criteria
            matches = True
            for key, value in criteria.items():
                # Handle special cases for different plugin types
                if key == "stage" and self.plugin_type == "transform":
                    if metadata.get("stage") != value:
                        matches = False
                        break
                elif key == "category" and self.plugin_type == "step":
                    if metadata.get("category") != value:
                        matches = False
                        break
                elif key == "kernel" and self.plugin_type == "backend":
                    if metadata.get("kernel") != value:
                        matches = False
                        break
                elif key in metadata:
                    if metadata[key] != value:
                        matches = False
                        break
                else:
                    # Criteria not found in metadata
                    matches = False
                    break
            
            if matches:
                matching_plugins.append(plugin_cls)
        
        return matching_plugins
    
    def list(self, **criteria) -> List[str]:
        """
        List plugin names matching criteria.
        
        Args:
            **criteria: Search criteria
            
        Returns:
            List of plugin names
        """
        if not criteria:
            return self._list_available()
        
        matching_classes = self.find(**criteria)
        # Get names from classes by looking them up in registry
        matching_names = []
        plugin_dict = getattr(self.registry, f"{self.plugin_type}s")
        for cls in matching_classes:
            for name, registered_cls in plugin_dict.items():
                if registered_cls is cls:
                    matching_names.append(name)
                    break
        return matching_names
    
    def get(self, **criteria) -> Optional[Type]:
        """
        Get first plugin matching criteria.
        
        Args:
            **criteria: Search criteria
            
        Returns:
            First matching plugin class or None
        """
        matches = self.find(**criteria)
        return matches[0] if matches else None
    
    def _get_plugin(self, name: str, framework: Optional[str] = None) -> Optional[Type]:
        """Get plugin by name with optional framework filter."""
        if self.plugin_type == "transform":
            return self.registry.get_transform(name, framework=framework)
        elif self.plugin_type == "kernel":
            return self.registry.get_kernel(name, framework=framework)
        elif self.plugin_type == "backend":
            return self.registry.get_backend(name, framework=framework)
        elif self.plugin_type == "step":
            return self.registry.get_step(name, framework=framework)
        return None
    
    def _list_available(self) -> List[str]:
        """List all available plugin names."""
        if self.plugin_type == "transform":
            return list(self.registry.transforms.keys())
        elif self.plugin_type == "kernel":
            return list(self.registry.kernels.keys())
        elif self.plugin_type == "backend":
            return list(self.registry.backends.keys())
        elif self.plugin_type == "step":
            return list(self.registry.steps.keys())
        return []
    
    def _get_frameworks(self) -> List[str]:
        """Get available frameworks for this plugin type."""
        if self.plugin_type == "transform":
            return list(self.registry.framework_transforms.keys())
        elif self.plugin_type == "kernel":
            return list(self.registry.framework_kernels.keys())
        elif self.plugin_type == "backend":
            return list(self.registry.framework_backends.keys())
        elif self.plugin_type == "step":
            return list(self.registry.framework_steps.keys())
        return []
    
    def _get_categories(self) -> List[str]:
        """Get available categories (for steps)."""
        if self.plugin_type == "step":
            return list(self.registry.steps_by_category.keys())
        return []


class FrameworkAccessor:
    """Provides framework-scoped access to plugins."""
    
    def __init__(self, plugin_type: str, framework: str, registry):
        self.plugin_type = plugin_type
        self.framework = framework
        self.registry = registry
    
    def __getattr__(self, name: str):
        """Get plugin from specific framework."""
        # Get the framework-specific plugins
        if self.plugin_type == "transform":
            framework_plugins = self.registry.get_framework_transforms(self.framework)
        elif self.plugin_type == "kernel":
            framework_plugins = self.registry.get_framework_kernels(self.framework)
        elif self.plugin_type == "backend":
            framework_plugins = self.registry.get_framework_backends(self.framework)
        elif self.plugin_type == "step":
            framework_plugins = self.registry.get_framework_steps(self.framework)
        else:
            framework_plugins = {}
        
        if name in framework_plugins:
            return framework_plugins[name]
        
        available = list(framework_plugins.keys())[:10]
        raise AttributeError(
            f"{self.plugin_type.title()} '{name}' not found in {self.framework}. "
            f"Available: {available}{' ...' if len(framework_plugins) > 10 else ''}"
        )
    
    def __dir__(self):
        """Support tab completion."""
        if self.plugin_type == "transform":
            framework_plugins = self.registry.get_framework_transforms(self.framework)
        elif self.plugin_type == "kernel":
            framework_plugins = self.registry.get_framework_kernels(self.framework)
        elif self.plugin_type == "backend":
            framework_plugins = self.registry.get_framework_backends(self.framework)
        elif self.plugin_type == "step":
            framework_plugins = self.registry.get_framework_steps(self.framework)
        else:
            framework_plugins = {}
        
        return sorted(framework_plugins.keys())


class CategoryAccessor:
    """Provides category-scoped access to steps."""
    
    def __init__(self, category: str, registry):
        self.category = category
        self.registry = registry
    
    def __getattr__(self, name: str):
        """Get step from specific category."""
        category_steps = self.registry.steps_by_category.get(self.category, {})
        
        if name in category_steps:
            return category_steps[name]
        
        available = list(category_steps.keys())[:10]
        raise AttributeError(
            f"Step '{name}' not found in category '{self.category}'. "
            f"Available: {available}{' ...' if len(category_steps) > 10 else ''}"
        )
    
    def __dir__(self):
        """Support tab completion."""
        return sorted(self.registry.steps_by_category.get(self.category, {}).keys())


def create_collections(registry=None):
    """
    Create collection instances with direct registry delegation.
    
    Perfect Code approach: Single collection class for all plugin types,
    no wrapper indirection needed.
    """
    if registry is None:
        from .registry import get_registry
        registry = get_registry()
    
    return {
        'transforms': PluginCollection("transform", registry),
        'kernels': PluginCollection("kernel", registry),
        'backends': PluginCollection("backend", registry),
        'steps': PluginCollection("step", registry)
    }