"""
Plugin System Data Models

Core data structures for the Pure Stevedore Plugin System.
Extracted from the monolithic manager.py for better separation of concerns.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Type


class DiscoveryStrategy(Enum):
    """Plugin discovery strategies."""
    STEVEDORE_ONLY = "stevedore_only"       # Use only entry points
    AUTO_DISCOVERY = "auto_discovery"        # Scan codebase + entry points
    HYBRID = "hybrid"                        # Best of both worlds (default)


class PluginType(Enum):
    """Supported plugin types."""
    TRANSFORM = "transform"
    KERNEL = "kernel"
    BACKEND = "backend"
    KERNEL_INFERENCE = "kernel_inference"
    STEP = "step"


class FrameworkType(Enum):
    """Supported frameworks."""
    BRAINSMITH = "brainsmith"
    QONNX = "qonnx"
    FINN = "finn"
    EXTERNAL = "external"


class DiscoveryMethod(Enum):
    """How a plugin was discovered."""
    STEVEDORE = "stevedore"              # Via entry points
    AUTO = "auto"                        # Via filesystem scanning
    FRAMEWORK_NATIVE = "framework_native" # Via framework's own registry
    DECORATOR = "decorator"              # Via decorator registration


@dataclass(frozen=True)
class PluginInfo:
    """
    Enhanced plugin information with all necessary metadata.
    
    This is the core data structure that represents a discovered plugin
    with all its metadata, discovery information, and loading capabilities.
    """
    name: str
    plugin_class: Type = field(hash=False)
    framework: str  # Use string for flexibility, validate against FrameworkType
    plugin_type: str  # Use string for flexibility, validate against PluginType
    metadata: Dict[str, Any] = field(default_factory=dict, hash=False)
    discovery_method: str = "auto"  # Use string, validate against DiscoveryMethod
    stevedore_extension: Optional[Any] = field(default=None, hash=False)  # Stevedore extension object if applicable
    
    def __post_init__(self):
        """Validate enum fields."""
        # Validate framework
        try:
            FrameworkType(self.framework.lower())
        except ValueError:
            if self.framework != "external":  # Allow external for flexibility
                raise ValueError(f"Invalid framework: {self.framework}")
        
        # Validate plugin_type
        try:
            PluginType(self.plugin_type.lower())
        except ValueError:
            raise ValueError(f"Invalid plugin_type: {self.plugin_type}")
        
        # Validate discovery_method
        try:
            DiscoveryMethod(self.discovery_method.lower())
        except ValueError:
            raise ValueError(f"Invalid discovery_method: {self.discovery_method}")
    
    @property
    def qualified_name(self) -> str:
        """Get the qualified name with framework prefix."""
        return f"{self.framework}:{self.name}"
    
    @property
    def is_unique(self) -> bool:
        """Check if this plugin name is unique across frameworks."""
        return self.metadata.get('is_unique', False)
    
    @property
    def is_loaded(self) -> bool:
        """Check if this plugin has been loaded."""
        return self.metadata.get('is_loaded', False)
    
    @property
    def stage(self) -> Optional[str]:
        """Get the compilation stage for transforms."""
        return self.metadata.get('stage')
    
    @property
    def kernel(self) -> Optional[str]:
        """Get the kernel name for kernel inference transforms or backends."""
        return self.metadata.get('kernel')
    
    @property
    def backend_type(self) -> Optional[str]:
        """Get the backend type (hls/rtl) for backends."""
        return self.metadata.get('backend_type')
    
    @property
    def description(self) -> Optional[str]:
        """Get the plugin description."""
        return self.metadata.get('description')
    
    @property
    def author(self) -> Optional[str]:
        """Get the plugin author."""
        return self.metadata.get('author')
    
    @property
    def version(self) -> Optional[str]:
        """Get the plugin version."""
        return self.metadata.get('version')
    
    def instantiate(self, **kwargs) -> Any:
        """
        Create an instance of the plugin using the best method.
        
        If discovered via Stevedore, use its optimized loading.
        Otherwise, instantiate directly from the class.
        """
        if self.stevedore_extension:
            # Use Stevedore's optimized loading
            return self.stevedore_extension.obj(**kwargs)
        else:
            # Direct instantiation
            return self.plugin_class(**kwargs)
    
    def matches(self, name: str = None, plugin_type: str = None, 
                framework: str = None, **kwargs) -> bool:
        """
        Check if this plugin matches the given criteria.
        
        Useful for filtering and searching plugins.
        """
        if name and self.name != name:
            return False
        if plugin_type and self.plugin_type != plugin_type:
            return False
        if framework and self.framework != framework:
            return False
        
        # Check additional metadata criteria
        for key, value in kwargs.items():
            if self.metadata.get(key) != value:
                return False
        
        return True
    
    def __str__(self) -> str:
        return f"{self.name} ({self.framework}, {self.discovery_method})"
    
    def __repr__(self) -> str:
        return (f"PluginInfo(name={self.name!r}, framework={self.framework!r}, "
                f"type={self.plugin_type!r}, method={self.discovery_method!r})")


@dataclass
class PluginCatalog:
    """
    Complete catalog of discovered plugins organized by multiple dimensions.
    
    This provides efficient access to plugins by name, type, framework,
    and handles conflict resolution for plugins with the same name.
    """
    plugins_by_name: Dict[str, List[PluginInfo]] = field(default_factory=dict)
    plugins_by_type: Dict[str, List[PluginInfo]] = field(default_factory=dict)
    plugins_by_framework: Dict[str, List[PluginInfo]] = field(default_factory=dict)
    conflicts: Dict[str, List[PluginInfo]] = field(default_factory=dict)
    unique_plugins: Dict[str, PluginInfo] = field(default_factory=dict)
    
    def add_plugin(self, plugin: PluginInfo) -> None:
        """Add a plugin to the catalog and update all indices."""
        # Add to name index
        if plugin.name not in self.plugins_by_name:
            self.plugins_by_name[plugin.name] = []
        self.plugins_by_name[plugin.name].append(plugin)
        
        # Add to type index
        if plugin.plugin_type not in self.plugins_by_type:
            self.plugins_by_type[plugin.plugin_type] = []
        self.plugins_by_type[plugin.plugin_type].append(plugin)
        
        # Add to framework index
        if plugin.framework not in self.plugins_by_framework:
            self.plugins_by_framework[plugin.framework] = []
        self.plugins_by_framework[plugin.framework].append(plugin)
        
        # Update conflict/unique status
        self._update_conflict_status(plugin.name)
    
    def _update_conflict_status(self, name: str) -> None:
        """Update conflict and unique status for plugins with given name."""
        plugins = self.plugins_by_name.get(name, [])
        
        if len(plugins) > 1:
            # Mark as conflict
            self.conflicts[name] = plugins
            # Remove from unique if it was there
            self.unique_plugins.pop(name, None)
            # Mark all as non-unique
            for plugin in plugins:
                plugin.metadata['is_unique'] = False
        elif len(plugins) == 1:
            # Mark as unique
            plugin = plugins[0]
            self.unique_plugins[name] = plugin
            plugin.metadata['is_unique'] = True
            # Remove from conflicts if it was there
            self.conflicts.pop(name, None)
    
    def get_plugin(self, name: str, framework: str = None) -> Optional[PluginInfo]:
        """
        Get specific plugin by name and optionally framework.
        
        Raises:
            ValueError: If plugin name is ambiguous and needs framework specification
        """
        if framework:
            # Framework-specific lookup
            for plugin in self.plugins_by_name.get(name, []):
                if plugin.framework == framework:
                    return plugin
        else:
            # Check if unique first
            if name in self.unique_plugins:
                return self.unique_plugins[name]
            # If not unique, need framework specification
            if name in self.conflicts:
                frameworks = [p.framework for p in self.conflicts[name]]
                raise ValueError(
                    f"Plugin '{name}' is ambiguous. Found in frameworks: {frameworks}. "
                    f"Use qualified name like 'qonnx:{name}' or 'finn:{name}'"
                )
        return None
    
    def find_plugins(self, **criteria) -> List[PluginInfo]:
        """Find all plugins matching the given criteria."""
        results = []
        for plugin_list in self.plugins_by_name.values():
            for plugin in plugin_list:
                if plugin.matches(**criteria):
                    results.append(plugin)
        return results
    
    def list_transforms(self, framework: str = None) -> List[PluginInfo]:
        """List all transforms, optionally filtered by framework."""
        transforms = self.plugins_by_type.get("transform", [])
        if framework:
            transforms = [t for t in transforms if t.framework == framework]
        return transforms
    
    def list_kernels(self, framework: str = None) -> List[PluginInfo]:
        """List all kernels, optionally filtered by framework."""
        kernels = self.plugins_by_type.get("kernel", [])
        if framework:
            kernels = [k for k in kernels if k.framework == framework]
        return kernels
    
    def list_backends(self, kernel: str = None) -> List[PluginInfo]:
        """List all backends, optionally filtered by kernel."""
        backends = self.plugins_by_type.get("backend", [])
        if kernel:
            backends = [b for b in backends if b.kernel == kernel]
        return backends
    
    def list_steps(self, category: str = None) -> List[PluginInfo]:
        """List all steps, optionally filtered by category."""
        steps = self.plugins_by_type.get("step", [])
        if category:
            steps = [s for s in steps if s.metadata.get('category') == category]
        return steps
    
    @property
    def total_plugins(self) -> int:
        """Get total number of plugins (counting duplicates)."""
        return sum(len(plist) for plist in self.plugins_by_name.values())
    
    @property
    def unique_count(self) -> int:
        """Get number of plugins with unique names."""
        return len(self.unique_plugins)
    
    @property
    def conflict_count(self) -> int:
        """Get number of plugin names with conflicts."""
        return len(self.conflicts)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the catalog contents."""
        return {
            'total_plugins': self.total_plugins,
            'unique_plugins': self.unique_count,
            'conflicted_plugins': self.conflict_count,
            'by_framework': {
                framework: len(plugins) 
                for framework, plugins in self.plugins_by_framework.items()
            },
            'by_type': {
                plugin_type: len(plugins)
                for plugin_type, plugins in self.plugins_by_type.items()
            }
        }
    
    def __repr__(self) -> str:
        summary = self.get_summary()
        return (f"PluginCatalog(total={summary['total_plugins']}, "
                f"unique={summary['unique_plugins']}, "
                f"conflicts={summary['conflicted_plugins']})")