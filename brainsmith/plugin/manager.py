"""
Pure Stevedore Plugin Manager

Uses Stevedore to its full potential for optimal plugin discovery and management.
Combines entry points (for external plugins) with intelligent auto-discovery 
(for development) to create the best possible system.
"""

import logging
import inspect
import importlib
import pkgutil
from typing import Dict, List, Set, Optional, Union, Any, Type
from dataclasses import dataclass
from threading import Lock, RLock
from enum import Enum
from pathlib import Path

try:
    from stevedore import extension, named, driver
except ImportError:
    raise ImportError(
        "Stevedore is required for the plugin system. "
        "Install with: pip install stevedore"
    )

logger = logging.getLogger(__name__)


class DiscoveryStrategy(Enum):
    """Plugin discovery strategies."""
    STEVEDORE_ONLY = "stevedore_only"       # Use only entry points
    AUTO_DISCOVERY = "auto_discovery"       # Scan codebase + entry points
    HYBRID = "hybrid"                       # Best of both worlds (default)


@dataclass
class PluginInfo:
    """
    Enhanced plugin information with Stevedore integration.
    """
    name: str
    plugin_class: type
    framework: str  # "qonnx", "finn", "brainsmith"
    plugin_type: str  # "transform", "kernel", "backend", "kernel_inference"
    metadata: Dict[str, Any]
    discovery_method: str  # "stevedore", "auto", "framework_native"
    stevedore_extension: Optional[Any] = None  # Stevedore extension object if applicable
    
    @property
    def qualified_name(self) -> str:
        """Get the qualified name with framework prefix."""
        return f"{self.framework}:{self.name}"
    
    @property
    def is_unique(self) -> bool:
        """Check if this plugin name is unique across frameworks."""
        return self.metadata.get('is_unique', False)
    
    def instantiate(self, **kwargs) -> Any:
        """Create an instance of the plugin using the best method."""
        if self.stevedore_extension:
            # Use Stevedore's optimized loading
            return self.stevedore_extension.obj(**kwargs)
        else:
            # Fallback to direct instantiation
            return self.plugin_class(**kwargs)
    
    def __str__(self) -> str:
        return f"{self.name} ({self.framework}, {self.discovery_method})"


@dataclass
class PluginCatalog:
    """
    Complete catalog of discovered plugins organized by type and framework.
    """
    plugins_by_name: Dict[str, List[PluginInfo]]  # name -> [plugin_info_list]
    plugins_by_type: Dict[str, List[PluginInfo]]  # type -> [plugin_info_list]
    plugins_by_framework: Dict[str, List[PluginInfo]]  # framework -> [plugin_info_list]
    conflicts: Dict[str, List[PluginInfo]]  # name -> [conflicting_plugins]
    unique_plugins: Dict[str, PluginInfo]  # name -> unique_plugin
    
    def get_plugin(self, name: str, framework: str = None) -> Optional[PluginInfo]:
        """Get specific plugin by name and optionally framework."""
        if framework:
            qualified_name = f"{framework}:{name}"
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


class PluginManager:
    """
    Pure Stevedore Plugin Manager - Uses Stevedore to its full potential.
    
    Combines entry points (external plugins) with intelligent auto-discovery
    (development convenience) to create the optimal plugin system.
    """
    
    # Entry point namespaces for different plugin types
    ENTRY_POINT_NAMESPACES = {
        'transform': [
            'brainsmith.transforms',
            'brainsmith.external.transforms',
        ],
        'kernel': [
            'brainsmith.kernels', 
            'brainsmith.external.kernels',
        ],
        'backend': [
            'brainsmith.backends',
            'brainsmith.external.backends',
        ]
    }
    
    def __init__(self, strategy: DiscoveryStrategy = DiscoveryStrategy.HYBRID):
        self.strategy = strategy
        self._catalog: Optional[PluginCatalog] = None
        self._stevedore_managers: Dict[str, extension.ExtensionManager] = {}
        self._lock = RLock()
        self._discovery_completed = False
        
        # Initialize Stevedore managers for each namespace
        self._init_stevedore_managers()
    
    def discover_all(self, force_refresh: bool = False) -> PluginCatalog:
        """
        Discover all available plugins using pure Stevedore approach.
        
        Args:
            force_refresh: Force rediscovery even if already completed
            
        Returns:
            Complete plugin catalog
        """
        with self._lock:
            if self._catalog is not None and not force_refresh:
                return self._catalog
            
            logger.info("Discovering plugins with pure Stevedore system...")
            
            all_plugins = []
            
            if self.strategy in [DiscoveryStrategy.STEVEDORE_ONLY, DiscoveryStrategy.HYBRID]:
                # Discover via Stevedore entry points
                stevedore_plugins = self._discover_via_stevedore()
                all_plugins.extend(stevedore_plugins)
                logger.info(f"Found {len(stevedore_plugins)} plugins via Stevedore entry points")
            
            if self.strategy in [DiscoveryStrategy.AUTO_DISCOVERY, DiscoveryStrategy.HYBRID]:
                # Auto-discover BrainSmith native plugins
                auto_plugins = self._discover_brainsmith_native()
                all_plugins.extend(auto_plugins)
                logger.info(f"Found {len(auto_plugins)} plugins via auto-discovery")
                
                # Direct framework integration
                qonnx_plugins = self._discover_qonnx_transforms()
                finn_plugins = self._discover_finn_transforms()
                all_plugins.extend(qonnx_plugins)
                all_plugins.extend(finn_plugins)
                logger.info(f"Found {len(qonnx_plugins)} QONNX + {len(finn_plugins)} FINN plugins")
            
            # Organize plugins into catalog structure
            plugins_by_name = {}
            plugins_by_type = {}
            plugins_by_framework = {}
            conflicts = {}
            unique_plugins = {}
            
            for plugin_info in all_plugins:
                # Organize by name
                if plugin_info.name not in plugins_by_name:
                    plugins_by_name[plugin_info.name] = []
                plugins_by_name[plugin_info.name].append(plugin_info)
                
                # Organize by type
                if plugin_info.plugin_type not in plugins_by_type:
                    plugins_by_type[plugin_info.plugin_type] = []
                plugins_by_type[plugin_info.plugin_type].append(plugin_info)
                
                # Organize by framework
                if plugin_info.framework not in plugins_by_framework:
                    plugins_by_framework[plugin_info.framework] = []
                plugins_by_framework[plugin_info.framework].append(plugin_info)
            
            # Identify conflicts and unique plugins
            for name, plugin_list in plugins_by_name.items():
                if len(plugin_list) > 1:
                    conflicts[name] = plugin_list
                    # Mark all as non-unique
                    for plugin in plugin_list:
                        plugin.metadata['is_unique'] = False
                else:
                    unique_plugins[name] = plugin_list[0]
                    plugin_list[0].metadata['is_unique'] = True
            
            self._catalog = PluginCatalog(
                plugins_by_name=plugins_by_name,
                plugins_by_type=plugins_by_type,
                plugins_by_framework=plugins_by_framework,
                conflicts=conflicts,
                unique_plugins=unique_plugins
            )
            
            self._discovery_completed = True
            
            # Log comprehensive summary
            total_plugins = sum(len(plist) for plist in plugins_by_name.values())
            unique_count = len(unique_plugins)
            conflict_count = len(conflicts)
            
            logger.info(f"Pure Stevedore discovery complete: {total_plugins} total plugins")
            logger.info(f"  - {unique_count} unique plugins (no prefix needed)")
            logger.info(f"  - {conflict_count} conflicted plugins (prefix required)")
            
            by_framework = {}
            for framework, plugin_list in plugins_by_framework.items():
                by_framework[framework] = len(plugin_list)
            logger.info(f"  - By framework: {by_framework}")
            
            return self._catalog
    
    def get_plugin(self, name: str, plugin_type: str = None, framework: str = None) -> Optional[PluginInfo]:
        """
        Get a specific plugin with intelligent conflict resolution.
        
        Args:
            name: Plugin name (can include framework prefix like "qonnx:RemoveIdentityOps")
            plugin_type: Optional plugin type filter
            framework: Optional framework filter
            
        Returns:
            PluginInfo if found, None otherwise
            
        Raises:
            ValueError: If plugin name is ambiguous and needs framework specification
        """
        # Ensure discovery is complete
        catalog = self.discover_all()
        
        # Handle framework prefix in name
        if ":" in name and framework is None:
            framework, name = name.split(":", 1)
        
        # Get plugin from catalog
        plugin_info = catalog.get_plugin(name, framework)
        
        if plugin_info and plugin_type and plugin_info.plugin_type != plugin_type:
            return None  # Type mismatch
        
        return plugin_info
    
    def load_plugin(self, name: str, plugin_type: str = None, framework: str = None) -> Optional[PluginInfo]:
        """
        Load a specific plugin and mark it as loaded.
        
        Args:
            name: Plugin name
            plugin_type: Optional plugin type filter
            framework: Optional framework filter
            
        Returns:
            Loaded PluginInfo if successful, None otherwise
        """
        with self._lock:
            # Initialize loaded plugins cache if needed
            if not hasattr(self, '_loaded_plugins'):
                self._loaded_plugins = {}
            
            # Check if already loaded
            cache_key = f"{framework or 'any'}:{plugin_type or 'any'}:{name}"
            if cache_key in self._loaded_plugins:
                return self._loaded_plugins[cache_key]
            
            # Get plugin info
            plugin_info = self.get_plugin(name, plugin_type, framework)
            if not plugin_info:
                return None
            
            # Mark as loaded
            plugin_info.metadata['is_loaded'] = True
            self._loaded_plugins[cache_key] = plugin_info
            
            logger.debug(f"Loaded plugin: {plugin_info}")
            return plugin_info
    
    def load_plugins(self, requirements: List[str]) -> Dict[str, PluginInfo]:
        """
        Load multiple plugins based on requirements.
        
        Args:
            requirements: List of plugin names (can include framework prefixes)
            
        Returns:
            Dict mapping requirement names to loaded PluginInfo objects
        """
        loaded = {}
        errors = []
        
        for requirement in requirements:
            try:
                plugin_info = self.load_plugin(requirement)
                if plugin_info:
                    loaded[requirement] = plugin_info
                else:
                    errors.append(f"Plugin '{requirement}' not found")
            except Exception as e:
                errors.append(f"Failed to load '{requirement}': {e}")
        
        if errors:
            logger.warning(f"Plugin loading errors: {'; '.join(errors)}")
        
        return loaded
    
    def get_transforms(self) -> 'TransformCollection':
        """Get transform collection for natural access."""
        from .collections import TransformCollection
        return TransformCollection(self)
    
    def get_kernels(self) -> 'KernelCollection':
        """Get kernel collection for natural access."""
        from .collections import KernelCollection
        return KernelCollection(self)
    
    def list_available(self, plugin_type: str = None, framework: str = None) -> List[PluginInfo]:
        """
        List all available plugins, optionally filtered.
        
        Args:
            plugin_type: Optional type filter ("transform", "kernel", etc.)
            framework: Optional framework filter ("qonnx", "finn", "brainsmith")
            
        Returns:
            List of matching PluginInfo objects
        """
        catalog = self.discover_all()
        
        if plugin_type:
            plugins = catalog.plugins_by_type.get(plugin_type, [])
        else:
            plugins = []
            for plugin_list in catalog.plugins_by_name.values():
                plugins.extend(plugin_list)
        
        if framework:
            plugins = [p for p in plugins if p.framework == framework]
        
        return plugins
    
    def analyze_conflicts(self) -> Dict[str, List[PluginInfo]]:
        """Get all naming conflicts."""
        catalog = self.discover_all()
        return catalog.conflicts.copy()
    
    def clear_loaded(self):
        """Clear all loaded plugins (useful for testing)."""
        with self._lock:
            if hasattr(self, '_loaded_plugins'):
                self._loaded_plugins.clear()
            logger.debug("Cleared all loaded plugins")
    
    def reset(self):
        """Reset manager to initial state (useful for testing)."""
        with self._lock:
            self._catalog = None
            if hasattr(self, '_loaded_plugins'):
                self._loaded_plugins.clear()
            self._discovery_completed = False
            logger.debug("Reset plugin manager")
    
    def _init_stevedore_managers(self):
        """Initialize Stevedore extension managers for each namespace."""
        for plugin_type, namespaces in self.ENTRY_POINT_NAMESPACES.items():
            for namespace in namespaces:
                try:
                    manager = extension.ExtensionManager(
                        namespace=namespace,
                        invoke_on_load=False,  # Don't instantiate immediately
                        on_load_failure_callback=self._on_stevedore_load_failure
                    )
                    self._stevedore_managers[namespace] = manager
                    logger.debug(f"Initialized Stevedore manager for namespace: {namespace}")
                except RuntimeError as e:
                    # No entry points found - this is normal during development
                    logger.debug(f"No entry points found for namespace {namespace}: {e}")
    
    def _on_stevedore_load_failure(self, manager, ep, err):
        """Handle Stevedore extension loading failures gracefully."""
        logger.warning(f"Failed to load entry point {ep.name} from {ep.dist}: {err}")
    
    def _discover_via_stevedore(self) -> List[PluginInfo]:
        """Discover plugins via Stevedore entry points."""
        plugins = []
        
        for namespace, manager in self._stevedore_managers.items():
            for extension_obj in manager.extensions:
                try:
                    # Determine plugin type and framework from namespace
                    plugin_type = self._extract_plugin_type_from_namespace(namespace)
                    framework = self._extract_framework_from_namespace(namespace)
                    
                    plugin_info = PluginInfo(
                        name=extension_obj.name,
                        plugin_class=extension_obj.obj,
                        framework=framework,
                        plugin_type=plugin_type,
                        metadata={
                            'entry_point': str(extension_obj.entry_point),
                            'distribution': str(extension_obj.entry_point.dist)
                        },
                        discovery_method="stevedore",
                        stevedore_extension=extension_obj
                    )
                    
                    plugins.append(plugin_info)
                    logger.debug(f"Discovered via Stevedore: {plugin_info.qualified_name}")
                    
                except Exception as e:
                    logger.warning(f"Error processing Stevedore extension {extension_obj.name}: {e}")
        
        return plugins
    
    def _discover_brainsmith_native(self) -> List[PluginInfo]:
        """Auto-discover BrainSmith native plugins."""
        plugins = []
        
        # Discover transforms in brainsmith.transforms
        transform_plugins = self._discover_brainsmith_transforms()
        plugins.extend(transform_plugins)
        
        # Discover kernels in brainsmith.kernels  
        kernel_plugins = self._discover_brainsmith_kernels()
        plugins.extend(kernel_plugins)
        
        # Discover steps in brainsmith.steps
        step_plugins = self._discover_brainsmith_steps()
        plugins.extend(step_plugins)
        
        return plugins
    
    def _discover_brainsmith_transforms(self) -> List[PluginInfo]:
        """Discover BrainSmith native transforms."""
        plugins = []
        
        try:
            import brainsmith.transforms
            
            # Look for transform modules and classes
            transform_modules = [
                'topology_opt.expand_norms',
                'model_specific.remove_bert_head', 
                'model_specific.remove_bert_tail',
                'kernel_opt.set_pumped_compute',
                'kernel_opt.temp_shuffle_fixer',
                'metadata.extract_shell_integration_metadata'
            ]
            
            for module_path in transform_modules:
                try:
                    module = importlib.import_module(f'brainsmith.transforms.{module_path}')
                    
                    # Look for transform classes (typically end with the module name)
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        
                        if (inspect.isclass(attr) and 
                            hasattr(attr, '__call__') and
                            not attr_name.startswith('_')):
                            
                            plugin_info = PluginInfo(
                                name=attr_name,
                                plugin_class=attr,
                                framework="brainsmith",
                                plugin_type="transform",
                                metadata={'module': module_path},
                                discovery_method="auto"
                            )
                            
                            plugins.append(plugin_info)
                            logger.debug(f"Auto-discovered BrainSmith transform: {attr_name}")
                            
                except ImportError as e:
                    logger.debug(f"Could not import BrainSmith transform module {module_path}: {e}")
                    
        except ImportError:
            logger.debug("BrainSmith transforms module not available")
        
        return plugins
    
    def _discover_qonnx_transforms(self) -> List[PluginInfo]:
        """Discover QONNX transforms from their native registry."""
        plugins = []
        
        try:
            # Import QONNX transformation modules directly
            qonnx_transform_modules = [
                'qonnx.transformation.general',
                'qonnx.transformation.remove',
                'qonnx.transformation.fold_constants',
                'qonnx.transformation.infer_data_layouts', 
                'qonnx.transformation.infer_datatypes',
                'qonnx.transformation.infer_shapes'
            ]
            
            for module_name in qonnx_transform_modules:
                try:
                    module = importlib.import_module(module_name)
                    
                    # Look for transformation classes
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        
                        if (inspect.isclass(attr) and 
                            hasattr(attr, '__call__') and
                            not attr_name.startswith('_') and
                            attr_name not in ['Transformation']):
                            
                            plugin_info = PluginInfo(
                                name=attr_name,
                                plugin_class=attr,
                                framework="qonnx",
                                plugin_type="transform",
                                metadata={'module': module_name},
                                discovery_method="framework_native"
                            )
                            
                            plugins.append(plugin_info)
                            logger.debug(f"Discovered QONNX transform: {attr_name}")
                            
                except ImportError as e:
                    logger.debug(f"Could not import QONNX module {module_name}: {e}")
                    
        except Exception as e:
            logger.debug(f"Error discovering QONNX transforms: {e}")
        
        return plugins
    
    def _discover_finn_transforms(self) -> List[PluginInfo]:
        """Discover FINN transforms from their native registry."""
        plugins = []
        
        try:
            # Import FINN transformation modules directly
            finn_transform_modules = [
                'finn.transformation.streamline',
                'finn.transformation.streamline.reorder',
                'finn.transformation.move_reshape',
                'finn.transformation.fpgadataflow.convert_to_hw_layers'
            ]
            
            for module_name in finn_transform_modules:
                try:
                    module = importlib.import_module(module_name)
                    
                    # Look for transformation classes
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        
                        if (inspect.isclass(attr) and 
                            hasattr(attr, '__call__') and
                            not attr_name.startswith('_') and
                            attr_name not in ['Transformation']):
                            
                            plugin_info = PluginInfo(
                                name=attr_name,
                                plugin_class=attr,
                                framework="finn",
                                plugin_type="transform",
                                metadata={'module': module_name},
                                discovery_method="framework_native"
                            )
                            
                            plugins.append(plugin_info)
                            logger.debug(f"Discovered FINN transform: {attr_name}")
                            
                except ImportError as e:
                    logger.debug(f"Could not import FINN module {module_name}: {e}")
                    
        except Exception as e:
            logger.debug(f"Error discovering FINN transforms: {e}")
        
        return plugins
    
    def _discover_brainsmith_kernels(self) -> List[PluginInfo]:
        """Discover BrainSmith kernel implementations."""
        plugins = []
        
        try:
            # Look in brainsmith.kernels for kernel modules
            import brainsmith.kernels
            
            kernel_modules = ['layernorm', 'matmul', 'softmax', 'shuffle', 'crop']
            
            for kernel_name in kernel_modules:
                try:
                    kernel_module = importlib.import_module(f'brainsmith.kernels.{kernel_name}')
                    
                    # Look for kernel classes
                    for attr_name in dir(kernel_module):
                        attr = getattr(kernel_module, attr_name)
                        
                        if (inspect.isclass(attr) and 
                            not attr_name.startswith('_')):
                            
                            plugin_info = PluginInfo(
                                name=attr_name,
                                plugin_class=attr,
                                framework="brainsmith",
                                plugin_type="kernel",
                                metadata={'kernel_module': kernel_name},
                                discovery_method="auto"
                            )
                            
                            plugins.append(plugin_info)
                            logger.debug(f"Auto-discovered BrainSmith kernel: {attr_name}")
                            
                except ImportError as e:
                    logger.debug(f"Could not import BrainSmith kernel module {kernel_name}: {e}")
                    
        except ImportError:
            logger.debug("BrainSmith kernels module not available")
        
        return plugins
    
    def _discover_brainsmith_steps(self) -> List[PluginInfo]:
        """Discover BrainSmith steps with @finn_step decorators."""
        plugins = []
        
        try:
            import brainsmith.steps.bert_steps as bert_steps
            
            # Look for functions with finn_step decorator
            for attr_name in dir(bert_steps):
                attr = getattr(bert_steps, attr_name)
                
                if (callable(attr) and 
                    hasattr(attr, '_finn_step_name') and
                    not attr_name.startswith('_')):
                    
                    plugin_info = PluginInfo(
                        name=attr._finn_step_name,
                        plugin_class=attr,
                        framework="brainsmith",
                        plugin_type="step",
                        metadata={
                            'category': getattr(attr, '_finn_step_category', 'unknown'),
                            'dependencies': getattr(attr, '_finn_step_dependencies', []),
                            'description': getattr(attr, '_finn_step_description', '')
                        },
                        discovery_method="auto"
                    )
                    
                    plugins.append(plugin_info)
                    logger.debug(f"Auto-discovered BrainSmith step: {attr._finn_step_name}")
                    
        except ImportError as e:
            logger.debug(f"Could not import BrainSmith steps: {e}")
        
        return plugins
    
    def _extract_plugin_type_from_namespace(self, namespace: str) -> str:
        """Extract plugin type from Stevedore namespace."""
        if 'transforms' in namespace:
            return 'transform'
        elif 'kernels' in namespace:
            return 'kernel'
        elif 'backends' in namespace:
            return 'backend'
        else:
            return 'unknown'
    
    def _extract_framework_from_namespace(self, namespace: str) -> str:
        """Extract framework from Stevedore namespace."""
        if 'external' in namespace:
            return 'external'
        else:
            return 'brainsmith'
    
    def _discover_and_load_all(self):
        """Internal method to discover and load all plugins (EAGER mode)."""
        catalog = self.discover_all()
        
        # Initialize loaded plugins cache if needed
        if not hasattr(self, '_loaded_plugins'):
            self._loaded_plugins = {}
        
        logger.info("Loading all plugins (EAGER mode)...")
        loaded_count = 0
        
        for name, plugin_list in catalog.plugins_by_name.items():
            for plugin_info in plugin_list:
                cache_key = f"{plugin_info.framework}:{plugin_info.plugin_type}:{plugin_info.name}"
                plugin_info.metadata['is_loaded'] = True
                self._loaded_plugins[cache_key] = plugin_info
                loaded_count += 1
        
        logger.info(f"Loaded {loaded_count} plugins in EAGER mode")


# Global plugin manager instance
_global_manager: Optional[PluginManager] = None
_manager_lock = Lock()


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance."""
    global _global_manager
    
    if _global_manager is None:
        with _manager_lock:
            if _global_manager is None:
                _global_manager = PluginManager(DiscoveryStrategy.HYBRID)
    
    return _global_manager


def set_plugin_manager(manager: PluginManager):
    """Set the global plugin manager (useful for testing)."""
    global _global_manager
    with _manager_lock:
        _global_manager = manager