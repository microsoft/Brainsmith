"""
Stevedore Adapters for QONNX and FINN Plugins

Bridges existing QONNX and FINN plugin systems to work with Stevedore
entry points for the blueprint-driven plugin manager.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass 
class AdaptedPlugin:
    """
    Adapter for external framework plugins to work with Stevedore.
    """
    name: str
    plugin_class: type
    framework: str
    plugin_type: str
    metadata: Dict[str, Any]
    
    def load(self):
        """Load the plugin class (Stevedore-compatible interface)."""
        return self.plugin_class


class QONNXDiscoveryAdapter:
    """
    Adapter to discover QONNX transforms and make them available
    as Stevedore-compatible entry points.
    """
    
    def __init__(self):
        self._discovered_plugins: Optional[List[AdaptedPlugin]] = None
    
    def discover_qonnx_transforms(self) -> List[AdaptedPlugin]:
        """
        Discover QONNX transforms using their registry system.
        
        Returns:
            List of adapted plugins compatible with Stevedore
        """
        if self._discovered_plugins is not None:
            return self._discovered_plugins
        
        logger.info("Discovering QONNX transforms...")
        self._discovered_plugins = []
        
        try:
            from qonnx.transformation.registry import list_transformations, get_transformation_info
            
            # Get all registered QONNX transforms
            transform_names = list_transformations()
            logger.debug(f"Found {len(transform_names)} QONNX transforms")
            
            for name in transform_names:
                try:
                    info = get_transformation_info(name)
                    transform_class = info.get('class')
                    
                    if transform_class:
                        adapted = AdaptedPlugin(
                            name=name,
                            plugin_class=transform_class,
                            framework="qonnx",
                            plugin_type="transform",
                            metadata={
                                "stage": self._infer_qonnx_stage(name, info),
                                "description": info.get('description', ''),
                                "tags": info.get('tags', []),
                                "author": info.get('author', ''),
                                "version": info.get('version', ''),
                                "framework": "qonnx"
                            }
                        )
                        self._discovered_plugins.append(adapted)
                        
                except Exception as e:
                    logger.warning(f"Failed to adapt QONNX transform '{name}': {e}")
                    
        except ImportError:
            logger.info("QONNX not available - skipping QONNX transform discovery")
        except Exception as e:
            logger.error(f"Error discovering QONNX transforms: {e}")
        
        logger.info(f"Discovered {len(self._discovered_plugins)} QONNX transforms")
        return self._discovered_plugins
    
    def _infer_qonnx_stage(self, name: str, info: Dict[str, Any]) -> str:
        """
        Infer BrainSmith compilation stage for QONNX transforms.
        
        QONNX doesn't have stages, so we use heuristics based on
        transform names and tags.
        """
        name_lower = name.lower()
        tags = info.get('tags', [])
        
        # Cleanup stage patterns
        if any(tag in ['cleanup', 'graph-simplification'] for tag in tags):
            return "cleanup"
        if any(keyword in name_lower for keyword in ['remove', 'clean', 'simplify']):
            return "cleanup"
            
        # Topology optimization patterns  
        if any(tag in ['optimization', 'graph-optimization'] for tag in tags):
            return "topology_opt"
        if any(keyword in name_lower for keyword in ['fold', 'merge', 'absorb', 'optimize']):
            return "topology_opt"
            
        # Default to cleanup for QONNX transforms
        return "cleanup"


class FINNDiscoveryAdapter:
    """
    Adapter to discover FINN plugins and make them available
    as Stevedore-compatible entry points.
    """
    
    def __init__(self):
        self._discovered_plugins: Optional[List[AdaptedPlugin]] = None
    
    def discover_finn_plugins(self) -> List[AdaptedPlugin]:
        """
        Discover FINN plugins using their registry system.
        
        Returns:
            List of adapted plugins compatible with Stevedore
        """
        if self._discovered_plugins is not None:
            return self._discovered_plugins
            
        logger.info("Discovering FINN plugins...")
        self._discovered_plugins = []
        
        try:
            # Import FINN's plugin registry
            from finn.plugin.registry import get_registry as get_finn_registry
            
            finn_registry = get_finn_registry()
            
            # Discover different plugin types
            self._discover_finn_transforms(finn_registry)
            self._discover_finn_kernels(finn_registry)
            self._discover_finn_backends(finn_registry)
            
        except ImportError:
            logger.info("FINN not available - skipping FINN plugin discovery")
        except Exception as e:
            logger.error(f"Error discovering FINN plugins: {e}")
        
        logger.info(f"Discovered {len(self._discovered_plugins)} FINN plugins")
        return self._discovered_plugins
    
    def _discover_finn_transforms(self, finn_registry):
        """Discover FINN transforms."""
        try:
            transforms = finn_registry.query(type="transform")
            for transform_info in transforms:
                adapted = AdaptedPlugin(
                    name=transform_info["name"],
                    plugin_class=transform_info["class"],
                    framework="finn",
                    plugin_type="transform",
                    metadata={
                        "stage": transform_info.get("stage", "cleanup"),
                        "description": transform_info.get("description", ""),
                        "author": transform_info.get("author", ""),
                        "version": transform_info.get("version", ""),
                        "framework": "finn"
                    }
                )
                self._discovered_plugins.append(adapted)
                
        except Exception as e:
            logger.warning(f"Failed to discover FINN transforms: {e}")
    
    def _discover_finn_kernels(self, finn_registry):
        """Discover FINN kernels."""
        try:
            kernels = finn_registry.query(type="kernel")
            for kernel_info in kernels:
                adapted = AdaptedPlugin(
                    name=kernel_info["name"],
                    plugin_class=kernel_info["class"],
                    framework="finn",
                    plugin_type="kernel",
                    metadata={
                        "op_type": kernel_info.get("op_type", ""),
                        "domain": kernel_info.get("domain", ""),
                        "description": kernel_info.get("description", ""),
                        "framework": "finn"
                    }
                )
                self._discovered_plugins.append(adapted)
                
        except Exception as e:
            logger.warning(f"Failed to discover FINN kernels: {e}")
    
    def _discover_finn_backends(self, finn_registry):
        """Discover FINN backends."""
        try:
            backends = finn_registry.query(type="backend")
            for backend_info in backends:
                adapted = AdaptedPlugin(
                    name=backend_info["name"],
                    plugin_class=backend_info["class"],
                    framework="finn",
                    plugin_type="backend",
                    metadata={
                        "kernel": backend_info.get("kernel", ""),
                        "backend_type": backend_info.get("backend_type", ""),
                        "description": backend_info.get("description", ""),
                        "framework": "finn"
                    }
                )
                self._discovered_plugins.append(adapted)
                
        except Exception as e:
            logger.warning(f"Failed to discover FINN backends: {e}")


class BrainSmithDiscoveryAdapter:
    """
    Adapter for native BrainSmith plugins to work with Stevedore.
    """
    
    def __init__(self):
        self._discovered_plugins: Optional[List[AdaptedPlugin]] = None
    
    def discover_brainsmith_plugins(self) -> List[AdaptedPlugin]:
        """
        Discover native BrainSmith plugins.
        
        Returns:
            List of adapted plugins compatible with Stevedore  
        """
        if self._discovered_plugins is not None:
            return self._discovered_plugins
            
        logger.info("Discovering BrainSmith plugins...")
        self._discovered_plugins = []
        
        try:
            from brainsmith.plugin.core import get_registry
            
            registry = get_registry()
            
            # Get all plugin types
            for plugin_type in ["transform", "kernel_inference", "kernel", "backend"]:
                plugins = registry.query(type=plugin_type)
                
                for plugin_info in plugins:
                    # Skip non-BrainSmith plugins (they'll be discovered by other adapters)
                    if plugin_info.get("framework", "brainsmith") != "brainsmith":
                        continue
                        
                    adapted = AdaptedPlugin(
                        name=plugin_info["name"],
                        plugin_class=plugin_info["class"],
                        framework="brainsmith",
                        plugin_type=plugin_type,
                        metadata=plugin_info
                    )
                    self._discovered_plugins.append(adapted)
                    
        except Exception as e:
            logger.error(f"Error discovering BrainSmith plugins: {e}")
        
        logger.info(f"Discovered {len(self._discovered_plugins)} BrainSmith plugins")
        return self._discovered_plugins


class UnifiedPluginDiscovery:
    """
    Unified discovery system that combines all frameworks
    and provides a Stevedore-compatible interface.
    """
    
    def __init__(self):
        self.qonnx_adapter = QONNXDiscoveryAdapter()
        self.finn_adapter = FINNDiscoveryAdapter()
        self.brainsmith_adapter = BrainSmithDiscoveryAdapter()
        self._all_plugins: Optional[List[AdaptedPlugin]] = None
    
    def discover_all_plugins(self) -> List[AdaptedPlugin]:
        """
        Discover plugins from all frameworks.
        
        Returns:
            Combined list of all adapted plugins
        """
        if self._all_plugins is not None:
            return self._all_plugins
            
        logger.info("Starting unified plugin discovery...")
        self._all_plugins = []
        
        # Discover from each framework
        self._all_plugins.extend(self.qonnx_adapter.discover_qonnx_transforms())
        self._all_plugins.extend(self.finn_adapter.discover_finn_plugins())
        self._all_plugins.extend(self.brainsmith_adapter.discover_brainsmith_plugins())
        
        # Log summary
        by_framework = {}
        for plugin in self._all_plugins:
            framework = plugin.framework
            if framework not in by_framework:
                by_framework[framework] = 0
            by_framework[framework] += 1
        
        logger.info(f"Unified discovery complete: {len(self._all_plugins)} total plugins")
        for framework, count in by_framework.items():
            logger.info(f"  - {framework}: {count} plugins")
        
        return self._all_plugins
    
    def get_plugins_by_framework(self, framework: str) -> List[AdaptedPlugin]:
        """Get all plugins from a specific framework."""
        all_plugins = self.discover_all_plugins()
        return [p for p in all_plugins if p.framework == framework]
    
    def get_plugins_by_type(self, plugin_type: str) -> List[AdaptedPlugin]:
        """Get all plugins of a specific type."""
        all_plugins = self.discover_all_plugins()
        return [p for p in all_plugins if p.plugin_type == plugin_type]
    
    def find_plugin(self, name: str, framework: Optional[str] = None) -> Optional[AdaptedPlugin]:
        """Find a specific plugin by name and optionally framework."""
        all_plugins = self.discover_all_plugins()
        
        for plugin in all_plugins:
            if plugin.name == name:
                if framework is None or plugin.framework == framework:
                    return plugin
        
        return None
    
    def analyze_conflicts(self) -> Dict[str, List[AdaptedPlugin]]:
        """
        Analyze naming conflicts across frameworks.
        
        Returns:
            Dict mapping conflicted names to lists of conflicting plugins
        """
        all_plugins = self.discover_all_plugins()
        conflicts = {}
        by_name = {}
        
        # Group by name
        for plugin in all_plugins:
            if plugin.name not in by_name:
                by_name[plugin.name] = []
            by_name[plugin.name].append(plugin)
        
        # Find conflicts
        for name, plugins in by_name.items():
            if len(plugins) > 1:
                conflicts[name] = plugins
        
        return conflicts


# Global discovery instance
_discovery = None


def get_unified_discovery() -> UnifiedPluginDiscovery:
    """Get the global unified discovery instance."""
    global _discovery
    if _discovery is None:
        _discovery = UnifiedPluginDiscovery()
    return _discovery