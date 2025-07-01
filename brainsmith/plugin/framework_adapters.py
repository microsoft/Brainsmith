"""
Framework-Specific Plugin Adapters

Provides clean integration with QONNX and FINN plugin systems through
adapter pattern, allowing graceful degradation when frameworks are unavailable.
"""

import logging
from typing import List, Dict, Any, Optional, Type
from abc import ABC, abstractmethod

from .data_models import PluginInfo

logger = logging.getLogger(__name__)


class FrameworkAdapter(ABC):
    """Base class for framework-specific plugin adapters."""
    
    @property
    @abstractmethod
    def framework_name(self) -> str:
        """Name of the framework this adapter handles."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the framework is available for discovery."""
        pass
    
    @abstractmethod
    def discover_plugins(self) -> List[PluginInfo]:
        """Discover plugins from the framework."""
        pass


class QONNXAdapter(FrameworkAdapter):
    """Adapter for QONNX transformation registry with manual fallback."""
    
    @property
    def framework_name(self) -> str:
        return "qonnx"
    
    def is_available(self) -> bool:
        """Check if QONNX is available."""
        try:
            import qonnx.transformation.base
            return True
        except ImportError:
            return False
    
    def discover_plugins(self) -> List[PluginInfo]:
        """Discover QONNX transforms using manual registry (no central QONNX registry exists)."""
        if not self.is_available():
            logger.debug("QONNX not available for plugin discovery")
            return []
        
        # QONNX doesn't have a central registry, use manual registration only
        plugins = self._discover_from_manual_registry()
        
        logger.info(f"Discovered {len(plugins)} QONNX transforms from manual registry")
        
        return plugins
    
    
    def _discover_from_manual_registry(self) -> List[PluginInfo]:
        """Discover transforms from our manual registry."""
        plugins = []
        try:
            from .qonnx_transforms import register_qonnx_transforms
            from .manager import PluginManager
            
            # Create temporary manager for manual registration
            temp_manager = PluginManager()
            
            # Register all QONNX transforms (no priority filter = all priorities)
            total_count = register_qonnx_transforms(temp_manager, priority_filter=None)
            logger.debug(f"Manual registry: registered {total_count} total QONNX transforms")
            
            # Extract plugins from temporary manager
            for plugin_info in temp_manager._plugins.values():
                if plugin_info.framework == "qonnx":
                    plugins.append(plugin_info)
            
        except Exception as e:
            logger.warning(f"Manual QONNX registry failed: {e}")
        
        return plugins
    
    
    def _adapt_qonnx_transform(self, name: str, transform_class: Type) -> Optional[PluginInfo]:
        """Adapt a QONNX transform to BrainSmith PluginInfo format."""
        try:
            # Extract metadata from QONNX transform
            metadata = {
                'discovery_source': 'qonnx_registry',
                'original_name': name
            }
            
            # Try to extract stage information from class or module
            stage = self._infer_qonnx_stage(transform_class)
            if stage:
                metadata['stage'] = stage
            
            # Add description if available
            if hasattr(transform_class, '__doc__') and transform_class.__doc__:
                metadata['description'] = transform_class.__doc__.strip().split('\n')[0]
            
            # Add author/version info if available
            if hasattr(transform_class, '__module__'):
                metadata['module'] = transform_class.__module__
            
            return PluginInfo(
                name=name,
                plugin_class=transform_class,
                plugin_type='transform',
                framework='qonnx',
                metadata=metadata
            )
            
        except Exception as e:
            logger.debug(f"Failed to adapt QONNX transform {name}: {e}")
            return None
    
    def _infer_qonnx_stage(self, transform_class: Type) -> Optional[str]:
        """Infer the BrainSmith stage from QONNX transform class."""
        # Map QONNX transform names/modules to BrainSmith stages
        class_name = transform_class.__name__.lower()
        module_name = getattr(transform_class, '__module__', '').lower()
        
        # Cleanup stage
        cleanup_patterns = [
            'remove', 'clean', 'eliminate', 'strip', 'prune',
            'identity', 'constant', 'reshape', 'squeeze', 'unsqueeze'
        ]
        
        # Topology optimization stage
        topology_patterns = [
            'fold', 'merge', 'combine', 'fuse', 'collapse',
            'streamline', 'normalize', 'absorb', 'extract'
        ]
        
        # Kernel optimization stage  
        kernel_patterns = [
            'lowering', 'kernel', 'convert', 'backend',
            'hardware', 'fpga', 'accelerator'
        ]
        
        # Dataflow optimization stage
        dataflow_patterns = [
            'dataflow', 'partition', 'split', 'parallel',
            'pipeline', 'schedule', 'memory'
        ]
        
        for pattern in cleanup_patterns:
            if pattern in class_name or pattern in module_name:
                return 'cleanup'
        
        for pattern in topology_patterns:
            if pattern in class_name or pattern in module_name:
                return 'topology_opt'
        
        for pattern in kernel_patterns:
            if pattern in class_name or pattern in module_name:
                return 'kernel_opt'
        
        for pattern in dataflow_patterns:
            if pattern in class_name or pattern in module_name:
                return 'dataflow_opt'
        
        # Default to cleanup if we can't determine
        return 'cleanup'


class FINNAdapter(FrameworkAdapter):
    """Adapter for FINN plugin system."""
    
    @property
    def framework_name(self) -> str:
        return "finn"
    
    def is_available(self) -> bool:
        """Check if FINN is available."""
        try:
            import finn
            return True
        except ImportError:
            return False
    
    def discover_plugins(self) -> List[PluginInfo]:
        """Discover FINN plugins and adapt them to BrainSmith format."""
        if not self.is_available():
            logger.debug("FINN not available for plugin discovery")
            return []
        
        plugins = []
        
        # Discover FINN transforms
        plugins.extend(self._discover_finn_transforms())
        
        # Discover FINN kernels (if available)
        plugins.extend(self._discover_finn_kernels())
        
        # Discover FINN backends (if available)
        plugins.extend(self._discover_finn_backends())
        
        logger.info(f"Discovered {len(plugins)} FINN plugins")
        return plugins
    
    def _discover_finn_transforms(self) -> List[PluginInfo]:
        """Discover FINN transformation plugins."""
        plugins = []
        
        try:
            # FINN may have a different registry structure
            # This is a placeholder for FINN-specific discovery
            logger.debug("FINN transform discovery not yet implemented")
            
            # Example implementation when FINN registry is available:
            # from finn.transformation.registry import get_all_transforms
            # transforms = get_all_transforms()
            # for name, transform_class in transforms.items():
            #     plugin_info = self._adapt_finn_transform(name, transform_class)
            #     if plugin_info:
            #         plugins.append(plugin_info)
            
        except Exception as e:
            logger.debug(f"FINN transform discovery failed: {e}")
        
        return plugins
    
    def _discover_finn_kernels(self) -> List[PluginInfo]:
        """Discover FINN kernel plugins."""
        plugins = []
        
        try:
            # Placeholder for FINN kernel discovery
            logger.debug("FINN kernel discovery not yet implemented")
            
        except Exception as e:
            logger.debug(f"FINN kernel discovery failed: {e}")
        
        return plugins
    
    def _discover_finn_backends(self) -> List[PluginInfo]:
        """Discover FINN backend plugins."""
        plugins = []
        
        try:
            # Placeholder for FINN backend discovery
            logger.debug("FINN backend discovery not yet implemented")
            
        except Exception as e:
            logger.debug(f"FINN backend discovery failed: {e}")
        
        return plugins
    
    def _adapt_finn_transform(self, name: str, transform_class: Type) -> Optional[PluginInfo]:
        """Adapt a FINN transform to BrainSmith PluginInfo format."""
        try:
            metadata = {
                'discovery_source': 'finn_registry',
                'original_name': name
            }
            
            # Add FINN-specific metadata extraction here
            
            return PluginInfo(
                name=name,
                plugin_class=transform_class,
                plugin_type='transform',
                framework='finn',
                metadata=metadata
            )
            
        except Exception as e:
            logger.debug(f"Failed to adapt FINN transform {name}: {e}")
            return None


class BrainSmithAdapter(FrameworkAdapter):
    """Adapter for native BrainSmith plugins."""
    
    @property
    def framework_name(self) -> str:
        return "brainsmith"
    
    def is_available(self) -> bool:
        """BrainSmith is always available."""
        return True
    
    def discover_plugins(self) -> List[PluginInfo]:
        """Discover native BrainSmith plugins through module scanning."""
        plugins = []
        
        # Scan BrainSmith modules for decorated plugins
        modules_to_scan = [
            'brainsmith.kernels',
            'brainsmith.transforms',
            'brainsmith.steps'
        ]
        
        for module_name in modules_to_scan:
            plugins.extend(self._scan_brainsmith_module(module_name))
        
        logger.info(f"Discovered {len(plugins)} BrainSmith plugins")
        return plugins
    
    def _scan_brainsmith_module(self, module_name: str) -> List[PluginInfo]:
        """Scan a BrainSmith module for plugin classes."""
        plugins = []
        
        try:
            import importlib
            import pkgutil
            
            module = importlib.import_module(module_name)
            
            for finder, name, ispkg in pkgutil.walk_packages(
                module.__path__, 
                prefix=f"{module_name}."
            ):
                if ispkg:
                    continue
                
                try:
                    submodule = importlib.import_module(name)
                    for attr_name in dir(submodule):
                        attr = getattr(submodule, attr_name)
                        if self._is_brainsmith_plugin_class(attr):
                            plugin_info = self._adapt_brainsmith_plugin(attr)
                            if plugin_info:
                                plugins.append(plugin_info)
                                
                except Exception as e:
                    logger.debug(f"Failed to scan {name}: {e}")
                    
        except Exception as e:
            logger.debug(f"Failed to scan module {module_name}: {e}")
        
        return plugins
    
    def _is_brainsmith_plugin_class(self, obj: Any) -> bool:
        """Check if an object is a BrainSmith plugin class."""
        return (
            isinstance(obj, type) and
            hasattr(obj, '_plugin_metadata') and
            isinstance(obj._plugin_metadata, dict)
        )
    
    def _adapt_brainsmith_plugin(self, plugin_class: Type) -> Optional[PluginInfo]:
        """Adapt a BrainSmith plugin class to PluginInfo format."""
        try:
            metadata = plugin_class._plugin_metadata.copy()
            
            # Add discovery source
            metadata['discovery_source'] = 'module_scan'
            
            return PluginInfo(
                name=metadata.pop('name'),
                plugin_class=plugin_class,
                plugin_type=metadata.pop('type'),
                framework=metadata.pop('framework', 'brainsmith'),
                metadata=metadata
            )
            
        except Exception as e:
            logger.debug(f"Failed to adapt BrainSmith plugin {plugin_class}: {e}")
            return None


class UnifiedFrameworkDiscovery:
    """
    Unified discovery system that coordinates all framework adapters.
    """
    
    def __init__(self):
        self.adapters = {
            'brainsmith': BrainSmithAdapter(),
            'qonnx': QONNXAdapter(),
            'finn': FINNAdapter()
        }
    
    def discover_all_plugins(self) -> List[PluginInfo]:
        """Discover plugins from all available frameworks."""
        all_plugins = []
        
        for framework, adapter in self.adapters.items():
            try:
                if adapter.is_available():
                    plugins = adapter.discover_plugins()
                    all_plugins.extend(plugins)
                else:
                    logger.debug(f"Framework {framework} not available")
            except Exception as e:
                logger.warning(f"Failed to discover plugins from {framework}: {e}")
        
        return all_plugins
    
    def discover_framework_plugins(self, frameworks: List[str]) -> List[PluginInfo]:
        """Discover plugins from specific frameworks only."""
        plugins = []
        
        for framework in frameworks:
            if framework in self.adapters:
                adapter = self.adapters[framework]
                try:
                    if adapter.is_available():
                        framework_plugins = adapter.discover_plugins()
                        plugins.extend(framework_plugins)
                    else:
                        logger.debug(f"Framework {framework} not available")
                except Exception as e:
                    logger.warning(f"Failed to discover plugins from {framework}: {e}")
            else:
                logger.warning(f"Unknown framework: {framework}")
        
        return plugins
    
    def get_available_frameworks(self) -> List[str]:
        """Get list of currently available frameworks."""
        return [
            framework for framework, adapter in self.adapters.items()
            if adapter.is_available()
        ]