"""
Auto-discovery for BrainSmith Native Plugins

Scans the BrainSmith codebase to discover plugins decorated with
the appropriate decorators or following naming conventions.
"""

import logging
import inspect
import importlib
import pkgutil
from typing import List, Dict, Any, Optional, Set
from pathlib import Path

from .base import DiscoveryInterface
from ..data_models import PluginInfo

logger = logging.getLogger(__name__)


class AutoDiscovery(DiscoveryInterface):
    """
    Discovers BrainSmith native plugins by scanning the codebase.
    
    This supports rapid development by automatically finding plugins
    without requiring entry point registration.
    """
    
    # Default modules to scan
    DEFAULT_SCAN_MODULES = {
        'transforms': [
            'brainsmith.transforms.topology_opt',
            'brainsmith.transforms.model_specific',
            'brainsmith.transforms.kernel_opt',
            'brainsmith.transforms.metadata',
            'brainsmith.transforms.graph_cleanup',
            'brainsmith.transforms.dataflow_opt',
            'brainsmith.transforms.kernel_inference',
        ],
        'kernels': [
            'brainsmith.kernels.layernorm',
            'brainsmith.kernels.matmul',
            'brainsmith.kernels.softmax',
            'brainsmith.kernels.shuffle',
            'brainsmith.kernels.crop',
        ],
        'steps': [
            'brainsmith.steps.bert_steps',
        ]
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.scan_modules = self.config.get('scan_modules', self.DEFAULT_SCAN_MODULES)
        self._discovered_classes: Set[type] = set()  # Avoid duplicates
    
    @property
    def name(self) -> str:
        return "AutoDiscovery"
    
    def discover(self) -> List[PluginInfo]:
        """Discover plugins by scanning configured modules."""
        plugins = []
        
        # Discover transforms
        transform_plugins = self._discover_transforms()
        plugins.extend(transform_plugins)
        
        # Discover kernels and backends
        kernel_plugins = self._discover_kernels()
        plugins.extend(kernel_plugins)
        
        # Discover steps
        step_plugins = self._discover_steps()
        plugins.extend(step_plugins)
        
        self.log_discovery_summary(plugins)
        return plugins
    
    def _discover_transforms(self) -> List[PluginInfo]:
        """Discover transform plugins."""
        plugins = []
        
        for module_name in self.scan_modules.get('transforms', []):
            plugins.extend(self._scan_module_for_transforms(module_name))
        
        return plugins
    
    def _scan_module_for_transforms(self, module_name: str) -> List[PluginInfo]:
        """Scan a specific module for transform classes."""
        plugins = []
        
        try:
            if '.' in module_name:
                # Import parent and scan submodules
                parent_name = module_name
                parent = importlib.import_module(parent_name)
                
                # Scan the parent module
                plugins.extend(self._extract_transforms_from_module(parent, parent_name))
                
                # Scan submodules
                if hasattr(parent, '__path__'):
                    for importer, modname, ispkg in pkgutil.iter_modules(parent.__path__):
                        full_name = f"{parent_name}.{modname}"
                        try:
                            submodule = importlib.import_module(full_name)
                            plugins.extend(self._extract_transforms_from_module(submodule, full_name))
                        except ImportError as e:
                            logger.debug(f"Could not import {full_name}: {e}")
            else:
                # Direct module import
                module = importlib.import_module(module_name)
                plugins.extend(self._extract_transforms_from_module(module, module_name))
                
        except ImportError as e:
            logger.debug(f"Could not import module {module_name}: {e}")
        
        return plugins
    
    def _extract_transforms_from_module(self, module, module_name: str) -> List[PluginInfo]:
        """Extract transform classes from a module."""
        plugins = []
        
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                not name.startswith('_') and
                obj not in self._discovered_classes and
                self._is_transform_class(obj)):
                
                self._discovered_classes.add(obj)
                
                # Extract metadata from decorator if available
                metadata = self._extract_plugin_metadata(obj)
                metadata['module'] = module_name
                
                # Determine plugin type from metadata
                plugin_type = metadata.get('type', 'transform')
                if metadata.get('kernel'):
                    plugin_type = 'kernel_inference'
                
                plugin_info = PluginInfo(
                    name=metadata.get('name', name),
                    plugin_class=obj,
                    framework="brainsmith",
                    plugin_type=plugin_type,
                    metadata=metadata,
                    discovery_method="auto"
                )
                
                plugins.append(plugin_info)
                logger.debug(f"Auto-discovered transform: {name} from {module_name}")
        
        return plugins
    
    def _is_transform_class(self, obj) -> bool:
        """Check if an object is a transform class."""
        # Check for decorator metadata
        if hasattr(obj, '_plugin_metadata'):
            return obj._plugin_metadata.get('type') in ['transform', 'kernel_inference']
        
        # Check for transform base class
        try:
            from qonnx.transformation.base import Transformation
            return issubclass(obj, Transformation)
        except:
            pass
        
        # Check for callable with apply method
        return (hasattr(obj, 'apply') and 
                callable(getattr(obj, 'apply', None)))
    
    def _discover_kernels(self) -> List[PluginInfo]:
        """Discover kernel and backend plugins."""
        plugins = []
        
        for module_name in self.scan_modules.get('kernels', []):
            try:
                module = importlib.import_module(module_name)
                
                # Look for kernel and backend classes
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        not name.startswith('_') and
                        obj not in self._discovered_classes):
                        
                        metadata = self._extract_plugin_metadata(obj)
                        plugin_type = metadata.get('type')
                        
                        if plugin_type in ['kernel', 'backend']:
                            self._discovered_classes.add(obj)
                            metadata['kernel_module'] = module_name.split('.')[-1]
                            
                            plugin_info = PluginInfo(
                                name=metadata.get('name', name),
                                plugin_class=obj,
                                framework="brainsmith",
                                plugin_type=plugin_type,
                                metadata=metadata,
                                discovery_method="auto"
                            )
                            
                            plugins.append(plugin_info)
                            logger.debug(f"Auto-discovered {plugin_type}: {name}")
                            
            except ImportError as e:
                logger.debug(f"Could not import kernel module {module_name}: {e}")
        
        return plugins
    
    def _discover_steps(self) -> List[PluginInfo]:
        """Discover FINN step plugins."""
        plugins = []
        
        for module_name in self.scan_modules.get('steps', []):
            try:
                module = importlib.import_module(module_name)
                
                # Look for functions with step metadata
                for name, obj in inspect.getmembers(module):
                    if (callable(obj) and 
                        hasattr(obj, '_finn_step_name') and
                        not name.startswith('_')):
                        
                        plugin_info = PluginInfo(
                            name=obj._finn_step_name,
                            plugin_class=obj,
                            framework="brainsmith",
                            plugin_type="step",
                            metadata={
                                'category': getattr(obj, '_finn_step_category', 'unknown'),
                                'dependencies': getattr(obj, '_finn_step_dependencies', []),
                                'description': getattr(obj, '_finn_step_description', ''),
                                'function_name': name,
                                'module': module_name
                            },
                            discovery_method="auto"
                        )
                        
                        plugins.append(plugin_info)
                        logger.debug(f"Auto-discovered step: {obj._finn_step_name}")
                        
            except ImportError as e:
                logger.debug(f"Could not import step module {module_name}: {e}")
        
        return plugins
    
    def _extract_plugin_metadata(self, obj) -> Dict[str, Any]:
        """Extract metadata from plugin class or function."""
        metadata = {}
        
        # Check for decorator metadata
        if hasattr(obj, '_plugin_metadata'):
            metadata.update(obj._plugin_metadata)
        
        # Extract from docstring if available
        if obj.__doc__:
            metadata['docstring'] = obj.__doc__.strip()
        
        # Extract module information
        if hasattr(obj, '__module__'):
            metadata['origin_module'] = obj.__module__
        
        return metadata