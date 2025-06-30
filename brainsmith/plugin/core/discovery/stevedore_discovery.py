"""
Stevedore-based Plugin Discovery

Uses Python entry points for plugin discovery, enabling external
plugins to be registered without modifying BrainSmith code.
"""

import logging
from typing import List, Dict, Any, Optional

try:
    from stevedore import extension
except ImportError:
    raise ImportError(
        "Stevedore is required for the plugin system. "
        "Install with: pip install stevedore"
    )

from .base import DiscoveryInterface
from ..data_models import PluginInfo

logger = logging.getLogger(__name__)


class StevedoreDiscovery(DiscoveryInterface):
    """
    Discovers plugins via Stevedore entry points.
    
    This enables external packages to register plugins with BrainSmith
    without modifying the core codebase.
    """
    
    # Default entry point namespaces
    DEFAULT_NAMESPACES = {
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
        ],
        'step': [
            'brainsmith.steps',
            'brainsmith.external.steps',
        ]
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.namespaces = self.config.get('namespaces', self.DEFAULT_NAMESPACES)
        self._managers: Dict[str, extension.ExtensionManager] = {}
    
    def _setup(self) -> None:
        """Initialize Stevedore extension managers."""
        for plugin_type, namespace_list in self.namespaces.items():
            for namespace in namespace_list:
                try:
                    manager = extension.ExtensionManager(
                        namespace=namespace,
                        invoke_on_load=False,  # Don't instantiate immediately
                        on_load_failure_callback=self._on_load_failure
                    )
                    self._managers[namespace] = manager
                    logger.debug(f"Initialized Stevedore manager for namespace: {namespace}")
                except RuntimeError as e:
                    # No entry points found - this is normal during development
                    logger.debug(f"No entry points found for namespace {namespace}: {e}")
    
    def _on_load_failure(self, manager, ep, err):
        """Handle Stevedore extension loading failures gracefully."""
        logger.warning(f"Failed to load entry point {ep.name} from {ep.dist}: {err}")
    
    @property
    def name(self) -> str:
        return "StevedoreDiscovery"
    
    def discover(self) -> List[PluginInfo]:
        """Discover plugins via Stevedore entry points."""
        plugins = []
        
        for namespace, manager in self._managers.items():
            plugin_type = self._extract_plugin_type(namespace)
            framework = self._extract_framework(namespace)
            
            for ext in manager.extensions:
                try:
                    plugin_info = self._create_plugin_info(
                        ext, plugin_type, framework, namespace
                    )
                    plugins.append(plugin_info)
                    logger.debug(f"Discovered via Stevedore: {plugin_info.qualified_name}")
                    
                except Exception as e:
                    logger.warning(f"Error processing extension {ext.name}: {e}")
        
        self.log_discovery_summary(plugins)
        return plugins
    
    def _create_plugin_info(self, ext: extension.Extension, 
                           plugin_type: str, framework: str,
                           namespace: str) -> PluginInfo:
        """Create PluginInfo from Stevedore extension."""
        # Extract metadata from the plugin class if available
        metadata = {}
        if hasattr(ext.obj, '_plugin_metadata'):
            metadata.update(ext.obj._plugin_metadata)
        
        # Add entry point information
        metadata.update({
            'entry_point': str(ext.entry_point),
            'distribution': str(ext.entry_point.dist),
            'namespace': namespace
        })
        
        return PluginInfo(
            name=ext.name,
            plugin_class=ext.obj,
            framework=framework,
            plugin_type=plugin_type,
            metadata=metadata,
            discovery_method="stevedore",
            stevedore_extension=ext
        )
    
    def _extract_plugin_type(self, namespace: str) -> str:
        """Extract plugin type from namespace."""
        for plugin_type, namespaces in self.namespaces.items():
            if namespace in namespaces:
                return plugin_type
        return "unknown"
    
    def _extract_framework(self, namespace: str) -> str:
        """Extract framework from namespace."""
        if 'external' in namespace:
            return 'external'
        return 'brainsmith'