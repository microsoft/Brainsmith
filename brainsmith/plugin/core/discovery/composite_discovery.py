"""
Composite Discovery Strategy

Combines multiple discovery strategies to provide comprehensive
plugin discovery across all sources.
"""

import logging
from typing import List, Dict, Any, Optional, Set

from .base import DiscoveryInterface
from ..data_models import PluginInfo, DiscoveryStrategy

logger = logging.getLogger(__name__)


class CompositeDiscovery(DiscoveryInterface):
    """
    Combines multiple discovery strategies into a unified approach.
    
    This is the main discovery implementation used by the plugin system,
    supporting different strategies based on configuration.
    """
    
    def __init__(self, strategy: DiscoveryStrategy = DiscoveryStrategy.HYBRID,
                 config: Optional[Dict[str, Any]] = None):
        self.strategy = strategy
        super().__init__(config)
        self._discoveries: List[DiscoveryInterface] = []
        self._seen_plugins: Set[str] = set()  # Track discovered plugins
    
    def _setup(self) -> None:
        """Initialize appropriate discovery strategies based on configuration."""
        from .stevedore_discovery import StevedoreDiscovery
        from .auto_discovery import AutoDiscovery
        from .framework_discovery import QONNXDiscovery, FINNDiscovery
        
        if self.strategy in [DiscoveryStrategy.STEVEDORE_ONLY, DiscoveryStrategy.HYBRID]:
            # Add Stevedore discovery
            stevedore_config = self.config.get('stevedore', {})
            self._discoveries.append(StevedoreDiscovery(stevedore_config))
        
        if self.strategy in [DiscoveryStrategy.AUTO_DISCOVERY, DiscoveryStrategy.HYBRID]:
            # Add auto-discovery
            auto_config = self.config.get('auto', {})
            self._discoveries.append(AutoDiscovery(auto_config))
            
            # Add framework discoveries
            qonnx_config = self.config.get('qonnx', {})
            self._discoveries.append(QONNXDiscovery(qonnx_config))
            
            finn_config = self.config.get('finn', {})
            self._discoveries.append(FINNDiscovery(finn_config))
    
    @property
    def name(self) -> str:
        return f"CompositeDiscovery({self.strategy.value})"
    
    def discover(self) -> List[PluginInfo]:
        """
        Discover plugins using all configured strategies.
        
        Handles deduplication and conflict detection across strategies.
        """
        all_plugins = []
        
        for discovery in self._discoveries:
            try:
                logger.info(f"Running {discovery.name}...")
                plugins = discovery.discover()
                
                # Filter out duplicates based on qualified name
                unique_plugins = []
                for plugin in plugins:
                    qualified_name = f"{plugin.framework}:{plugin.plugin_type}:{plugin.name}"
                    if qualified_name not in self._seen_plugins:
                        self._seen_plugins.add(qualified_name)
                        unique_plugins.append(plugin)
                    else:
                        logger.debug(
                            f"Skipping duplicate plugin: {qualified_name} "
                            f"from {discovery.name}"
                        )
                
                all_plugins.extend(unique_plugins)
                
            except Exception as e:
                logger.error(f"Error in {discovery.name}: {e}", exc_info=True)
        
        # Log final summary
        self.log_discovery_summary(all_plugins)
        self._log_detailed_summary(all_plugins)
        
        return all_plugins
    
    def _log_detailed_summary(self, plugins: List[PluginInfo]) -> None:
        """Log detailed summary of all discovered plugins."""
        by_discovery = {}
        by_combo = {}
        
        for plugin in plugins:
            # Count by discovery method
            method = plugin.discovery_method
            if method not in by_discovery:
                by_discovery[method] = 0
            by_discovery[method] += 1
            
            # Count by framework/type combination
            combo = f"{plugin.framework}:{plugin.plugin_type}"
            if combo not in by_combo:
                by_combo[combo] = 0
            by_combo[combo] += 1
        
        logger.info(f"Composite discovery complete: {len(plugins)} total plugins")
        logger.info(f"  By discovery method: {by_discovery}")
        logger.info(f"  By framework/type: {by_combo}")
    
    def add_discovery(self, discovery: DiscoveryInterface) -> None:
        """
        Add an additional discovery strategy.
        
        Useful for extending the system with custom discoveries.
        """
        self._discoveries.append(discovery)
        logger.info(f"Added discovery strategy: {discovery.name}")