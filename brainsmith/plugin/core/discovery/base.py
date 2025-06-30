"""
Base Discovery Interface

Defines the contract for all plugin discovery implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

from ..data_models import PluginInfo

logger = logging.getLogger(__name__)


class DiscoveryInterface(ABC):
    """
    Abstract base class for plugin discovery strategies.
    
    All discovery implementations must inherit from this class
    and implement the discover() method.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize discovery with optional configuration.
        
        Args:
            config: Discovery-specific configuration
        """
        self.config = config or {}
        self._setup()
    
    def _setup(self) -> None:
        """
        Optional setup method for subclasses.
        
        Called after __init__ to perform any necessary initialization.
        """
        pass
    
    @abstractmethod
    def discover(self) -> List[PluginInfo]:
        """
        Discover plugins using this strategy.
        
        Returns:
            List of discovered PluginInfo objects
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of this discovery strategy.
        
        Used for logging and debugging.
        """
        pass
    
    def filter_plugins(self, plugins: List[PluginInfo], 
                      **criteria) -> List[PluginInfo]:
        """
        Filter plugins based on given criteria.
        
        Args:
            plugins: List of plugins to filter
            **criteria: Filter criteria (name, type, framework, etc.)
            
        Returns:
            Filtered list of plugins
        """
        return [p for p in plugins if p.matches(**criteria)]
    
    def log_discovery_summary(self, plugins: List[PluginInfo]) -> None:
        """Log a summary of discovered plugins."""
        if not plugins:
            logger.info(f"{self.name}: No plugins discovered")
            return
        
        by_type = {}
        by_framework = {}
        
        for plugin in plugins:
            # Count by type
            if plugin.plugin_type not in by_type:
                by_type[plugin.plugin_type] = 0
            by_type[plugin.plugin_type] += 1
            
            # Count by framework
            if plugin.framework not in by_framework:
                by_framework[plugin.framework] = 0
            by_framework[plugin.framework] += 1
        
        logger.info(
            f"{self.name}: Discovered {len(plugins)} plugins - "
            f"by type: {by_type}, by framework: {by_framework}"
        )