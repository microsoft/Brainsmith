"""
Plugin Discovery Module

Discovers and imports plugins from external frameworks like FINN and QONNX.
Uses specialized discovery modules for detailed metadata mapping.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class PluginDiscovery:
    """Discovers and imports plugins from QONNX and FINN using specialized modules."""
    
    def __init__(self, registry=None):
        """
        Initialize discovery with a registry.
        
        Args:
            registry: Plugin registry to use. If None, uses global registry.
        """
        if registry is None:
            from .core import get_registry
            registry = get_registry()
        self.registry = registry
    
    def discover_all(self):
        """Discover all available plugins from external frameworks."""
        qonnx_count = self._discover_qonnx()
        finn_count = self._discover_finn()
        
        total = qonnx_count + finn_count
        logger.info(f"Discovered {total} plugins ({qonnx_count} QONNX, {finn_count} FINN)")
        return total
    
    def _discover_qonnx(self) -> int:
        """Discover QONNX transformations using specialized discovery module."""
        try:
            from .qonnx_discovery import discover_qonnx_transforms
            return discover_qonnx_transforms()
        except ImportError:
            logger.debug("QONNX not available for discovery")
            return 0
        except Exception as e:
            logger.debug(f"QONNX discovery error: {e}")
            return 0
    
    def _discover_finn(self) -> int:
        """Discover FINN transformations using specialized discovery module."""
        try:
            from .finn_discovery import discover_finn_transforms
            return discover_finn_transforms()
        except ImportError:
            logger.debug("FINN not available for discovery")
            return 0
        except Exception as e:
            logger.debug(f"FINN discovery error: {e}")
            return 0


# Convenience function
def discover_plugins(registry=None):
    """
    Discover all available plugins.
    
    Args:
        registry: Registry to populate. If None, uses global registry.
        
    Returns:
        Number of plugins discovered
    """
    discovery = PluginDiscovery(registry)
    return discovery.discover_all()