"""
Core Plugin Infrastructure

Provides the foundational components for the Pure Stevedore Plugin System.
"""

from .data_models import (
    DiscoveryStrategy,
    PluginInfo, 
    PluginCatalog,
    PluginType,
    FrameworkType,
    DiscoveryMethod
)

__all__ = [
    'DiscoveryStrategy',
    'PluginInfo',
    'PluginCatalog',
    'PluginType',
    'FrameworkType',
    'DiscoveryMethod'
]