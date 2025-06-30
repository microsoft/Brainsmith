"""
Plugin Discovery System

Provides multiple discovery strategies for finding plugins:
- Stevedore entry points
- Auto-discovery via filesystem
- Framework-native discovery (QONNX/FINN)
- Composite discovery combining multiple strategies
"""

from .base import DiscoveryInterface
from .stevedore_discovery import StevedoreDiscovery
from .auto_discovery import AutoDiscovery
from .framework_discovery import QONNXDiscovery, FINNDiscovery
from .composite_discovery import CompositeDiscovery

__all__ = [
    'DiscoveryInterface',
    'StevedoreDiscovery',
    'AutoDiscovery',
    'QONNXDiscovery',
    'FINNDiscovery',
    'CompositeDiscovery'
]