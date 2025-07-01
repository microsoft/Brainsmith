"""
Plugin System Data Models

Core data structures for the plugin system.
"""

from typing import Dict, List, Optional, Type, Any
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PluginInfo:
    """Essential plugin information."""
    name: str
    plugin_class: Type
    plugin_type: str     # "transform", "kernel", "backend", "step"
    framework: str       # "brainsmith", "qonnx", "finn"
    metadata: Dict[str, Any] = field(default_factory=dict, hash=False)


def create_plugin_info(name: str, plugin_class: Type, plugin_type: str, 
                      framework: str = "brainsmith", **metadata) -> PluginInfo:
    """Create a PluginInfo object with metadata."""
    return PluginInfo(
        name=name,
        plugin_class=plugin_class,
        plugin_type=plugin_type,
        framework=framework,
        metadata=metadata
    )