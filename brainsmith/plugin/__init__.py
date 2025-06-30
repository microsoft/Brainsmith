"""
BrainSmith Plugin System

This package provides the unified plugin system for BrainSmith, including:
- Pure Stevedore-based discovery and management
- Unified decorator system for all plugin types
- Natural access patterns via collections
- Integration with QONNX and FINN frameworks

Usage:
    # Plugin registration
    from brainsmith.plugin.decorators import plugin
    
    @plugin(type="transform", name="MyTransform", stage="topology_opt")
    class MyTransform(Transformation):
        pass
    
    # Plugin access
    from brainsmith.plugins import transforms, kernels, steps
    
    model = transforms.MyTransform()(model)
    layer = kernels.LayerNorm.hls()
    
    # Advanced usage
    from brainsmith.plugin.manager import get_plugin_manager
    from brainsmith.plugin.core.registry import get_plugin_registry
"""

# Import key components for convenience
from .decorators import (
    # Main unified decorator
    plugin,
    # Convenience decorators
    transform,
    kernel, 
    backend,
    step,
    # Configuration
    configure_plugin_decorator,
    PluginDecoratorConfig,
    # Validation
    PluginMetadataValidator,
    ValidationError
)

from .manager import (
    get_plugin_manager,
    PluginManager
)

from .core.registry import (
    get_plugin_registry,
    PluginRegistry
)

from .core.data_models import (
    PluginInfo,
    PluginCatalog,
    PluginType,
    FrameworkType,
    DiscoveryMethod,
    DiscoveryStrategy
)

# Legacy imports removed - these modules now raise ImportError with helpful messages

__all__ = [
    # Main decorators
    "plugin",
    "transform",
    "kernel", 
    "backend",
    "step",
    # Configuration
    "configure_plugin_decorator",
    "PluginDecoratorConfig",
    # Validation
    "PluginMetadataValidator", 
    "ValidationError",
    # Management
    "get_plugin_manager",
    "PluginManager",
    "get_plugin_registry",
    "PluginRegistry",
    # Data models
    "PluginInfo",
    "PluginCatalog", 
    "PluginType",
    "FrameworkType",
    "DiscoveryMethod",
    "DiscoveryStrategy",
    # Legacy imports removed - see migration guide in removed modules
]