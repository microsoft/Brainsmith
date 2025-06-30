"""
Transform Collections

Provides natural access to transform plugins across all frameworks.
"""

import logging
from typing import Dict, Optional, List, TYPE_CHECKING

from .base import BaseCollection, FrameworkCollection
from .wrappers import TransformWrapper

if TYPE_CHECKING:
    from ..core.data_models import PluginInfo
    from ..core.registry import PluginRegistry
    from ..core.loader import PluginLoader

logger = logging.getLogger(__name__)


class FrameworkTransformCollection(FrameworkCollection):
    """
    Transform collection for a specific framework.
    
    Provides natural access: transforms.qonnx.RemoveIdentityOps()
    """
    
    @property
    def plugin_type(self) -> str:
        return "transform"
    
    def _create_wrapper(self, plugin_info: 'PluginInfo') -> TransformWrapper:
        """Create transform wrapper."""
        return TransformWrapper(plugin_info, self.loader)
    
    def list_by_stage(self, stage: str) -> List[str]:
        """List transforms for a specific compilation stage."""
        transforms = self.registry.list_transforms(self.framework, stage)
        return sorted([t.name for t in transforms])
    
    @property
    def stages(self) -> List[str]:
        """Get all compilation stages used by transforms in this framework."""
        transforms = self.registry.list_transforms(self.framework)
        stages = set(t.stage for t in transforms if t.stage)
        return sorted(stages)


class TransformCollection(BaseCollection):
    """
    Main transform collection providing natural access to all transforms.
    
    Usage:
        transforms = TransformCollection(registry, loader)
        
        # Unique transform (no framework prefix needed)
        model = transforms.ExpandNorms()(model)
        
        # Framework-specific (for conflicts or clarity)
        model = transforms.qonnx.RemoveIdentityOps()(model)
        model = transforms.finn.ConvertToHWLayers()(model)
    """
    
    def __init__(self, registry: 'PluginRegistry', loader: 'PluginLoader'):
        super().__init__(registry, loader)
        self._framework_collections: Dict[str, FrameworkTransformCollection] = {}
    
    @property
    def plugin_type(self) -> str:
        return "transform"
    
    def _create_wrapper(self, plugin_info: 'PluginInfo') -> TransformWrapper:
        """Create transform wrapper."""
        return TransformWrapper(plugin_info, self.loader)
    
    @property
    def qonnx(self) -> FrameworkTransformCollection:
        """Access QONNX transforms."""
        if 'qonnx' not in self._framework_collections:
            self._framework_collections['qonnx'] = FrameworkTransformCollection(
                'qonnx', self.registry, self.loader
            )
        return self._framework_collections['qonnx']
    
    @property
    def finn(self) -> FrameworkTransformCollection:
        """Access FINN transforms."""
        if 'finn' not in self._framework_collections:
            self._framework_collections['finn'] = FrameworkTransformCollection(
                'finn', self.registry, self.loader
            )
        return self._framework_collections['finn']
    
    @property
    def brainsmith(self) -> FrameworkTransformCollection:
        """Access BrainSmith transforms."""
        if 'brainsmith' not in self._framework_collections:
            self._framework_collections['brainsmith'] = FrameworkTransformCollection(
                'brainsmith', self.registry, self.loader
            )
        return self._framework_collections['brainsmith']
    
    def __getattr__(self, name: str) -> TransformWrapper:
        """
        Direct access for unique transforms.
        
        This works when the transform name is unique across all frameworks.
        If the name conflicts, user must use framework prefix.
        """
        if name.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
        
        # Try to get unique transform
        return self._get_wrapper(name)
    
    def __dir__(self) -> List[str]:
        """Support tab completion with unique transforms and frameworks."""
        # Get unique plugin names
        unique_names = [
            name for name, plugin in self.registry.get_unique_plugins().items()
            if plugin.plugin_type == "transform"
        ]
        
        # Add framework names
        framework_names = ['qonnx', 'finn', 'brainsmith']
        
        return sorted(unique_names + framework_names)
    
    def list_conflicts(self) -> Dict[str, List[str]]:
        """List all naming conflicts between frameworks."""
        conflicts = {}
        
        for name, plugins in self.registry.get_conflicts().items():
            # Filter for transforms only
            transform_plugins = [p for p in plugins if p.plugin_type == "transform"]
            if len(transform_plugins) > 1:
                conflicts[name] = [p.framework for p in transform_plugins]
        
        return conflicts
    
    def list_by_stage(self, stage: str, framework: Optional[str] = None) -> List[str]:
        """List all transforms for a specific compilation stage."""
        transforms = self.registry.list_transforms(framework, stage)
        return sorted([t.name for t in transforms])
    
    @property
    def stages(self) -> List[str]:
        """Get all compilation stages across all frameworks."""
        all_transforms = self.registry.list_transforms()
        stages = set(t.stage for t in all_transforms if t.stage)
        return sorted(stages)
    
    def __repr__(self) -> str:
        return "TransformCollection(qonnx, finn, brainsmith)"