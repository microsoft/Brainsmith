"""
Kernel Collections

Provides natural access to kernel plugins and their backends.
"""

import logging
from typing import Dict, Optional, List, TYPE_CHECKING

from .base import BaseCollection, FrameworkCollection
from .wrappers import KernelWrapper

if TYPE_CHECKING:
    from ..core.data_models import PluginInfo
    from ..core.registry import PluginRegistry
    from ..core.loader import PluginLoader

logger = logging.getLogger(__name__)


class FrameworkKernelCollection(FrameworkCollection):
    """
    Kernel collection for a specific framework.
    
    Provides natural access: kernels.brainsmith.LayerNorm.hls()
    """
    
    @property
    def plugin_type(self) -> str:
        return "kernel"
    
    def _create_wrapper(self, plugin_info: 'PluginInfo') -> KernelWrapper:
        """Create kernel wrapper."""
        return KernelWrapper(plugin_info, self.loader)
    
    def list_with_backends(self) -> Dict[str, List[str]]:
        """List kernels with their available backends."""
        result = {}
        
        for kernel in self.registry.list_kernels(self.framework):
            backends = self.registry.list_backends(kernel.name)
            backend_types = sorted(set(b.backend_type for b in backends if b.backend_type))
            if backend_types:
                result[kernel.name] = backend_types
        
        return result


class KernelCollection(BaseCollection):
    """
    Main kernel collection providing natural access to all kernels.
    
    Usage:
        kernels = KernelCollection(registry, loader)
        
        # Access kernel and its backends
        layer_norm_hls = kernels.LayerNorm.hls()
        softmax_rtl = kernels.Softmax.rtl()
        
        # Direct kernel instantiation
        custom_kernel = kernels.CustomKernel(param=value)
        
        # Framework-specific (if needed)
        finn_kernel = kernels.finn.SomeKernel()
    """
    
    def __init__(self, registry: 'PluginRegistry', loader: 'PluginLoader'):
        super().__init__(registry, loader)
        self._framework_collections: Dict[str, FrameworkKernelCollection] = {}
    
    @property
    def plugin_type(self) -> str:
        return "kernel"
    
    def _create_wrapper(self, plugin_info: 'PluginInfo') -> KernelWrapper:
        """Create kernel wrapper."""
        return KernelWrapper(plugin_info, self.loader)
    
    @property
    def qonnx(self) -> FrameworkKernelCollection:
        """Access QONNX kernels."""
        if 'qonnx' not in self._framework_collections:
            self._framework_collections['qonnx'] = FrameworkKernelCollection(
                'qonnx', self.registry, self.loader
            )
        return self._framework_collections['qonnx']
    
    @property
    def finn(self) -> FrameworkKernelCollection:
        """Access FINN kernels."""
        if 'finn' not in self._framework_collections:
            self._framework_collections['finn'] = FrameworkKernelCollection(
                'finn', self.registry, self.loader
            )
        return self._framework_collections['finn']
    
    @property
    def brainsmith(self) -> FrameworkKernelCollection:
        """Access BrainSmith kernels."""
        if 'brainsmith' not in self._framework_collections:
            self._framework_collections['brainsmith'] = FrameworkKernelCollection(
                'brainsmith', self.registry, self.loader
            )
        return self._framework_collections['brainsmith']
    
    def __getattr__(self, name: str) -> KernelWrapper:
        """
        Direct access for unique kernels.
        
        This works when the kernel name is unique across all frameworks.
        If the name conflicts, user must use framework prefix.
        """
        if name.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
        
        # Try to get unique kernel
        return self._get_wrapper(name)
    
    def __dir__(self) -> List[str]:
        """Support tab completion with unique kernels and frameworks."""
        # Get unique plugin names
        unique_names = [
            name for name, plugin in self.registry.get_unique_plugins().items()
            if plugin.plugin_type == "kernel"
        ]
        
        # Add framework names
        framework_names = ['qonnx', 'finn', 'brainsmith']
        
        return sorted(unique_names + framework_names)
    
    def list_with_backends(self) -> Dict[str, List[str]]:
        """List all kernels with their available backends."""
        result = {}
        
        for kernel in self.registry.list_kernels():
            backends = self.registry.list_backends(kernel.name)
            backend_types = sorted(set(b.backend_type for b in backends if b.backend_type))
            if backend_types:
                result[kernel.name] = backend_types
        
        return result
    
    def list_by_backend_type(self, backend_type: str) -> List[str]:
        """List kernels that have a specific backend type (hls/rtl)."""
        kernels_with_backend = []
        
        for kernel in self.registry.list_kernels():
            backends = self.registry.list_backends(kernel.name, backend_type)
            if backends:
                kernels_with_backend.append(kernel.name)
        
        return sorted(kernels_with_backend)
    
    def __repr__(self) -> str:
        return "KernelCollection(qonnx, finn, brainsmith)"