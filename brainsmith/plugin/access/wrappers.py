"""
Plugin Wrappers

Provides natural calling interfaces for different plugin types.
These wrappers make plugins feel like native Python objects.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Callable, List, TYPE_CHECKING
import inspect

if TYPE_CHECKING:
    from ..core.data_models import PluginInfo
    from ..core.loader import PluginLoader

logger = logging.getLogger(__name__)


class PluginWrapper(ABC):
    """
    Base class for all plugin wrappers.
    
    Provides common functionality for wrapping plugins with
    natural Python interfaces.
    """
    
    def __init__(self, plugin_info: 'PluginInfo', loader: 'PluginLoader'):
        self.plugin_info = plugin_info
        self.loader = loader
        self._instance_cache = None
        self._instance_kwargs_hash = None
    
    @property
    def name(self) -> str:
        """Get the plugin name."""
        return self.plugin_info.name
    
    @property
    def framework(self) -> str:
        """Get the plugin framework."""
        return self.plugin_info.framework
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get plugin metadata."""
        return self.plugin_info.metadata.copy()
    
    def _get_instance(self, **kwargs) -> Any:
        """
        Get plugin instance with intelligent caching.
        
        Caches instances for stateless plugins when no kwargs provided.
        """
        # Check if we can use cached instance
        kwargs_hash = hash(frozenset(kwargs.items())) if kwargs else None
        
        if (self._instance_cache is not None and 
            self._instance_kwargs_hash == kwargs_hash):
            return self._instance_cache
        
        # Create new instance via loader
        instance = self.loader.instantiate(
            self.plugin_info.name,
            self.plugin_info.plugin_type,
            self.plugin_info.framework,
            **kwargs
        )
        
        if not instance:
            raise RuntimeError(
                f"Failed to instantiate {self.plugin_info.plugin_type} "
                f"'{self.plugin_info.name}'"
            )
        
        # Cache if appropriate (no kwargs for stateless plugins)
        if not kwargs and self._is_cacheable():
            self._instance_cache = instance
            self._instance_kwargs_hash = kwargs_hash
        
        return instance
    
    def _is_cacheable(self) -> bool:
        """Determine if this plugin instance can be cached."""
        # Use loader's stateless detection
        return self.loader._is_stateless(self.plugin_info)
    
    @abstractmethod
    def __repr__(self) -> str:
        """String representation."""
        pass


class TransformWrapper(PluginWrapper):
    """
    Wrapper for transform plugins.
    
    Provides natural calling interface:
        transform = transforms.ExpandNorms()
        model = transform(model)
        
    Or direct application:
        model = transforms.ExpandNorms()(model)
    """
    
    def __call__(self, *args, **kwargs):
        """
        Natural calling interface for transforms.
        
        Behavior:
        - No args: Returns transform instance
        - With model arg: Applies transform and returns result
        """
        if len(args) == 0 and len(kwargs) == 0:
            # Return transform instance for manual use
            return self._get_instance()
        
        # Check if this is instantiation kwargs vs model + kwargs
        if len(args) == 0:
            # Just kwargs, return configured instance
            return self._get_instance(**kwargs)
        
        # First arg is model, remaining kwargs are for instantiation
        model = args[0]
        instance = self._get_instance(**kwargs)
        
        # Apply transform using appropriate method
        if hasattr(instance, 'apply'):
            return instance.apply(model)
        elif callable(instance):
            return instance(model)
        else:
            raise RuntimeError(
                f"Transform '{self.name}' has no apply() method or __call__"
            )
    
    @property
    def stage(self) -> Optional[str]:
        """Get the compilation stage for this transform."""
        return self.plugin_info.stage
    
    @property
    def description(self) -> Optional[str]:
        """Get transform description."""
        return self.plugin_info.description
    
    def __repr__(self) -> str:
        return f"Transform({self.name}, {self.framework})"


class KernelWrapper(PluginWrapper):
    """
    Wrapper for kernel plugins.
    
    Provides access to different backends:
        kernel = kernels.LayerNorm
        hls_impl = kernel.hls()
        rtl_impl = kernel.rtl()
    """
    
    def __init__(self, plugin_info: 'PluginInfo', loader: 'PluginLoader'):
        super().__init__(plugin_info, loader)
        self._backend_cache: Dict[str, BackendWrapper] = {}
    
    @property
    def hls(self) -> 'BackendWrapper':
        """Get HLS backend for this kernel."""
        return self._get_backend("hls")
    
    @property
    def rtl(self) -> 'BackendWrapper':
        """Get RTL backend for this kernel."""
        return self._get_backend("rtl")
    
    def _get_backend(self, backend_type: str) -> 'BackendWrapper':
        """Get specific backend type."""
        if backend_type in self._backend_cache:
            return self._backend_cache[backend_type]
        
        # Find backend in registry
        from ..core.registry import get_plugin_registry
        registry = get_plugin_registry()
        
        backends = registry.list_backends(
            kernel=self.name,
            backend_type=backend_type
        )
        
        if not backends:
            raise AttributeError(
                f"Kernel '{self.name}' has no {backend_type} backend. "
                f"Available backends: {self._list_available_backends()}"
            )
        
        # Use first matching backend
        backend_info = backends[0]
        backend_wrapper = BackendWrapper(backend_info, self.loader)
        self._backend_cache[backend_type] = backend_wrapper
        
        return backend_wrapper
    
    def _list_available_backends(self) -> List[str]:
        """List available backend types for this kernel."""
        from ..core.registry import get_plugin_registry
        registry = get_plugin_registry()
        
        backends = registry.list_backends(kernel=self.name)
        return sorted(set(b.backend_type for b in backends if b.backend_type))
    
    def __call__(self, **kwargs):
        """Direct instantiation of kernel."""
        return self._get_instance(**kwargs)
    
    def __repr__(self) -> str:
        return f"Kernel({self.name}, {self.framework})"


class BackendWrapper(PluginWrapper):
    """
    Wrapper for backend plugins.
    
    Provides natural calling interface for backends:
        backend = kernels.LayerNorm.hls
        instance = backend(param1=value1)
    """
    
    def __call__(self, **kwargs):
        """Instantiate backend with parameters."""
        return self._get_instance(**kwargs)
    
    @property
    def backend_type(self) -> Optional[str]:
        """Get backend type (hls/rtl)."""
        return self.plugin_info.backend_type
    
    @property
    def kernel(self) -> Optional[str]:
        """Get parent kernel name."""
        return self.plugin_info.kernel
    
    def __repr__(self) -> str:
        return f"Backend({self.name}, {self.backend_type}, {self.framework})"


class StepWrapper(PluginWrapper):
    """
    Wrapper for FINN build step plugins.
    
    Steps are functions, so the wrapper acts as a callable:
        step = steps.shell_metadata_handover
        model = step(model, cfg)
    """
    
    def __call__(self, *args, **kwargs):
        """Execute the step function."""
        # Steps are functions, not classes
        step_func = self.plugin_info.plugin_class
        
        if not callable(step_func):
            raise RuntimeError(f"Step '{self.name}' is not callable")
        
        return step_func(*args, **kwargs)
    
    @property
    def category(self) -> Optional[str]:
        """Get step category."""
        return self.metadata.get('category', 'unknown')
    
    @property
    def dependencies(self) -> List[str]:
        """Get step dependencies."""
        return self.metadata.get('dependencies', [])
    
    @property
    def description(self) -> Optional[str]:
        """Get step description."""
        return self.plugin_info.description or self.metadata.get('description')
    
    def __repr__(self) -> str:
        return f"Step({self.name}, category={self.category})"