"""
Natural Plugin Access Collections

Provides clean, natural access to plugins without boilerplate.
Usage: transforms.ExpandNorms() or transforms.qonnx.RemoveIdentityOps()
"""

import logging
from typing import Dict, List, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .manager import PluginManager, PluginInfo

logger = logging.getLogger(__name__)


class Transform:
    """
    Wrapper for a transform plugin that provides natural calling interface.
    """
    
    def __init__(self, plugin_info: 'PluginInfo', manager: 'PluginManager'):
        self._plugin_info = plugin_info
        self._manager = manager
        self._instance_cache = None
    
    def __call__(self, *args, **kwargs):
        """
        Natural calling interface: transforms.ExpandNorms()(model)
        
        If called with no arguments, returns a transform instance.
        If called with arguments, creates instance and calls apply() method.
        """
        if len(args) == 0 and len(kwargs) == 0:
            # Return transform instance for manual use
            return self._get_instance()
        
        # Assume first argument is model, call apply method
        if len(args) >= 1:
            model = args[0]
            instance = self._get_instance(**kwargs)
            
            # Try different calling patterns
            if hasattr(instance, 'apply'):
                return instance.apply(model)
            elif callable(instance):
                return instance(model)
            else:
                raise RuntimeError(f"Transform {self._plugin_info.name} is not callable")
        
        raise ValueError("Transform must be called with a model as first argument")
    
    def _get_instance(self, **kwargs):
        """Get transform instance, using cache if no kwargs provided."""
        if kwargs or self._instance_cache is None:
            # Load the plugin if not already loaded
            loaded_plugin = self._manager.load_plugin(
                self._plugin_info.name,
                self._plugin_info.plugin_type,
                self._plugin_info.framework
            )
            if not loaded_plugin:
                raise RuntimeError(f"Failed to load transform {self._plugin_info.name}")
            
            instance = loaded_plugin.instantiate(**kwargs)
            
            # Cache only if no custom kwargs
            if not kwargs:
                self._instance_cache = instance
            
            return instance
        
        return self._instance_cache
    
    def __repr__(self):
        return f"Transform({self._plugin_info.name}, {self._plugin_info.framework})"


class Kernel:
    """
    Wrapper for a kernel plugin that provides access to backends.
    """
    
    def __init__(self, plugin_info: 'PluginInfo', manager: 'PluginManager'):
        self._plugin_info = plugin_info
        self._manager = manager
        self._backends = None
    
    @property
    def hls(self):
        """Get HLS backend for this kernel."""
        return self._get_backend("hls")
    
    @property
    def rtl(self):
        """Get RTL backend for this kernel."""
        return self._get_backend("rtl")
    
    def _get_backend(self, backend_type: str):
        """Get specific backend type."""
        if self._backends is None:
            self._load_backends()
        
        if backend_type in self._backends:
            return self._backends[backend_type]
        
        raise AttributeError(f"Kernel {self._plugin_info.name} has no {backend_type} backend")
    
    def _load_backends(self):
        """Load all backends for this kernel."""
        self._backends = {}
        
        # Find all backends for this kernel
        all_plugins = self._manager.list_available("backend")
        kernel_backends = [
            p for p in all_plugins 
            if p.metadata.get("kernel") == self._plugin_info.name
        ]
        
        for backend_info in kernel_backends:
            backend_type = backend_info.metadata.get("backend_type")
            if backend_type:
                self._backends[backend_type] = Transform(backend_info, self._manager)
    
    def __repr__(self):
        return f"Kernel({self._plugin_info.name}, {self._plugin_info.framework})"


class FrameworkTransforms:
    """
    Transform collection for a specific framework.
    Provides natural access: transforms.qonnx.RemoveIdentityOps()
    """
    
    def __init__(self, framework: str, manager: 'PluginManager'):
        self.framework = framework
        self._manager = manager
        self._transform_cache = {}
    
    def __getattr__(self, name: str) -> Transform:
        """Get transform by name from this framework."""
        if name.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        # Check cache first
        if name in self._transform_cache:
            return self._transform_cache[name]
        
        # Get plugin info
        plugin_info = self._manager.get_plugin(name, "transform", self.framework)
        if not plugin_info:
            available = [p.name for p in self._manager.list_available("transform", self.framework)]
            raise AttributeError(
                f"Transform '{name}' not found in {self.framework} framework. "
                f"Available: {available[:5]}{'...' if len(available) > 5 else ''}"
            )
        
        # Create and cache transform wrapper
        transform = Transform(plugin_info, self._manager)
        self._transform_cache[name] = transform
        return transform
    
    def __dir__(self):
        """Support tab completion by listing available transforms."""
        transforms = self._manager.list_available("transform", self.framework)
        return [t.name for t in transforms]
    
    def __repr__(self):
        return f"FrameworkTransforms({self.framework})"


class TransformCollection:
    """
    Main transform collection providing natural access to all transforms.
    
    Usage:
        transforms.ExpandNorms()  # Unique transform, no framework needed
        transforms.qonnx.RemoveIdentityOps()  # Framework-specific
        transforms.finn.ConvertQONNXtoFINN()  # Framework-specific
    """
    
    def __init__(self, manager: 'PluginManager'):
        self._manager = manager
        self._framework_collections = {}
        self._unique_transform_cache = {}
    
    @property
    def qonnx(self) -> FrameworkTransforms:
        """Access QONNX transforms: transforms.qonnx.RemoveIdentityOps()"""
        if 'qonnx' not in self._framework_collections:
            self._framework_collections['qonnx'] = FrameworkTransforms('qonnx', self._manager)
        return self._framework_collections['qonnx']
    
    @property
    def finn(self) -> FrameworkTransforms:
        """Access FINN transforms: transforms.finn.ConvertQONNXtoFINN()"""
        if 'finn' not in self._framework_collections:
            self._framework_collections['finn'] = FrameworkTransforms('finn', self._manager)
        return self._framework_collections['finn']
    
    @property
    def brainsmith(self) -> FrameworkTransforms:
        """Access BrainSmith transforms: transforms.brainsmith.ExpandNorms()"""
        if 'brainsmith' not in self._framework_collections:
            self._framework_collections['brainsmith'] = FrameworkTransforms('brainsmith', self._manager)
        return self._framework_collections['brainsmith']
    
    def __getattr__(self, name: str) -> Transform:
        """
        Direct access for unique transforms: transforms.FoldConstants()
        
        This works when the transform name is unique across all frameworks.
        If the name conflicts, user must use framework prefix.
        """
        if name.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        # Check cache first
        if name in self._unique_transform_cache:
            return self._unique_transform_cache[name]
        
        # Try to get unique transform
        try:
            plugin_info = self._manager.get_plugin(name, "transform")
            if plugin_info:
                transform = Transform(plugin_info, self._manager)
                self._unique_transform_cache[name] = transform
                return transform
        except ValueError as e:
            # This means the name is ambiguous
            if "ambiguous" in str(e).lower():
                raise AttributeError(
                    f"Transform '{name}' is ambiguous. {str(e)} "
                    f"Use transforms.qonnx.{name}() or transforms.finn.{name}() instead."
                )
        
        # Not found
        conflicts = self._manager.analyze_conflicts()
        if name in conflicts:
            frameworks = [p.framework for p in conflicts[name]]
            raise AttributeError(
                f"Transform '{name}' exists in multiple frameworks: {frameworks}. "
                f"Use transforms.{frameworks[0]}.{name}() or similar."
            )
        
        # Suggest similar names
        all_transforms = self._manager.list_available("transform")
        suggestions = [t.name for t in all_transforms if name.lower() in t.name.lower()][:3]
        
        error_msg = f"Transform '{name}' not found."
        if suggestions:
            error_msg += f" Similar transforms: {suggestions}"
        
        raise AttributeError(error_msg)
    
    def __dir__(self):
        """Support tab completion with all available unique transforms."""
        catalog = self._manager.discover_all()
        unique_names = list(catalog.unique_plugins.keys())
        framework_names = ['qonnx', 'finn', 'brainsmith']
        return unique_names + framework_names
    
    def list_available(self, framework: str = None) -> List[str]:
        """List all available transforms, optionally filtered by framework."""
        transforms = self._manager.list_available("transform", framework)
        return [t.name for t in transforms]
    
    def list_conflicts(self) -> Dict[str, List[str]]:
        """List all naming conflicts between frameworks."""
        conflicts = self._manager.analyze_conflicts()
        return {
            name: [p.framework for p in plugin_list]
            for name, plugin_list in conflicts.items()
        }
    
    def __repr__(self):
        return "TransformCollection(qonnx, finn, brainsmith)"


class FrameworkKernels:
    """
    Kernel collection for a specific framework.
    """
    
    def __init__(self, framework: str, manager: 'PluginManager'):
        self.framework = framework
        self._manager = manager
        self._kernel_cache = {}
    
    def __getattr__(self, name: str) -> Kernel:
        """Get kernel by name from this framework."""
        if name.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        # Check cache first
        if name in self._kernel_cache:
            return self._kernel_cache[name]
        
        # Get plugin info
        plugin_info = self._manager.get_plugin(name, "kernel", self.framework)
        if not plugin_info:
            available = [p.name for p in self._manager.list_available("kernel", self.framework)]
            raise AttributeError(
                f"Kernel '{name}' not found in {self.framework} framework. "
                f"Available: {available[:5]}{'...' if len(available) > 5 else ''}"
            )
        
        # Create and cache kernel wrapper
        kernel = Kernel(plugin_info, self._manager)
        self._kernel_cache[name] = kernel
        return kernel
    
    def __dir__(self):
        """Support tab completion by listing available kernels."""
        kernels = self._manager.list_available("kernel", self.framework)
        return [k.name for k in kernels]


class KernelCollection:
    """
    Main kernel collection providing natural access to all kernels.
    
    Usage:
        kernels.LayerNorm.hls()  # Get HLS backend for LayerNorm
        kernels.Softmax.rtl()    # Get RTL backend for Softmax
    """
    
    def __init__(self, manager: 'PluginManager'):
        self._manager = manager
        self._framework_collections = {}
        self._unique_kernel_cache = {}
    
    @property
    def qonnx(self) -> FrameworkKernels:
        """Access QONNX kernels."""
        if 'qonnx' not in self._framework_collections:
            self._framework_collections['qonnx'] = FrameworkKernels('qonnx', self._manager)
        return self._framework_collections['qonnx']
    
    @property
    def finn(self) -> FrameworkKernels:
        """Access FINN kernels."""
        if 'finn' not in self._framework_collections:
            self._framework_collections['finn'] = FrameworkKernels('finn', self._manager)
        return self._framework_collections['finn']
    
    @property
    def brainsmith(self) -> FrameworkKernels:
        """Access BrainSmith kernels."""
        if 'brainsmith' not in self._framework_collections:
            self._framework_collections['brainsmith'] = FrameworkKernels('brainsmith', self._manager)
        return self._framework_collections['brainsmith']
    
    def __getattr__(self, name: str) -> Kernel:
        """Direct access for unique kernels: kernels.LayerNorm.hls()"""
        if name.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        # Check cache first
        if name in self._unique_kernel_cache:
            return self._unique_kernel_cache[name]
        
        # Try to get unique kernel
        try:
            plugin_info = self._manager.get_plugin(name, "kernel")
            if plugin_info:
                kernel = Kernel(plugin_info, self._manager)
                self._unique_kernel_cache[name] = kernel
                return kernel
        except ValueError as e:
            if "ambiguous" in str(e).lower():
                raise AttributeError(f"Kernel '{name}' is ambiguous. {str(e)}")
        
        # Not found
        all_kernels = self._manager.list_available("kernel")
        suggestions = [k.name for k in all_kernels if name.lower() in k.name.lower()][:3]
        
        error_msg = f"Kernel '{name}' not found."
        if suggestions:
            error_msg += f" Similar kernels: {suggestions}"
        
        raise AttributeError(error_msg)
    
    def __dir__(self):
        """Support tab completion with all available unique kernels."""
        catalog = self._manager.discover_all()
        kernel_plugins = catalog.plugins_by_type.get("kernel", [])
        unique_names = [p.name for p in kernel_plugins if p.is_unique]
        framework_names = ['qonnx', 'finn', 'brainsmith']
        return unique_names + framework_names
    
    def list_available(self, framework: str = None) -> List[str]:
        """List all available kernels, optionally filtered by framework."""
        kernels = self._manager.list_available("kernel", framework)
        return [k.name for k in kernels]
    
    def __repr__(self):
        return "KernelCollection(qonnx, finn, brainsmith)"