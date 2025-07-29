# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Plugin System
"""
from typing import Dict, Any, Type, List, Optional, Tuple
import logging
import sys

logger = logging.getLogger(__name__)

def kernel_inference(**metadata):
    """Decorator for kernel inference transforms (FINN compatibility)."""
    return plugin('transform', **metadata)

class Registry:
    def __init__(self):
        self._plugins: Dict[str, Dict[str, Tuple[Type, Dict[str, Any]]]] = {
            'transform': {}, 'kernel': {}, 'backend': {}, 'step': {}
        }
    
    def register(self, plugin_type: str, name: str, cls: Type, 
                 framework: str = 'brainsmith', **metadata) -> None:
        """Register a plugin with optional framework namespace."""
        key = f"{framework}:{name}" if framework != 'brainsmith' else name
        self._plugins[plugin_type][key] = (cls, {**metadata, 'framework': framework})
    
    def _ensure_external_plugins(self):
        """Ensure external plugins are loaded."""
        if 'brainsmith.core.plugins.framework_adapters' in sys.modules:
            from . import framework_adapters
            framework_adapters.ensure_initialized()
    
    def get(self, plugin_type: str, name: str) -> Type:
        """Get plugin by name (with or without framework prefix).
        """
        self._ensure_external_plugins()
        
        # Direct lookup first
        entry = self._plugins[plugin_type].get(name)
        if entry:
            return entry[0]
        
        # If no colon, try each framework
        if ':' not in name:
            # First try without any prefix (for brainsmith plugins registered without framework)
            for key, value in self._plugins[plugin_type].items():
                if key == name:
                    return value[0]
            
            # Then try with brainsmith prefix
            for key, value in self._plugins[plugin_type].items():
                if key == f'brainsmith:{name}':
                    return value[0]
            
            # Finally try other frameworks
            for key, value in self._plugins[plugin_type].items():
                if ':' in key and key.endswith(f':{name}'):
                    return value[0]
        
        # Plugin not found - fail fast
        available = list(self._plugins[plugin_type].keys())
        raise KeyError(
            f"Plugin {plugin_type}:{name} not found. "
            f"Available ({len(available)}): {available[:10] if available else 'none'}"
        )
    
    
    def find(self, plugin_type: str, **criteria) -> List[Type]:
        """Find plugins matching criteria."""
        results = []
        for name, (cls, metadata) in self._plugins[plugin_type].items():
            if all(metadata.get(k) == v for k, v in criteria.items()):
                results.append(cls)
        return results
    
    def all(self, plugin_type: str) -> Dict[str, Type]:
        """Get all plugins of a type."""
        return {name: cls for name, (cls, _) in self._plugins[plugin_type].items()}
    
    def reset(self) -> None:
        """Reset registry and reload all plugins.
        
        This is primarily for testing to ensure a clean state.
        """
        # Clear all plugins
        self._plugins = {
            'transform': {}, 'kernel': {}, 'backend': {}, 'step': {}
        }
        
        # Force reload of plugin modules to re-register everything
        import importlib
        import sys
        
        # List of modules that register plugins
        # Need to reload child modules before parent modules
        plugin_modules = [
            # Transform modules
            'brainsmith.transforms.cleanup.remove_identity',
            'brainsmith.transforms.dataflow_opt.infer_finn_loop_op',
            'brainsmith.transforms.kernel_opt.set_pumped_compute',
            'brainsmith.transforms.kernel_opt.temp_shuffle_fixer',
            'brainsmith.transforms.metadata.extract_shell_integration_metadata',
            'brainsmith.transforms.metadata.ensure_custom_opset_imports',
            'brainsmith.transforms.topology_opt.expand_norms',
            'brainsmith.transforms',
            # Kernel modules
            'brainsmith.kernels.crop.crop',
            'brainsmith.kernels.crop.crop_hls',
            'brainsmith.kernels.layernorm.layernorm',
            'brainsmith.kernels.layernorm.layernorm_hls',
            'brainsmith.kernels.shuffle.shuffle',
            'brainsmith.kernels.shuffle.shuffle_hls',
            'brainsmith.kernels.softmax.hwsoftmax',
            'brainsmith.kernels.softmax.hwsoftmax_hls',
            'brainsmith.kernels',
            # Step modules
            'brainsmith.steps.core_steps',
            'brainsmith.steps.bert_custom_steps',
            'brainsmith.steps.kernel_inference',
            'brainsmith.steps',
            # Framework adapters
            'brainsmith.core.plugins.framework_adapters'
        ]
        
        for module_name in plugin_modules:
            if module_name in sys.modules:
                # Reload the module to trigger re-registration
                module = sys.modules[module_name]
                importlib.reload(module)
                
        logger.debug("Registry reset and plugins reloaded")
    

# Singleton
_registry = Registry()

# Public API
def get_registry() -> Registry:
    return _registry

# Convenience functions
def get_transform(name: str) -> Type:
    return _registry.get('transform', name)

def get_kernel(name: str) -> Type:
    return _registry.get('kernel', name)

def get_backend(name: str) -> Type:
    return _registry.get('backend', name)

def get_step(name: str) -> Type:
    return _registry.get('step', name)


# Registration decorators
def plugin(plugin_type: str, **metadata):
    def decorator(cls: Type) -> Type:
        framework = metadata.pop('framework', 'brainsmith')
        name = metadata.pop('name', cls.__name__)
        _registry.register(plugin_type, name, cls, framework, **metadata)
        return cls
    return decorator

transform = lambda **kw: plugin('transform', **kw)
kernel = lambda **kw: plugin('kernel', **kw)
backend = lambda **kw: plugin('backend', **kw)
step = lambda **kw: plugin('step', **kw)
kernel_inference = lambda **kw: plugin('transform', kernel_inference=True, **kw)

# List functions
def list_transforms() -> List[str]:
    """List all transform names."""
    return list(_registry._plugins['transform'].keys())

def list_kernels() -> List[str]:
    """List all kernel names."""
    return list(_registry._plugins['kernel'].keys())

def list_backends() -> List[str]:
    """List all backend names."""
    return list(_registry._plugins['backend'].keys())

def list_steps() -> List[str]:
    """List all step names."""
    return list(_registry._plugins['step'].keys())

# "Has" functions
def has_transform(name: str) -> bool:
    """Check if transform exists."""
    try:
        _registry.get('transform', name)
        return True
    except KeyError:
        return False

def has_kernel(name: str) -> bool:
    """Check if kernel exists."""
    try:
        _registry.get('kernel', name)
        return True
    except KeyError:
        return False

def has_backend(name: str) -> bool:
    """Check if backend exists."""
    try:
        _registry.get('backend', name)
        return True
    except KeyError:
        return False

def has_step(name: str) -> bool:
    """Check if step exists."""
    try:
        _registry.get('step', name)
        return True
    except KeyError:
        return False

# Metadata query functions
def _get_names_for_classes(plugin_type: str, classes: List[Type]) -> List[str]:
    """Convert plugin classes back to their registered names."""
    names = []
    for cls in classes:
        for key, (plugin_cls, metadata) in _registry._plugins[plugin_type].items():
            if plugin_cls == cls:
                names.append(key)
                break
    return names

def get_transforms_by_metadata(**criteria) -> List[str]:
    """Get transforms matching metadata criteria."""
    return _get_names_for_classes('transform', _registry.find('transform', **criteria))

def get_kernels_by_metadata(**criteria) -> List[str]:
    """Get kernels matching metadata criteria."""
    return _get_names_for_classes('kernel', _registry.find('kernel', **criteria))

def get_backends_by_metadata(**criteria) -> List[str]:
    """Get backends matching metadata criteria."""
    return _get_names_for_classes('backend', _registry.find('backend', **criteria))

def get_steps_by_metadata(**criteria) -> List[str]:
    """Get steps matching metadata criteria."""
    return _get_names_for_classes('step', _registry.find('step', **criteria))

# Blueprint compatibility functions (used by explorer)
def list_backends_by_kernel(kernel: str) -> List[str]:
    """List all backends for a given kernel."""
    backends = []
    for name, (cls, metadata) in _registry._plugins['backend'].items():
        if metadata.get('kernel') == kernel:
            backends.append(name.split(':')[-1])
    return backends

def get_default_backend(kernel: str) -> Optional[str]:
    """Get the default backend for a kernel."""
    for name, (cls, metadata) in _registry._plugins['backend'].items():
        if metadata.get('kernel') == kernel and metadata.get('default'):
            return name.split(':')[-1]
    backends = list_backends_by_kernel(kernel)
    return backends[0] if backends else None