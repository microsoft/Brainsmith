"""
Natural Access Collections - Perfect Code Implementation

Direct registry delegation for zero-overhead plugin access.
Preserves exact API patterns while eliminating complex caching.
"""

import logging
from typing import Dict, Any, Optional, Type

logger = logging.getLogger(__name__)


class TransformWrapper:
    """Wrapper that provides natural calling interface for transforms."""
    
    def __init__(self, transform_cls: Type, name: str, registry):
        self.transform_cls = transform_cls
        self.name = name
        self.registry = registry
    
    def __call__(self, *args, **kwargs):
        """Create and return transform instance."""
        return self.transform_cls(*args, **kwargs)
    
    def __repr__(self):
        metadata = self.registry.get_plugin_metadata(self.name)
        return f"<Transform: {self.name} (stage={metadata.get('stage')}, framework={metadata.get('framework')})>"


class FrameworkAccessor:
    """Provides access to transforms from a specific framework."""
    
    def __init__(self, framework: str, registry):
        self.framework = framework
        self.registry = registry
    
    def __getattr__(self, name: str):
        """Get transform by name from this framework."""
        # Direct registry lookup for maximum performance
        transform_cls = self.registry.get_transform(name, framework=self.framework)
        if transform_cls:
            return TransformWrapper(transform_cls, name, self.registry)
        
        # Provide helpful error with available options
        available = list(self.registry.get_framework_transforms(self.framework).keys())
        raise AttributeError(
            f"Transform '{name}' not found in {self.framework} framework. "
            f"Available: {available[:10]}{' ...' if len(available) > 10 else ''}"
        )
    
    def __dir__(self):
        """Support tab completion."""
        return list(self.registry.get_framework_transforms(self.framework).keys())


class TransformCollection:
    """Collection providing natural access to transforms."""
    
    def __init__(self, registry):
        self.registry = registry
        
        # Framework accessors - created on demand
        self._framework_accessors = {}
    
    @property
    def qonnx(self):
        """Access QONNX transforms."""
        if 'qonnx' not in self._framework_accessors:
            self._framework_accessors['qonnx'] = FrameworkAccessor('qonnx', self.registry)
        return self._framework_accessors['qonnx']
    
    @property
    def finn(self):
        """Access FINN transforms."""
        if 'finn' not in self._framework_accessors:
            self._framework_accessors['finn'] = FrameworkAccessor('finn', self.registry)
        return self._framework_accessors['finn']
    
    @property
    def brainsmith(self):
        """Access BrainSmith transforms."""
        if 'brainsmith' not in self._framework_accessors:
            self._framework_accessors['brainsmith'] = FrameworkAccessor('brainsmith', self.registry)
        return self._framework_accessors['brainsmith']
    
    def __getattr__(self, name: str):
        """Direct transform access - searches all frameworks."""
        # Direct registry lookup - maximum performance
        transform_cls = self.registry.get_transform(name)
        if transform_cls:
            return TransformWrapper(transform_cls, name, self.registry)
        
        # Provide helpful error message
        available = list(self.registry.transforms.keys())[:10]  # Show first 10
        raise AttributeError(
            f"Transform '{name}' not found. "
            f"Available: {available}{' ...' if len(self.registry.transforms) > 10 else ''}"
        )
    
    def list_by_stage(self, stage: str):
        """List transforms for a specific stage."""
        return self.registry.list_transforms_by_stage(stage)
    
    def get_by_stage(self, stage: str):
        """Get all transforms for a stage as wrapped instances."""
        stage_transforms = self.registry.transforms_by_stage.get(stage, {})
        return {name: TransformWrapper(cls, name, self.registry) 
                for name, cls in stage_transforms.items()}


class KernelWrapper:
    """Wrapper that provides natural calling interface for kernels."""
    
    def __init__(self, kernel_cls: Type, name: str, registry):
        self.kernel_cls = kernel_cls
        self.name = name
        self.registry = registry
    
    def __call__(self, *args, **kwargs):
        """Create and return default backend instance."""
        backend_cls = self.registry.get_default_backend(self.name)
        if backend_cls:
            return backend_cls(*args, **kwargs)
        else:
            raise RuntimeError(f"No backends registered for kernel {self.name}")
    
    def get_backend(self, backend_name: str, **config):
        """Get specific backend by name."""
        backend_cls = self.registry.get_backend(backend_name)
        if backend_cls:
            return backend_cls(**config) if config else backend_cls()
        else:
            available = self.registry.list_backends_by_kernel(self.name)
            raise AttributeError(
                f"Backend '{backend_name}' not found. "
                f"Available for {self.name}: {available}"
            )
    
    def find_backend(self, **criteria):
        """Find backend by criteria."""
        # Always filter by this kernel
        criteria['kernel'] = self.name
        backend_names = self.registry.find_backends(**criteria)
        if backend_names:
            # Return first match
            return self.registry.get_backend(backend_names[0])
        else:
            raise AttributeError(
                f"No backend found for {self.name} matching criteria: {criteria}"
            )
    
    def list_backends(self):
        """List all backend names for this kernel."""
        return self.registry.list_backends_by_kernel(self.name)
    
    def list_backends_with_metadata(self):
        """List all backends with their metadata."""
        backend_names = self.registry.list_backends_by_kernel(self.name)
        return [
            {
                'name': name,
                'metadata': self.registry.get_plugin_metadata(name)
            }
            for name in backend_names
        ]
    
    # Convenience methods for common languages
    def hls(self, **config):
        """Get HLS backend (or first HLS backend if multiple)."""
        backend = self.find_backend(language='hls')
        return backend(**config) if config else backend()
    
    def rtl(self, **config):
        """Get RTL/Verilog backend."""
        # Try verilog first, then rtl for backward compatibility
        try:
            backend = self.find_backend(language='verilog')
        except AttributeError:
            backend = self.find_backend(language='rtl')
        return backend(**config) if config else backend()
    
    def __repr__(self):
        available_backends = self.list_backends()
        return f"<Kernel: {self.name} (backends: {available_backends})>"


class KernelCollection:
    """Collection providing natural access to kernels and their backends."""
    
    def __init__(self, registry):
        self.registry = registry
    
    def __getattr__(self, kernel_name: str):
        """Get kernel accessor."""
        kernel_cls = self.registry.get_kernel(kernel_name)
        if kernel_cls:
            return KernelWrapper(kernel_cls, kernel_name, self.registry)
        else:
            available = list(self.registry.kernels.keys())
            raise AttributeError(
                f"Kernel '{kernel_name}' not found. Available: {available}"
            )
    
    def list_all(self):
        """List all available kernels."""
        return list(self.registry.kernels.keys())


class BackendWrapper:
    """Wrapper for backend instances."""
    
    def __init__(self, backend_cls: Type, name: str, registry):
        self.backend_cls = backend_cls
        self.name = name
        self.registry = registry
    
    def __call__(self, *args, **kwargs):
        """Create and return backend instance."""
        return self.backend_cls(*args, **kwargs)
    
    def __repr__(self):
        metadata = self.registry.get_plugin_metadata(self.name)
        language = metadata.get('language', metadata.get('backend_type', 'unknown'))
        return f"<Backend: {self.name} (kernel={metadata.get('kernel')}, language={language})>"


class BackendCollection:
    """Collection providing access to backend implementations."""
    
    def __init__(self, registry):
        self.registry = registry
    
    def __getattr__(self, backend_name: str):
        """Get backend by name."""
        if backend_name in self.registry.backends:
            backend_cls = self.registry.backends[backend_name]
            return BackendWrapper(backend_cls, backend_name, self.registry)
        else:
            available = list(self.registry.backends.keys())
            raise AttributeError(
                f"Backend '{backend_name}' not found. Available: {available}"
            )
    
    def list_all(self):
        """List all available backends."""
        return list(self.registry.backends.keys())


class StepWrapper:
    """Wrapper for step functions."""
    
    def __init__(self, step_cls: Type, name: str, registry):
        self.step_cls = step_cls
        self.name = name
        self.registry = registry
    
    def __call__(self, *args, **kwargs):
        """Execute step function."""
        return self.step_cls(*args, **kwargs)
    
    def __repr__(self):
        metadata = self.registry.get_plugin_metadata(self.name)
        return f"<Step: {self.name} (category={metadata.get('category')})>"


class CategoryAccessor:
    """Provides access to steps from a specific category."""
    
    def __init__(self, category: str, collection):
        self.category = category
        self.collection = collection
    
    def __getattr__(self, name: str):
        """Get step by name from this category."""
        # Look for steps in this category
        all_steps = self.collection.registry.list_all_plugins()
        for plugin in all_steps:
            if (plugin['metadata'].get('type') in ['step', 'kernel_inference'] and
                plugin['metadata'].get('category') == self.category and
                plugin['name'] == name):
                return StepWrapper(plugin['class'], name, self.collection.registry)
        
        # Show available steps in this category
        available = []
        for plugin in all_steps:
            if (plugin['metadata'].get('type') in ['step', 'kernel_inference'] and
                plugin['metadata'].get('category') == self.category):
                available.append(plugin['name'])
        
        raise AttributeError(
            f"Step '{name}' not found in category '{self.category}'. "
            f"Available: {available}"
        )
    
    def __dir__(self):
        """Support tab completion."""
        all_steps = self.collection.registry.list_all_plugins()
        names = []
        for plugin in all_steps:
            if (plugin['metadata'].get('type') in ['step', 'kernel_inference'] and
                plugin['metadata'].get('category') == self.category):
                names.append(plugin['name'])
        return sorted(names)


class StepCollection:
    """Collection providing access to step functions organized by category."""
    
    def __init__(self, registry):
        self.registry = registry
        self._category_accessors = {}
    
    def __getattr__(self, name: str):
        """Get step function or category accessor by name."""
        # Check if it's a category name first
        all_steps = self.registry.list_all_plugins()
        categories = set()
        for plugin in all_steps:
            if plugin['metadata'].get('type') in ['step', 'kernel_inference']:
                category = plugin['metadata'].get('category', 'general')
                categories.add(category)
        
        if name in categories:
            if name not in self._category_accessors:
                self._category_accessors[name] = CategoryAccessor(name, self)
            return self._category_accessors[name]
        
        # Look for step by name directly
        for plugin in all_steps:
            if (plugin['metadata'].get('type') in ['step', 'kernel_inference'] and
                plugin['name'] == name):
                return StepWrapper(plugin['class'], name, self.registry)
        
        # Show available options
        available_steps = []
        for plugin in all_steps:
            if plugin['metadata'].get('type') in ['step', 'kernel_inference']:
                available_steps.append(plugin['name'])
        
        raise AttributeError(
            f"Step '{name}' not found. Available: {available_steps[:10]}{' ...' if len(available_steps) > 10 else ''}"
        )
    
    def __dir__(self):
        """Support tab completion."""
        all_steps = self.registry.list_all_plugins()
        names = set()
        
        # Add category names
        for plugin in all_steps:
            if plugin['metadata'].get('type') in ['step', 'kernel_inference']:
                category = plugin['metadata'].get('category', 'general')
                names.add(category)
                # Also add step names directly
                names.add(plugin['name'])
        
        return sorted(names)


def create_collections(registry=None):
    """
    Create collection instances with direct registry delegation.
    
    Perfect Code approach: Collections are thin wrappers over registry,
    no caching needed since registry lookups are already optimized.
    """
    from .registry import get_registry
    
    reg = registry or get_registry()
    return {
        'transforms': TransformCollection(reg),
        'kernels': KernelCollection(reg),
        'backends': BackendCollection(reg),
        'steps': StepCollection(reg)
    }