"""
High-Performance Plugin Registry - Perfect Code Implementation

Direct dictionary lookups with pre-computed indexes for maximum performance.
Eliminates discovery overhead through decoration-time registration.
"""

import logging
from typing import Dict, List, Optional, Type, Any

logger = logging.getLogger(__name__)


class BrainsmithPluginRegistry:
    """
    High-performance plugin registry with pre-computed indexes.
    
    Core principle: Register at decoration time, lookup with direct dict access.
    No discovery, no caching - the registry IS the optimized data structure.
    """
    
    def __init__(self):
        # Main registries - direct dict access for maximum performance
        self.transforms: Dict[str, Type] = {}           # name -> class
        self.kernels: Dict[str, Type] = {}             # name -> class  
        self.backends: Dict[str, Type] = {}            # backend_name -> class
        self.steps: Dict[str, Type] = {}               # name -> class (NEW: separate from transforms)
        
        # Performance indexes - pre-computed for fast queries
        self.transforms_by_stage: Dict[str, Dict[str, Type]] = {}     # stage -> {name -> class}
        self.backends_by_kernel: Dict[str, List[str]] = {}            # kernel -> [backend_names]
        self.steps_by_category: Dict[str, Dict[str, Type]] = {}       # category -> {name -> class}
        
        # Framework indexes for ALL plugin types
        self.framework_transforms: Dict[str, Dict[str, Type]] = {}    # framework -> {name -> class}
        self.framework_kernels: Dict[str, Dict[str, Type]] = {}       # framework -> {name -> class}
        self.framework_backends: Dict[str, Dict[str, Type]] = {}      # framework -> {name -> class}
        self.framework_steps: Dict[str, Dict[str, Type]] = {}         # framework -> {name -> class}
        
        # Metadata storage for rich plugin information
        self.plugin_metadata: Dict[str, Dict[str, Any]] = {}         # name -> metadata dict
        
        # Additional indexes for efficient backend queries
        self.default_backends: Dict[str, str] = {}                   # kernel -> default_backend_name
        self.backend_indexes: Dict[str, Dict[str, List[str]]] = {   # attribute -> {value -> [backend_names]}
            'language': {},
            'optimization': {},
            'architecture': {},
            # More indexes added dynamically
        }
    
    def register_transform(self, name: str, transform_class: Type, stage: Optional[str] = None, 
                          framework: str = 'brainsmith', **metadata) -> None:
        """Register transform with automatic stage and framework indexing."""
        # Store in main registry
        self.transforms[name] = transform_class
        
        # Index by stage for blueprint performance
        if stage:
            if stage not in self.transforms_by_stage:
                self.transforms_by_stage[stage] = {}
            self.transforms_by_stage[stage][name] = transform_class
        
        # Index by framework for organized access
        if framework not in self.framework_transforms:
            self.framework_transforms[framework] = {}
        self.framework_transforms[framework][name] = transform_class
        
        # Store metadata
        self.plugin_metadata[name] = {
            'type': 'transform',
            'stage': stage,
            'framework': framework,
            **metadata
        }
        
        logger.debug(f"Registered transform: {name} (stage={stage}, framework={framework})")
    
    def register_kernel(self, name: str, kernel_class: Type, framework: str = 'brainsmith', **metadata) -> None:
        """Register kernel with framework indexing."""
        # Store in main registry
        self.kernels[name] = kernel_class
        
        # Index by framework
        if framework not in self.framework_kernels:
            self.framework_kernels[framework] = {}
        self.framework_kernels[framework][name] = kernel_class
        
        # Store metadata
        self.plugin_metadata[name] = {
            'type': 'kernel',
            'framework': framework,
            **metadata
        }
        
        logger.debug(f"Registered kernel: {name} ({framework})")
    
    def register_backend(self, name: str, backend_class: Type, kernel: str, framework: str = 'brainsmith', **metadata) -> None:
        """Register backend with automatic kernel and framework indexing."""
        # Store in main registry by name
        self.backends[name] = backend_class
        
        # Index by kernel for fast backend lookup
        if kernel not in self.backends_by_kernel:
            self.backends_by_kernel[kernel] = []
        self.backends_by_kernel[kernel].append(name)
        
        # Index by framework
        if framework not in self.framework_backends:
            self.framework_backends[framework] = {}
        self.framework_backends[framework][name] = backend_class
        
        # Handle default backend
        if metadata.get('default', False):
            self.default_backends[kernel] = name
        
        # Update attribute indexes for efficient queries
        for attr, index_dict in self.backend_indexes.items():
            if attr in metadata:
                value = metadata[attr]
                if value not in index_dict:
                    index_dict[value] = []
                index_dict[value].append(name)
        
        
        # Store all metadata
        self.plugin_metadata[name] = {
            'type': 'backend',
            'kernel': kernel,
            'framework': framework,
            **metadata
        }
        
        logger.debug(f"Registered backend: {name} for {kernel} ({framework}, language={metadata.get('language', 'unknown')})")
    
    def register_step(self, name: str, step_class: Type, category: Optional[str] = None,
                     framework: str = 'brainsmith', **metadata) -> None:
        """Register step with category and framework indexing."""
        # Store in main registry
        self.steps[name] = step_class
        
        # Index by category
        if category:
            if category not in self.steps_by_category:
                self.steps_by_category[category] = {}
            self.steps_by_category[category][name] = step_class
        
        # Index by framework
        if framework not in self.framework_steps:
            self.framework_steps[framework] = {}
        self.framework_steps[framework][name] = step_class
        
        # Store metadata
        self.plugin_metadata[name] = {
            'type': 'step',
            'category': category,
            'framework': framework,
            **metadata
        }
        
        logger.debug(f"Registered step: {name} (category={category}, framework={framework})")
    
    # Fast lookup methods - direct dict access
    def get_transform(self, name: str, stage: Optional[str] = None, framework: Optional[str] = None) -> Optional[Type]:
        """Get transform with optional stage/framework filter."""
        if framework and framework in self.framework_transforms:
            return self.framework_transforms[framework].get(name)
        if stage and stage in self.transforms_by_stage:
            return self.transforms_by_stage[stage].get(name)
        return self.transforms.get(name)
    
    def get_kernel(self, name: str, framework: Optional[str] = None) -> Optional[Type]:
        """Get kernel by name with optional framework filter."""
        if framework and framework in self.framework_kernels:
            return self.framework_kernels[framework].get(name)
        return self.kernels.get(name)
    
    def get_backend(self, name: str, framework: Optional[str] = None) -> Optional[Type]:
        """Get backend by name with optional framework filter."""
        if framework and framework in self.framework_backends:
            return self.framework_backends[framework].get(name)
        return self.backends.get(name)
    
    def get_step(self, name: str, category: Optional[str] = None, framework: Optional[str] = None) -> Optional[Type]:
        """Get step with optional category/framework filter."""
        if framework and framework in self.framework_steps:
            return self.framework_steps[framework].get(name)
        if category and category in self.steps_by_category:
            return self.steps_by_category[category].get(name)
        return self.steps.get(name)
    
    def get_default_backend(self, kernel: str) -> Optional[Type]:
        """Get default backend for kernel."""
        backend_name = self.default_backends.get(kernel)
        if backend_name:
            return self.backends.get(backend_name)
        # Fallback to first backend if no default specified
        backend_names = self.backends_by_kernel.get(kernel, [])
        if backend_names:
            return self.backends.get(backend_names[0])
        return None
    
    def find_backends(self, **criteria) -> List[str]:
        """Find backends matching all criteria using indexes."""
        results = None
        
        # Check kernel criterion specially
        if 'kernel' in criteria:
            kernel = criteria.pop('kernel')
            results = set(self.backends_by_kernel.get(kernel, []))
            if not results:
                return []
        
        # Use indexes for each remaining criterion
        for attr, value in criteria.items():
            if attr in self.backend_indexes:
                candidates = set(self.backend_indexes[attr].get(value, []))
            else:
                # Fallback to metadata scan for non-indexed attributes
                candidates = {
                    name for name, meta in self.plugin_metadata.items()
                    if meta.get('type') == 'backend' and meta.get(attr) == value
                }
            
            # Intersect with previous results
            if results is None:
                results = candidates
            else:
                results &= candidates
                
            # Early exit if no results
            if not results:
                return []
        
        # If no criteria, return all backends
        if results is None:
            results = set(self.backends.keys())
            
        return list(results)
    
    # Blueprint optimization methods
    def list_transforms_by_stage(self, stage: str) -> List[str]:
        """List all transform names for a stage (for blueprint loading)."""
        return list(self.transforms_by_stage.get(stage, {}).keys())
    
    def list_backends_by_kernel(self, kernel: str) -> List[str]:
        """List all backend names for a kernel."""
        return self.backends_by_kernel.get(kernel, [])
    
    def get_framework_transforms(self, framework: str) -> Dict[str, Type]:
        """Get all transforms for a framework."""
        return self.framework_transforms.get(framework, {})
    
    def get_framework_kernels(self, framework: str) -> Dict[str, Type]:
        """Get all kernels for a framework."""
        return self.framework_kernels.get(framework, {})
    
    def get_framework_backends(self, framework: str) -> Dict[str, Type]:
        """Get all backends for a framework."""
        return self.framework_backends.get(framework, {})
    
    def get_framework_steps(self, framework: str) -> Dict[str, Type]:
        """Get all steps for a framework."""
        return self.framework_steps.get(framework, {})
    
    def get_plugin_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a plugin."""
        return self.plugin_metadata.get(name, {})
    
    def list_all_plugins(self) -> List[Dict[str, Any]]:
        """List all registered plugins with metadata."""
        all_plugins = []
        
        # Add transforms
        for name, cls in self.transforms.items():
            metadata = self.get_plugin_metadata(name)
            all_plugins.append({'name': name, 'class': cls, 'metadata': metadata})
        
        # Add kernels
        for name, cls in self.kernels.items():
            metadata = self.get_plugin_metadata(name)
            all_plugins.append({'name': name, 'class': cls, 'metadata': metadata})
            
        # Add backends
        for name, cls in self.backends.items():
            metadata = self.get_plugin_metadata(name)
            all_plugins.append({'name': name, 'class': cls, 'metadata': metadata})
            
        # Add steps
        for name, cls in self.steps.items():
            metadata = self.get_plugin_metadata(name)
            all_plugins.append({'name': name, 'class': cls, 'metadata': metadata})
        
        return all_plugins
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        # Collect all frameworks across all plugin types
        all_frameworks = set()
        all_frameworks.update(self.framework_transforms.keys())
        all_frameworks.update(self.framework_kernels.keys())
        all_frameworks.update(self.framework_backends.keys())
        all_frameworks.update(self.framework_steps.keys())
        
        return {
            'total_plugins': len(self.transforms) + len(self.kernels) + len(self.backends) + len(self.steps),
            'transforms': len(self.transforms),
            'kernels': len(self.kernels),
            'backends': len(self.backends),
            'steps': len(self.steps),
            'stages': list(self.transforms_by_stage.keys()),
            'categories': list(self.steps_by_category.keys()),
            'frameworks': list(all_frameworks),
            'indexed_backends': len(self.backends_by_kernel)
        }
    
    # Discovery methods for Phase 1 integration
    def list_available_kernels(self) -> List[str]:
        """List all registered kernel names."""
        return list(self.kernels.keys())
    
    def list_available_transforms(self) -> List[str]:
        """List all registered transform names."""
        return list(self.transforms.keys())
    
    def list_available_steps(self) -> List[str]:
        """List all registered step names."""
        return list(self.steps.keys())
    
    def list_steps_by_category(self, category: str) -> List[str]:
        """List all step names for a category."""
        return list(self.steps_by_category.get(category, {}).keys())
    
    def get_valid_stages(self) -> List[str]:
        """Get list of valid transform stages."""
        return list(self.transforms_by_stage.keys())
    
    def validate_kernel_backends(self, kernel: str, backends: List[str]) -> List[str]:
        """
        Validate backends exist for kernel, return list of invalid ones.
        
        Args:
            kernel: Kernel name
            backends: List of backend names to validate
            
        Returns:
            List of backend names that are not available for the kernel
        """
        available = self.list_backends_by_kernel(kernel)
        return [b for b in backends if b not in available]
    
    def create_subset(self, requirements: Dict[str, List[str]]) -> 'BrainsmithPluginRegistry':
        """Create optimized subset registry for blueprint loading."""
        subset = BrainsmithPluginRegistry()
        
        # Copy only required transforms
        for transform_name in requirements.get('transforms', []):
            if transform_name in self.transforms:
                transform_class = self.transforms[transform_name]
                metadata = self.plugin_metadata[transform_name]
                subset.register_transform(
                    transform_name,
                    transform_class,
                    stage=metadata.get('stage'),
                    framework=metadata.get('framework', 'brainsmith'),
                    **{k: v for k, v in metadata.items() if k not in ['type', 'stage', 'framework']}
                )
        
        # Copy only required kernels
        for kernel_name in requirements.get('kernels', []):
            if kernel_name in self.kernels:
                kernel_class = self.kernels[kernel_name]
                metadata = self.plugin_metadata[kernel_name]
                subset.register_kernel(
                    kernel_name,
                    kernel_class,
                    **{k: v for k, v in metadata.items() if k != 'type'}
                )
        
        # Copy only required backends
        for backend_name in requirements.get('backends', []):
            if backend_name in self.backends:
                backend_class = self.backends[backend_name]
                metadata = self.plugin_metadata[backend_name]
                subset.register_backend(
                    backend_name,
                    backend_class,
                    kernel=metadata['kernel'],
                    **{k: v for k, v in metadata.items() if k not in ['type', 'kernel']}
                )
        
        # Copy only required steps
        for step_name in requirements.get('steps', []):
            if step_name in self.steps:
                step_class = self.steps[step_name]
                metadata = self.plugin_metadata[step_name]
                subset.register_step(
                    step_name,
                    step_class,
                    category=metadata.get('category'),
                    framework=metadata.get('framework', 'brainsmith'),
                    **{k: v for k, v in metadata.items() if k not in ['type', 'category', 'framework']}
                )
        
        return subset
    
    def clear(self) -> None:
        """Clear all registered plugins."""
        self.transforms.clear()
        self.kernels.clear()
        self.backends.clear()
        self.steps.clear()
        self.transforms_by_stage.clear()
        self.backends_by_kernel.clear()
        self.steps_by_category.clear()
        self.framework_transforms.clear()
        self.framework_kernels.clear()
        self.framework_backends.clear()
        self.framework_steps.clear()
        self.plugin_metadata.clear()
        self.default_backends.clear()
        for index in self.backend_indexes.values():
            index.clear()
        logger.debug("Cleared all plugin registrations")


# Global registry instance - Perfect Code pattern
_global_registry: Optional[BrainsmithPluginRegistry] = None


def get_registry() -> BrainsmithPluginRegistry:
    """Get the global plugin registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = BrainsmithPluginRegistry()
    return _global_registry


def reset_registry() -> None:
    """Reset the global registry (useful for testing)."""
    global _global_registry
    if _global_registry is not None:
        _global_registry.clear()
    _global_registry = None