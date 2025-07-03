"""
Unified Plugin Decorators - Perfect Code Implementation

Auto-registration at decoration time eliminates discovery overhead.
Preserves exact API while using optimized registry backend.
"""

import logging
from typing import Dict, Any, Optional, Type, Callable, List
from functools import wraps

logger = logging.getLogger(__name__)


def plugin(
    type: str,
    name: Optional[str] = None,
    framework: str = "brainsmith",
    **metadata
) -> Callable:
    """
    Unified plugin decorator for all plugin types.
    
    Registers plugin immediately at decoration time for zero discovery overhead.
    
    Args:
        type: Plugin type ("transform", "kernel", "backend", "step", "kernel_inference")
        name: Plugin name (defaults to class name)
        framework: Framework name (defaults to "brainsmith")
        **metadata: Additional plugin metadata
    
    Usage:
        @plugin(type="transform", name="MyTransform", stage="topology_opt")
        @plugin(type="kernel", name="MyKernel", op_type="MyOp")
        @plugin(type="backend", name="MyBackend", kernel="MyKernel", backend_type="hls")
        @plugin(type="step", name="MyStep", category="metadata")
    """
    
    def decorator(cls: Type) -> Type:
        # Determine plugin name
        plugin_name = name or cls.__name__
        
        # Validate plugin type
        valid_types = {"transform", "kernel", "backend", "step", "kernel_inference"}
        if type not in valid_types:
            logger.warning(f"Unknown plugin type '{type}' for {plugin_name}. Valid types: {valid_types}")
        
        # Type-specific validation
        validation_errors = _validate_metadata(type, metadata)
        if validation_errors:
            for error in validation_errors:
                logger.warning(f"Plugin {plugin_name}: {error}")
        
        # Create metadata dict
        plugin_metadata = {
            'name': plugin_name,
            'type': type,
            'framework': framework,
            **metadata
        }
        
        # Store metadata on class for backward compatibility
        cls._plugin_metadata = plugin_metadata
        
        # Auto-register with the registry immediately
        _auto_register_plugin(cls, plugin_metadata)
        
        logger.debug(f"Registered {type} plugin: {plugin_name} ({framework})")
        return cls
    
    return decorator


def _validate_metadata(plugin_type: str, metadata: Dict[str, Any]) -> List[str]:
    """
    Validate plugin metadata based on type.
    
    Args:
        plugin_type: Type of plugin
        metadata: Metadata to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Transform validation
    if plugin_type == "transform":
        has_stage = "stage" in metadata and metadata["stage"] is not None
        has_kernel = "kernel" in metadata and metadata["kernel"] is not None
        
        if not has_stage and not has_kernel:
            errors.append("Transform must specify either 'stage' or 'kernel'")
        elif has_stage and has_kernel:
            errors.append("Transform cannot specify both 'stage' and 'kernel'")
        elif has_stage:
            valid_stages = {"pre_proc", "cleanup", "topology_opt", "kernel_opt", "dataflow_opt", "post_proc"}
            if metadata["stage"] not in valid_stages:
                errors.append(f"Invalid stage '{metadata['stage']}'. Valid: {valid_stages}")
    
    # Backend validation
    elif plugin_type == "backend":
        if "kernel" not in metadata:
            errors.append("Backend must specify 'kernel'")
        # backend_type is now optional, but if provided, validate it
        if "backend_type" in metadata:
            valid_backend_types = {"hls", "rtl"}
            if metadata["backend_type"] not in valid_backend_types:
                errors.append(f"Invalid backend_type '{metadata['backend_type']}'. Valid: {valid_backend_types}")
    
    # Kernel inference validation
    elif plugin_type == "kernel_inference":
        if "kernel" not in metadata:
            errors.append("Kernel inference transform must specify 'kernel'")
    
    # Step validation
    elif plugin_type == "step":
        if "category" not in metadata:
            errors.append("Step must specify 'category'")
    
    return errors


def _auto_register_plugin(cls: Type, metadata: Dict[str, Any]) -> None:
    """
    Auto-register plugin with the global registry at decoration time.
    
    This is the key Perfect Code improvement: registration happens immediately,
    eliminating all discovery overhead.
    
    Args:
        cls: Plugin class
        metadata: Plugin metadata
    """
    try:
        # Import here to avoid circular imports
        from .registry import get_registry
        
        registry = get_registry()
        plugin_type = metadata['type']
        plugin_name = metadata['name']
        framework = metadata['framework']
        
        # Register based on plugin type
        if plugin_type == "transform":
            stage = metadata.get('stage')
            # Filter out type-specific metadata
            transform_metadata = {k: v for k, v in metadata.items() 
                                if k not in ['name', 'type', 'framework', 'stage']}
            
            registry.register_transform(
                plugin_name, 
                cls, 
                stage=stage,
                framework=framework,
                **transform_metadata
            )
            
        elif plugin_type == "kernel":
            # Filter out type-specific metadata
            kernel_metadata = {k: v for k, v in metadata.items() 
                             if k not in ['name', 'type', 'framework']}
            
            registry.register_kernel(plugin_name, cls, framework=framework, **kernel_metadata)
            
        elif plugin_type == "backend":
            kernel = metadata['kernel']
            # Filter out type-specific metadata
            backend_metadata = {k: v for k, v in metadata.items() 
                              if k not in ['name', 'type', 'framework', 'kernel']}
            
            registry.register_backend(
                plugin_name, 
                cls, 
                kernel=kernel,
                framework=framework,
                **backend_metadata
            )
            
        elif plugin_type in ["step", "kernel_inference"]:
            # Treat steps and kernel_inference as special transforms
            stage = metadata.get('stage') or metadata.get('category', 'general')
            # Filter out type-specific metadata
            transform_metadata = {k: v for k, v in metadata.items() 
                                if k not in ['name', 'type', 'framework', 'stage', 'category']}
            
            registry.register_transform(
                plugin_name, 
                cls, 
                stage=stage,
                framework=framework,
                plugin_type=plugin_type,  # Preserve original type
                **transform_metadata
            )
        
        logger.debug(f"Auto-registered {plugin_type}: {plugin_name}")
        
    except Exception as e:
        logger.warning(f"Auto-registration failed for {metadata['name']}: {e}")


# Convenience decorators for specific plugin types
def transform(name: Optional[str] = None, stage: Optional[str] = None, 
              kernel: Optional[str] = None, framework: str = "brainsmith", **kwargs) -> Callable:
    """Convenience decorator for transform plugins."""
    return plugin(type="transform", name=name, stage=stage, kernel=kernel, framework=framework, **kwargs)


def kernel(name: Optional[str] = None, op_type: Optional[str] = None, 
           framework: str = "brainsmith", **kwargs) -> Callable:
    """Convenience decorator for kernel plugins."""
    return plugin(type="kernel", name=name, op_type=op_type, framework=framework, **kwargs)


def backend(name: Optional[str] = None, kernel: Optional[str] = None, 
            backend_type: Optional[str] = None, language: Optional[str] = None,
            default: bool = False, framework: str = "brainsmith", **kwargs) -> Callable:
    """Convenience decorator for backend plugins.
    
    Args:
        name: Backend name (must be unique)
        kernel: Kernel this backend implements
        backend_type: Deprecated, use 'language' instead
        language: Implementation language (hls, verilog, etc.)
        default: Whether this is the default backend for the kernel
        framework: Framework name
        **kwargs: Additional metadata (optimization, architecture, etc.)
    """
    # Handle backward compatibility
    if backend_type and not language:
        language = backend_type
    
    return plugin(type="backend", name=name, kernel=kernel, backend_type=backend_type,
                  language=language, default=default, framework=framework, **kwargs)


def step(name: Optional[str] = None, category: Optional[str] = None, 
         framework: str = "brainsmith", **kwargs) -> Callable:
    """Convenience decorator for step plugins."""
    return plugin(type="step", name=name, category=category, framework=framework, **kwargs)


def kernel_inference(name: Optional[str] = None, kernel: Optional[str] = None, 
                    framework: str = "brainsmith", **kwargs) -> Callable:
    """Convenience decorator for kernel inference plugins."""
    return plugin(type="kernel_inference", name=name, kernel=kernel, framework=framework, **kwargs)


# Utility functions for backward compatibility
def get_plugin_metadata(cls: Type) -> Optional[Dict[str, Any]]:
    """Get plugin metadata from a class."""
    return getattr(cls, '_plugin_metadata', None)


def is_plugin_class(cls: Type) -> bool:
    """Check if a class is a plugin class."""
    return hasattr(cls, '_plugin_metadata')


def list_plugin_classes(module) -> List[Type]:
    """List all plugin classes in a module."""
    plugins = []
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and is_plugin_class(obj):
            plugins.append(obj)
    return plugins