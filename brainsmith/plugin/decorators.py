"""
Unified Plugin Decorator - Simplified

Single @plugin decorator that replaces all convenience decorators.
Eliminates complex validation system while preserving essential functionality.
"""

import logging
from typing import Dict, Any, Optional, Type, Union, Callable
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
        
        # Basic type-specific validation
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
        
        # Store metadata on class
        cls._plugin_metadata = plugin_metadata
        
        # Auto-register with the plugin manager
        _auto_register_plugin(cls, plugin_metadata)
        
        # Register with external frameworks if needed
        _register_with_external_frameworks(cls, plugin_metadata)
        
        logger.debug(f"Registered {type} plugin: {plugin_name} ({framework})")
        return cls
    
    return decorator


def _validate_metadata(plugin_type: str, metadata: Dict[str, Any]) -> list:
    """
    Simple validation of plugin metadata.
    
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
            valid_stages = {"cleanup", "topology_opt", "kernel_opt", "dataflow_opt"}
            if metadata["stage"] not in valid_stages:
                errors.append(f"Invalid stage '{metadata['stage']}'. Valid: {valid_stages}")
    
    # Backend validation
    elif plugin_type == "backend":
        if "kernel" not in metadata:
            errors.append("Backend must specify 'kernel'")
        if "backend_type" not in metadata:
            errors.append("Backend must specify 'backend_type'")
        else:
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
    Auto-register plugin with the global manager when decorated.
    
    Args:
        cls: Plugin class
        metadata: Plugin metadata
    """
    try:
        # Import here to avoid circular imports
        from .manager import get_plugin_manager
        from .data_models import create_plugin_info
        
        manager = get_plugin_manager()
        
        plugin_info = create_plugin_info(
            name=metadata['name'],
            plugin_class=cls,
            plugin_type=metadata['type'],
            framework=metadata['framework'],
            **{k: v for k, v in metadata.items() if k not in ['name', 'type', 'framework']}
        )
        
        manager.register_plugin(plugin_info)
        logger.debug(f"Auto-registered plugin: {metadata['name']}")
        
    except Exception as e:
        logger.warning(f"Auto-registration failed for {metadata['name']}: {e}")
        import traceback
        logger.warning(traceback.format_exc())


def _register_with_external_frameworks(cls: Type, metadata: Dict[str, Any]) -> None:
    """
    Register plugin with external frameworks (QONNX, FINN) if needed.
    
    Args:
        cls: Plugin class
        metadata: Plugin metadata
    """
    plugin_type = metadata.get('type')
    framework = metadata.get('framework', 'brainsmith')
    
    # Only register external frameworks
    if framework == 'brainsmith':
        return
    
    try:
        if framework == 'qonnx' and plugin_type == 'transform':
            _register_qonnx_transform(cls, metadata)
        elif framework == 'finn' and plugin_type == 'transform':
            _register_finn_transform(cls, metadata)
        elif framework == 'finn' and plugin_type == 'step':
            _register_finn_step(cls, metadata)
    except Exception as e:
        logger.debug(f"Failed to register {metadata['name']} with {framework}: {e}")


def _register_qonnx_transform(cls: Type, metadata: Dict[str, Any]) -> None:
    """Register transform with QONNX registry."""
    try:
        from qonnx.transformation.registry import register_transformation
        
        # Filter metadata to avoid duplicate arguments
        qonnx_metadata = metadata.copy()
        # Remove fields that might conflict with QONNX registration
        qonnx_metadata.pop('name', None)
        qonnx_metadata.pop('type', None)
        qonnx_metadata.pop('framework', None)
        
        register_transformation(cls, metadata['name'], **qonnx_metadata)
        logger.debug(f"Registered QONNX transform: {metadata['name']}")
    except ImportError:
        logger.debug("QONNX not available for registration")
    except Exception as e:
        logger.debug(f"QONNX registration failed for {metadata['name']}: {e}")


def _register_finn_transform(cls: Type, metadata: Dict[str, Any]) -> None:
    """Register transform with FINN registry."""
    try:
        # FINN may have different registration mechanism
        # Add FINN-specific registration here if needed
        logger.debug(f"FINN transform registration not implemented for: {metadata['name']}")
    except Exception as e:
        logger.debug(f"FINN transform registration failed for {metadata['name']}: {e}")


def _register_finn_step(cls: Type, metadata: Dict[str, Any]) -> None:
    """Register step with FINN registry."""
    try:
        # FINN may have different registration mechanism for steps
        # Add FINN-specific step registration here if needed
        logger.debug(f"FINN step registration not implemented for: {metadata['name']}")
    except Exception as e:
        logger.debug(f"FINN step registration failed for {metadata['name']}: {e}")


# Convenience functions for specific plugin types (optional)
def transform(name: Optional[str] = None, stage: Optional[str] = None, 
              kernel: Optional[str] = None, **kwargs) -> Callable:
    """Convenience decorator for transform plugins."""
    return plugin(type="transform", name=name, stage=stage, kernel=kernel, **kwargs)


def kernel(name: Optional[str] = None, op_type: Optional[str] = None, **kwargs) -> Callable:
    """Convenience decorator for kernel plugins."""
    return plugin(type="kernel", name=name, op_type=op_type, **kwargs)


def backend(name: Optional[str] = None, kernel: Optional[str] = None, 
            backend_type: Optional[str] = None, **kwargs) -> Callable:
    """Convenience decorator for backend plugins."""
    return plugin(type="backend", name=name, kernel=kernel, backend_type=backend_type, **kwargs)


def step(name: Optional[str] = None, category: Optional[str] = None, **kwargs) -> Callable:
    """Convenience decorator for step plugins."""
    return plugin(type="step", name=name, category=category, **kwargs)


def kernel_inference(name: Optional[str] = None, kernel: Optional[str] = None, **kwargs) -> Callable:
    """Convenience decorator for kernel inference plugins."""
    return plugin(type="kernel_inference", name=name, kernel=kernel, **kwargs)


# For backward compatibility during transition
def get_plugin_metadata(cls: Type) -> Optional[Dict[str, Any]]:
    """Get plugin metadata from a class."""
    return getattr(cls, '_plugin_metadata', None)


def is_plugin_class(cls: Type) -> bool:
    """Check if a class is a plugin class."""
    return hasattr(cls, '_plugin_metadata')


def list_plugin_classes(module) -> list:
    """List all plugin classes in a module."""
    plugins = []
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and is_plugin_class(obj):
            plugins.append(obj)
    return plugins