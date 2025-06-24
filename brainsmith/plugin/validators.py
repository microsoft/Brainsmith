"""
Plugin Validators

Utilities for validating plugin implementations.
"""

import logging
from typing import Type, List, Tuple, Any

from .exceptions import PluginValidationError

logger = logging.getLogger(__name__)


def validate_transform(transform_class: Type) -> Tuple[bool, List[str]]:
    """
    Validate a transform plugin class.
    
    Args:
        transform_class: Transform class to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check for metadata
    if not hasattr(transform_class, '_plugin_metadata'):
        errors.append("Missing _plugin_metadata attribute")
        return False, errors
    
    metadata = transform_class._plugin_metadata
    
    # Check required metadata fields
    if not metadata.get('name'):
        errors.append("Missing required metadata field: name")
    if not metadata.get('stage'):
        errors.append("Missing required metadata field: stage")
    
    # Check inheritance
    try:
        from qonnx.transformation.base import Transformation
        if not issubclass(transform_class, Transformation):
            errors.append("Transform must inherit from qonnx.transformation.base.Transformation")
    except ImportError:
        logger.warning("QONNX not available, skipping inheritance check")
    
    # Check for apply method
    if not hasattr(transform_class, 'apply'):
        errors.append("Transform must have an 'apply' method")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def validate_kernel(kernel_class: Type) -> Tuple[bool, List[str]]:
    """
    Validate a kernel plugin class.
    
    Args:
        kernel_class: Kernel class to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check for metadata
    if not hasattr(kernel_class, '_plugin_metadata'):
        errors.append("Missing _plugin_metadata attribute")
        return False, errors
    
    metadata = kernel_class._plugin_metadata
    
    # Check required metadata fields
    if not metadata.get('name'):
        errors.append("Missing required metadata field: name")
    
    # Kernel implementation validation would go here
    # For now, just basic checks since kernels are stubs
    
    is_valid = len(errors) == 0
    return is_valid, errors


def validate_backend(backend_class: Type) -> Tuple[bool, List[str]]:
    """
    Validate a backend plugin class.
    
    Args:
        backend_class: Backend class to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check for metadata
    if not hasattr(backend_class, '_plugin_metadata'):
        errors.append("Missing _plugin_metadata attribute")
        return False, errors
    
    metadata = backend_class._plugin_metadata
    
    # Check required metadata fields
    if not metadata.get('name'):
        errors.append("Missing required metadata field: name")
    
    # Backend implementation validation would go here
    # For now, just basic checks since backends are stubs
    
    is_valid = len(errors) == 0
    return is_valid, errors


def validate_hw_transform(hw_transform_class: Type) -> Tuple[bool, List[str]]:
    """
    Validate a hardware transform plugin class.
    
    Args:
        hw_transform_class: Hardware transform class to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check for metadata
    if not hasattr(hw_transform_class, '_plugin_metadata'):
        errors.append("Missing _plugin_metadata attribute")
        return False, errors
    
    metadata = hw_transform_class._plugin_metadata
    
    # Check required metadata fields
    if not metadata.get('name'):
        errors.append("Missing required metadata field: name")
    
    # Hardware transform implementation validation would go here
    # For now, just basic checks since hw_transforms are stubs
    
    is_valid = len(errors) == 0
    return is_valid, errors


def validate_plugin(plugin_class: Type) -> Tuple[bool, List[str]]:
    """
    Validate any plugin class based on its type.
    
    Args:
        plugin_class: Plugin class to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    if not hasattr(plugin_class, '_plugin_metadata'):
        return False, ["Missing _plugin_metadata attribute"]
    
    plugin_type = plugin_class._plugin_metadata.get('type')
    
    if plugin_type == 'transform':
        return validate_transform(plugin_class)
    elif plugin_type == 'kernel':
        return validate_kernel(plugin_class)
    elif plugin_type == 'backend':
        return validate_backend(plugin_class)
    elif plugin_type == 'hw_transform':
        return validate_hw_transform(plugin_class)
    else:
        return False, [f"Unknown plugin type: {plugin_type}"]