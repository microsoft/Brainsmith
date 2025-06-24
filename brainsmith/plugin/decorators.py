"""
Plugin Decorators

Decorators for registering transforms, kernels, backends, and hardware transforms.
"""

import logging
from typing import Type, Optional, List, Any
from functools import wraps

from .exceptions import PluginRegistrationError, PluginValidationError

logger = logging.getLogger(__name__)


def transform(
    name: str,
    stage: str,
    description: Optional[str] = None,
    author: Optional[str] = None,
    version: Optional[str] = None,
    requires: Optional[List[str]] = None
):
    """
    Decorator for registering transforms.
    
    Args:
        name: Name of the transform (required)
        stage: Compilation stage where transform applies (required)
        description: Human-readable description
        author: Author name or organization
        version: Version string (e.g., "1.0.0")
        requires: List of requirements (kernels, libraries)
    
    Example:
        @transform(
            name="ExpandNorms",
            stage="topology_optimization",
            description="Expand LayerNorms into functional components",
            requires=["qonnx>=0.1.0"]
        )
        class ExpandNorms(Transformation):
            ...
    """
    def decorator(cls: Type) -> Type:
        # Late import to avoid circular dependency
        from .registry import PluginRegistry
        
        # Validate that it's a QONNX Transformation
        try:
            from qonnx.transformation.base import Transformation
            if not issubclass(cls, Transformation):
                raise PluginValidationError(
                    f"Transform '{name}' must inherit from qonnx.transformation.base.Transformation"
                )
        except ImportError:
            # If QONNX not available, skip validation but log warning
            logger.warning("QONNX not available, skipping transform validation")
        
        # Validate stage
        valid_stages = [
            "graph_cleanup",
            "topology_optimization", 
            "kernel_mapping",
            "kernel_optimization",
            "graph_optimization",
            "metadata",
            "model_specific"
        ]
        if stage not in valid_stages:
            raise PluginValidationError(
                f"Invalid stage '{stage}' for transform '{name}'. "
                f"Valid stages: {valid_stages}"
            )
        
        # Add metadata to class
        cls._plugin_metadata = {
            "type": "transform",
            "name": name,
            "stage": stage,
            "description": description,
            "author": author,
            "version": version,
            "requires": requires or []
        }
        
        # Register with global registry
        try:
            PluginRegistry.register(cls)
            logger.info(f"Registered transform: {name} (stage: {stage})")
        except Exception as e:
            raise PluginRegistrationError(f"Failed to register transform '{name}': {e}")
        
        return cls
    
    return decorator


def kernel(
    name: str,
    description: Optional[str] = None,
    author: Optional[str] = None,
    version: Optional[str] = None
):
    """
    Decorator for registering kernels.
    
    Args:
        name: Name of the kernel (required)
        description: Human-readable description
        author: Author name or organization
        version: Version string (e.g., "1.0.0")
    
    Example:
        @kernel(
            name="MatMul",
            description="Matrix multiplication kernel"
        )
        class MatMulKernel:
            ...
    """
    def decorator(cls: Type) -> Type:
        # Late import to avoid circular dependency
        from .registry import PluginRegistry
        
        # Add metadata to class
        cls._plugin_metadata = {
            "type": "kernel",
            "name": name,
            "description": description,
            "author": author,
            "version": version
        }
        
        # Register with global registry
        try:
            PluginRegistry.register(cls)
            logger.info(f"Registered kernel: {name}")
        except Exception as e:
            raise PluginRegistrationError(f"Failed to register kernel '{name}': {e}")
        
        return cls
    
    return decorator


def backend(
    name: str,
    description: Optional[str] = None,
    author: Optional[str] = None,
    version: Optional[str] = None
):
    """
    Decorator for registering backends.
    
    Args:
        name: Name of the backend (required)
        description: Human-readable description
        author: Author name or organization
        version: Version string (e.g., "1.0.0")
    
    Example:
        @backend(
            name="MatMulHLS",
            description="HLS backend for MatMul kernel"
        )
        class MatMulHLSBackend:
            ...
    """
    def decorator(cls: Type) -> Type:
        # Late import to avoid circular dependency
        from .registry import PluginRegistry
        
        # Add metadata to class
        cls._plugin_metadata = {
            "type": "backend",
            "name": name,
            "description": description,
            "author": author,
            "version": version
        }
        
        # Register with global registry
        try:
            PluginRegistry.register(cls)
            logger.info(f"Registered backend: {name}")
        except Exception as e:
            raise PluginRegistrationError(f"Failed to register backend '{name}': {e}")
        
        return cls
    
    return decorator


def hw_transform(
    name: str,
    description: Optional[str] = None,
    author: Optional[str] = None,
    version: Optional[str] = None
):
    """
    Decorator for registering hardware transforms.
    
    Args:
        name: Name of the hardware transform (required)
        description: Human-readable description
        author: Author name or organization
        version: Version string (e.g., "1.0.0")
    
    Example:
        @hw_transform(
            name="OptimizeDSPUsage",
            description="Optimize DSP usage in hardware kernels"
        )
        class OptimizeDSPUsageTransform:
            ...
    """
    def decorator(cls: Type) -> Type:
        # Late import to avoid circular dependency
        from .registry import PluginRegistry
        
        # Add metadata to class
        cls._plugin_metadata = {
            "type": "hw_transform",
            "name": name,
            "description": description,
            "author": author,
            "version": version
        }
        
        # Register with global registry
        try:
            PluginRegistry.register(cls)
            logger.info(f"Registered hw_transform: {name}")
        except Exception as e:
            raise PluginRegistrationError(f"Failed to register hw_transform '{name}': {e}")
        
        return cls
    
    return decorator