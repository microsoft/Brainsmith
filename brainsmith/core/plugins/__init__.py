# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Plugin System
"""
from .registry import (
    get_registry, get_transform, get_kernel, get_backend, get_step,
    transform, kernel, backend, step, kernel_inference,
    list_transforms, list_kernels, list_backends, list_steps,
    has_transform, has_kernel, has_backend, has_step,
    get_transforms_by_metadata, get_kernels_by_metadata,
    get_backends_by_metadata, get_steps_by_metadata,
    list_all_steps, list_all_kernels
)

# Framework adapters are loaded lazily when needed to avoid slow startup
# Plugins are discovered lazily on first access to avoid circular imports
# See registry.py _load_plugins() for the implementation

__all__ = [
    # Registry access
    "get_registry",
    "get_transform",
    "get_kernel", 
    "get_backend",
    "get_step",
    
    # Decorators
    "transform",
    "kernel",
    "backend", 
    "step",
    "kernel_inference",
    
    # List functions
    "list_transforms",
    "list_kernels",
    "list_backends",
    "list_steps",
    
    # Has functions
    "has_transform",
    "has_kernel",
    "has_backend",
    "has_step",
    
    # Metadata queries
    "get_transforms_by_metadata",
    "get_kernels_by_metadata",
    "get_backends_by_metadata",
    "get_steps_by_metadata",
    
    # CLI helpers
    "list_all_steps",
    "list_all_kernels"
]