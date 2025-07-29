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
    get_backends_by_metadata, get_steps_by_metadata
)

# Import framework adapters to register external plugins
from . import framework_adapters

# Import BrainSmith modules to trigger registrations
import brainsmith.transforms
import brainsmith.kernels  
import brainsmith.steps

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
]