# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Brainsmith core components.

This module provides the base classes and registration system for
Brainsmith plugins:

Base Classes:
- Step: Transformation/compilation steps
- CustomOp: Custom ONNX operations (kernels)
- Backend: Hardware backend implementations

Registration:
- registry: Global component registry
- source_context: Context manager for source detection

Examples:
    >>> from brainsmith.core import CustomOp, Backend, registry
    >>>
    >>> @registry.kernel
    ... class MyKernel(CustomOp):
    ...     op_type = "MyOp"
    ...     infer_transform = InferMyOp
    >>>
    >>> @registry.backend
    ... class MyKernelHLS(Backend):
    ...     target_kernel = 'brainsmith:MyKernel'
    ...     language = 'hls'
"""

from .base import Step, CustomOp, Backend, step_from_function
from brainsmith.registry import registry, source_context

__all__ = [
    # Base classes
    'Step',
    'CustomOp',
    'Backend',

    # Registration
    'registry',
    'source_context',

    # Utilities
    'step_from_function',
]
