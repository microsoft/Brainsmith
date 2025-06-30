"""
Plugin Access Layer

Provides natural, object-oriented access to plugins through collections
and wrappers. This layer sits on top of the core plugin infrastructure
to provide a user-friendly API.

Usage:
    from brainsmith.plugin.access import TransformCollection, KernelCollection
    
    # Or use the pre-configured global collections:
    from brainsmith.plugins import transforms, kernels, steps
"""

from .wrappers import (
    PluginWrapper,
    TransformWrapper,
    KernelWrapper,
    BackendWrapper,
    StepWrapper
)

from .transforms import (
    TransformCollection,
    FrameworkTransformCollection
)

from .kernels import (
    KernelCollection,
    FrameworkKernelCollection
)

from .steps import (
    StepCollection,
    CategoryStepCollection
)

from .factory import CollectionFactory

__all__ = [
    # Wrappers
    'PluginWrapper',
    'TransformWrapper',
    'KernelWrapper',
    'BackendWrapper',
    'StepWrapper',
    
    # Transform collections
    'TransformCollection',
    'FrameworkTransformCollection',
    
    # Kernel collections
    'KernelCollection',
    'FrameworkKernelCollection',
    
    # Step collections
    'StepCollection',
    'CategoryStepCollection',
    
    # Factory
    'CollectionFactory'
]