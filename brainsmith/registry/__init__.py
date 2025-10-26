# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Brainsmith Component Registry.

Public API for component registration, discovery, and lookup.

Registration (for component authors):
    from brainsmith.registry import kernel, backend, step, registry, source_context

    @kernel
    class MyKernel(HWCustomOp):
        ...

    @backend(target_kernel='MyKernel', language='hls')
    class MyKernel_hls:
        ...

    @step
    def my_step(model, config):
        ...

Discovery and Lookup (for users):
    from brainsmith.registry import discover_components, get_kernel, list_steps

    discover_components()  # Usually automatic on first use

    LayerNorm = get_kernel('LayerNorm')
    steps = list_steps()

Lazy Loading (for plugin/package authors):
    from brainsmith.registry import create_lazy_module

    # In your __init__.py:
    COMPONENTS = {
        'kernels': {'MyKernel': '.my_kernel'},
        'steps': {'my_step': '.my_step'},
    }

    __getattr__, __dir__ = create_lazy_module(COMPONENTS, __name__)
"""

# Public constants
from .constants import (
    SOURCE_BRAINSMITH,
    SOURCE_FINN,
    SOURCE_PROJECT,
    SOURCE_USER,
    PROTECTED_SOURCES,
    DEFAULT_SOURCE_PRIORITY,
    SOURCE_MODULE_PREFIXES,
)

# Metadata structures
from ._metadata import ComponentMetadata, ImportSpec

# Registration decorators and registry
from ._decorators import (
    kernel,
    backend,
    step,
    registry,
    source_context,
    Registry,
)

# Lazy loading helper
from ._loading import create_lazy_module, ComponentsDict

# Discovery and lookup - import from specialized modules
from ._discovery import discover_components
from ._state import _component_index

from ._lookup import (
    # Lookup - Steps
    get_step,
    has_step,
    list_steps,

    # Lookup - Kernels
    get_kernel,
    get_kernel_infer,
    has_kernel,
    list_kernels,

    # Lookup - Backends
    get_backend,
    get_backend_metadata,
    list_backends,
    list_backends_for_kernel,

    # Metadata Access (for CLI and inspection)
    get_component_metadata,
    get_all_component_metadata,
)

__all__ = [
    # Constants
    'SOURCE_BRAINSMITH',
    'SOURCE_FINN',
    'SOURCE_PROJECT',
    'SOURCE_USER',
    'PROTECTED_SOURCES',
    'DEFAULT_SOURCE_PRIORITY',
    'SOURCE_MODULE_PREFIXES',

    # Metadata
    'ComponentMetadata',
    'ImportSpec',

    # Registration
    'kernel',
    'backend',
    'step',
    'registry',
    'source_context',
    'Registry',

    # Lazy loading
    'create_lazy_module',
    'ComponentsDict',

    # Discovery
    'discover_components',

    # Lookup - Steps
    'get_step',
    'has_step',
    'list_steps',

    # Lookup - Kernels
    'get_kernel',
    'get_kernel_infer',
    'has_kernel',
    'list_kernels',

    # Lookup - Backends
    'get_backend',
    'get_backend_metadata',
    'list_backends',
    'list_backends_for_kernel',

    # Metadata Access
    'get_component_metadata',
    'get_all_component_metadata',
]
