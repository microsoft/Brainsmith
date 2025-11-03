# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Brainsmith Component Registry.

Public API for component registration, discovery, and lookup.

Registration (for component authors):
    from brainsmith.registry import kernel, backend, step, source_context

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
"""

# Public constants
from .constants import (
    CORE_NAMESPACE,
    SOURCE_BRAINSMITH,
    SOURCE_FINN,
    SOURCE_PROJECT,
    SOURCE_CUSTOM,
    KNOWN_ENTRY_POINTS,
    PROTECTED_SOURCES,
    DEFAULT_SOURCE_PRIORITY,
    SOURCE_MODULE_PREFIXES,
)

# Metadata structures and helpers
from ._metadata import ComponentMetadata, ComponentType, ImportSpec, resolve_lazy_class

# Registration decorators
from ._decorators import (
    kernel,
    backend,
    step,
    source_context,
)

# Discovery and lookup - import from specialized modules
from ._discovery import discover_components
from ._state import _component_index


# ============================================================================
# Registry Lifecycle Management
# ============================================================================

def reset_registry() -> None:
    """Reset registry to uninitialized state.

    Clears all discovered components and resets discovery flag. Useful for
    testing and when switching configurations.

    Example:
        >>> from brainsmith.registry import reset_registry
        >>> reset_registry()  # Clear state
        >>> # ... change configuration ...
        >>> from brainsmith.registry import discover_components
        >>> discover_components()  # Re-discover with new config

    Note:
        This is the recommended way to reset registry state in tests.
        Do not directly manipulate private _component_index or
        _components_discovered variables.
    """
    import brainsmith.registry._state as registry_state
    import brainsmith.registry._discovery as discovery_module

    _component_index.clear()
    registry_state._components_discovered = False
    discovery_module._components_discovered = False


def is_initialized() -> bool:
    """Check if registry has been initialized.

    Returns:
        True if component discovery has completed

    Example:
        >>> from brainsmith.registry import is_initialized
        >>> is_initialized()
        False
        >>> from brainsmith.registry import discover_components
        >>> discover_components()
        >>> is_initialized()
        True
    """
    from ._state import _components_discovered
    return _components_discovered


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
    'CORE_NAMESPACE',
    'SOURCE_BRAINSMITH',
    'SOURCE_FINN',
    'SOURCE_PROJECT',
    'SOURCE_CUSTOM',
    'KNOWN_ENTRY_POINTS',
    'PROTECTED_SOURCES',
    'DEFAULT_SOURCE_PRIORITY',
    'SOURCE_MODULE_PREFIXES',

    # Metadata
    'ComponentMetadata',
    'ComponentType',
    'ImportSpec',

    # Registration
    'kernel',
    'backend',
    'step',
    'source_context',

    # Discovery and Lifecycle
    'discover_components',
    'reset_registry',
    'is_initialized',

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
