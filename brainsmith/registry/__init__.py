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
# Registration decorators
from ._decorators import (
    backend,
    kernel,
    source_context,
    step,
)

# Discovery and lookup - import from specialized modules
from ._discovery import discover_components

# Metadata structures and helpers
from ._metadata import ComponentMetadata, ComponentType, ImportSpec
from .constants import (
    CORE_NAMESPACE,
    DEFAULT_SOURCE_PRIORITY,
    KNOWN_ENTRY_POINTS,
    PROTECTED_SOURCES,
    SOURCE_BRAINSMITH,
    SOURCE_CUSTOM,
    SOURCE_FINN,
    SOURCE_MODULE_PREFIXES,
    SOURCE_PROJECT,
)

# ============================================================================
# Registry Lifecycle Management
# ============================================================================


def reset_registry() -> None:
    """Reset registry to uninitialized state.

    Clears all discovered components and resets discovery flag.
    Primarily used for testing.

    Example:
        >>> from brainsmith.registry import reset_registry
        >>> reset_registry()
        >>> # ... change configuration ...
        >>> from brainsmith.registry import discover_components
        >>> discover_components()  # Re-discover with new config
    """
    import brainsmith.registry._discovery as discovery_module
    import brainsmith.registry._state as registry_state

    registry_state._component_index.clear()
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
        >>> discover_components()
        >>> is_initialized()
        True
    """
    from ._state import _components_discovered

    return _components_discovered


from ._lookup import (  # noqa: E402
    get_all_component_metadata,
    # Lookup - Backends
    get_backend,
    get_backend_metadata,
    # Metadata Access (for CLI and inspection)
    get_component_metadata,
    # Domain Resolution
    get_domain_for_backend,
    # Lookup - Kernels
    get_kernel,
    get_kernel_infer,
    # Lookup - Steps
    get_step,
    has_kernel,
    has_step,
    list_backends,
    list_backends_for_kernel,
    list_kernels,
    list_steps,
)

__all__ = [
    # Constants
    "CORE_NAMESPACE",
    "SOURCE_BRAINSMITH",
    "SOURCE_FINN",
    "SOURCE_PROJECT",
    "SOURCE_CUSTOM",
    "KNOWN_ENTRY_POINTS",
    "PROTECTED_SOURCES",
    "DEFAULT_SOURCE_PRIORITY",
    "SOURCE_MODULE_PREFIXES",
    # Metadata Structures
    "ComponentMetadata",
    "ComponentType",
    "ImportSpec",
    # Registration
    "kernel",
    "backend",
    "step",
    "source_context",
    # Discovery and Lifecycle
    "discover_components",
    "reset_registry",
    "is_initialized",
    # Lookup - Steps
    "get_step",
    "has_step",
    "list_steps",
    # Lookup - Kernels
    "get_kernel",
    "get_kernel_infer",
    "has_kernel",
    "list_kernels",
    # Lookup - Backends
    "get_backend",
    "get_backend_metadata",
    "list_backends",
    "list_backends_for_kernel",
    # Metadata Access
    "get_component_metadata",
    "get_all_component_metadata",
    # Domain Resolution
    "get_domain_for_backend",
]
