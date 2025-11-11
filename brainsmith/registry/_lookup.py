# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Component lookup and public API for the registry system.

Provides type-specific public API functions for accessing components:
- Steps: get_step(), has_step(), list_steps()
- Kernels: get_kernel(), get_kernel_infer(), has_kernel(), list_kernels()
- Backends: get_backend(), get_backend_metadata(), list_backends(), list_backends_for_kernel()

All functions support both short names (e.g., 'LayerNorm') and fully-qualified
names (e.g., 'brainsmith:LayerNorm'). Short names use source priority resolution.
"""

import logging
from typing import Any

from ._discovery import (
    _load_component,
    _measure_load,
    _resolve_component_name,
    discover_components,
)
from ._metadata import ComponentType, resolve_lazy_class
from ._state import _component_index, _components_discovered

logger = logging.getLogger(__name__)


# ============================================================================
# Error Formatting Helpers
# ============================================================================

def _format_not_found_error(
    component_type: str,
    full_name: str,
    available: list[str]
) -> str:
    """Format helpful error message with troubleshooting guide.

    Args:
        component_type: Component type string ('kernel', 'backend', 'step')
        full_name: Fully-qualified component name that wasn't found
        available: List of available components of this type

    Returns:
        Formatted error message with troubleshooting steps
    """
    # Extract requested name without source prefix
    requested_name = full_name.split(':', 1)[1] if ':' in full_name else full_name

    # Check if name exists in other sources
    similar = [a for a in available if a.endswith(f':{requested_name}')]

    type_str = component_type
    msg = f"{type_str.title()} '{full_name}' not found.\n"

    if similar:
        # Name exists in different source - show alternatives
        msg += f"\nHint: Found '{requested_name}' in other sources:\n"
        for s in similar:
            msg += f"   • {s}\n"

        try:
            from brainsmith.settings import get_config
            source_priority = get_config().source_priority
            msg += f"\nCurrent source priority: {source_priority}\n"
        except Exception:
            pass

        msg += f"\n→ Try: get_{type_str}('{similar[0]}')  # Use fully-qualified name\n"
    else:
        # Name doesn't exist anywhere - show troubleshooting
        msg += f"\n✗ No {type_str} named '{requested_name}' found in any source.\n"
        msg += "\nTroubleshooting:\n"
        msg += f"  1. List all: list_{type_str}s()\n"
        msg += f"  2. Check decorator: @{type_str} applied?\n"
        msg += "  3. Check paths: Verify component_sources config\n"

    msg += f"\nAvailable {type_str}s: {', '.join(available[:10])}"
    if len(available) > 10:
        msg += f" ... and {len(available) - 10} more"

    return msg


# ============================================================================
# Unified Component Access (Arete: Deduplicated Public API)
# ============================================================================

def _get_component(name: str, component_type: str):
    """Unified component lookup - single source of truth.

    All public get_*() functions delegate to this implementation.
    Type-specific behavior is minimal - all lookups use _component_index.

    Args:
        name: Component name (with or without source prefix)
        component_type: One of 'step', 'kernel', 'backend'

    Returns:
        Loaded component (class or function)

    Raises:
        KeyError: If component not found
    """
    with _measure_load(f'get_{component_type}', name):
        if not _components_discovered:
            discover_components()

        full_name = _resolve_component_name(name, component_type)
        logger.debug(f"Looking up {component_type}: {name} → {full_name}")

        # Lookup in component index
        meta = _component_index.get(full_name)
        if not meta:
            available = _list_components(component_type)
            raise KeyError(_format_not_found_error(component_type, full_name, available))

        # Load component and return it directly
        # Note: Must return _load_component() result, not meta.loaded_obj, because
        # the decorator may replace ComponentMetadata in _component_index during import,
        # making our 'meta' reference stale (loaded_obj=None even after successful load).
        return _load_component(meta)


def _has_component(name: str, component_type: str) -> bool:
    """Unified component existence check.

    All public has_*() functions delegate to this implementation.

    Args:
        name: Component name (with or without source prefix)
        component_type: One of 'step', 'kernel', 'backend'

    Returns:
        True if component exists in index
    """
    if not _components_discovered:
        discover_components()

    full_name = _resolve_component_name(name, component_type)
    return full_name in _component_index


def _list_components(component_type: str, source: str | None = None) -> list[str]:
    """Unified component listing.

    All public list_*() functions delegate to this implementation.

    Args:
        component_type: One of 'step', 'kernel', 'backend'
        source: Optional source filter (e.g., 'user', 'brainsmith')

    Returns:
        Sorted list of component names (with source prefixes)
    """
    if not _components_discovered:
        discover_components()

    # Convert string to enum for comparison
    component_type_enum = ComponentType.from_string(component_type)

    components = [
        meta.full_name for meta in _component_index.values()
        if meta.component_type == component_type_enum and (source is None or meta.source == source)
    ]
    return sorted(components)


# ============================================================================
# Public API: Component Access Functions
# ============================================================================
# Design Note: Type-specific wrappers (get_step, get_kernel, get_backend)
# are intentional API design for better discoverability and IDE support.
# All wrappers delegate to unified internal implementations (_get_component,
# _has_component, _list_components) for maintainability.
#
# This design prioritizes user experience over internal code brevity:
# - Better IDE autocomplete: get_kernel() appears in suggestions
# - More Pythonic: Explicit functions > type enums
# - Clearer docs: Dedicated docstrings for each component type
# ============================================================================

# === Steps ===

def get_step(name: str):
    """Get step callable by name.

    Accepts both short names ('streamline') and qualified names ('user:custom_step').
    Short names are resolved using source priority from settings.

    Args:
        name: Step name - either short ('streamline') or qualified ('user:custom_step')

    Returns:
        Callable step function or Step instance

    Raises:
        KeyError: If step not found

    Examples:
        >>> streamline = get_step('streamline')  # Short name, uses source priority
        >>> custom = get_step('user:custom_step')  # Qualified name, explicit source
    """
    return _get_component(name, 'step')


def has_step(name: str) -> bool:
    """Check if step exists without importing it.

    Args:
        name: Step name (with or without source prefix)

    Returns:
        True if step exists
    """
    return _has_component(name, 'step')


def list_steps(source: str | None = None) -> list[str]:
    """List all available steps.

    Args:
        source: Optional source filter (e.g., 'user', 'brainsmith')

    Returns:
        Sorted list of step names (with source prefixes)

    Examples:
        >>> steps = list_steps()  # All steps
        >>> user_steps = list_steps(source='user')  # Only user steps
    """
    return _list_components('step', source)


# === Kernels ===

def get_kernel(name: str) -> type:
    """Get kernel class.

    Accepts both short names ('LayerNorm') and qualified names ('user:LayerNorm').
    Short names are resolved using source priority from settings.

    Args:
        name: Kernel name - either short ('LayerNorm') or qualified ('user:LayerNorm')

    Returns:
        Kernel class

    Raises:
        KeyError: If kernel not found in any source

    Examples:
        >>> LayerNorm = get_kernel('LayerNorm')  # Short name, uses source priority
        >>> kernel = LayerNorm(onnx_node)
        >>> CustomKernel = get_kernel('user:CustomKernel')  # Qualified name, explicit source
    """
    return _get_component(name, 'kernel')


def get_kernel_infer(name: str) -> type:
    """Get kernel's inference transform class.

    Args:
        name: Kernel name (short or qualified)

    Returns:
        InferTransform class for ONNX → kernel conversion

    Raises:
        KeyError: If kernel not found or has no InferTransform

    Examples:
        >>> InferLayerNorm = get_kernel_infer('LayerNorm')
        >>> model = model.transform(InferLayerNorm())
    """
    if not _components_discovered:
        discover_components()

    full_name = _resolve_component_name(name, 'kernel')

    # Lookup in unified component index
    meta = _component_index.get(full_name)
    if not meta:
        available = list_kernels()
        raise KeyError(_format_not_found_error('kernel', full_name, available))

    # Load component to ensure metadata is populated
    _load_component(meta)

    # Check metadata for infer transform
    if meta.kernel_infer is None:
        raise KeyError(
            f"Kernel '{full_name}' has no InferTransform.\n"
            f"\nThis kernel does not have an associated shape inference transform.\n"
            f"Check that the kernel class defines 'infer_transform' attribute."
        )

    # Resolve lazy import specs and cache
    infer = resolve_lazy_class(meta.kernel_infer)
    if infer != meta.kernel_infer:
        # Cache the resolved class
        meta.kernel_infer = infer

    return infer


def has_kernel(name: str) -> bool:
    """Check if kernel exists.

    Args:
        name: Kernel name (with or without source prefix)

    Returns:
        True if kernel exists
    """
    return _has_component(name, 'kernel')


def list_kernels(source: str | None = None) -> list[str]:
    """List all available kernels.

    Args:
        source: Optional source filter (e.g., 'user', 'brainsmith')

    Returns:
        Sorted list of kernel names (with source prefixes)

    Examples:
        >>> kernels = list_kernels()  # All kernels
        >>> user_kernels = list_kernels(source='user')  # Only user kernels
    """
    return _list_components('kernel', source)


# === Backends ===

def get_backend(name: str) -> type:
    """Get backend class by name.

    Accepts both short names ('LayerNorm_HLS') and qualified names ('user:LayerNorm_HLS_Fast').
    Short names are resolved using source priority from settings.

    Args:
        name: Backend name - either short ('LayerNorm_HLS') or qualified ('user:LayerNorm_HLS_Fast')

    Returns:
        Backend class

    Raises:
        KeyError: If backend not found in any source

    Examples:
        >>> backend = get_backend('LayerNorm_HLS')  # Short name, uses source priority
        >>> custom = get_backend('user:LayerNorm_HLS_Fast')  # Qualified name, explicit source
    """
    return _get_component(name, 'backend')


def get_backend_metadata(name: str) -> dict[str, Any]:
    """Get backend metadata.

    Args:
        name: Backend name (with or without source prefix)

    Returns:
        Backend metadata dict with 'class', 'target_kernel', 'language' keys

    Raises:
        KeyError: If backend not found
    """
    if not _components_discovered:
        discover_components()

    full_name = _resolve_component_name(name, 'backend')
    logger.debug(f"Looking up backend metadata: {name} → {full_name}")

    # Lookup in unified component index
    meta = _component_index.get(full_name)
    if not meta:
        available = list_backends()
        raise KeyError(_format_not_found_error('backend', full_name, available))

    # Load component to ensure metadata is populated
    _load_component(meta)

    # Build metadata dict from ComponentMetadata fields
    return {
        'class': meta.loaded_obj,
        'target_kernel': meta.backend_target,
        'language': meta.backend_language
    }


def list_backends_for_kernel(
    kernel: str,
    language: str | None = None,
    sources: list[str] | None = None
) -> list[str]:
    """List backends that target a specific kernel.

    Backends are stored directly on kernel metadata during discovery, so this
    is a simple field access followed by optional filtering.

    Args:
        kernel: Kernel name (with or without source prefix)
        language: Optional language filter ('hls' or 'rtl')
        sources: Optional list of sources to search (default: all sources)

    Returns:
        Sorted list of backend names (with source prefixes)

    Examples:
        >>> # All backends for LayerNorm
        >>> backends = list_backends_for_kernel('LayerNorm')
        >>> print(backends)  # ['brainsmith:LayerNorm_hls', 'user:LayerNorm_rtl', ...]

        >>> # Only HLS backends for LayerNorm
        >>> hls_backends = list_backends_for_kernel('LayerNorm', language='hls')

        >>> # Only user-provided backends
        >>> user_backends = list_backends_for_kernel('LayerNorm', sources=['user'])
    """
    if not _components_discovered:
        discover_components()

    # Resolve kernel name to full source:name format
    kernel_full = _resolve_component_name(kernel, 'kernel')

    # Get kernel metadata
    kernel_meta = _component_index.get(kernel_full)
    if not kernel_meta:
        available = list_kernels()
        raise KeyError(_format_not_found_error('kernel', kernel_full, available))

    # Get backends list from kernel metadata (populated during discovery)
    candidate_backends = kernel_meta.kernel_backends or []

    matching = []

    # Filter candidates by language and sources (typically small set: 1-5 backends)
    for backend_name in candidate_backends:
        meta = _component_index[backend_name]

        # Filter by sources if specified
        if sources and meta.source not in sources:
            continue

        # Filter by language if specified
        # Metadata already populated during discovery - no loading needed!
        if language and meta.backend_language != language:
            continue

        matching.append(backend_name)

    return sorted(matching)


def list_backends(source: str | None = None) -> list[str]:
    """List all available backends.

    Args:
        source: Optional source filter (e.g., 'user', 'brainsmith')

    Returns:
        Sorted list of backend names (with source prefixes)

    Examples:
        >>> backends = list_backends()  # All backends
        >>> user_backends = list_backends(source='user')  # Only user backends
    """
    return _list_components('backend', source)


# ============================================================================
# Public API: Component Metadata Access
# ============================================================================

def get_component_metadata(name: str, component_type: str):
    """Get component metadata without loading.

    Args:
        name: Component name (short or qualified)
        component_type: Component type ('step', 'kernel', 'backend')

    Returns:
        ComponentMetadata with source, import spec, and type-specific fields

    Raises:
        KeyError: If component not found

    Examples:
        >>> meta = get_component_metadata('LayerNorm', 'kernel')
        >>> print(meta.source, meta.import_spec.module)
        brainsmith brainsmith.kernels.layernorm
    """
    if not _components_discovered:
        discover_components()

    full_name = _resolve_component_name(name, component_type)

    if full_name not in _component_index:
        available = _list_components(component_type)
        raise KeyError(_format_not_found_error(component_type, full_name, available))

    return _component_index[full_name]


def get_all_component_metadata() -> dict[str, Any]:
    """Get all component metadata.

    Returns:
        Dict mapping full_name (source:name) to ComponentMetadata

    Examples:
        >>> all_components = get_all_component_metadata()
        >>> for name, meta in all_components.items():
        ...     print(f"{name}: {meta.component_type}")
    """
    if not _components_discovered:
        discover_components()

    return dict(_component_index)


# ============================================================================
# Domain Resolution
# ============================================================================

def get_domain_for_backend(backend_name: str) -> str:
    """Get ONNX domain for backend by deriving from module path.

    Derives domain directly from the backend class's __module__ attribute,
    eliminating the need for reverse lookup through source_module_prefixes.

    Args:
        backend_name: Backend name (short or qualified)

    Returns:
        ONNX domain string

    Examples:
        >>> get_domain_for_backend('brainsmith:LayerNorm_hls')
        'brainsmith.kernels'
        >>> get_domain_for_backend('finn:MVAU_hls')
        'finn.custom_op.fpgadataflow.hls'
    """
    from brainsmith.registry._domain_utils import derive_domain_from_module

    # Get backend metadata and load the class
    meta = get_component_metadata(backend_name, 'backend')

    # Load the backend class to access its __module__ attribute
    backend_class = _load_component(meta)

    # Derive domain directly from the backend class's module path
    domain = derive_domain_from_module(backend_class.__module__)

    logger.debug(f"Derived domain '{domain}' for backend '{backend_name}' from module '{backend_class.__module__}'")
    return domain
