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
from typing import Type, Optional, Dict, List, Any

from ._state import _component_index, _components_discovered
from ._discovery import (
    discover_components,
    _resolve_component_name,
    _load_component,
    _measure_load,
)
from ._metadata import resolve_lazy_class

logger = logging.getLogger(__name__)


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
            raise KeyError(
                f"{component_type.title()} '{full_name}' not found.\n"
                f"Available: {', '.join(available[:10])}" +
                (f" ... and {len(available) - 10} more" if len(available) > 10 else "")
            )

        # Load component
        _load_component(meta)

        # Return loaded object from component index
        return meta.loaded_obj


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


def _list_components(component_type: str, source: Optional[str] = None) -> List[str]:
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

    components = [
        meta.full_name for meta in _component_index.values()
        if meta.component_type == component_type and (source is None or meta.source == source)
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


def list_steps(source: Optional[str] = None) -> List[str]:
    """List all available steps.

    Args:
        source: Optional source filter (e.g., 'user', 'brainsmith')

    Returns:
        Sorted list of step names (with source prefixes)

    Example:
        >>> steps = list_steps()  # All steps
        >>> user_steps = list_steps(source='user')  # Only user steps
    """
    return _list_components('step', source)


# === Kernels ===

def get_kernel(name: str) -> Type:
    """Get kernel class.

    Accepts both short names ('LayerNorm') and qualified names ('user:LayerNorm').
    Short names are resolved using source priority from settings.

    Args:
        name: Kernel name - either short ('LayerNorm') or qualified ('user:LayerNorm')

    Returns:
        Kernel class

    Examples:
        >>> LayerNorm = get_kernel('LayerNorm')  # Short name, uses source priority
        >>> kernel = LayerNorm(onnx_node)
        >>> CustomKernel = get_kernel('user:CustomKernel')  # Qualified name, explicit source
    """
    return _get_component(name, 'kernel')


def get_kernel_infer(name: str) -> Type:
    """Get kernel's InferTransform class.

    Accepts both short names (uses default_source) and fully-qualified names (source:name).

    Args:
        name: Kernel name (e.g., 'LayerNorm' or 'user:LayerNorm')

    Returns:
        InferTransform class

    Raises:
        KeyError: If kernel has no InferTransform

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
        raise KeyError(f"Kernel '{full_name}' not found")

    # Load component to ensure metadata is populated
    _load_component(meta)

    # Check metadata for infer transform
    if meta.kernel_infer is None:
        raise KeyError(f"Kernel '{full_name}' has no InferTransform")

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


def list_kernels(source: Optional[str] = None) -> List[str]:
    """List all available kernels.

    Args:
        source: Optional source filter (e.g., 'user', 'brainsmith')

    Returns:
        Sorted list of kernel names (with source prefixes)

    Example:
        >>> kernels = list_kernels()  # All kernels
        >>> user_kernels = list_kernels(source='user')  # Only user kernels
    """
    return _list_components('kernel', source)


# === Backends ===

def get_backend(name: str) -> Type:
    """Get independent backend by name.

    Accepts both short names ('LayerNorm_HLS') and qualified names ('user:LayerNorm_HLS_Fast').
    Short names are resolved using source priority from settings.

    Args:
        name: Backend name - either short ('LayerNorm_HLS') or qualified ('user:LayerNorm_HLS_Fast')

    Returns:
        Backend class

    Examples:
        >>> backend = get_backend('LayerNorm_HLS')  # Short name, uses source priority
        >>> custom = get_backend('user:LayerNorm_HLS_Fast')  # Qualified name, explicit source
    """
    return _get_component(name, 'backend')


def get_backend_metadata(name: str) -> Dict[str, Any]:
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
        raise KeyError(
            f"Backend '{full_name}' not found.\n"
            f"Available backends: {', '.join(available[:10])}" +
            (f" ... and {len(available) - 10} more" if len(available) > 10 else "")
        )

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
    language: Optional[str] = None,
    sources: Optional[List[str]] = None
) -> List[str]:
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
        raise KeyError(
            f"Kernel '{kernel_full}' not found. "
            f"Available kernels: {', '.join(list_kernels()[:10])}"
        )

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


def list_backends(source: Optional[str] = None) -> List[str]:
    """List all available backends.

    Args:
        source: Optional source filter (e.g., 'user', 'brainsmith')

    Returns:
        Sorted list of backend names (with source prefixes)

    Example:
        >>> backends = list_backends()  # All backends
        >>> user_backends = list_backends(source='user')  # Only user backends
    """
    return _list_components('backend', source)


# ============================================================================
# Public API: Component Metadata Access
# ============================================================================

def get_component_metadata(name: str, component_type: str):
    """Get metadata for a component without loading it.

    Useful for inspection and CLI commands that need component information
    without triggering imports.

    Args:
        name: Component name (with or without source prefix)
        component_type: Type of component ('step', 'kernel', 'backend')

    Returns:
        ComponentMetadata object

    Raises:
        KeyError: If component not found

    Examples:
        >>> meta = get_component_metadata('LayerNorm', 'kernel')
        >>> print(meta.source, meta.import_spec.module)
        brainsmith brainsmith.kernels.layernorm.layernorm
    """
    if not _components_discovered:
        discover_components()

    full_name = _resolve_component_name(name, component_type)

    if full_name not in _component_index:
        raise KeyError(f"Component '{full_name}' not found")

    return _component_index[full_name]


def get_all_component_metadata() -> Dict[str, Any]:
    """Get all component metadata (for CLI/inspection).

    Returns a copy of the component index for safe iteration and inspection.

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
