# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Namespace-based plugin loader for Brainsmith.

This module provides plugin discovery and lookup for Brainsmith components.
All plugins are namespace-based - they must have __init__.py and explicitly
register components.

Discovery sources (in order):
1. Core (brainsmith, finn, qonnx) - from brainsmith.plugins
2. User plugins - from configured plugin_sources
3. Entry points - from pip-installed packages

All components use source-prefixed names (e.g., 'brainsmith:LayerNorm').
"""

import os
import sys
import importlib
import logging
from pathlib import Path
from importlib.metadata import entry_points
from typing import Type, Optional, Dict, List, Any

from brainsmith.registry import registry, source_context

logger = logging.getLogger(__name__)

# Discovery state
_plugins_discovered = False


def _resolve_component_name(name: str, component_type: str = 'step') -> str:
    """Resolve component name to source:name format.

    If name already has source prefix (contains ':'), return as-is.
    Otherwise, prepend with default_source from config.

    Args:
        name: Component name (e.g., 'LayerNorm' or 'user:LayerNorm')
        component_type: Type of component for logging (not used functionally)

    Returns:
        Full name with source prefix (e.g., 'brainsmith:LayerNorm')

    Examples:
        >>> _resolve_component_name('streamline')
        'brainsmith:streamline'
        >>> _resolve_component_name('user:custom')
        'user:custom'
    """
    if ':' in name:
        return name  # Already has source prefix

    try:
        from brainsmith.settings import get_config
        default_source = get_config().default_source
    except Exception:
        # Fallback if config not available
        default_source = 'brainsmith'

    return f"{default_source}:{name}"


def discover_plugins():
    """Discover and load all plugins from configured sources.

    This loads:
    1. Core plugins (brainsmith, finn, qonnx) via import of brainsmith.plugins
    2. User plugins from plugin_sources (must have __init__.py)
    3. Entry points from pip-installed packages

    Plugins self-register during import using the global registry.

    This is called automatically on first component lookup.
    """
    global _plugins_discovered
    if _plugins_discovered:
        return

    logger.info("Discovering plugins...")

    # 1. Load core plugins (brainsmith, finn, qonnx)
    _load_core_plugins()

    # 2. Load user plugins from configured sources
    _load_user_plugins()

    # 3. Load entry point plugins
    _load_entry_point_plugins()

    _plugins_discovered = True

    logger.info(
        f"Plugin discovery complete: {registry}"
    )


def _load_core_plugins():
    """Load core brainsmith/finn/qonnx plugins.

    Core plugins are loaded by importing brainsmith.plugins, which
    registers all core components.
    """
    try:
        import brainsmith.plugins  # noqa: F401
        logger.debug("Loaded core plugins")
    except ImportError as e:
        logger.warning(f"Failed to load core plugins: {e}")


def _load_user_plugins():
    """Load user plugins from configured plugin sources.

    Scans plugin_sources from config, imports each package that has
    an __init__.py file. The __init__.py must import and register
    its components.
    """
    try:
        from brainsmith.settings import get_config
        plugin_sources = get_config().effective_plugin_sources
    except Exception as e:
        logger.warning(f"Could not load plugin sources config: {e}")
        return

    for source_name, source_path in plugin_sources.items():
        # Skip core sources (already loaded)
        if source_name in ('brainsmith', 'finn', 'qonnx'):
            continue

        if not source_path.exists():
            logger.debug(f"Plugin source '{source_name}' does not exist: {source_path}")
            continue

        _load_plugin_package(source_name, source_path)


def _load_plugin_package(source_name: str, source_path: Path):
    """Load a plugin package from filesystem path.

    Args:
        source_name: Source name for registration (e.g., 'user', 'team')
        source_path: Path to plugin package directory

    The package must have __init__.py which registers components.
    """
    init_file = source_path / '__init__.py'
    if not init_file.exists():
        logger.warning(
            f"Plugin source '{source_name}' has no __init__.py, skipping. "
            f"Namespace-based plugins must have __init__.py that registers components. "
            f"Path: {source_path}"
        )
        return

    logger.debug(f"Loading plugin source '{source_name}' from {source_path}")

    # Add parent directory to sys.path for imports
    parent = str(source_path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
        logger.debug(f"Added to sys.path: {parent}")

    # Import package within source context
    module_name = source_path.name

    with source_context(source_name):
        try:
            importlib.import_module(module_name)
            logger.info(f"Loaded plugin source '{source_name}' from {source_path}")
        except Exception as e:
            logger.error(f"Failed to load plugin source '{source_name}': {e}")

            # Check if strict mode
            try:
                from brainsmith.settings import get_config
                if get_config().plugins_strict:
                    raise
            except ImportError:
                pass  # Config not available, don't fail


def _load_entry_point_plugins():
    """Load plugins from pip package entry points.

    Scans entry points in group 'brainsmith.plugins'. Each entry point
    should point to a module that registers components when imported.

    Entry points are registered with their distribution name as source.
    """
    logger.debug("Scanning entry points")

    try:
        eps = entry_points(group='brainsmith.plugins')

        for ep in eps:
            source_name = ep.dist.name if hasattr(ep, 'dist') else 'pkg'

            with source_context(source_name):
                try:
                    ep.load()  # Import module, which self-registers
                    logger.info(f"Loaded entry point plugin '{source_name}' from {ep.value}")
                except Exception as e:
                    logger.error(f"Failed to load entry point '{ep.name}': {e}")

    except Exception as e:
        logger.warning(f"Entry point discovery failed: {e}")


# === Steps ===

def get_step(name: str):
    """Get step callable by name.

    Accepts both short names (uses default_source) and fully-qualified names (source:name).

    Args:
        name: Step name (e.g., 'streamline' or 'user:custom_step')

    Returns:
        Callable step function or Step instance

    Raises:
        KeyError: If step not found

    Examples:
        >>> streamline = get_step('streamline')  # Uses default_source
        >>> custom = get_step('user:custom_step')  # Explicit source
    """
    if not _plugins_discovered:
        discover_plugins()

    full_name = _resolve_component_name(name, 'step')
    logger.debug(f"Looking up step: {name} → {full_name}")

    if full_name not in registry._steps:
        available = list_steps()
        raise KeyError(
            f"Step '{full_name}' not found.\n"
            f"Available steps: {', '.join(available[:10])}" +
            (f" ... and {len(available) - 10} more" if len(available) > 10 else "")
        )

    return registry._steps[full_name]


def has_step(name: str) -> bool:
    """Check if step exists without importing it.

    Args:
        name: Step name (with or without source prefix)

    Returns:
        True if step exists

    Examples:
        >>> if has_step('streamline'):
        ...     step = get_step('streamline')
    """
    if not _plugins_discovered:
        discover_plugins()

    full_name = _resolve_component_name(name, 'step')
    return full_name in registry._steps


def list_steps(source: Optional[str] = None) -> List[str]:
    """List all available steps.

    Args:
        source: Optional source filter (e.g., 'user', 'brainsmith')

    Returns:
        Sorted list of step names (with source prefixes)

    Examples:
        >>> steps = list_steps()
        >>> print(steps[:3])  # ['brainsmith:qonnx_to_finn', ...]
        >>> user_steps = list_steps(source='user')
        >>> print(user_steps)  # ['user:custom_step', ...]
    """
    if not _plugins_discovered:
        discover_plugins()

    all_steps = set(registry._steps.keys())

    # Filter by source if requested
    if source:
        all_steps = {s for s in all_steps if s.startswith(f"{source}:")}

    return sorted(all_steps)


# === Kernels ===

def get_kernel(name: str) -> Type:
    """Get kernel class.

    Accepts both short names (uses default_source) and fully-qualified names (source:name).

    Args:
        name: Kernel name (e.g., 'LayerNorm' or 'user:LayerNorm')

    Returns:
        Kernel class

    Examples:
        >>> LayerNorm = get_kernel('LayerNorm')  # Uses default_source
        >>> kernel = LayerNorm(onnx_node)
        >>> CustomKernel = get_kernel('user:CustomKernel')  # Explicit source
    """
    meta = _get_kernel_metadata(name)
    return meta['class']


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
    meta = _get_kernel_metadata(name)

    if meta['infer'] is None:
        full_name = _resolve_component_name(name, 'kernel')
        raise KeyError(f"Kernel '{full_name}' has no InferTransform")

    return meta['infer']


def has_kernel(name: str) -> bool:
    """Check if kernel exists.

    Args:
        name: Kernel name (with or without source prefix)

    Returns:
        True if kernel exists

    Examples:
        >>> if has_kernel('LayerNorm'):
        ...     kernel = get_kernel('LayerNorm')
    """
    if not _plugins_discovered:
        discover_plugins()

    full_name = _resolve_component_name(name, 'kernel')
    return full_name in registry._kernels


def list_kernels(source: Optional[str] = None) -> List[str]:
    """List all available kernels.

    Args:
        source: Optional source filter (e.g., 'user', 'brainsmith')

    Returns:
        Sorted list of kernel names (with source prefixes)

    Examples:
        >>> kernels = list_kernels()
        >>> print(kernels[:3])  # ['brainsmith:LayerNorm', ...]
        >>> user_kernels = list_kernels(source='user')
        >>> print(user_kernels)  # ['user:CustomKernel', ...]
    """
    if not _plugins_discovered:
        discover_plugins()

    all_kernels = set(registry._kernels.keys())

    # Filter by source if requested
    if source:
        all_kernels = {k for k in all_kernels if k.startswith(f"{source}:")}

    return sorted(all_kernels)


def _get_kernel_metadata(name: str) -> Dict[str, Any]:
    """Get kernel metadata from registry.

    Args:
        name: Kernel name (with or without source prefix)

    Returns:
        Kernel metadata dict with 'class', 'infer', 'op_type' keys

    Raises:
        KeyError: If kernel not found
    """
    if not _plugins_discovered:
        discover_plugins()

    full_name = _resolve_component_name(name, 'kernel')
    logger.debug(f"Looking up kernel metadata: {name} → {full_name}")

    if full_name not in registry._kernels:
        available = list_kernels()
        raise KeyError(
            f"Kernel '{full_name}' not found.\n"
            f"Available kernels: {', '.join(available[:10])}" +
            (f" ... and {len(available) - 10} more" if len(available) > 10 else "")
        )

    return registry._kernels[full_name]


# === Backends ===

def get_backend(name: str) -> Type:
    """Get independent backend by name.

    Accepts both short names (uses default_source) and fully-qualified names (source:name).

    Args:
        name: Backend name (e.g., 'LayerNorm_HLS' or 'user:LayerNorm_HLS_Fast')

    Returns:
        Backend class

    Examples:
        >>> backend = get_backend('LayerNorm_HLS')  # Uses default_source
        >>> custom = get_backend('user:LayerNorm_HLS_Fast')  # Explicit source
    """
    meta = get_backend_metadata(name)
    return meta['class']


def get_backend_metadata(name: str) -> Dict[str, Any]:
    """Get backend metadata.

    Args:
        name: Backend name (with or without source prefix)

    Returns:
        Backend metadata dict with 'class', 'target_kernel', 'language' keys

    Raises:
        KeyError: If backend not found
    """
    if not _plugins_discovered:
        discover_plugins()

    full_name = _resolve_component_name(name, 'backend')
    logger.debug(f"Looking up backend metadata: {name} → {full_name}")

    if full_name not in registry._backends:
        available = list_all_backends()
        raise KeyError(
            f"Backend '{full_name}' not found.\n"
            f"Available backends: {', '.join(available[:10])}" +
            (f" ... and {len(available) - 10} more" if len(available) > 10 else "")
        )

    return registry._backends[full_name]


def list_backends_for_kernel(
    kernel: str,
    language: Optional[str] = None,
    sources: Optional[List[str]] = None
) -> List[str]:
    """List backends that target a specific kernel.

    Args:
        kernel: Kernel name (with or without source prefix)
        language: Optional language filter ('hls' or 'rtl')
        sources: Optional list of sources to search (default: all sources)

    Returns:
        Sorted list of backend names (with source prefixes)

    Examples:
        >>> # All backends for LayerNorm
        >>> backends = list_backends_for_kernel('LayerNorm')
        >>> print(backends)  # ['brainsmith:LayerNorm_HLS', 'user:LayerNorm_HLS_Fast', ...]

        >>> # Only HLS backends for LayerNorm
        >>> hls_backends = list_backends_for_kernel('LayerNorm', language='hls')

        >>> # Only user-provided backends
        >>> user_backends = list_backends_for_kernel('LayerNorm', sources=['user'])
    """
    if not _plugins_discovered:
        discover_plugins()

    # Resolve kernel name to full source:name format
    kernel_full = _resolve_component_name(kernel, 'kernel')

    # Filter backends
    matching = []
    for backend_name, backend_meta in registry._backends.items():
        if backend_meta['target_kernel'] != kernel_full:
            continue

        # Filter by language if specified
        if language and backend_meta['language'] != language:
            continue

        # Filter by sources if specified
        if sources:
            backend_source = backend_name.split(':', 1)[0]
            if backend_source not in sources:
                continue

        matching.append(backend_name)

    return sorted(matching)


def list_all_backends(source: Optional[str] = None) -> List[str]:
    """List all available backends.

    Args:
        source: Optional source filter (e.g., 'user', 'brainsmith')

    Returns:
        Sorted list of backend names (with source prefixes)

    Examples:
        >>> backends = list_all_backends()
        >>> print(backends[:3])  # ['brainsmith:LayerNorm_HLS', ...]
        >>> user_backends = list_all_backends(source='user')
    """
    if not _plugins_discovered:
        discover_plugins()

    all_backends = set(registry._backends.keys())

    # Filter by source if requested
    if source:
        all_backends = {b for b in all_backends if b.startswith(f"{source}:")}

    return sorted(all_backends)


# Auto-discover on module import (lazy)
# Actual discovery happens on first component lookup
