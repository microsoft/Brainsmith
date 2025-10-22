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
from typing import Type, Optional, Dict, List, Any, Union

from brainsmith.registry import registry, source_context, kernel, backend, step

logger = logging.getLogger(__name__)

# Discovery state
_plugins_discovered = False

# Discovery mode cache
_discovery_mode = None  # 'editable', 'installed', or None


def is_editable_install() -> bool:
    """Detect if brainsmith is installed in editable mode.

    Editable mode (pip install -e .) is used during development, while
    regular installs are used in production. This distinction allows us to
    use different discovery strategies:

    - Editable: Use runtime discovery (existing deferred registry)
    - Installed: Use pre-generated entry points (fast)

    Returns:
        True if running from editable install, False otherwise

    Detection strategy:
        1. Check PEP 610 direct_url.json for editable marker
        2. Fallback: Check if brainsmith.__file__ is in site-packages

    Examples:
        >>> # During development (pip install -e .)
        >>> is_editable_install()
        True

        >>> # In production (pip install brainsmith)
        >>> is_editable_install()
        False
    """
    global _discovery_mode

    # Use cached result if available
    if _discovery_mode is not None:
        return _discovery_mode == 'editable'

    # Strategy 1: Check PEP 610 direct_url.json (most reliable)
    try:
        from importlib.metadata import distribution
        import json

        dist = distribution('brainsmith')
        direct_url_data = dist.read_text('direct_url.json')

        if direct_url_data:
            direct_url = json.loads(direct_url_data)
            if direct_url.get('dir_info', {}).get('editable'):
                _discovery_mode = 'editable'
                logger.debug("Detected editable install (via direct_url.json)")
                return True

    except Exception as e:
        logger.debug(f"Could not check direct_url.json: {e}")

    # Strategy 2: Check if brainsmith.__file__ is in site-packages (fallback)
    try:
        import brainsmith
        brainsmith_file = Path(brainsmith.__file__)

        # In editable install, __file__ points to source tree
        # In regular install, __file__ points to site-packages
        if 'site-packages' in str(brainsmith_file):
            _discovery_mode = 'installed'
            logger.debug("Detected regular install (brainsmith in site-packages)")
            return False
        else:
            _discovery_mode = 'editable'
            logger.debug("Detected editable install (brainsmith not in site-packages)")
            return True

    except Exception as e:
        logger.warning(f"Could not detect install mode: {e}, assuming installed")
        _discovery_mode = 'installed'
        return False


def _resolve_component_name(name: str, component_type: str = 'step') -> str:
    """Resolve component name to source:name format with intelligent fallback.

    If name already has source prefix (contains ':'), return as-is.
    Otherwise, try sources in priority order until found:
    1. default_source (from config, usually 'brainsmith')
    2. finn
    3. custom
    4. First match in other registered sources

    Args:
        name: Component name (e.g., 'LayerNorm' or 'user:LayerNorm')
        component_type: Type of component ('step', 'kernel', 'backend')

    Returns:
        Full name with source prefix (e.g., 'brainsmith:LayerNorm')

    Examples:
        >>> _resolve_component_name('qonnx_to_finn', 'step')
        'brainsmith:qonnx_to_finn'  # Found in brainsmith
        >>> _resolve_component_name('create_dataflow_partition', 'step')
        'finn:create_dataflow_partition'  # Not in brainsmith, fallback to finn
    """
    if ':' in name:
        return name  # Already has source prefix

    try:
        from brainsmith.settings import get_config
        default_source = get_config().default_source
    except Exception:
        # Fallback if config not available
        default_source = 'brainsmith'

    # Get the appropriate registry dict
    if component_type == 'step':
        registry_dict = registry._steps
    elif component_type == 'kernel':
        registry_dict = registry._kernels
    elif component_type == 'backend':
        registry_dict = registry._backends
    else:
        # Unknown type, just use default source
        return f"{default_source}:{name}"

    # Try sources in priority order
    sources_to_try = [default_source, 'finn', 'custom']

    # Add other registered sources (from plugin_sources)
    try:
        from brainsmith.settings import get_config
        for source_name in get_config().plugin_sources.keys():
            if source_name not in sources_to_try:
                sources_to_try.append(source_name)
    except Exception:
        pass

    # Try each source until we find the component
    for source in sources_to_try:
        full_name = f"{source}:{name}"
        if full_name in registry_dict:
            return full_name

    # Not found in any source, return default
    return f"{default_source}:{name}"


def discover_plugins():
    """Discover and load all plugins from configured sources.

    This loads:
    1. Core brainsmith plugins (kernels, steps) via direct imports
    2. User plugins from plugin_sources (must have __init__.py)
    3. Entry points from pip-installed packages (e.g., FINN)

    Plugins self-register during import using the global registry.

    This is called automatically on first component lookup.
    """
    global _plugins_discovered
    if _plugins_discovered:
        return

    logger.info("Discovering plugins...")

    # 1. Load core brainsmith plugins (triggers self-registration)
    import brainsmith.kernels
    import brainsmith.steps

    # 2. Load user plugins from configured sources
    _load_user_plugins()

    # 3. Load entry point plugins (FINN, etc.)
    _load_entry_point_plugins()

    # 4. Process all deferred registrations (brainsmith components use decorators)
    _process_deferred_registrations()

    _plugins_discovered = True

    logger.info(
        f"Plugin discovery complete: {registry}"
    )


def _load_user_plugins():
    """Load user plugins from configured plugin sources.

    Scans plugin_sources from config, imports each package that has
    an __init__.py file. The __init__.py must import and register
    its components.
    """
    try:
        from brainsmith.settings import get_config
        plugin_sources = get_config().plugin_sources
    except Exception as e:
        logger.warning(f"Could not load plugin sources config: {e}")
        return

    for source_name, source_path in plugin_sources.items():
        # Skip protected sources:
        # - brainsmith: loaded via direct import
        # - finn: loaded via entry points
        # - project, user: optional __init__.py-based plugins
        from brainsmith.settings.schema import PROTECTED_SOURCES
        if source_name in ('brainsmith', 'finn'):
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
    init_path = source_path / '__init__.py'

    with source_context(source_name):
        try:
            # Handle module name collisions using spec-based import
            # Multiple plugin sources may have the same directory name (e.g., 'plugins')
            # Use importlib.util to import from specific file path to avoid sys.path ambiguity

            import importlib.util

            # Create a unique module name to avoid cache collisions
            unique_module_name = f"{source_name}__{module_name}"

            spec = importlib.util.spec_from_file_location(unique_module_name, init_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[unique_module_name] = module
                spec.loader.exec_module(module)
                logger.info(f"Loaded plugin source '{source_name}' from {source_path}")
            else:
                raise ImportError(f"Could not create module spec for {init_path}")

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
    should return a dict of component metadata that we register.

    Entry points are registered with their entry point name as source.
    """
    logger.debug("Scanning entry points")

    try:
        eps = entry_points(group='brainsmith.plugins')

        for ep in eps:
            source_name = ep.name  # Use entry point name (e.g., 'finn') as source

            try:
                # Load the entry point function
                register_func = ep.load()

                # Call it to get component metadata
                # Plugin returns: {'kernels': [...], 'backends': [...], 'steps': [...]}
                components = register_func()

                if not isinstance(components, dict):
                    logger.error(f"Entry point '{ep.name}' returned {type(components)}, expected dict")
                    continue

                logger.info(f"Loading plugin source: {source_name}")

                # Register all components under this source
                with source_context(source_name):
                    # Register kernels
                    for kernel_meta in components.get('kernels', []):
                        # Debug: check if infer_transform is present
                        if kernel_meta.get('infer_transform'):
                            logger.debug(f"  Kernel {kernel_meta['name']} has infer_transform: {kernel_meta['infer_transform']}")
                        registry.kernel(
                            kernel_meta['class'],
                            name=kernel_meta['name'],
                            infer_transform=kernel_meta.get('infer_transform')
                        )

                    # Register backends
                    for backend_meta in components.get('backends', []):
                        registry.backend(
                            backend_meta['class'],
                            name=backend_meta['name'],
                            target_kernel=backend_meta['target_kernel'],
                            language=backend_meta['language']
                        )

                    # Register steps
                    for step_meta in components.get('steps', []):
                        registry.step(
                            step_meta['func'],
                            name=step_meta['name']
                        )

                logger.info(
                    f"✓ Loaded {source_name}: "
                    f"{len(components.get('kernels', []))} kernels, "
                    f"{len(components.get('backends', []))} backends, "
                    f"{len(components.get('steps', []))} steps"
                )

            except Exception as e:
                logger.error(f"Failed to load entry point '{ep.name}': {e}")

                # Check if strict mode
                try:
                    from brainsmith.settings import get_config
                    if get_config().plugins_strict:
                        raise
                except ImportError:
                    pass  # Config not available, don't fail

    except Exception as e:
        logger.warning(f"Entry point discovery failed: {e}")


# === Deferred Registration Processing ===

def _detect_source_for_component(obj):
    """Detect source from component's module.

    Similar to Registry._detect_source but works for any object.

    Args:
        obj: Function or class to detect source for

    Returns:
        Source name (e.g., 'brainsmith', 'finn', 'custom')

    Source detection priority:
    1. brainsmith.* → 'brainsmith'
    2. finn.* → 'finn'
    3. qonnx.* → 'qonnx'
    4. Under plugin_sources path → source name
    5. Otherwise → 'custom' (for eager-registered components not in known packages)
    """
    import inspect

    # Get module
    module = inspect.getmodule(obj)
    if module is None:
        logger.warning(f"Could not detect module for {obj}, using 'custom' source")
        return 'custom'

    module_name = module.__name__

    # Check for core packages
    if module_name.startswith('brainsmith.'):
        return 'brainsmith'
    elif module_name.startswith('finn.'):
        return 'finn'
    elif module_name.startswith('qonnx.'):
        return 'qonnx'

    # Check plugin_sources
    try:
        from brainsmith.settings import get_config
        module_file = getattr(module, '__file__', None)

        if module_file:
            for source_name, source_path in get_config().plugin_sources.items():
                # Skip core sources
                if source_name in ('brainsmith', 'finn', 'qonnx'):
                    continue

                # Check if module is under this plugin source
                module_path = Path(module_file)
                try:
                    module_path.relative_to(source_path)
                    return source_name
                except ValueError:
                    continue  # Not relative to this source

        # Fallback for eager-registered components not in known packages
        return 'custom'

    except Exception as e:
        logger.warning(f"Could not detect source for {obj}: {e}, using 'custom'")
        return 'custom'


def _process_deferred_registrations():
    """Process all deferred registrations.

    Called lazily on first component lookup.
    Thread-safe and idempotent.
    """
    # Get the actual registry module from sys.modules
    # (bypasses lazy loading in brainsmith/__init__.py)
    import sys
    reg_module = sys.modules['brainsmith.registry']

    # Quick check without lock - only skip if there's nothing to process
    if reg_module._registration_processed and not (
        reg_module._deferred_steps or
        reg_module._deferred_kernels or
        reg_module._deferred_backends
    ):
        return

    # Acquire lock for processing
    with reg_module._registration_lock:
        # Double-check after acquiring lock
        if not (reg_module._deferred_steps or reg_module._deferred_kernels or reg_module._deferred_backends):
            return

        logger.info(
            f"Processing deferred registrations: "
            f"{len(reg_module._deferred_steps)} steps, "
            f"{len(reg_module._deferred_kernels)} kernels, "
            f"{len(reg_module._deferred_backends)} backends"
        )

        conflicts = {
            'steps': [],
            'kernels': [],
            'backends': []
        }

        # Process steps
        for func in reg_module._deferred_steps:
            meta = func.__brainsmith_step__
            source = _detect_source_for_component(func)

            full_name = f"{source}:{meta['name']}"

            # Check for conflicts
            if full_name in registry._steps:
                conflicts['steps'].append({
                    'name': full_name,
                    'existing': registry._steps[full_name],
                    'new': func
                })

            # Register (using source context)
            with source_context(source):
                registry.step(func, name=meta['name'])

        # Process kernels
        for cls in reg_module._deferred_kernels:
            meta = cls.__brainsmith_kernel__
            source = _detect_source_for_component(cls)

            full_name = f"{source}:{meta['name']}"

            if full_name in registry._kernels:
                conflicts['kernels'].append({
                    'name': full_name,
                    'existing': registry._kernels[full_name],
                    'new': cls
                })

            with source_context(source):
                registry.kernel(
                    cls,
                    name=meta['name'],
                    infer_transform=meta.get('infer_transform'),
                    domain=meta.get('domain')
                )

        # Process backends
        for cls in reg_module._deferred_backends:
            meta = cls.__brainsmith_backend__
            source = _detect_source_for_component(cls)

            full_name = f"{source}:{meta['name']}"

            if full_name in registry._backends:
                conflicts['backends'].append({
                    'name': full_name,
                    'existing': registry._backends[full_name],
                    'new': cls
                })

            with source_context(source):
                registry.backend(
                    cls,
                    name=meta['name'],
                    target_kernel=meta.get('target_kernel'),
                    language=meta.get('language'),
                    variant=meta.get('variant')
                )

        # Log conflicts
        total_conflicts = sum(len(v) for v in conflicts.values())
        if total_conflicts > 0:
            logger.warning(f"Found {total_conflicts} registration conflicts:")
            for component_type, conflict_list in conflicts.items():
                for c in conflict_list:
                    existing = c['existing']
                    new = c['new']

                    if hasattr(existing, '__module__'):
                        existing_loc = f"{existing.__module__}.{existing.__name__}"
                    else:
                        existing_loc = str(existing)

                    logger.warning(
                        f"  {c['name']}: {existing_loc} "
                        f"overridden by {new.__module__}.{new.__name__}"
                    )

        # Mark as processed
        reg_module._registration_processed = True

        # Clear deferred lists (free memory)
        reg_module._deferred_steps.clear()
        reg_module._deferred_kernels.clear()
        reg_module._deferred_backends.clear()

        logger.info("Deferred registration complete")


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
    # Process deferred registrations first
    _process_deferred_registrations()

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

    step_callable = registry._steps[full_name]
    return step_callable


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

    # Process deferred registrations after plugins discovered
    _process_deferred_registrations()

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
    # Process deferred registrations first
    _process_deferred_registrations()

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
    kernel_class = meta['class']
    return kernel_class


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

    infer_class = meta['infer']
    return infer_class


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
    # Process deferred registrations first
    _process_deferred_registrations()

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
    backend_class = meta['class']
    return backend_class


def get_backend_metadata(name: str) -> Dict[str, Any]:
    """Get backend metadata.

    Args:
        name: Backend name (with or without source prefix)

    Returns:
        Backend metadata dict with 'class', 'target_kernel', 'language' keys

    Raises:
        KeyError: If backend not found
    """
    # Process deferred registrations first
    _process_deferred_registrations()

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
