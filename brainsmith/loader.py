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
import time
from contextlib import contextmanager
from pathlib import Path
from importlib.metadata import entry_points
from typing import Type, Optional, Dict, List, Any, Union

from brainsmith.registry import registry, source_context, kernel, backend, step

logger = logging.getLogger(__name__)

# ============================================================================
# Performance Instrumentation
# ============================================================================

# Collection of load metrics for performance analysis
_load_metrics: List[Dict[str, Any]] = []


@contextmanager
def _measure_load(operation: str, component: str = ""):
    """Measure and log component loading performance.

    Args:
        operation: Description of operation (e.g., 'get_kernel', 'discover_plugins')
        component: Optional component name being loaded

    Example:
        >>> with _measure_load('get_kernel', 'LayerNorm'):
        ...     kernel = get_kernel('LayerNorm')
    """
    label = f"{operation}({component})" if component else operation
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"⏱️  {label}: {duration_ms:.1f}ms")

        # Collect metrics for analysis
        _load_metrics.append({
            'operation': operation,
            'component': component,
            'duration_ms': duration_ms,
            'timestamp': time.time()
        })


def get_load_metrics() -> List[Dict[str, Any]]:
    """Return collected load metrics for analysis.

    Returns:
        List of metric dicts with keys: operation, component, duration_ms, timestamp
    """
    return _load_metrics.copy()


def reset_load_metrics():
    """Clear collected metrics (useful for testing)."""
    _load_metrics.clear()

# === Plugin Discovery State Management ===

class PluginDiscovery:
    """Centralized plugin discovery state.

    Manages discovery status, install mode, and loaded plugin modules.
    Replaces scattered module-level globals with a cohesive state object.
    """

    def __init__(self):
        self.discovered = False
        self.install_mode = None  # 'editable', 'installed', or None
        self.loaded_modules = {}  # source_name -> module or dict of modules

    def is_editable_install(self) -> bool:
        """Check if brainsmith is installed in editable mode.

        Uses PEP 610 direct_url.json as the standard detection method.
        Caches result for performance.

        Returns:
            True if running from editable install, False otherwise
        """
        # Use cached result if available
        if self.install_mode is not None:
            return self.install_mode == 'editable'

        # Check PEP 610 direct_url.json
        try:
            from importlib.metadata import distribution
            import json

            dist = distribution('brainsmith')
            direct_url_data = dist.read_text('direct_url.json')

            if direct_url_data:
                direct_url = json.loads(direct_url_data)
                if direct_url.get('dir_info', {}).get('editable'):
                    self.install_mode = 'editable'
                    logger.debug("Detected editable install")
                    return True

        except Exception as e:
            logger.debug(f"Could not detect editable install: {e}")

        # Default to regular install
        self.install_mode = 'installed'
        return False


# Global plugin discovery singleton
_discovery = PluginDiscovery()

# Lazy entry point steps - deferred imports for expensive modules (e.g. FINN)
# Maps "source:name" -> {'name': str, 'module': str, 'func_name': str}
# Steps stored here are NOT in registry yet - import deferred until get_step()
_lazy_entry_point_steps: Dict[str, Dict[str, str]] = {}

# Maps "source:name" -> {'name': str, 'module': str, 'class_name': str, 'infer_transform': {...}}
# Kernels stored here are NOT in registry yet - import deferred until get_kernel()
_lazy_entry_point_kernels: Dict[str, Dict[str, Any]] = {}

# Maps "source:name" -> {'name': str, 'module': str, 'class_name': str, 'target_kernel': str, 'language': str}
# Backends stored here are NOT in registry yet - import deferred until get_backend()
_lazy_entry_point_backends: Dict[str, Dict[str, str]] = {}


def is_editable_install() -> bool:
    """Detect if brainsmith is installed in editable mode.

    Delegates to the global PluginDiscovery singleton.

    Returns:
        True if running from editable install, False otherwise
    """
    return _discovery.is_editable_install()


def _resolve_component_name(name: str, component_type: str = 'step') -> str:
    """Resolve component name to source:name format.

    If name contains ':', return as-is (explicit source).
    Otherwise, search sources in priority order from config.

    Args:
        name: Component name (e.g., 'LayerNorm' or 'user:LayerNorm')
        component_type: Type of component ('step', 'kernel', 'backend')

    Returns:
        Full name with source prefix (e.g., 'brainsmith:LayerNorm' or 'finn:DuplicateStreams')

    Note: Uses source_priority from config (default: project > user > brainsmith > finn).
    """
    if ':' in name:
        return name  # Already has source prefix

    # Get source priority from config
    try:
        from brainsmith.settings import get_config
        source_priority = get_config().source_priority
    except Exception:
        source_priority = ['project', 'user', 'brainsmith', 'finn']  # Ultimate fallback

    # Helper to check if component exists in a specific source
    def check_source(source: str) -> bool:
        # Map component type to registry and lazy storage
        if component_type == 'kernel':
            registry_dict = registry._kernels
            lazy_dict = _lazy_entry_point_kernels
            components_key = 'kernels'
        elif component_type == 'backend':
            registry_dict = registry._backends
            lazy_dict = _lazy_entry_point_backends
            components_key = 'backends'
        else:  # step
            registry_dict = registry._steps
            lazy_dict = _lazy_entry_point_steps
            components_key = 'steps'

        full_name = f"{source}:{name}"

        # Check registry
        if full_name in registry_dict:
            return True

        # Check lazy entry points
        if full_name in lazy_dict:
            return True

        # Check lazy COMPONENTS
        if source in _discovery.loaded_modules:
            modules_or_module = _discovery.loaded_modules[source]

            # Core brainsmith: dict with component type keys
            if isinstance(modules_or_module, dict):
                module = modules_or_module.get(components_key)
            # User plugins: single module
            else:
                module = modules_or_module

            if module and hasattr(module, 'COMPONENTS'):
                if name in module.COMPONENTS.get(components_key, {}):
                    return True

        return False

    # Check sources in priority order
    if _discovery.discovered:
        for source in source_priority:
            if check_source(source):
                return f"{source}:{name}"

    # Not found - use first priority source for error messages
    return f"{source_priority[0]}:{name}"


def discover_plugins():
    """Discover and load all plugins from configured sources.

    This loads:
    1. Core brainsmith plugins (kernels, steps) via direct imports
    2. User plugins from plugin_sources (must have __init__.py)
    3. Entry points from pip-installed packages (e.g., FINN)

    Plugins self-register during import using the global registry.

    This is called automatically on first component lookup.
    """
    if _discovery.discovered:
        return

    with _measure_load('discover_plugins'):
        logger.info("Discovering plugins...")

        # 1. Load core brainsmith plugins (triggers self-registration)
        import brainsmith.kernels
        import brainsmith.steps

        # Cache core module references for lazy loading
        _discovery.loaded_modules['brainsmith'] = {
            'kernels': brainsmith.kernels,
            'steps': brainsmith.steps,
        }

        # 2. Load user plugins from configured sources
        _load_user_plugins()

        # 3. Load entry point plugins (FINN, etc.)
        _load_entry_point_plugins()

        _discovery.discovered = True

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


# Note: Plugin modules are now tracked in _discovery.loaded_modules
# Maps source_name -> module reference or dict of module references
# - Core brainsmith: {'kernels': module, 'steps': module}
# - User plugins: module (single __init__.py with COMPONENTS)
# - Entry points: Not tracked here (register components directly)


def _load_plugin_package(source_name: str, source_path: Path):
    """Load a plugin package from filesystem path.

    Supports two patterns:
    1. Lazy: Plugin has COMPONENTS dict -> register with lazy loaders
    2. Eager: Plugin uses decorators -> import triggers registration

    Args:
        source_name: Source name for registration (e.g., 'user', 'team')
        source_path: Path to plugin package directory

    The package must have __init__.py which either:
    - Exports COMPONENTS dict (lazy loading, recommended)
    - Imports modules with @kernel/@step decorators (eager, legacy)
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

    # Import package
    module_name = source_path.name
    init_path = source_path / '__init__.py'

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

            # Store module reference for lazy loading
            _discovery.loaded_modules[source_name] = module
        else:
            raise ImportError(f"Could not create module spec for {init_path}")

        # Both lazy and eager patterns work:
        # - Lazy: Plugin has COMPONENTS dict, components loaded on first access
        # - Eager: Decorators already fired during import, components in registry

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
                        # Support two patterns:
                        # 1. Eager: {'name': 'x', 'class': <class>, 'infer_transform': <class>}
                        # 2. Lazy: {'name': 'x', 'module': '...', 'class_name': '...', 'infer_transform': {...}}
                        #    Store metadata only - defer import until get_kernel() is called

                        if 'class' in kernel_meta:
                            # Pattern 1: Class already loaded - register immediately (legacy)
                            registry.kernel(
                                kernel_meta['class'],
                                name=kernel_meta['name'],
                                infer_transform=kernel_meta.get('infer_transform')
                            )

                        elif 'module' in kernel_meta and 'class_name' in kernel_meta:
                            # Pattern 2: Store metadata for lazy loading
                            # Import will happen on first get_kernel() call (avoids expensive imports)
                            full_name = f"{source_name}:{kernel_meta['name']}"
                            _lazy_entry_point_kernels[full_name] = kernel_meta
                            logger.debug(f"Registered lazy kernel: {full_name}")

                        else:
                            logger.error(
                                f"Invalid kernel metadata for {kernel_meta.get('name', 'unknown')}: "
                                f"missing 'class' or ('module' and 'class_name')"
                            )

                    # Register backends
                    for backend_meta in components.get('backends', []):
                        # Support two patterns:
                        # 1. Eager: {'name': 'x', 'class': <class>, 'target_kernel': ..., 'language': ...}
                        # 2. Lazy: {'name': 'x', 'module': '...', 'class_name': '...', 'target_kernel': ..., 'language': ...}
                        #    Store metadata only - defer import until get_backend() is called

                        if 'class' in backend_meta:
                            # Pattern 1: Class already loaded - register immediately (legacy)
                            registry.backend(
                                backend_meta['class'],
                                name=backend_meta['name'],
                                target_kernel=backend_meta['target_kernel'],
                                language=backend_meta['language']
                            )

                        elif 'module' in backend_meta and 'class_name' in backend_meta:
                            # Pattern 2: Store metadata for lazy loading
                            # Import will happen on first get_backend() call (avoids expensive imports)
                            full_name = f"{source_name}:{backend_meta['name']}"
                            _lazy_entry_point_backends[full_name] = backend_meta
                            logger.debug(f"Registered lazy backend: {full_name}")

                        else:
                            logger.error(
                                f"Invalid backend metadata for {backend_meta.get('name', 'unknown')}: "
                                f"missing 'class' or ('module' and 'class_name')"
                            )

                    # Register steps
                    for step_meta in components.get('steps', []):
                        # Support two patterns:
                        # 1. Eager: {'name': 'x', 'func': <function>}
                        # 2. Lazy: {'name': 'x', 'module': 'path.to.module', 'func_name': 'func_name'}
                        #    Store metadata only - defer import until get_step() is called

                        if 'func' in step_meta:
                            # Pattern 1: Function already loaded - register immediately
                            registry.step(step_meta['func'], name=step_meta['name'])

                        elif 'module' in step_meta and 'func_name' in step_meta:
                            # Pattern 2: Store metadata for lazy loading
                            # Import will happen on first get_step() call (avoids expensive imports)
                            full_name = f"{source_name}:{step_meta['name']}"
                            _lazy_entry_point_steps[full_name] = step_meta
                            logger.debug(f"Registered lazy step: {full_name}")

                        else:
                            logger.error(
                                f"Invalid step metadata for {step_meta.get('name', 'unknown')}: "
                                f"missing 'func' or ('module' and 'func_name')"
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
    with _measure_load('get_step', name):
        if not _discovery.discovered:
            discover_plugins()

        full_name = _resolve_component_name(name, 'step')
        logger.debug(f"Looking up step: {name} → {full_name}")

        # Check if already registered
        if full_name in registry._steps:
            return registry._steps[full_name]

        # Check if this is a lazy entry point step (e.g. FINN)
        if full_name in _lazy_entry_point_steps:
            logger.debug(f"Loading lazy entry point step: {full_name}")
            meta = _lazy_entry_point_steps[full_name]

            # Import module NOW (only when actually requested)
            module = importlib.import_module(meta['module'])
            func = getattr(module, meta['func_name'])

            # Register it (migrate from lazy dict to registry)
            source = full_name.split(':', 1)[0]
            with source_context(source):
                registry.step(func, name=meta['name'])

            # Remove from lazy dict (now cached in registry)
            del _lazy_entry_point_steps[full_name]

            return registry._steps[full_name]

        # Try lazy loading from plugin module
        source, component_name = full_name.split(':', 1)

        if source in _discovery.loaded_modules:
            logger.debug(f"Lazy loading step: {full_name}")

            # Get the appropriate module
            modules_or_module = _discovery.loaded_modules[source]

            # Core brainsmith: dict with 'steps' key
            if isinstance(modules_or_module, dict):
                module = modules_or_module.get('steps')
            # User plugins: single module
            else:
                module = modules_or_module

            if module:
                # Trigger import via plugin's __getattr__ - decorator fires and registers immediately
                with source_context(source):
                    getattr(module, component_name)

                # Check if it's now registered
                if full_name in registry._steps:
                    return registry._steps[full_name]

        # Not found
        available = list_steps()
        raise KeyError(
            f"Step '{full_name}' not found.\n"
            f"Available steps: {', '.join(available[:10])}" +
            (f" ... and {len(available) - 10} more" if len(available) > 10 else "")
        )


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
    if not _discovery.discovered:
        discover_plugins()

    full_name = _resolve_component_name(name, 'step')

    # Check already-registered steps
    if full_name in registry._steps:
        return True

    # Check lazy entry point steps
    if full_name in _lazy_entry_point_steps:
        return True

    # Check lazy components in loaded modules
    source, component_name = full_name.split(':', 1)
    if source in _discovery.loaded_modules:
        modules_or_module = _discovery.loaded_modules[source]

        # Core brainsmith: dict with 'steps' key
        if isinstance(modules_or_module, dict):
            module = modules_or_module.get('steps')
        # User plugins: single module
        else:
            module = modules_or_module

        if module and hasattr(module, 'COMPONENTS'):
            if component_name in module.COMPONENTS.get('steps', {}):
                return True

    return False


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
    if not _discovery.discovered:
        discover_plugins()

    all_steps = set(registry._steps.keys())

    # Add lazy entry point steps (not yet imported, e.g. FINN)
    all_steps.update(_lazy_entry_point_steps.keys())

    # Include lazy components not yet loaded from plugin modules
    for source_name, modules_or_module in _discovery.loaded_modules.items():
        if source and source != source_name:
            continue

        # Core brainsmith: dict with 'steps' key
        if isinstance(modules_or_module, dict):
            module = modules_or_module.get('steps')
        # User plugins: single module with COMPONENTS
        else:
            module = modules_or_module

        if module and hasattr(module, 'COMPONENTS'):
            # Read from COMPONENTS dict
            for step_name in module.COMPONENTS.get('steps', {}).keys():
                full_name = f'{source_name}:{step_name}'
                if full_name not in registry._steps:  # Not yet loaded
                    all_steps.add(full_name)

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
    with _measure_load('get_kernel', name):
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
    if not _discovery.discovered:
        discover_plugins()

    full_name = _resolve_component_name(name, 'kernel')

    # Check already-registered kernels
    if full_name in registry._kernels:
        return True

    # Check lazy entry point kernels
    if full_name in _lazy_entry_point_kernels:
        return True

    # Check lazy components in loaded modules
    source, component_name = full_name.split(':', 1)
    if source in _discovery.loaded_modules:
        modules_or_module = _discovery.loaded_modules[source]

        # Core brainsmith: dict with 'kernels' key
        if isinstance(modules_or_module, dict):
            module = modules_or_module.get('kernels')
        # User plugins: single module
        else:
            module = modules_or_module

        if module and hasattr(module, 'COMPONENTS'):
            if component_name in module.COMPONENTS.get('kernels', {}):
                return True

    return False


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
    if not _discovery.discovered:
        discover_plugins()

    all_kernels = set(registry._kernels.keys())

    # Add lazy entry point kernels (not yet imported, e.g. FINN)
    all_kernels.update(_lazy_entry_point_kernels.keys())

    # Include lazy components not yet loaded from plugin modules
    for source_name, modules_or_module in _discovery.loaded_modules.items():
        if source and source != source_name:
            continue

        # Core brainsmith: dict with 'kernels' key
        if isinstance(modules_or_module, dict):
            module = modules_or_module.get('kernels')
        # User plugins: single module with COMPONENTS
        else:
            module = modules_or_module

        if module and hasattr(module, 'COMPONENTS'):
            # Read from COMPONENTS dict
            for kernel_name in module.COMPONENTS.get('kernels', {}).keys():
                full_name = f'{source_name}:{kernel_name}'
                if full_name not in registry._kernels:  # Not yet loaded
                    all_kernels.add(full_name)

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
    if not _discovery.discovered:
        discover_plugins()

    full_name = _resolve_component_name(name, 'kernel')
    logger.debug(f"Looking up kernel metadata: {name} → {full_name}")

    # Check if already registered
    if full_name in registry._kernels:
        return registry._kernels[full_name]

    # Try lazy loading from plugin module
    source, component_name = full_name.split(':', 1)

    if source in _discovery.loaded_modules:
        logger.debug(f"Lazy loading kernel: {full_name}")

        # Get the appropriate module
        modules_or_module = _discovery.loaded_modules[source]

        # Core brainsmith: dict with 'kernels' key
        if isinstance(modules_or_module, dict):
            module = modules_or_module.get('kernels')
        # User plugins: single module
        else:
            module = modules_or_module

        if module:
            # Trigger import via plugin's __getattr__ - decorator fires and registers immediately
            with source_context(source):
                getattr(module, component_name)

            # Check if it's now registered
            if full_name in registry._kernels:
                return registry._kernels[full_name]

    # Check if this is a lazy entry point kernel (e.g. FINN)
    if full_name in _lazy_entry_point_kernels:
        logger.debug(f"Loading lazy entry point kernel: {full_name}")
        meta = _lazy_entry_point_kernels[full_name]

        # Import module NOW (only when actually requested)
        module = importlib.import_module(meta['module'])
        kernel_class = getattr(module, meta['class_name'])

        # Import infer_transform if present (also lazy)
        infer_transform = None
        if 'infer_transform' in meta:
            transform_meta = meta['infer_transform']
            if isinstance(transform_meta, dict) and 'module' in transform_meta:
                # Lazy transform: import module + class
                transform_module = importlib.import_module(transform_meta['module'])
                infer_transform = getattr(transform_module, transform_meta['class_name'])
            else:
                # Eager transform: already a class reference
                infer_transform = transform_meta

        # Register with registry (migrate from lazy to loaded)
        source = full_name.split(':', 1)[0]
        with source_context(source):
            registry.kernel(
                kernel_class,
                name=meta['name'],
                infer_transform=infer_transform
            )

        # Remove from lazy dict (now cached in registry)
        del _lazy_entry_point_kernels[full_name]

        return registry._kernels[full_name]

    # Not found
    available = list_kernels()
    raise KeyError(
        f"Kernel '{full_name}' not found.\n"
        f"Available kernels: {', '.join(available[:10])}" +
        (f" ... and {len(available) - 10} more" if len(available) > 10 else "")
    )


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
    if not _discovery.discovered:
        discover_plugins()

    full_name = _resolve_component_name(name, 'backend')
    logger.debug(f"Looking up backend metadata: {name} → {full_name}")

    # Check if already registered
    if full_name in registry._backends:
        return registry._backends[full_name]

    # Try lazy loading from plugin module
    source, component_name = full_name.split(':', 1)

    if source in _discovery.loaded_modules:
        logger.debug(f"Lazy loading backend: {full_name}")

        # Get the appropriate module (core brainsmith doesn't have backends module)
        modules_or_module = _discovery.loaded_modules[source]

        # User plugins: single module
        module = modules_or_module if not isinstance(modules_or_module, dict) else None

        if module:
            # Trigger import via plugin's __getattr__ - decorator fires and registers immediately
            with source_context(source):
                getattr(module, component_name)

            # Check if it's now registered
            if full_name in registry._backends:
                return registry._backends[full_name]

    # Check if this is a lazy entry point backend (e.g. FINN)
    if full_name in _lazy_entry_point_backends:
        logger.debug(f"Loading lazy entry point backend: {full_name}")
        meta = _lazy_entry_point_backends[full_name]

        # Import module NOW (only when actually requested)
        module = importlib.import_module(meta['module'])
        backend_class = getattr(module, meta['class_name'])

        # Register with registry (migrate from lazy to loaded)
        source = full_name.split(':', 1)[0]
        with source_context(source):
            registry.backend(
                backend_class,
                name=meta['name'],
                target_kernel=meta['target_kernel'],
                language=meta['language']
            )

        # Remove from lazy dict (now cached in registry)
        del _lazy_entry_point_backends[full_name]

        return registry._backends[full_name]

    # Not found
    available = list_all_backends()
    raise KeyError(
        f"Backend '{full_name}' not found.\n"
        f"Available backends: {', '.join(available[:10])}" +
        (f" ... and {len(available) - 10} more" if len(available) > 10 else "")
    )


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
    if not _discovery.discovered:
        discover_plugins()

    # Resolve kernel name to full source:name format
    kernel_full = _resolve_component_name(kernel, 'kernel')

    # Filter backends from registry (already-registered)
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

    # Also check lazy entry point backends (e.g., FINN)
    for backend_name, backend_meta in _lazy_entry_point_backends.items():
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

    # Also check lazy components in loaded modules
    for source_name, modules_or_module in _discovery.loaded_modules.items():
        # Filter by sources if specified
        if sources and source_name not in sources:
            continue

        # Core brainsmith: dict with 'kernels', 'steps' keys
        # Note: brainsmith.kernels module has both kernels and backends in COMPONENTS
        if isinstance(modules_or_module, dict):
            module = modules_or_module.get('kernels')
        # User plugins: single module
        else:
            module = modules_or_module

        if module and hasattr(module, 'COMPONENTS'):
            for backend_name, module_path in module.COMPONENTS.get('backends', {}).items():
                full_name = f"{source_name}:{backend_name}"

                # Need to import to check target_kernel
                # Import via module's __getattr__ to trigger registration
                try:
                    with source_context(source_name):
                        getattr(module, backend_name)

                    # Now check if it's registered and matches our kernel
                    if full_name in registry._backends:
                        backend_meta = registry._backends[full_name]
                        if backend_meta['target_kernel'] != kernel_full:
                            continue
                        if language and backend_meta['language'] != language:
                            continue
                        matching.append(full_name)
                except Exception:
                    # Import failed, skip this backend
                    pass

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
    if not _discovery.discovered:
        discover_plugins()

    all_backends = set(registry._backends.keys())

    # Add lazy entry point backends (not yet imported, e.g. FINN)
    all_backends.update(_lazy_entry_point_backends.keys())

    # Include lazy components not yet loaded from plugin modules
    for source_name, modules_or_module in _discovery.loaded_modules.items():
        if source and source != source_name:
            continue

        # Core brainsmith: dict with 'kernels', 'backends', 'steps' keys
        if isinstance(modules_or_module, dict):
            # Check if there's a 'backends' key in the dict
            # Note: brainsmith.kernels module has both kernels and backends
            kernels_module = modules_or_module.get('kernels')
            if kernels_module and hasattr(kernels_module, 'COMPONENTS'):
                for backend_name in kernels_module.COMPONENTS.get('backends', {}).keys():
                    full_name = f'{source_name}:{backend_name}'
                    if full_name not in registry._backends:  # Not yet loaded
                        all_backends.add(full_name)
        # User plugins: single module
        else:
            module = modules_or_module
            if module and hasattr(module, 'COMPONENTS'):
                # Read from COMPONENTS dict
                for backend_name in module.COMPONENTS.get('backends', {}).keys():
                    full_name = f'{source_name}:{backend_name}'
                    if full_name not in registry._backends:  # Not yet loaded
                        all_backends.add(full_name)

    # Filter by source if requested
    if source:
        all_backends = {b for b in all_backends if b.startswith(f"{source}:")}

    return sorted(all_backends)


# Auto-discover on module import (lazy)
# Actual discovery happens on first component lookup
