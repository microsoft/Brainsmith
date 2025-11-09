# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Component discovery for the registry system.

Discovers components from multiple sources:
- Brainsmith core components (built-in kernels, steps)
- User and project components (filesystem-based packages)
- Plugin components (entry points from pip packages like FINN)

Main entry point is discover_components(), which populates the global
_component_index with metadata for lazy loading.

Logging Strategy:
    - DEBUG: Individual component indexing, file operations, loading details
    - INFO: Discovery phases (start/complete), manifest load/save, source loading
    - WARNING: Recoverable errors (manifest stale, source missing, import errors in lenient mode)
    - ERROR: Import failures in strict mode, entry point load failures
"""

import os
import sys
import importlib
import importlib.util
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from importlib.metadata import entry_points
from typing import Dict, List, Any

from ._state import _component_index, _components_discovered, _discovered_sources
from ._decorators import _register_kernel, _register_backend, _register_step, source_context, _convert_lazy_import_spec
from ._metadata import ComponentMetadata, ComponentType, ImportSpec, resolve_lazy_class
from .constants import (
    SOURCE_BRAINSMITH,
    SOURCE_FINN,
    SOURCE_PROJECT,
    DEFAULT_SOURCE_PRIORITY,
)
from ._manifest import (
    _build_manifest_from_index,
    _save_manifest,
    _load_manifest,
    _is_manifest_stale,
    _populate_index_from_manifest,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Performance Instrumentation
# ============================================================================

@contextmanager
def _measure_load(operation: str, component: str = ""):
    """Time and log component loading (only when BRAINSMITH_PROFILE is set).

    Set BRAINSMITH_PROFILE=1 to enable performance timing logs.
    """
    # Only measure when explicitly enabled
    if not os.environ.get('BRAINSMITH_PROFILE'):
        yield
        return

    label = f"{operation}({component})" if component else operation
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"{label}: {duration_ms:.1f}ms")


# ============================================================================
# Registry Access Helpers
# ============================================================================


def _is_strict_mode() -> bool:
    """Check if components_strict setting is enabled.

    Returns:
        True if strict mode is enabled, False otherwise (or if config unavailable)
    """
    try:
        from brainsmith.settings import get_config
        return get_config().components_strict
    except (ImportError, AttributeError):
        return False


# ============================================================================
# Component Name Resolution
# ============================================================================

def _resolve_component_name(name_or_qualified: str, component_type: str = 'step') -> str:
    """Resolve 'LayerNorm' or 'brainsmith:LayerNorm' → 'brainsmith:LayerNorm'.

    Returns the full name even if component doesn't exist - the caller is
    responsible for validation. Falls through to source_priority[0] when
    the component is not found in the index (this allows callers to provide
    better error messages about which sources were searched).

    Args:
        name_or_qualified: Short name ('LayerNorm') or qualified ('brainsmith:LayerNorm')
        component_type: Type of component to resolve ('step', 'kernel', 'backend')

    Returns:
        Fully-qualified name in 'source:name' format
    """
    # Convert string to enum for comparison
    component_type_enum = ComponentType.from_string(component_type)

    # Already qualified?
    if ':' in name_or_qualified:
        return name_or_qualified

    # Get source priority from config
    try:
        from brainsmith.settings import get_config
        source_priority = get_config().source_priority
    except (ImportError, AttributeError) as e:
        logger.debug(f"Using default source priority (config unavailable): {e}")
        source_priority = DEFAULT_SOURCE_PRIORITY

    # Search discovered components if available
    if _components_discovered:
        for source in source_priority:
            full_name = f"{source}:{name_or_qualified}"
            meta = _component_index.get(full_name)
            if meta and meta.component_type == component_type_enum:
                return full_name

    # Not found - return default source (caller will raise KeyError if needed)
    default_name = f"{source_priority[0]}:{name_or_qualified}"
    logger.debug(f"Component '{name_or_qualified}' not in index, assuming: {default_name}")
    return default_name


# ============================================================================
# Component Indexing
# ============================================================================

def _index_entry_point_components(
    source: str,
    component_type: str,
    metas: List[Dict]
):
    """Index entry point components - unified for all types.

    Supports both patterns (no AST parsing needed):
    - Eager: Component already in registry (decorator fired)
    - Lazy: Component has module/class_name (or func_name) for later import

    Args:
        source: Entry point source name (e.g., 'finn')
        component_type: Component type ('kernel', 'backend', 'step')
        metas: List of component metadata dicts from entry point

    Side effects:
        - Populates _component_index with ComponentMetadata entries
    """
    # Convert string to enum
    component_type_enum = ComponentType.from_string(component_type)

    attr_field = 'func_name' if component_type_enum == ComponentType.STEP else 'class_name'

    for meta in metas:
        name = meta['name']
        full_name = f"{source}:{name}"

        # Check if already registered via decorator
        if full_name in _component_index and _component_index[full_name].loaded_obj is not None:
            # Eager pattern: component already registered
            existing = _component_index[full_name].loaded_obj
            module_name = existing.__module__
            class_name = existing.__name__

            # Build metadata
            metadata = ComponentMetadata(
                name=name,
                source=source,
                component_type=component_type_enum,
                import_spec=ImportSpec(
                    module=module_name,
                    attr=class_name
                ),
                loaded_obj=existing
            )

            # Populate type-specific metadata fields (inline)
            if component_type_enum == ComponentType.KERNEL:
                infer_spec = meta.get('infer_transform')
                metadata.kernel_infer = _convert_lazy_import_spec(infer_spec)
                metadata.is_infrastructure = meta.get('is_infrastructure', False)
            elif component_type_enum == ComponentType.BACKEND:
                metadata.backend_target = meta.get('target_kernel')
                metadata.backend_language = meta.get('language')

            _component_index[full_name] = metadata
            logger.debug(f"Indexed eager {component_type}: {full_name}")

        else:
            # Lazy pattern: Component not registered yet, has module path
            metadata = ComponentMetadata(
                name=name,
                source=source,
                component_type=component_type_enum,
                import_spec=ImportSpec(
                    module=meta['module'],
                    attr=meta[attr_field]
                )
            )

            # Populate type-specific metadata fields (inline)
            if component_type_enum == ComponentType.KERNEL:
                infer_spec = meta.get('infer_transform')
                metadata.kernel_infer = _convert_lazy_import_spec(infer_spec)
                metadata.is_infrastructure = meta.get('is_infrastructure', False)
            elif component_type_enum == ComponentType.BACKEND:
                metadata.backend_target = meta.get('target_kernel')
                metadata.backend_language = meta.get('language')

            _component_index[full_name] = metadata
            logger.debug(f"Indexed lazy {component_type}: {full_name}")


def _link_backends_to_kernels() -> None:
    """Link backends to their target kernels after all components indexed.

    Scans all backends in _component_index and populates the kernel_backends
    field on their target kernels. Called once after discovery completes.

    This builds the kernel->backends relationship naturally from the
    backend->kernel metadata, avoiding the need for a separate inverted index.
    """
    for full_name, meta in _component_index.items():
        if meta.component_type != ComponentType.BACKEND:
            continue

        target = meta.backend_target
        if not target:
            logger.warning(
                f"Backend {full_name} missing target_kernel metadata. "
                "Skipping backend->kernel linking."
            )
            continue

        kernel_meta = _component_index.get(target)
        if not kernel_meta:
            logger.warning(
                f"Backend {full_name} targets unknown kernel {target}. "
                "This backend will not be discoverable via list_backends_for_kernel()."
            )
            continue

        # Initialize backends list if needed
        if kernel_meta.kernel_backends is None:
            kernel_meta.kernel_backends = []

        # Add backend to kernel's list (avoid duplicates from decorator registration)
        if full_name not in kernel_meta.kernel_backends:
            kernel_meta.kernel_backends.append(full_name)
            logger.debug(f"Linked backend {full_name} to kernel {target}")
        else:
            logger.debug(f"Backend {full_name} already linked to kernel {target} (skipping duplicate)")

    logger.debug("Linked backends to their target kernels")


# ============================================================================
# Component Loading
# ============================================================================

def _load_component(meta: ComponentMetadata) -> Any:
    """Load a component on demand - unified direct import.

    All components now use direct imports with absolute paths. The COMPONENTS dict
    pattern (with __getattr__) handles lazy loading at the package level, but once
    we're loading, we just import the module directly.

    Args:
        meta: Component metadata from _component_index

    Returns:
        Loaded component (kernel class, backend class, or step function)

    Side effects:
        - Sets meta.loaded_obj to loaded component
        - Registers component with registry if not already registered
    """
    # Already loaded? Return cached
    if meta.loaded_obj is not None:
        return meta.loaded_obj

    logger.debug(f"Loading component: {meta.full_name}")

    spec = meta.import_spec

    # Direct import using absolute module path
    module = importlib.import_module(spec.module)

    # After import, decorator may have replaced the ComponentMetadata in _component_index
    # Re-fetch to get the updated metadata
    meta = _component_index[meta.full_name]

    # Check again if already loaded (decorator may have just loaded it during import).
    # This happens when modules use @step/@kernel/@backend decorators - the decorator
    # fires during module execution and calls _register_*() which sets loaded_obj.
    if meta.loaded_obj is not None:
        return meta.loaded_obj

    try:
        obj = getattr(module, spec.attr)
    except AttributeError:
        # For steps: decorator name might differ from function name
        # After import, check if decorator already registered it in component index
        if meta.component_type == ComponentType.STEP:
            # Check if decorator already populated loaded_obj
            if meta.loaded_obj is not None:
                obj = meta.loaded_obj
            else:
                raise AttributeError(
                    f"Step '{meta.name}' not found in module '{spec.module}' "
                    f"and not registered via decorator"
                )
        else:
            raise

    # Register if needed (plugin components that weren't eagerly registered)
    # Check component index, not registry dicts
    with source_context(meta.source):
        if meta.loaded_obj is None:
            _register_component(obj, meta)

    meta.loaded_obj = obj
    logger.debug(f"Loaded component: {meta.full_name}")
    return obj


def _register_component(obj: Any, meta: ComponentMetadata) -> None:
    """Register plugin component with appropriate registry method.

    Plugin components need explicit registration (unlike filesystem components
    which auto-register via decorators during import).

    Args:
        obj: Loaded component (class or function)
        meta: Component metadata containing type and extra info

    Side effects:
        Registers component with global registry using @kernel/@backend/@step logic
    """
    if meta.component_type == ComponentType.KERNEL:
        # Resolve lazy infer_transform if needed
        infer_transform = resolve_lazy_class(meta.kernel_infer)

        _register_kernel(
            obj,
            name=meta.name,
            infer_transform=infer_transform,
            is_infrastructure=meta.is_infrastructure
        )
        logger.debug(f"Registered kernel: {meta.full_name}")

    elif meta.component_type == ComponentType.BACKEND:
        _register_backend(
            obj,
            name=meta.name,
            target_kernel=meta.backend_target,
            language=meta.backend_language
        )
        logger.debug(f"Registered backend: {meta.full_name}")

    elif meta.component_type == ComponentType.STEP:
        _register_step(obj, name=meta.name)
        logger.debug(f"Registered step: {meta.full_name}")

    else:
        raise ValueError(f"Unknown component type: {meta.component_type}")


# ============================================================================
# Main Discovery Entry Point
# ============================================================================

def discover_components(use_cache: bool = True, force_refresh: bool = False):
    """Discover components from all configured sources.

    Discovers kernels, backends, and steps from:
    - Core brainsmith components
    - Project components (if configured)
    - Entry points from installed packages (e.g., FINN)
    - Custom component sources

    Called automatically on first component lookup.

    Args:
        use_cache: Use cached manifest if available
        force_refresh: Ignore cache and regenerate manifest
    """
    global _components_discovered

    # Handle force refresh - reset discovery state to allow re-discovery
    if force_refresh and _components_discovered:
        logger.debug("Force refresh requested - resetting discovery state")
        _components_discovered = False
        _component_index.clear()
        _discovered_sources.clear()

    # Skip if already discovered (and not forcing refresh)
    if _components_discovered:
        return

    with _measure_load('discover_components'):
        # Check cache_components setting and get project_dir
        from brainsmith.settings import get_config
        config = get_config()
        cache_enabled = config.cache_components

        # Use project_dir for manifest location (not CWD)
        manifest_path = config.project_dir / '.brainsmith' / 'component_manifest.json'

        if not cache_enabled:
            logger.debug("Component caching disabled via cache_components setting")
            use_cache = False

        # Try cached manifest
        if use_cache and not force_refresh and manifest_path.exists():
            try:
                logger.debug(f"Loading component manifest from {manifest_path}")
                manifest = _load_manifest(manifest_path)

                # Check if cache is stale
                if _is_manifest_stale(manifest):
                    logger.debug("Manifest is stale - performing full discovery")
                    # Don't return, fall through to full discovery
                else:
                    _populate_index_from_manifest(manifest)
                    _components_discovered = True

                    logger.debug(
                        f"Loaded {len(_component_index)} components from cache"
                    )
                    return
            except Exception as e:
                logger.warning(f"Failed to load manifest cache: {e}")
                logger.debug("Falling back to full discovery...")

        # Full discovery
        logger.info("Discovering components from all sources...")

        # Get config for component sources
        config = get_config()

        # 1. Core brainsmith components (eager imports with source_context)
        # Decorators fire during import and auto-populate registry + index
        _discovered_sources.add(SOURCE_BRAINSMITH)
        with source_context(SOURCE_BRAINSMITH):
            import brainsmith.kernels  # noqa: E402
            import brainsmith.steps    # noqa: E402

        logger.debug(f"Loaded core brainsmith components")

        # 2. Entry point components (FINN, etc.)
        _load_entry_point_components()

        # 3. Active project components (SOURCE_PROJECT at project root)
        _load_project_components(config)

        # 4. Other filesystem-based component sources (custom sources)
        _load_other_component_sources(config)

        _components_discovered = True

        # Link backends to their target kernels after all components indexed
        _link_backends_to_kernels()

        # Count components by type
        counts = {'step': 0, 'kernel': 0, 'backend': 0}
        for meta in _component_index.values():
            counts[str(meta.component_type)] += 1

        logger.info(
            f"Component discovery complete: "
            f"{counts['step']} steps, "
            f"{counts['kernel']} kernels, "
            f"{counts['backend']} backends"
        )

        # Save manifest for next time (only if caching is enabled)
        if cache_enabled:
            try:
                manifest = _build_manifest_from_index()
                _save_manifest(manifest, manifest_path)
                logger.debug(f"Regenerated and saved manifest cache to {manifest_path}")
            except Exception as e:
                logger.warning(f"Failed to save manifest cache: {e}")
        else:
            logger.debug("Skipping manifest save (caching disabled)")


def _load_project_components(config):
    """Load active project components from SOURCE_PROJECT.

    Project components use structured layout:
        project_dir/kernels/__init__.py → project kernels
        project_dir/steps/__init__.py → project steps

    This is phase 3 of discovery, after brainsmith core and entry points.

    Args:
        config: SystemConfig instance with component_sources configured
    """
    source_path = config.component_sources.get(SOURCE_PROJECT)

    if not source_path:
        logger.debug("No SOURCE_PROJECT configured, skipping project components")
        return

    if not source_path.exists():
        logger.debug(f"Project component source does not exist: {source_path}")
        return

    # Register project source as discovered
    _discovered_sources.add(SOURCE_PROJECT)

    # Load structured layout (project_dir/kernels/, project_dir/steps/)
    for component_type in ['kernels', 'steps']:
        type_dir = source_path / component_type
        init_file = type_dir / '__init__.py'

        if init_file.exists():
            _load_component_package(SOURCE_PROJECT, type_dir)
            logger.debug(f"Loaded project {component_type} from {type_dir}")
        else:
            logger.debug(f"No project {component_type} at {type_dir}")


def _load_other_component_sources(config):
    """Load custom filesystem-based component sources.

    Loads components from component_sources, excluding protected sources
    (brainsmith, finn) and SOURCE_PROJECT (loaded separately).
    This is phase 4 of discovery, after project components.

    Args:
        config: SystemConfig instance with component_sources configured
    """
    for source_name, source_path in config.component_sources.items():
        # Skip protected sources (loaded separately):
        # - brainsmith: loaded via direct import in phase 1
        # - finn: loaded via entry points in phase 2
        # - project: loaded in phase 3
        if source_name in (SOURCE_BRAINSMITH, SOURCE_FINN, SOURCE_PROJECT):
            continue

        if not source_path or not source_path.exists():
            logger.debug(f"Component source '{source_name}' does not exist: {source_path}")
            continue

        # Register custom source as discovered
        _discovered_sources.add(source_name)
        _load_component_package(source_name, source_path)


def _load_component_package(source_name: str, source_path: Path):
    """Load a component package from filesystem path.

    Supports two patterns:
    1. Lazy: Package has COMPONENTS dict -> register with lazy loaders
    2. Eager: Package uses decorators -> import triggers registration

    Args:
        source_name: Source name for registration (e.g., 'user', 'team')
        source_path: Path to component package directory

    The package must have __init__.py which either:
    - Exports COMPONENTS dict (lazy loading, recommended)
    - Imports modules with @kernel/@step decorators (eager, legacy)
    """
    init_file = source_path / '__init__.py'
    if not init_file.exists():
        logger.debug(
            f"Component source '{source_name}' has no __init__.py, skipping. "
            f"Component packages must have __init__.py that registers components. "
            f"Path: {source_path}"
        )
        return

    logger.debug(f"Loading component source '{source_name}' from {source_path}")

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
        # Multiple component sources may have the same directory name (e.g., 'kernels', 'steps')
        # Use importlib.util to import from specific file path to avoid sys.path ambiguity

        # Create a unique module name to avoid cache collisions
        unique_module_name = f"{source_name}__{module_name}"

        spec = importlib.util.spec_from_file_location(unique_module_name, init_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[unique_module_name] = module

            # Execute module with source_context to ensure decorators inherit source
            with source_context(source_name):
                spec.loader.exec_module(module)

            logger.debug(f"Loaded component source '{source_name}' from {source_path}")
        else:
            raise ImportError(f"Could not create module spec for {init_path}")

        # Both lazy and eager patterns work:
        # - Lazy: Package has COMPONENTS dict, components loaded on first access
        # - Eager: Decorators already fired during import, components in registry

    except Exception as e:
        logger.error(f"Failed to load component source '{source_name}': {e}")

        # Re-raise in strict mode
        if _is_strict_mode():
            raise


def _load_entry_point_components():
    """Load components from pip package entry points.

    Scans entry points in group 'brainsmith.plugins' (a Python packaging
    entry point group, not a filesystem directory). Each entry point
    should return a dict of component metadata that we register.

    Entry points are registered with their entry point name as source
    (e.g., 'finn' from the finn package).
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
                # Returns: {'kernels': [...], 'backends': [...], 'steps': [...]}
                components = register_func()

                if not isinstance(components, dict):
                    logger.error(f"Entry point '{ep.name}' returned {type(components)}, expected dict")
                    continue

                # Register this source as discovered
                _discovered_sources.add(source_name)
                logger.debug(f"Loading component source: {source_name}")

                # Register all components under this source
                with source_context(source_name):
                    # Index all component types using unified helper
                    _index_entry_point_components(source_name, 'kernel', components.get('kernels', []))
                    _index_entry_point_components(source_name, 'backend', components.get('backends', []))
                    _index_entry_point_components(source_name, 'step', components.get('steps', []))

                logger.debug(
                    f"✓ Loaded {source_name}: "
                    f"{len(components.get('kernels', []))} kernels, "
                    f"{len(components.get('backends', []))} backends, "
                    f"{len(components.get('steps', []))} steps"
                )

            except Exception as e:
                logger.error(f"Failed to load entry point '{ep.name}': {e}")

                # Re-raise in strict mode
                if _is_strict_mode():
                    raise

    except Exception as e:
        logger.warning(f"Entry point discovery failed: {e}")
