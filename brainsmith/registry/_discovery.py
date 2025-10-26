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

from ._state import _component_index, _components_discovered
from ._decorators import registry, source_context
from ._metadata import ComponentMetadata, ImportSpec
from .constants import (
    SOURCE_BRAINSMITH,
    SOURCE_FINN,
    SOURCE_PROJECT,
    SOURCE_USER,
    DEFAULT_SOURCE_PRIORITY,
    COMPONENT_TYPE_PLURALS,
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
    """Time and log component loading."""
    label = f"{operation}({component})" if component else operation
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"{label}: {duration_ms:.1f}ms")


# ============================================================================
# String Transformation Helpers
# ============================================================================

def _resolve_module_path(base: str, relative: str) -> str:
    """Join base module path with relative import using stdlib.

    Uses importlib.util.resolve_name() for proper relative import resolution.
    Handles edge cases like '..sibling' correctly.
    """
    return importlib.util.resolve_name(relative, base)


# ============================================================================
# Registry Access Helpers
# ============================================================================

# Type-to-registry mapping for unified component access
_COMPONENT_REGISTRIES = {
    'step': lambda: registry._steps,
    'kernel': lambda: registry._kernels,
    'backend': lambda: registry._backends,
}


def _get_registry_for_type(component_type: str) -> dict:
    """Get the appropriate registry dict for a component type."""
    try:
        return _COMPONENT_REGISTRIES[component_type]()
    except KeyError:
        valid = ', '.join(_COMPONENT_REGISTRIES.keys())
        raise ValueError(f"Unknown component type: {component_type}. Valid: {valid}")


# ============================================================================
# Component Name Resolution
# ============================================================================

def _resolve_component_name(name: str, component_type: str = 'step') -> str:
    """Resolve 'LayerNorm' → 'brainsmith:LayerNorm' using source priority.

    Returns the full name even if component doesn't exist - the caller is
    responsible for validation. Falls through to source_priority[0] when
    the component is not found in the index (this allows callers to provide
    better error messages about which sources were searched).

    Args:
        name: Component name (short or fully-qualified)
        component_type: Type of component to resolve ('step', 'kernel', 'backend')

    Returns:
        Fully-qualified name (source:name format)
    """
    # Already qualified?
    if ':' in name:
        return name

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
            full_name = f"{source}:{name}"
            meta = _component_index.get(full_name)
            if meta and meta.component_type == component_type:
                return full_name

    # Not found - return default source (caller will raise KeyError if needed)
    default_name = f"{source_priority[0]}:{name}"
    logger.debug(f"Component '{name}' not in index, assuming: {default_name}")
    return default_name


# ============================================================================
# Component Indexing
# ============================================================================

def _index_filesystem_components(source: str, modules: Dict[str, Any]):
    """Index components from filesystem-based source (core, user, project).

    This helper populates the unified component index during discovery.
    Works for both core brainsmith components and user/project components
    that use the COMPONENTS dict pattern.

    Args:
        source: Source name ('brainsmith', 'user', 'project')
        modules: Dict of module references with COMPONENTS dicts
                Example: {'kernels': brainsmith.kernels, 'steps': brainsmith.steps}

    Side effects:
        - Populates _component_index with ComponentMetadata entries

    Note:
        A single module can contain multiple component types. For example,
        brainsmith.kernels has both 'kernels' and 'backends' in its COMPONENTS dict.
    """

    for module_key, module in modules.items():
        if not hasattr(module, 'COMPONENTS'):
            continue

        # Index ALL component types from this module's COMPONENTS dict
        # (A module can have kernels, backends, steps, etc.)
        for component_type_plural, components in module.COMPONENTS.items():
            component_type = COMPONENT_TYPE_PLURALS.get(component_type_plural, component_type_plural.rstrip('s'))

            for name, spec in components.items():
                full_name = f"{source}:{name}"

                # Support both formats: string (old) and dict (new with metadata)
                # This enables backwards compatibility with old COMPONENTS dicts
                if isinstance(spec, str):
                    # Old format: just module path
                    module_path = spec
                    metadata = {}
                else:
                    # New format: dict with module + metadata (Issue #9 Phase 2)
                    module_path = spec.get('module', spec)
                    metadata = {k: v for k, v in spec.items() if k != 'module'}

                # Derive full module path for import
                full_module = _resolve_module_path(module.__name__, module_path)

                # Create ComponentMetadata with type-specific fields from COMPONENTS dict
                component_meta = ComponentMetadata(
                    name=name,
                    source=source,
                    component_type=component_type,
                    import_spec=ImportSpec(
                        module=full_module,
                        attr=name,
                        extra={}
                    )
                )

                # Populate type-specific metadata from COMPONENTS dict (Issue #9 Phase 2!)
                # Metadata is now available at discovery time, no loading needed
                if component_type == 'kernel':
                    infer_spec = metadata.get('infer_transform')
                    if infer_spec and isinstance(infer_spec, str) and ':' in infer_spec:
                        # Parse 'module.path:ClassName' format
                        module_path, class_name = infer_spec.split(':', 1)
                        component_meta.kernel_infer = {'module': module_path, 'class_name': class_name}
                    else:
                        component_meta.kernel_infer = infer_spec
                    component_meta.kernel_domain = metadata.get('domain', 'finn.custom')
                elif component_type == 'backend':
                    component_meta.backend_target = metadata.get('target_kernel')
                    component_meta.backend_language = metadata.get('language')

                _component_index[full_name] = component_meta
                logger.debug(f"Indexed {source} {component_type}: {full_name} with metadata")


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
    registry_dict = _get_registry_for_type(component_type)
    attr_field = 'func_name' if component_type == 'step' else 'class_name'

    for meta in metas:
        name = meta['name']
        full_name = f"{source}:{name}"

        # Check if already registered via decorator
        if full_name in registry_dict:
            # Eager pattern: component already registered
            # Registry now stores objects directly (not dicts)
            existing = registry_dict[full_name]
            module_name = existing.__module__
            class_name = existing.__name__

            # Build metadata
            metadata = ComponentMetadata(
                name=name,
                source=source,
                component_type=component_type,
                import_spec=ImportSpec(
                    module=module_name,
                    attr=class_name,
                    extra={}
                ),
                loaded_obj=existing
            )

            # Populate type-specific metadata fields
            if component_type == 'kernel':
                # For kernels, try to get infer metadata from meta dict
                metadata.kernel_infer = meta.get('infer_transform')
                metadata.kernel_domain = meta.get('domain', 'finn.custom')
            elif component_type == 'backend':
                metadata.backend_target = meta.get('target_kernel')
                metadata.backend_language = meta.get('language')

            _component_index[full_name] = metadata
            logger.debug(f"Indexed eager {component_type}: {full_name}")

        else:
            # Lazy pattern: Component not registered yet, has module path
            metadata = ComponentMetadata(
                name=name,
                source=source,
                component_type=component_type,
                import_spec=ImportSpec(
                    module=meta['module'],
                    attr=meta[attr_field],
                    extra={}
                )
            )

            # Populate type-specific metadata fields
            if component_type == 'kernel':
                metadata.kernel_infer = meta.get('infer_transform')
                metadata.kernel_domain = meta.get('domain', 'finn.custom')
            elif component_type == 'backend':
                metadata.backend_target = meta.get('target_kernel')
                metadata.backend_language = meta.get('language')

            _component_index[full_name] = metadata
            logger.debug(f"Indexed lazy {component_type}: {full_name}")


# ============================================================================
# Component Loading
# ============================================================================

def _load_component(meta: ComponentMetadata) -> Any:
    """Load a component on demand - unified single-path loading.

    All components use importlib for loading. Filesystem components (brainsmith, user)
    delegate to their module's __getattr__ for name resolution.

    Args:
        meta: Component metadata from _component_index

    Returns:
        Loaded component (kernel class, backend class, or step function)

    Side effects:
        - Sets meta.loaded_obj to loaded component
        - Registers component with registry if not already registered
    """
    # Already loaded? Return cached
    if meta.is_loaded:
        return meta.loaded_obj

    logger.debug(f"Loading component: {meta.full_name}")

    # Load via importlib
    spec = meta.import_spec

    # For filesystem components (brainsmith, user), use module's __getattr__
    # This handles name mapping (e.g., 'qonnx_to_finn' -> 'qonnx_to_finn_step')
    if meta.source in (SOURCE_BRAINSMITH, SOURCE_USER, SOURCE_PROJECT):
        # Import the top-level module that has __getattr__ defined
        # For brainsmith: 'brainsmith.kernels' or 'brainsmith.steps'
        # module path like 'brainsmith.kernels.layernorm.layernorm' -> 'brainsmith.kernels'
        module_parts = spec.module.split('.')
        if meta.source == SOURCE_BRAINSMITH:
            # Top-level is source.{component_type}s (e.g., brainsmith.kernels)
            parent_module_path = f"{module_parts[0]}.{module_parts[1]}"
        else:
            # User/project components - assume similar structure
            parent_module_path = f"{meta.source}.{meta.component_type}s"

        parent_module = importlib.import_module(parent_module_path)

        # Use __getattr__ to get component (handles name mapping)
        with source_context(meta.source):
            obj = getattr(parent_module, meta.name)

        # For steps, __getattr__ might return None (decorator already registered it)
        if obj is None:
            # Component registered via decorator during import
            # Registry now stores objects directly (no unwrapping needed)
            obj = _get_registry_for_type(meta.component_type).get(meta.full_name)
            if not obj:
                raise RuntimeError(f"Component {meta.full_name} registered but not found in registry")
    else:
        # Plugin components: direct import with exact attr name
        module = importlib.import_module(spec.module)
        obj = getattr(module, spec.attr)

        # Register plugin component if needed
        with source_context(meta.source):
            if meta.full_name not in _get_registry_for_type(meta.component_type):
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
    extra = meta.import_spec.extra if meta.import_spec else {}

    if meta.component_type == 'kernel':
        # Handle nested lazy infer_transform
        infer_transform = extra.get('infer_transform')
        if isinstance(infer_transform, dict) and 'module' in infer_transform:
            # Lazy infer_transform: import it now
            transform_module = importlib.import_module(infer_transform['module'])
            infer_transform = getattr(transform_module, infer_transform['class_name'])

        registry.kernel(obj, name=meta.name, infer_transform=infer_transform)
        logger.debug(f"Registered kernel: {meta.full_name}")

    elif meta.component_type == 'backend':
        registry.backend(
            obj,
            name=meta.name,
            target_kernel=extra['target_kernel'],
            language=extra['language']
        )
        logger.debug(f"Registered backend: {meta.full_name}")

    elif meta.component_type == 'step':
        registry.step(obj, name=meta.name)
        logger.debug(f"Registered step: {meta.full_name}")

    else:
        raise ValueError(f"Unknown component type: {meta.component_type}")


# ============================================================================
# Main Discovery Entry Point
# ============================================================================

def discover_components(use_cache: bool = True, force_refresh: bool = False):
    """Discover and load all components from configured sources.

    This loads:
    1. Core brainsmith components (kernels, steps) via direct imports
    2. Component packages from component_sources (must have __init__.py)
    3. Entry points from pip-installed packages (e.g., FINN)

    Components self-register during import using the global registry.

    This is called automatically on first component lookup.

    Args:
        use_cache: If True, try to load from cached manifest
        force_refresh: If True, ignore cache and regenerate manifest
    """
    global _components_discovered

    # Handle force refresh - reset discovery state to allow re-discovery
    if force_refresh and _components_discovered:
        logger.info("Force refresh requested - resetting discovery state")
        _components_discovered = False
        _component_index.clear()

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
                logger.info(f"Loading component manifest from {manifest_path}")
                manifest = _load_manifest(manifest_path)

                # Check if cache is stale
                if _is_manifest_stale(manifest):
                    logger.info("Manifest is stale - performing full discovery")
                    # Don't return, fall through to full discovery
                else:
                    _populate_index_from_manifest(manifest)
                    _components_discovered = True

                    logger.info(
                        f"Loaded {len(_component_index)} components from cache"
                    )
                    return
            except Exception as e:
                logger.warning(f"Failed to load manifest cache: {e}")
                logger.info("Falling back to full discovery...")

        # Full discovery
        logger.info("Discovering components from all sources...")

        # 1. Core brainsmith components (filesystem)
        import brainsmith.kernels
        import brainsmith.steps

        _index_filesystem_components(SOURCE_BRAINSMITH, {
            'kernels': brainsmith.kernels,
            'steps': brainsmith.steps,
        })

        # 2. User/project components (filesystem)
        _load_component_sources()

        # 3. Entry point components (FINN, etc.)
        _load_entry_point_components()

        _components_discovered = True

        logger.info(
            f"Component discovery complete: {registry}"
        )
        logger.info(
            f"Component index: {len(_component_index)} components indexed"
        )

        # Save manifest for next time (only if caching is enabled)
        if cache_enabled:
            try:
                manifest = _build_manifest_from_index()
                _save_manifest(manifest, manifest_path)
                logger.info(f"Saved manifest cache to {manifest_path}")
            except Exception as e:
                logger.warning(f"Failed to save manifest cache: {e}")
        else:
            logger.debug("Skipping manifest save (caching disabled)")


def _load_component_sources():
    """Load component packages from configured component sources.

    Scans component_sources from config, imports each package that has
    an __init__.py file. The __init__.py must import and register
    its components.
    """
    try:
        from brainsmith.settings import get_config
        component_sources = get_config().component_sources
    except Exception as e:
        logger.warning(f"Could not load component sources config: {e}")
        return

    for source_name, source_path in component_sources.items():
        # Skip protected sources:
        # - brainsmith: loaded via direct import
        # - finn: loaded via entry points
        # - project, user: optional __init__.py-based component packages
        if source_name in (SOURCE_BRAINSMITH, SOURCE_FINN):
            continue

        if not source_path.exists():
            logger.debug(f"Component source '{source_name}' does not exist: {source_path}")
            continue

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
        logger.warning(
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
        # Multiple plugin sources may have the same directory name (e.g., 'plugins')
        # Use importlib.util to import from specific file path to avoid sys.path ambiguity

        # Create a unique module name to avoid cache collisions
        unique_module_name = f"{source_name}__{module_name}"

        spec = importlib.util.spec_from_file_location(unique_module_name, init_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[unique_module_name] = module
            spec.loader.exec_module(module)
            logger.info(f"Loaded component source '{source_name}' from {source_path}")
        else:
            raise ImportError(f"Could not create module spec for {init_path}")

        # Both lazy and eager patterns work:
        # - Lazy: Package has COMPONENTS dict, components loaded on first access
        # - Eager: Decorators already fired during import, components in registry

    except Exception as e:
        logger.error(f"Failed to load component source '{source_name}': {e}")

        # Check if strict mode
        try:
            from brainsmith.settings import get_config
            if get_config().components_strict:
                raise
        except ImportError:
            pass  # Config not available, don't fail


def _load_entry_point_components():
    """Load components from pip package entry points.

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
                # Returns: {'kernels': [...], 'backends': [...], 'steps': [...]}
                components = register_func()

                if not isinstance(components, dict):
                    logger.error(f"Entry point '{ep.name}' returned {type(components)}, expected dict")
                    continue

                logger.info(f"Loading component source: {source_name}")

                # Register all components under this source
                with source_context(source_name):
                    # Index all component types using unified helper
                    _index_entry_point_components(source_name, 'kernel', components.get('kernels', []))
                    _index_entry_point_components(source_name, 'backend', components.get('backends', []))
                    _index_entry_point_components(source_name, 'step', components.get('steps', []))

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
                    if get_config().components_strict:
                        raise
                except ImportError:
                    pass  # Config not available, don't fail

    except Exception as e:
        logger.warning(f"Entry point discovery failed: {e}")
