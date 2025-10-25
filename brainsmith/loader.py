# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Plugin discovery and component loading for Brainsmith.

Provides functions for accessing registered components:
- get_kernel(), list_kernels(), has_kernel()
- get_backend(), list_backends(), has_backend()
- get_step(), list_steps(), has_step()

Components are discovered from multiple sources (brainsmith, finn, user, project)
and use source-prefixed names (e.g., 'brainsmith:LayerNorm').

See docs/ARCHITECTURE.md for discovery flow and lazy loading details.
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
from brainsmith.constants import (
    SOURCE_BRAINSMITH,
    SOURCE_FINN,
    SOURCE_PROJECT,
    SOURCE_USER,
    PROTECTED_SOURCES,
    DEFAULT_SOURCE_PRIORITY,
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

def _normalize_component_type(plural: str) -> str:
    """Convert 'kernels' → 'kernel'."""
    return plural.rstrip('s')


def _resolve_module_path(base: str, relative: str) -> str:
    """Join base module path with relative import."""
    clean_path = relative.lstrip('.').replace('/', '.')
    return f"{base}.{clean_path}" if clean_path else base


# ============================================================================
# Unified Component Loading Infrastructure (Arete Refactor)
# ============================================================================

from dataclasses import dataclass, field
from typing import Literal

@dataclass
class ImportSpec:
    """Lazy import specification: module + attr + metadata."""
    module: str
    attr: str
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ComponentMetadata:
    """Component metadata for unified lazy loading."""
    name: str
    source: str
    component_type: Literal['kernel', 'backend', 'step']
    import_spec: ImportSpec
    loaded_obj: Optional[Any] = None

    @property
    def full_name(self) -> str:
        return f"{self.source}:{self.name}"

    @property
    def is_loaded(self) -> bool:
        return self.loaded_obj is not None


# Global component index - single source of truth for component metadata
# Maps "source:name" -> ComponentMetadata
# Unified index for all components from all sources (core, user, plugins)
_component_index: Dict[str, ComponentMetadata] = {}


# ============================================================================
# Manifest Caching (Arete: Eager Discovery + Optional Performance Cache)
# ============================================================================

def _build_manifest_from_index() -> dict:
    """Build manifest dict from current component index.

    Converts _component_index to JSON-serializable format for caching.
    Includes file mtimes for cache invalidation.

    Returns:
        Manifest dict with schema version, component metadata, and file mtimes
    """
    import datetime
    import importlib.util
    import os

    manifest = {
        "version": "1.0",
        "generated_at": datetime.datetime.now().isoformat(),
        "components": {}
    }

    for full_name, meta in _component_index.items():
        # Resolve module to file path and get mtime
        file_path = None
        mtime = None
        try:
            spec = importlib.util.find_spec(meta.import_spec.module)
            if spec and spec.origin:
                file_path = spec.origin
                mtime = os.path.getmtime(file_path)
        except Exception as e:
            logger.debug(f"Could not get mtime for {meta.import_spec.module}: {e}")

        # All components now have import_spec - simple conversion
        # Note: source is embedded in full_name (source:name), state is runtime-only
        manifest["components"][full_name] = {
            "type": meta.component_type,
            "module": meta.import_spec.module,
            "class_name": meta.import_spec.attr,
            "metadata": meta.import_spec.extra,
            "file_path": file_path,
            "mtime": mtime
        }

    return manifest


def _save_manifest(manifest: dict, path: Path) -> None:
    """Save manifest to JSON file.

    Args:
        manifest: Manifest dict from _build_manifest_from_index()
        path: Path to save manifest (typically .brainsmith/component_manifest.json)
    """
    import json

    # Create parent directory if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.debug(f"Saved manifest with {len(manifest['components'])} components to {path}")


def _load_manifest(path: Path) -> dict:
    """Load manifest from JSON file.

    Args:
        path: Path to manifest file

    Returns:
        Manifest dict

    Raises:
        FileNotFoundError: If manifest doesn't exist
        json.JSONDecodeError: If manifest is corrupted
    """
    import json

    with open(path, 'r') as f:
        manifest = json.load(f)

    logger.debug(f"Loaded manifest with {len(manifest['components'])} components from {path}")
    return manifest


def _is_manifest_stale(manifest: dict) -> bool:
    """Check if manifest is stale by comparing file mtimes.

    A manifest is stale if any component file has been modified since the
    manifest was generated. This enables automatic cache invalidation when
    code changes.

    Note: This only detects file modifications, not new files. Users must
    manually refresh (--refresh) when adding new component files.

    Args:
        manifest: Manifest dict from _load_manifest()

    Returns:
        True if manifest is stale and should be regenerated
    """
    import os

    for full_name, comp_data in manifest.get("components", {}).items():
        file_path = comp_data.get("file_path")
        cached_mtime = comp_data.get("mtime")

        # Skip if we don't have mtime data (old manifest format)
        if file_path is None or cached_mtime is None:
            continue

        try:
            current_mtime = os.path.getmtime(file_path)
            if current_mtime > cached_mtime:
                logger.info(
                    f"Cache stale: {full_name} modified "
                    f"(cached: {cached_mtime}, current: {current_mtime})"
                )
                return True
        except OSError as e:
            # File doesn't exist or can't be accessed - treat as stale
            logger.info(f"Cache stale: cannot access {file_path}: {e}")
            return True

    return False


def _populate_index_from_manifest(manifest: dict) -> None:
    """Populate component index from cached manifest.

    Rebuilds _component_index from manifest without importing components.
    Components are marked as DISCOVERED and will be lazy-loaded on first use.

    Args:
        manifest: Manifest dict from _load_manifest()
    """
    for full_name, component in manifest["components"].items():
        # Parse source and name from key (source:name)
        source, name = full_name.split(':', 1)

        # All components in manifest have module + class_name for import
        _component_index[full_name] = ComponentMetadata(
            name=name,
            source=source,
            component_type=component["type"],
            import_spec=ImportSpec(
                module=component["module"],
                attr=component["class_name"],
                extra=component.get("metadata", {})
            )
        )

        logger.debug(f"Indexed from manifest: {full_name}")


# === Plugin Discovery State Management ===

# Module-level state
_plugins_discovered = False


def _resolve_component_name(name: str, component_type: str = 'step') -> str:
    """Resolve 'LayerNorm' → 'brainsmith:LayerNorm' using source priority."""
    if ':' in name:
        return name

    try:
        from brainsmith.settings import get_config
        source_priority = get_config().source_priority
    except (ImportError, AttributeError):
        source_priority = DEFAULT_SOURCE_PRIORITY

    if _plugins_discovered:
        for source in source_priority:
            full_name = f"{source}:{name}"
            if full_name in _component_index:
                if _component_index[full_name].component_type == component_type:
                    return full_name

    return f"{source_priority[0]}:{name}"


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
            component_type = _normalize_component_type(component_type_plural)

            for name, module_path in components.items():
                full_name = f"{source}:{name}"

                # Derive full module path for import
                full_module = _resolve_module_path(module.__name__, module_path)

                _component_index[full_name] = ComponentMetadata(
                    name=name,
                    source=source,
                    component_type=component_type,
                    import_spec=ImportSpec(
                        module=full_module,
                        attr=name,
                        extra={}
                    )
                )

                logger.debug(f"Indexed {source} {component_type}: {full_name}")


def _index_plugin_components(
    source: str,
    component_type: str,
    metas: List[Dict]
):
    """Index plugin components from entry point - unified for all types.

    Supports both patterns (no AST parsing needed):
    - Eager: Component already in registry (decorator fired)
    - Lazy: Component has module/class_name (or func_name) for later import

    Args:
        source: Plugin source name (e.g., 'finn')
        component_type: Component type ('kernel', 'backend', 'step')
        metas: List of component metadata dicts from entry point

    Side effects:
        - Populates _component_index with ComponentMetadata entries
    """
    registry_dict = _get_registry_for_type(component_type)
    attr_field = 'func_name' if component_type == 'step' else 'class_name'

    for meta in metas:
        full_name = f"{source}:{meta['name']}"

        # Check if already loaded (eager pattern)
        if full_name in registry_dict:
            # Index from registry
            if component_type == 'step':
                # Steps store function directly
                obj = registry_dict[full_name]
                extra = {}
            else:
                # Kernels/backends store dict with 'class' key
                obj_data = registry_dict[full_name]
                obj = obj_data['class']

                # Extract type-specific metadata
                if component_type == 'kernel':
                    extra = {'infer_transform': obj_data.get('infer')}
                else:  # backend
                    extra = {
                        'target_kernel': obj_data['target_kernel'],
                        'language': obj_data['language']
                    }
                    if 'variant' in obj_data:
                        extra['variant'] = obj_data['variant']

            _component_index[full_name] = ComponentMetadata(
                name=meta['name'],
                source=source,
                component_type=component_type,
                loaded_obj=obj,
                import_spec=ImportSpec(
                    module=obj.__module__,
                    attr=obj.__name__,
                    extra=extra
                )
            )
            logger.debug(f"Indexed eager {component_type}: {full_name}")

        # Lazy pattern - store import spec
        elif 'module' in meta and attr_field in meta:
            # Extract type-specific metadata
            if component_type == 'kernel':
                extra = {'infer_transform': meta['infer_transform']} if 'infer_transform' in meta else {}
            elif component_type == 'backend':
                extra = {
                    'target_kernel': meta['target_kernel'],
                    'language': meta['language']
                }
                if 'variant' in meta:
                    extra['variant'] = meta['variant']
            else:  # step
                extra = {}

            _component_index[full_name] = ComponentMetadata(
                name=meta['name'],
                source=source,
                component_type=component_type,
                import_spec=ImportSpec(
                    module=meta['module'],
                    attr=meta[attr_field],
                    extra=extra
                )
            )
            logger.debug(f"Indexed lazy {component_type}: {full_name}")


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
            obj = _get_registry_for_type(meta.component_type).get(meta.full_name)
            if not obj:
                raise RuntimeError(f"Component {meta.full_name} registered but not found in registry")
            # For steps, registry stores function directly
            if meta.component_type == 'step':
                pass  # obj is the function
            else:
                obj = obj['class']  # For kernels/backends, unwrap from dict
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


def _get_registry_for_type(component_type: str) -> dict:
    """Get the appropriate registry dict for a component type."""
    try:
        return _COMPONENT_REGISTRIES[component_type]()
    except KeyError:
        valid = ', '.join(_COMPONENT_REGISTRIES.keys())
        raise ValueError(f"Unknown component type: {component_type}. Valid: {valid}")


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


def discover_plugins(use_cache: bool = True, force_refresh: bool = False):
    """Discover and load all plugins from configured sources.

    This loads:
    1. Core brainsmith plugins (kernels, steps) via direct imports
    2. User plugins from plugin_sources (must have __init__.py)
    3. Entry points from pip-installed packages (e.g., FINN)

    Plugins self-register during import using the global registry.

    This is called automatically on first component lookup.

    Args:
        use_cache: If True, try to load from cached manifest
        force_refresh: If True, ignore cache and regenerate manifest
    """
    global _plugins_discovered

    # Handle force refresh - reset discovery state to allow re-discovery
    if force_refresh and _plugins_discovered:
        logger.info("Force refresh requested - resetting discovery state")
        _plugins_discovered = False
        _component_index.clear()

    # Skip if already discovered (and not forcing refresh)
    if _plugins_discovered:
        return

    with _measure_load('discover_plugins'):
        # Check cache_plugins setting and get project_dir
        from brainsmith.settings import get_config
        config = get_config()
        cache_enabled = config.cache_plugins

        # Use project_dir for manifest location (not CWD)
        manifest_path = config.project_dir / '.brainsmith' / 'component_manifest.json'

        if not cache_enabled:
            logger.debug("Plugin caching disabled via cache_plugins setting")
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
                    _plugins_discovered = True

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
        _load_user_plugins()

        # 3. Plugin components (entry points - FINN, etc.)
        _load_entry_point_plugins()

        _plugins_discovered = True

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
        if source_name in (SOURCE_BRAINSMITH, SOURCE_FINN):
            continue

        if not source_path.exists():
            logger.debug(f"Plugin source '{source_name}' does not exist: {source_path}")
            continue

        _load_plugin_package(source_name, source_path)


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
                    # Index all component types using unified helper
                    _index_plugin_components(source_name, 'kernel', components.get('kernels', []))
                    _index_plugin_components(source_name, 'backend', components.get('backends', []))
                    _index_plugin_components(source_name, 'step', components.get('steps', []))

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


# ============================================================================
# Unified Component Access (Arete: Deduplicated Public API)
# ============================================================================

# Type-to-registry mapping for unified component access
_COMPONENT_REGISTRIES = {
    'step': lambda: registry._steps,
    'kernel': lambda: registry._kernels,
    'backend': lambda: registry._backends,
}

# Type-to-unwrapper mapping (kernels/backends store dicts, steps store functions)
_COMPONENT_UNWRAPPERS = {
    'step': lambda obj: obj,  # Steps stored directly
    'kernel': lambda obj: obj['class'],  # Kernels in dict with 'class' key
    'backend': lambda obj: obj['class'],  # Backends in dict with 'class' key
}


def _get_component(name: str, component_type: str):
    """Unified component lookup - single source of truth.

    All public get_*() functions delegate to this implementation.
    Type-specific behavior is driven by _COMPONENT_REGISTRIES and
    _COMPONENT_UNWRAPPERS mappings.

    Args:
        name: Component name (with or without source prefix)
        component_type: One of 'step', 'kernel', 'backend'

    Returns:
        Loaded component (class or function)

    Raises:
        KeyError: If component not found
    """
    with _measure_load(f'get_{component_type}', name):
        if not _plugins_discovered:
            discover_plugins()

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

        # Get from registry and unwrap
        registry_dict = _COMPONENT_REGISTRIES[component_type]()
        obj = registry_dict[full_name]
        return _COMPONENT_UNWRAPPERS[component_type](obj)


def _has_component(name: str, component_type: str) -> bool:
    """Unified component existence check.

    All public has_*() functions delegate to this implementation.

    Args:
        name: Component name (with or without source prefix)
        component_type: One of 'step', 'kernel', 'backend'

    Returns:
        True if component exists in index
    """
    if not _plugins_discovered:
        discover_plugins()

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
    if not _plugins_discovered:
        discover_plugins()

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
    if not _plugins_discovered:
        discover_plugins()

    full_name = _resolve_component_name(name, 'kernel')

    # Lookup in unified component index
    meta = _component_index.get(full_name)
    if not meta:
        raise KeyError(f"Kernel '{full_name}' not found")

    # Load component
    _load_component(meta)

    # Check registry for infer transform
    kernel_meta = registry._kernels[full_name]
    if kernel_meta['infer'] is None:
        raise KeyError(f"Kernel '{full_name}' has no InferTransform")

    return kernel_meta['infer']


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

    Accepts both short names (uses default_source) and fully-qualified names (source:name).

    Args:
        name: Backend name (e.g., 'LayerNorm_HLS' or 'user:LayerNorm_HLS_Fast')

    Returns:
        Backend class

    Examples:
        >>> backend = get_backend('LayerNorm_HLS')  # Uses default_source
        >>> custom = get_backend('user:LayerNorm_HLS_Fast')  # Explicit source
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
    if not _plugins_discovered:
        discover_plugins()

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

    # Load component (handles both filesystem and plugin sources)
    _load_component(meta)

    # Return from registry (where _load_component registered it)
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

    matching = []

    # Iterate through component index
    for full_name, meta in _component_index.items():
        # Filter to backends only
        if meta.component_type != 'backend':
            continue

        # Filter by sources if specified
        if sources and meta.source not in sources:
            continue

        # Load backend to get metadata (target_kernel, language)
        # For filesystem components, metadata isn't available until decorator fires
        try:
            _load_component(meta)
        except Exception:
            # Import failed, skip this backend
            continue

        # Check metadata in registry (populated after loading)
        if full_name not in registry._backends:
            continue

        backend_meta = registry._backends[full_name]
        if backend_meta['target_kernel'] != kernel_full:
            continue
        if language and backend_meta['language'] != language:
            continue

        matching.append(full_name)

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


# Auto-discover on module import (lazy)
# Actual discovery happens on first component lookup
