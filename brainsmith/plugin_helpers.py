# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Plugin Helper Utilities - Lazy Loading Pattern

Provides reusable lazy loading infrastructure for both core brainsmith
components and user plugins. Everyone uses the same pattern.

Example (user plugins):
    >>> # plugins/__init__.py
    >>> from brainsmith.plugin_helpers import create_lazy_module
    >>>
    >>> COMPONENTS = {
    ...     'kernels': {'MyKernel': '.my_kernel'},
    ...     'backends': {'MyKernel_hls': '.my_backend'},
    ...     'steps': {'my_step': '.my_step'},
    ... }
    >>>
    >>> __getattr__, __dir__ = create_lazy_module(COMPONENTS, __name__)

Example (core brainsmith):
    >>> # brainsmith/kernels/__init__.py (simplified)
    >>> COMPONENTS = {
    ...     'kernels': {
    ...         'LayerNorm': '.layernorm.layernorm',
    ...         'Crop': '.crop.crop',
    ...     }
    ... }
    >>> __getattr__, __dir__ = create_lazy_module(COMPONENTS, __name__)
"""

from typing import Dict, Tuple, Callable, Any, List, TypedDict, Optional
from importlib import import_module
import os
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Type Definitions
# ============================================================================


class ComponentsDict(TypedDict, total=False):
    """Type definition for COMPONENTS dictionary structure.

    All keys are optional. Each key maps to a dict of component_name -> module_path.
    Module paths should be relative (start with '.') for proper package imports.

    Example:
        >>> COMPONENTS: ComponentsDict = {
        ...     'kernels': {'MyKernel': '.my_kernel'},
        ...     'backends': {'MyKernel_hls': '.my_backend'},
        ...     'steps': {'my_step': '.my_step'},
        ...     'modules': {'utils': '.utils'},
        ... }
    """
    kernels: Dict[str, str]
    backends: Dict[str, str]
    steps: Dict[str, str]
    modules: Dict[str, str]


# ============================================================================
# Validation
# ============================================================================


def validate_components(components: Dict[str, Dict[str, str]],
                       package_name: str = "",
                       strict: bool = False) -> List[str]:
    """Validate COMPONENTS dictionary structure and module paths.

    Checks for common issues:
    - Module paths that don't start with '.' (should be relative)
    - Duplicate component names across types
    - Invalid module paths (strict mode only - attempts import)

    Args:
        components: COMPONENTS dict to validate
        package_name: Package name for resolving relative imports (strict mode only)
        strict: If True, attempt to import modules to verify they exist

    Returns:
        List of validation error messages (empty if valid)

    Example:
        >>> COMPONENTS = {'kernels': {'K1': 'absolute.path'}}  # Bad: absolute path
        >>> errors = validate_components(COMPONENTS)
        >>> print(errors)
        ["Module path 'absolute.path' should start with '.' (relative import)"]
    """
    errors: List[str] = []
    seen_names: Dict[str, str] = {}  # component_name -> component_type

    valid_types = {'kernels', 'backends', 'steps', 'modules'}

    for component_type, items in components.items():
        # Validate component type
        if component_type not in valid_types:
            errors.append(
                f"Unknown component type '{component_type}'. "
                f"Valid types: {', '.join(sorted(valid_types))}"
            )
            continue

        if not isinstance(items, dict):
            errors.append(
                f"Component type '{component_type}' must be a dict, got {type(items).__name__}"
            )
            continue

        for name, module_path in items.items():
            # Check for string paths
            if not isinstance(module_path, str):
                errors.append(
                    f"COMPONENTS['{component_type}']['{name}']: "
                    f"module path must be string, got {type(module_path).__name__}"
                )
                continue

            # Check for relative imports
            if not module_path.startswith('.'):
                errors.append(
                    f"COMPONENTS['{component_type}']['{name}']: "
                    f"module path '{module_path}' should start with '.' (relative import)"
                )

            # Check for duplicates across component types
            if name in seen_names:
                errors.append(
                    f"Duplicate component name '{name}' in '{component_type}' "
                    f"(already defined in '{seen_names[name]}')"
                )
            else:
                seen_names[name] = component_type

            # Strict mode: attempt import
            if strict and package_name:
                try:
                    import_module(module_path, package=package_name)
                except ImportError as e:
                    errors.append(
                        f"COMPONENTS['{component_type}']['{name}']: "
                        f"module '{module_path}' cannot be imported: {e}"
                    )
                except Exception as e:
                    errors.append(
                        f"COMPONENTS['{component_type}']['{name}']: "
                        f"unexpected error importing '{module_path}': {e}"
                    )

    return errors


def create_lazy_module(components: Dict[str, Dict[str, str]], package_name: str) -> Tuple[Callable, Callable]:
    """Create lazy loading functions for a plugin module.

    This helper implements PEP 562 lazy loading pattern. Components are
    imported only when accessed, avoiding expensive upfront imports.

    Args:
        components: Component metadata organized by type:
            {
                'kernels': {'KernelName': '.module.path'},
                'backends': {'BackendName': '.module.path'},
                'steps': {'step_name': '.module.path'},
            }
        package_name: Package name for relative imports (use __name__)

    Returns:
        Tuple of (__getattr__, __dir__) functions for module-level use

    Design:
        - Metadata-only at import time (fast discovery)
        - Lazy import on first access (defer heavy deps)
        - Cache loaded components (fast subsequent access)
        - Standard PEP 562 pattern (familiar to users)

    Example:
        >>> # In your __init__.py
        >>> from brainsmith.plugin_helpers import create_lazy_module
        >>>
        >>> COMPONENTS = {
        ...     'kernels': {'MyKernel': '.my_kernel'},
        ...     'steps': {'my_step': '.my_step'},
        ... }
        >>>
        >>> __getattr__, __dir__ = create_lazy_module(COMPONENTS, __name__)
        >>>
        >>> # Now your module supports lazy loading:
        >>> # - dir(module) lists all components (fast)
        >>> # - module.MyKernel imports only when accessed
    """
    # Runtime validation (opt-in via environment variable)
    if os.environ.get('BRAINSMITH_VALIDATE_PLUGINS'):
        errors = validate_components(components, package_name, strict=False)
        if errors:
            logger.warning(
                f"COMPONENTS validation warnings in '{package_name}':\n  " +
                "\n  ".join(errors)
            )

    # Flatten component types into single lookup dict, tracking source type
    _modules = {}  # component_name -> module_path
    _component_types = {}  # component_name -> component_type (for error messages)

    for component_type, items in components.items():
        for name, path in items.items():
            _modules[name] = path
            _component_types[name] = component_type

    # Cache for loaded components
    _loaded: Dict[str, Any] = {}

    def __getattr__(name: str):
        """Lazy import components on first access (PEP 562).

        Called automatically when attribute not found in module namespace.
        Imports the specific component module and caches the result.

        Args:
            name: Component name to load

        Returns:
            Component class or function

        Raises:
            AttributeError: If component doesn't exist or fails to load
        """
        if name in _modules:
            if name not in _loaded:
                module_path = _modules[name]
                component_type = _component_types.get(name, 'component')

                try:
                    # Import the module
                    module = import_module(module_path, package=package_name)
                except ImportError as e:
                    # Module doesn't exist or has import errors
                    raise AttributeError(
                        f"Component '{name}' failed to load: module '{module_path}' not found.\n"
                        f"Check COMPONENTS['{component_type}']['{name}'] = '{module_path}'\n"
                        f"Error: {e}"
                    ) from e
                except Exception as e:
                    # Other import failures (syntax errors, etc)
                    raise AttributeError(
                        f"Component '{name}' failed to load: error importing '{module_path}'.\n"
                        f"Check COMPONENTS['{component_type}']['{name}'] = '{module_path}'\n"
                        f"Error: {type(e).__name__}: {e}"
                    ) from e

                try:
                    # Get the component from the module
                    _loaded[name] = getattr(module, name)
                except AttributeError as e:
                    # Module exists but doesn't contain the component
                    raise AttributeError(
                        f"Component '{name}' not found in module '{module_path}'.\n"
                        f"The module loaded successfully but doesn't define '{name}'.\n"
                        f"Check that {module_path.lstrip('.')} contains 'class {name}' or 'def {name}'."
                    ) from e

            return _loaded[name]

        # Component name not in COMPONENTS dict
        available = list(_modules.keys())
        raise AttributeError(
            f"module '{package_name}' has no attribute '{name}'. "
            f"Available: {', '.join(available[:5])}" +
            (f" ... and {len(available) - 5} more" if len(available) > 5 else "")
        )

    def __dir__():
        """List available components without importing them.

        Called by dir(), IDE autocomplete, and discovery tools.
        Returns component names from metadata (no imports).

        Returns:
            List of available component names
        """
        return list(_modules.keys())

    return __getattr__, __dir__
