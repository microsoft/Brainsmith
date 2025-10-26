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


def create_lazy_module(components: Dict[str, Dict[str, str]], package_name: str) -> Tuple[Callable, Callable]:
    """Create lazy loading functions for a plugin module.

    This helper implements PEP 562 lazy loading pattern using absolute import paths.
    Components are imported only when accessed, avoiding expensive upfront imports.

    IMPORTANT: Module paths must be ABSOLUTE (e.g., 'brainsmith.kernels.layernorm.layernorm'),
    not relative (e.g., '.layernorm.layernorm'). This simplifies loading and makes paths
    explicit.

    Args:
        components: Component metadata organized by type:
            {
                'kernels': {'LayerNorm': 'brainsmith.kernels.layernorm.layernorm'},
                'backends': {'LayerNorm_hls': 'brainsmith.kernels.layernorm.layernorm_hls'},
                'steps': {'streamline': 'brainsmith.steps.streamline'},
            }
        package_name: Package name (for error messages, not used for imports)

    Returns:
        Tuple of (__getattr__, __dir__) functions for module-level use

    Design:
        - Metadata-only at import time (fast discovery)
        - Lazy import on first access (defer heavy deps)
        - Direct imports using absolute paths (simple, explicit)
        - Cache loaded components (fast subsequent access)
        - Standard PEP 562 pattern (QONNX compatible)

    Example:
        >>> # In your __init__.py
        >>> from brainsmith.registry import create_lazy_module
        >>>
        >>> COMPONENTS = {
        ...     'kernels': {
        ...         'LayerNorm': 'brainsmith.kernels.layernorm.layernorm',
        ...     },
        ... }
        >>>
        >>> __getattr__, __dir__ = create_lazy_module(COMPONENTS, __name__)
        >>>
        >>> # Now getattr(module, 'LayerNorm') works (needed for QONNX integration)
    """

    # Flatten component types into single lookup dict
    _modules = {}  # component_name -> absolute_module_path
    _component_types = {}  # component_name -> component_type (for error messages)

    for component_type, items in components.items():
        for name, spec in items.items():
            # Extract module path (support both string and dict format for backwards compat)
            if isinstance(spec, str):
                module_path = spec
            else:
                # Dict format: {'module': 'path', ...metadata...}
                module_path = spec.get('module', spec)

            _modules[name] = module_path
            _component_types[name] = component_type

    # Cache for loaded components
    _loaded: Dict[str, Any] = {}

    def __getattr__(name: str):
        """Lazy import components on first access (PEP 562).

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
                    # Direct import using absolute path
                    module = import_module(module_path)
                except ImportError as e:
                    raise AttributeError(
                        f"Component '{name}' failed to load: module '{module_path}' not found.\n"
                        f"Check COMPONENTS['{component_type}']['{name}'] = '{module_path}'\n"
                        f"Error: {e}"
                    ) from e

                try:
                    # Get component from module
                    _loaded[name] = getattr(module, name)
                except AttributeError:
                    # For steps: decorator might have registered under different name
                    if component_type == 'steps':
                        _loaded[name] = None  # Marker for decorator pattern
                    else:
                        raise AttributeError(
                            f"Component '{name}' not found in module '{module_path}'.\n"
                            f"Module loaded but doesn't define '{name}'."
                        )

            return _loaded[name]

        # Component not in COMPONENTS dict
        available = list(_modules.keys())
        raise AttributeError(
            f"module '{package_name}' has no attribute '{name}'. "
            f"Available: {', '.join(available[:5])}" +
            (f" ... and {len(available) - 5} more" if len(available) > 5 else "")
        )

    def __dir__():
        """List available components without importing them."""
        return list(_modules.keys())

    return __getattr__, __dir__
