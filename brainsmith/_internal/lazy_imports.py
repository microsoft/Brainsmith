# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Reusable PEP 562 lazy import implementation.

This module provides utilities for implementing lazy module loading to improve
import performance. By deferring imports until first access, modules can avoid
loading heavy dependencies during initial import.

Example usage in a module's __init__.py:

    from brainsmith._internal.lazy_imports import LazyModuleLoader

    _lazy_loader = LazyModuleLoader({
        'SystemConfig': 'brainsmith.settings',
        'get_config': 'brainsmith.settings',
    }, package=__name__)

    def __getattr__(name):
        return _lazy_loader.get_attribute(name)

    def __dir__():
        return _lazy_loader.dir()
"""

from importlib import import_module
from typing import Any


class LazyModuleLoader:
    """Reusable PEP 562 lazy import implementation.

    This class provides a clean, testable implementation of lazy module loading
    using Python's __getattr__ mechanism (PEP 562). It caches loaded modules
    to avoid repeated imports.

    Attributes:
        lazy_map: Mapping of attribute names to module paths
        package: Package name for relative imports (optional)
        cache: Internal cache of loaded attributes

    Example:
        >>> loader = LazyModuleLoader({
        ...     'some_function': 'mypackage.module'
        ... }, package='mypackage')
        >>> func = loader.get_attribute('some_function')
    """

    def __init__(self, lazy_map: dict[str, str], package: str | None = None):
        """Initialize the lazy loader.

        Example:
            {'Config': 'myapp.config'}
        """
        self._lazy_map = lazy_map
        self._cache: dict[str, Any] = {}
        self._package = package

    def get_attribute(self, name: str) -> Any:
        """Get an attribute, loading it lazily if needed.

        Raises:
            AttributeError: If the attribute is not in the lazy_map
        """
        if name not in self._lazy_map:
            raise AttributeError(f"module {self._package!r} has no attribute {name!r}")

        if name not in self._cache:
            module_path = self._lazy_map[name]

            if self._package:
                module = import_module(f".{module_path}", package=self._package)
            else:
                module = import_module(module_path)

            self._cache[name] = getattr(module, name)

        return self._cache[name]

    def dir(self) -> list[str]:
        return list(self._lazy_map.keys())

    def clear_cache(self) -> None:
        self._cache.clear()
