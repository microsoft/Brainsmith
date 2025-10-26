# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Component registration via @step, @kernel, @backend decorators.

See docs/ARCHITECTURE.md for details.

Logging Strategy:
    - DEBUG: Individual component registrations and metadata extraction
    - INFO: Not used in this module (registration is low-level)
    - WARNING: Overwriting existing components
    - ERROR: Registration failures (missing required fields)
"""

import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Type, Union
from pathlib import Path

from .constants import (
    SOURCE_BRAINSMITH,
    SOURCE_FINN,
    SOURCE_MODULE_PREFIXES,
    DEFAULT_SOURCE_PRIORITY,
)

logger = logging.getLogger(__name__)

# Current source context (set during plugin discovery)
_current_source: Optional[str] = None


# === Decorator Functions (User API) ===
# These decorators support both @decorator and @decorator(...) syntax


class Registry:
    """Central registry for Brainsmith components.

    Stores steps, kernels, and backends with source-prefixed names.
    Components register via @step, @kernel, @backend decorators.

    Internal implementation - users should use decorators, not registry methods directly.
    """

    def __init__(self):
        """Initialize empty registries."""
        self._steps: Dict[str, Any] = {}
        self._kernels: Dict[str, Dict[str, Any]] = {}
        self._backends: Dict[str, Dict[str, Any]] = {}

    def step(
        self,
        func_or_class: Union[Callable, Type],
        *,
        source: Optional[str] = None,
        name: Optional[str] = None
    ) -> Union[Callable, Type]:
        """Register step in global registry."""
        source = source or self._detect_source(func_or_class)
        name = name or self._extract_step_name(func_or_class)

        full_name = f"{source}:{name}"

        if full_name in self._steps:
            logger.warning(f"Overriding existing step: {full_name}")

        self._steps[full_name] = func_or_class

        # ALWAYS create/update component index entry (Option 1)
        from ._state import _component_index
        from ._metadata import ComponentMetadata, ImportSpec

        if full_name not in _component_index:
            # Create new index entry for standalone decorator usage
            logger.debug(f"Creating index entry for standalone step: {full_name}")
            import_spec = ImportSpec(
                module=func_or_class.__module__,
                attr=func_or_class.__name__,
                extra={}
            )
            _component_index[full_name] = ComponentMetadata(
                name=name,
                source=source,
                component_type='step',
                import_spec=import_spec,
                loaded_obj=func_or_class  # Already loaded
            )

        logger.debug(f"Registered step: {full_name}")

        return func_or_class

    def kernel(
        self,
        cls: Type,
        *,
        source: Optional[str] = None,
        name: Optional[str] = None,
        infer_transform: Optional[Type] = None,
        domain: Optional[str] = None
    ) -> Type:
        """Register kernel in global registry."""
        source = source or self._detect_source(cls)
        name = name or self._extract_kernel_name(cls)

        full_name = f"{source}:{name}"

        if full_name in self._kernels:
            logger.warning(f"Overriding existing kernel: {full_name}")

        # Extract infer_transform (class attribute or parameter)
        # Supports both class references and string-based lazy import specs
        if infer_transform is None:
            infer_transform = getattr(cls, 'infer_transform', None)

        # Handle string-based lazy import specs (format: 'module.path:ClassName')
        if isinstance(infer_transform, str) and ':' in infer_transform:
            module_path, class_name = infer_transform.split(':', 1)
            infer_transform = {'module': module_path, 'class_name': class_name}

        # Store kernel class directly in registry
        self._kernels[full_name] = cls

        # ALWAYS create/update component index entry (Option 1)
        from ._state import _component_index
        from ._metadata import ComponentMetadata, ImportSpec

        if full_name not in _component_index:
            # Create new index entry for standalone decorator usage
            logger.debug(f"Creating index entry for standalone kernel: {full_name}")
            import_spec = ImportSpec(
                module=cls.__module__,
                attr=cls.__name__,
                extra={}
            )
            _component_index[full_name] = ComponentMetadata(
                name=name,
                source=source,
                component_type='kernel',
                import_spec=import_spec,
                loaded_obj=cls  # Already loaded
            )

        # Populate type-specific metadata (always works now)
        meta = _component_index[full_name]
        meta.kernel_infer = infer_transform
        meta.kernel_domain = domain or getattr(cls, 'domain', 'finn.custom')

        logger.debug(f"Registered kernel: {full_name}")

        return cls

    def backend(
        self,
        cls: Type,
        *,
        source: Optional[str] = None,
        name: Optional[str] = None,
        target_kernel: Optional[str] = None,
        language: Optional[str] = None,
        variant: Optional[str] = None
    ) -> Type:
        """Register backend in global registry."""
        source = source or self._detect_source(cls)
        name = name or cls.__name__

        full_name = f"{source}:{name}"

        if full_name in self._backends:
            logger.warning(f"Overriding existing backend: {full_name}")

        # Extract metadata - parameters override class attributes
        target = target_kernel or getattr(cls, 'target_kernel', None)
        lang = language or getattr(cls, 'language', None)

        # Validate required fields
        if not target:
            raise ValueError(f"Backend {full_name} missing 'target_kernel' attribute")
        if not lang:
            raise ValueError(f"Backend {full_name} missing 'language' attribute")

        # Store backend class directly in registry
        self._backends[full_name] = cls

        # ALWAYS create/update component index entry (Option 1)
        from ._state import _component_index
        from ._metadata import ComponentMetadata, ImportSpec

        if full_name not in _component_index:
            # Create new index entry for standalone decorator usage
            logger.debug(f"Creating index entry for standalone backend: {full_name}")
            import_spec = ImportSpec(
                module=cls.__module__,
                attr=cls.__name__,
                extra={}
            )
            _component_index[full_name] = ComponentMetadata(
                name=name,
                source=source,
                component_type='backend',
                import_spec=import_spec,
                loaded_obj=cls  # Already loaded
            )

        # Populate type-specific metadata (always works now)
        meta = _component_index[full_name]
        meta.backend_target = target
        meta.backend_language = lang

        logger.debug(
            f"Registered backend: {full_name} "
            f"(target={target}, lang={lang})"
        )

        return cls

    def _detect_source(self, obj: Any) -> str:
        """Auto-detect source: context > module prefix > config default."""
        if _current_source:
            return _current_source

        if module := inspect.getmodule(obj):
            for prefix, source in SOURCE_MODULE_PREFIXES.items():
                if module.__name__.startswith(prefix):
                    return source

        try:
            from brainsmith.settings import get_config
            return get_config().source_priority[0]
        except (ImportError, AttributeError):
            return DEFAULT_SOURCE_PRIORITY[0]

    def _extract_step_name(self, func_or_class: Any) -> str:
        """Extract step name from function/class (prefers .name attribute)."""
        # Check for explicit name attribute
        if hasattr(func_or_class, 'name') and func_or_class.name:
            return func_or_class.name

        # Use __name__
        return func_or_class.__name__

    def _extract_kernel_name(self, cls: Type) -> str:
        """Extract kernel name (prefers cls.op_type over __name__)."""
        # Prefer op_type class attribute if defined (for ONNX compatibility)
        # Note: This is just for name extraction, not stored in metadata
        if hasattr(cls, 'op_type') and cls.op_type:
            return cls.op_type

        # Fallback to class name
        return cls.__name__

    def clear(self):
        """Clear all registrations (for testing)."""
        self._steps.clear()
        self._kernels.clear()
        self._backends.clear()
        logger.debug("Registry cleared")

    def __repr__(self) -> str:
        """String representation showing registry stats."""
        return (
            f"<Registry: "
            f"{len(self._steps)} steps, "
            f"{len(self._kernels)} kernels, "
            f"{len(self._backends)} backends>"
        )


# Global singleton registry
registry = Registry()


# === Public Decorator API ===
# Direct decorator implementations (no factory needed)

def step(obj=None, **kwargs):
    """Register a step function or class.

    Supports both @step and @step(name='custom_name') syntax.

    Args:
        obj: Function or class to register (when used as @step)
        **kwargs: Additional arguments passed to Registry.step()

    Returns:
        Decorated function/class (unchanged)

    Examples:
        >>> @step
        ... def my_step(model, config):
        ...     pass

        >>> @step(name='custom_name')
        ... def my_step_func(model, config):
        ...     pass
    """
    def register(func_or_class):
        kwargs.setdefault('name', func_or_class.__name__)
        registry.step(func_or_class, **kwargs)
        return func_or_class
    return register(obj) if obj is not None else register


def kernel(obj=None, **kwargs):
    """Register a kernel class.

    Supports both @kernel and @kernel(name='custom_name') syntax.

    Args:
        obj: Class to register (when used as @kernel)
        **kwargs: Additional arguments passed to Registry.kernel()

    Returns:
        Decorated class (unchanged)

    Examples:
        >>> @kernel
        ... class MyKernel(HWCustomOp):
        ...     op_type = 'MyKernel'

        >>> @kernel(name='CustomName', domain='custom.domain')
        ... class MyKernelClass(HWCustomOp):
        ...     pass
    """
    def register(cls):
        kwargs.setdefault('name', getattr(cls, 'op_type', cls.__name__))
        registry.kernel(cls, **kwargs)
        return cls
    return register(obj) if obj is not None else register


def backend(obj=None, **kwargs):
    """Register a backend class.

    Supports both @backend and @backend(...) syntax.

    Args:
        obj: Class to register (when used as @backend)
        **kwargs: Additional arguments passed to Registry.backend()

    Returns:
        Decorated class (unchanged)

    Examples:
        >>> @backend(target_kernel='MyKernel', language='hls')
        ... class MyKernel_hls:
        ...     pass
    """
    def register(cls):
        kwargs.setdefault('name', cls.__name__)
        registry.backend(cls, **kwargs)
        return cls
    return register(obj) if obj is not None else register


# Context manager for source detection
class source_context:
    """Context manager for setting current source during discovery.

    Examples:
        >>> with source_context('user'):
        ...     import user_plugin  # Auto-detects source='user'
    """

    def __init__(self, source_name: str):
        """Initialize context with source name.

        Args:
            source_name: Source to set as current
        """
        self.source_name = source_name
        self.old_source = None

    def __enter__(self):
        """Enter context, set current source."""
        global _current_source
        self.old_source = _current_source
        _current_source = self.source_name
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context, restore previous source."""
        global _current_source
        _current_source = self.old_source
        return False


# Export decorators for public API
__all__ = ['Registry', 'registry', 'source_context', 'step', 'kernel', 'backend']
