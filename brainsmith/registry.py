# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Component registration via @step, @kernel, @backend decorators.

See docs/ARCHITECTURE.md for details.
"""

import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Type, Union
from pathlib import Path

from brainsmith.constants import (
    SOURCE_BRAINSMITH,
    SOURCE_FINN,
    SOURCE_MODULE_PREFIXES,
    DEFAULT_SOURCE_PRIORITY,
)

logger = logging.getLogger(__name__)

# Current source context (set during plugin discovery)
_current_source: Optional[str] = None


# === Decorator Functions (User API) ===

def _make_decorator(
    component_type: str,
    registry_method: Callable,
    name_extractor: Callable[[Any], str]
):
    """Factory for @step, @kernel, @backend decorators."""
    def decorator(obj=None, **kwargs):
        def register(o):
            kwargs.setdefault('name', name_extractor(o))
            registry_method(o, **kwargs)
            return o
        return register(obj) if obj is not None else register

    decorator.__name__ = component_type
    return decorator


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

        # Extract infer_transform, handling @property decorators
        # Properties return descriptors when accessed on class, we need the actual value
        if infer_transform is None:
            infer_attr = getattr(cls, 'infer_transform', None)
            if isinstance(infer_attr, property):
                # Call property getter to get the actual class
                # Properties on brainsmith kernels are used to avoid circular imports
                try:
                    infer_transform = infer_attr.fget(None)
                except (TypeError, AttributeError):
                    # Property requires instance, can't unwrap
                    infer_transform = None
            else:
                # Class attribute or None, use directly
                infer_transform = infer_attr

        # Extract metadata - parameters override class attributes
        metadata = {
            'class': cls,
            'infer': infer_transform,
            'domain': domain or getattr(cls, 'domain', 'finn.custom')
        }

        self._kernels[full_name] = metadata
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
        metadata = {
            'class': cls,
            'target_kernel': target_kernel or getattr(cls, 'target_kernel', None),
            'language': language or getattr(cls, 'language', None),
            'variant': variant or getattr(cls, 'variant', None)
        }

        # Validate required fields
        if not metadata['target_kernel']:
            raise ValueError(f"Backend {full_name} missing 'target_kernel' attribute")
        if not metadata['language']:
            raise ValueError(f"Backend {full_name} missing 'language' attribute")

        self._backends[full_name] = metadata
        logger.debug(
            f"Registered backend: {full_name} "
            f"(target={metadata['target_kernel']}, lang={metadata['language']})"
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


# Create decorators using factory (after registry is defined)
step = _make_decorator(
    'step',
    registry.step,
    lambda f: f.__name__
)

kernel = _make_decorator(
    'kernel',
    registry.kernel,
    lambda c: getattr(c, 'op_type', c.__name__)
)

backend = _make_decorator(
    'backend',
    registry.backend,
    lambda c: c.__name__
)


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
