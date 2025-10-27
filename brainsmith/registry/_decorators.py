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

import logging
from contextvars import ContextVar
from typing import Any, Callable, Dict, List, Optional, Type, Union

from ._state import _component_index
from ._metadata import ComponentMetadata, ComponentType, ImportSpec

logger = logging.getLogger(__name__)

# Current source context (set during plugin discovery)
# Using ContextVar for thread-safe source tracking
_current_source: ContextVar[Optional[str]] = ContextVar('current_source', default=None)


# ============================================================================
# Helper Functions
# ============================================================================

def _convert_lazy_import_spec(spec: Optional[Union[str, Dict, Type]]) -> Optional[Union[Dict, Type]]:
    """Convert 'module:ClassName' string to {'module': ..., 'class_name': ...}.

    Passes through None, class references, or dicts unchanged.
    """
    if isinstance(spec, str) and ':' in spec:
        module_path, class_name = spec.split(':', 1)
        return {'module': module_path, 'class_name': class_name}
    return spec


# ============================================================================
# Registration Functions (Internal API)
# ============================================================================

def _register_step(
    func_or_class: Union[Callable, Type],
    *,
    source: Optional[str] = None,
    name: Optional[str] = None
) -> Union[Callable, Type]:
    """Register step in global component index."""
    source = source or _current_source.get() or 'custom'
    name = name or getattr(func_or_class, 'name', None) or func_or_class.__name__

    full_name = f"{source}:{name}"

    # Check if we're overriding an existing step
    # Don't warn if we're just loading a cached entry (loaded_obj is None)
    if full_name in _component_index:
        existing = _component_index[full_name]
        if existing.loaded_obj is not None:
            logger.warning(f"Overriding existing step: {full_name}")

    # Create/update index entry for standalone decorator usage
    logger.debug(f"Registering step: {full_name}")
    import_spec = ImportSpec(
        module=func_or_class.__module__,
        attr=func_or_class.__name__,
        extra={}
    )
    _component_index[full_name] = ComponentMetadata(
        name=name,
        source=source,
        component_type=ComponentType.STEP,
        import_spec=import_spec,
        loaded_obj=func_or_class  # Already loaded
    )

    return func_or_class


def _register_kernel(
    cls: Type,
    *,
    source: Optional[str] = None,
    name: Optional[str] = None,
    infer_transform: Optional[Type] = None,
    domain: Optional[str] = None,
    **kwargs  # Accept and ignore extra parameters (e.g., op_type) for backwards compat
) -> Type:
    """Register kernel in global component index."""
    source = source or _current_source.get() or 'custom'
    name = name or getattr(cls, 'op_type', None) or cls.__name__

    full_name = f"{source}:{name}"

    # Check if we're overriding an existing kernel
    # Don't warn if we're just loading a cached entry (loaded_obj is None)
    if full_name in _component_index:
        existing = _component_index[full_name]
        if existing.loaded_obj is not None:
            logger.warning(f"Overriding existing kernel: {full_name}")

    # Extract infer_transform (class attribute or parameter)
    # Supports both class references and string-based lazy import specs
    # Note: infer_transform is optional - not all kernels need it (e.g., test kernels)
    if infer_transform is None:
        infer_transform = getattr(cls, 'infer_transform', None)

    # Handle string-based lazy import specs (format: 'module.path:ClassName')
    infer_transform = _convert_lazy_import_spec(infer_transform)

    # Create/update index entry for standalone decorator usage
    logger.debug(f"Registering kernel: {full_name}")
    import_spec = ImportSpec(
        module=cls.__module__,
        attr=cls.__name__,
        extra={}
    )
    _component_index[full_name] = ComponentMetadata(
        name=name,
        source=source,
        component_type=ComponentType.KERNEL,
        import_spec=import_spec,
        loaded_obj=cls  # Already loaded
    )

    # Populate type-specific metadata (inline)
    from .constants import DEFAULT_KERNEL_DOMAIN

    meta = _component_index[full_name]
    # Kernel-specific metadata (infer_transform already converted above)
    meta.kernel_infer = infer_transform
    meta.kernel_domain = domain or getattr(cls, 'domain', DEFAULT_KERNEL_DOMAIN)

    return cls


def _register_backend(
    cls: Type,
    *,
    source: Optional[str] = None,
    name: Optional[str] = None,
    target_kernel: Optional[str] = None,
    language: Optional[str] = None,
    variant: Optional[str] = None
) -> Type:
    """Register backend in global component index."""
    source = source or _current_source.get() or 'custom'
    name = name or cls.__name__

    full_name = f"{source}:{name}"

    # Check if we're overriding an existing backend
    # Don't warn if we're just loading a cached entry (loaded_obj is None)
    if full_name in _component_index:
        existing = _component_index[full_name]
        if existing.loaded_obj is not None:
            # Actually overriding a loaded backend - this is unusual
            logger.warning(f"Overriding existing backend: {full_name}")

    # Extract metadata - parameters override class attributes
    target = target_kernel or getattr(cls, 'target_kernel', None)
    lang = language or getattr(cls, 'language', None)

    # Validate required fields
    if not target:
        raise ValueError(f"Backend {full_name} missing 'target_kernel' attribute")
    if not lang:
        raise ValueError(f"Backend {full_name} missing 'language' attribute")

    # Create/update index entry for standalone decorator usage
    logger.debug(f"Registering backend: {full_name} (target={target}, lang={lang})")
    import_spec = ImportSpec(
        module=cls.__module__,
        attr=cls.__name__,
        extra={}
    )
    _component_index[full_name] = ComponentMetadata(
        name=name,
        source=source,
        component_type=ComponentType.BACKEND,
        import_spec=import_spec,
        loaded_obj=cls  # Already loaded
    )

    # Populate type-specific metadata (always works now)
    meta = _component_index[full_name]
    meta.backend_target = target
    meta.backend_language = lang

    # Link backend to its target kernel (if kernel exists)
    kernel_meta = _component_index.get(target)
    if kernel_meta:
        if kernel_meta.kernel_backends is None:
            kernel_meta.kernel_backends = []
        if full_name not in kernel_meta.kernel_backends:
            kernel_meta.kernel_backends.append(full_name)

    return cls


# ============================================================================
# Public Decorator API ===
# Direct decorator implementations (no factory needed)

def step(obj=None, **kwargs):
    """Register a step function or class.

    Supports both @step and @step(name='custom_name') syntax.
    """
    def register(func_or_class):
        kwargs.setdefault('name', func_or_class.__name__)
        _register_step(func_or_class, **kwargs)
        return func_or_class
    return register(obj) if obj is not None else register


def kernel(obj=None, **kwargs):
    """Register a kernel class.

    Supports both @kernel and @kernel(name='custom_name') syntax.
    """
    def register(cls):
        kwargs.setdefault('name', getattr(cls, 'op_type', cls.__name__))
        _register_kernel(cls, **kwargs)
        return cls
    return register(obj) if obj is not None else register


def backend(obj=None, **kwargs):
    """Register a backend class.

    Supports both @backend and @backend(...) syntax.
    """
    def register(cls):
        kwargs.setdefault('name', cls.__name__)
        _register_backend(cls, **kwargs)
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
        self.token = None

    def __enter__(self):
        """Enter context, set current source."""
        self.token = _current_source.set(self.source_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context, restore previous source."""
        _current_source.reset(self.token)
        return False


# Export decorators for public API
__all__ = ['source_context', 'step', 'kernel', 'backend']
