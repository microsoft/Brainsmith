# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Component registration system.

This module provides the central registry for all Brainsmith components:
steps, kernels, and backends.

Registration Pattern:
- **Decorators**: @step, @kernel, @backend
  - User-facing, ergonomic API
  - Immediate registration during import
  - Example: @kernel(name='LayerNorm', op_type='LayerNorm')

Internal Implementation:
- Registry.step(), Registry.kernel(), Registry.backend() methods are internal
- Called by decorators during component import
- Users should NOT call these methods directly
"""

import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Type, Union
from pathlib import Path

logger = logging.getLogger(__name__)

# Current source context (set during plugin discovery)
_current_source: Optional[str] = None


# === Decorator Functions (User API) ===

def step(
    _func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    **metadata
) -> Callable:
    """Register a step function via decorator or explicit call.

    Supports syntax patterns:
    - @step (decorator without params)
    - @step(...) (decorator with params)
    - step(my_function) (explicit function registration)

    Args:
        _func: Function to register
        name: Step name (defaults to function name)
        **metadata: Additional metadata (reserved for future use)

    Returns:
        Original function (for decorator use)

    Examples:
        >>> # Decorator without params
        >>> @step
        ... def my_step(model, cfg):
        ...     return model

        >>> # Decorator with params
        >>> @step(name='custom_name')
        ... def my_step(model, cfg):
        ...     return model

        >>> # Explicit function registration
        >>> step(my_step_function, name='my_step')
    """

    def register_step(func: Callable) -> Callable:
        # Store metadata on function for potential later access
        func.__brainsmith_step__ = {
            'name': name or func.__name__,
            'metadata': metadata
        }

        # Register immediately
        registry.step(func, name=name or func.__name__)

        logger.debug(
            f"Registered step: {func.__module__}.{func.__name__}"
        )

        return func

    # Support decorator syntax: @step, @step(), @step(name='...')
    if _func is not None:
        # Called as: @step or step(my_func)
        return register_step(_func)
    else:
        # Called as: @step(...) with parameters only
        return register_step


def kernel(
    _cls: Optional[Type] = None,
    *,
    name: Optional[str] = None,
    infer_transform: Optional[Callable] = None,
    domain: Optional[str] = None,
    **metadata
) -> Type:
    """Register a kernel class via decorator or explicit call.

    Supports syntax patterns:
    - @kernel (decorator without params)
    - @kernel(...) (decorator with params)
    - kernel(MyClass) (explicit class registration)

    Args:
        _cls: Class to register
        name: Kernel name (defaults to class name)
        infer_transform: Optional inference transform callable
        domain: ONNX domain (defaults to class.domain or 'finn.custom')
        **metadata: Additional metadata

    Returns:
        Original class (for decorator use)

    Examples:
        >>> # Decorator without params
        >>> @kernel
        ... class LayerNorm(CustomOp):
        ...     op_type = "LayerNorm"

        >>> # Decorator with params
        >>> @kernel(name='CustomName')
        ... class MyKernel(CustomOp):
        ...     pass

        >>> # Explicit class registration
        >>> kernel(MyKernelClass, name='MyKernel')
    """

    def register_kernel(cls: Type) -> Type:
        # Store metadata on class for potential later access
        # Parameters override class attributes
        # Note: Don't extract infer_transform here - let registry.kernel() handle it
        cls.__brainsmith_kernel__ = {
            'name': name or getattr(cls, 'op_type', cls.__name__),
            'infer_transform': infer_transform,  # Store decorator parameter only
            'domain': domain or getattr(cls, 'domain', None),
            'metadata': metadata
        }

        # Register immediately
        # Pass decorator parameter (may be None), let registry extract from class if needed
        registry.kernel(
            cls,
            name=name or getattr(cls, 'op_type', cls.__name__),
            infer_transform=infer_transform,  # Let registry.kernel() handle extraction
            domain=domain or getattr(cls, 'domain', None)
        )

        logger.debug(
            f"Registered kernel: {cls.__module__}.{cls.__name__}"
        )

        return cls

    # Support decorator syntax: @kernel, @kernel(), @kernel(name='...')
    if _cls is not None:
        # Called as: @kernel or kernel(MyClass)
        return register_kernel(_cls)
    else:
        # Called as: @kernel(...) with parameters only
        return register_kernel


def backend(
    _cls: Optional[Type] = None,
    *,
    name: Optional[str] = None,
    target_kernel: Optional[str] = None,
    language: Optional[str] = None,
    variant: Optional[str] = None,
    **metadata
) -> Type:
    """Register a backend class via decorator or explicit call.

    Supports syntax patterns:
    - @backend (decorator without params)
    - @backend(...) (decorator with params)
    - backend(MyClass) (explicit class registration)

    Args:
        _cls: Class to register
        name: Backend name (defaults to class name)
        target_kernel: Target kernel full name (e.g., 'brainsmith:LayerNorm')
        language: Backend language ('hls' or 'rtl')
        variant: Optional variant name
        **metadata: Additional metadata

    Returns:
        Original class (for decorator use)

    Examples:
        >>> # Decorator without params
        >>> @backend
        ... class LayerNorm_hls(LayerNorm, HLSBackend):
        ...     target_kernel = 'brainsmith:LayerNorm'
        ...     language = 'hls'

        >>> # Decorator with params
        >>> @backend(target_kernel='user:Custom', language='hls')
        ... class CustomBackend(HLSBackend):
        ...     pass

        >>> # Explicit class registration
        >>> backend(MyBackendClass, name='MyBackend_hls')
    """

    def register_backend(cls: Type) -> Type:
        # Store metadata on class for potential later access
        # Parameters override class attributes
        cls.__brainsmith_backend__ = {
            'name': name or cls.__name__,
            'target_kernel': target_kernel or getattr(cls, 'target_kernel', None),
            'language': language or getattr(cls, 'language', None),
            'variant': variant or getattr(cls, 'variant', None),
            'metadata': metadata
        }

        # Register immediately
        registry.backend(
            cls,
            name=name or cls.__name__,
            target_kernel=target_kernel or getattr(cls, 'target_kernel', None),
            language=language or getattr(cls, 'language', None),
            variant=variant or getattr(cls, 'variant', None)
        )

        logger.debug(
            f"Registered backend: {cls.__module__}.{cls.__name__}"
        )

        return cls

    # Support decorator syntax: @backend, @backend(), @backend(name='...')
    if _cls is not None:
        # Called as: @backend or backend(MyClass)
        return register_backend(_cls)
    else:
        # Called as: @backend(...) with parameters only
        return register_backend


class Registry:
    """Central registry for Brainsmith components.

    Stores steps, kernels, and backends with source-prefixed names.

    User API (Decorators):
        Use decorators to register components:

        >>> from brainsmith.registry import step, kernel, backend
        >>>
        >>> @step(name='my_step')
        ... def my_step_function(model, cfg):
        ...     return model
        >>>
        >>> @kernel(name='LayerNorm', op_type='LayerNorm')
        ... class LayerNorm(HWCustomOp):
        ...     pass
        >>>
        >>> @backend(name='LayerNorm_hls', target_kernel='brainsmith:LayerNorm', language='hls')
        ... class LayerNorm_hls(LayerNorm, HLSBackend):
        ...     pass

    Internal Methods:
        The registry.step(), registry.kernel(), registry.backend() methods
        are called internally by decorators during import.
        Users should NOT call these directly - use decorators instead.
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
        """Register a step (internal method).

        INTERNAL: Called by @step decorator during import.
        Users should use @step decorator instead.

        Args:
            func_or_class: Step function or Step class
            source: Source name (auto-detected if None)
            name: Step name (uses func/class name if None)

        Returns:
            The original func_or_class
        """
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
        """Register a kernel (internal method).

        INTERNAL: Called by @kernel decorator during import.
        Users should use @kernel decorator instead.

        Metadata can come from class attributes or parameters.
        Parameters take precedence over class attributes.

        Args:
            cls: CustomOp subclass (or any class with kernel metadata)
            source: Source name (auto-detected if None)
            name: Kernel name (uses cls.__name__ if None)
            infer_transform: InferTransform class (overrides cls.infer_transform)
            domain: ONNX domain (overrides cls.domain, defaults to 'finn.custom')

        Returns:
            The original class
        """
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
        """Register a backend (internal method).

        INTERNAL: Called by @backend decorator during import.
        Users should use @backend decorator instead.

        Metadata can come from class attributes or parameters.
        Parameters take precedence over class attributes.

        Args:
            cls: Backend class with required metadata attributes
            source: Source name (auto-detected if None)
            name: Backend name (uses cls.__name__ if None)
            target_kernel: Full kernel name this backend targets (overrides cls.target_kernel)
            language: Backend language 'hls' or 'rtl' (overrides cls.language)
            variant: Optional variant name (overrides cls.variant)

        Returns:
            The original class
        """
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
        """Auto-detect source from context or module.

        Args:
            obj: Component to detect source for

        Returns:
            Source name

        Detection order:
        1. Current source context (_current_source) set during discovery
        2. Module path analysis (brainsmith.* → 'brainsmith')
        3. Default source from config
        """
        # Priority 1: Explicit context
        if _current_source:
            return _current_source

        # Priority 2: Module path analysis
        module = inspect.getmodule(obj)
        if module:
            module_name = module.__name__

            # Check for brainsmith core
            if module_name.startswith('brainsmith.'):
                return 'brainsmith'

            # Check for FINN
            if module_name.startswith('finn.'):
                return 'finn'

            # Check for QONNX
            if module_name.startswith('qonnx.'):
                return 'qonnx'

        # Priority 3: First source in priority list
        try:
            from brainsmith.settings import get_config
            return get_config().source_priority[0]
        except Exception:
            return 'project'  # Ultimate fallback (first in default priority)

    def _extract_step_name(self, func_or_class: Any) -> str:
        """Extract step name from function or class.

        Args:
            func_or_class: Step function or class

        Returns:
            Step name
        """
        # Check for explicit name attribute
        if hasattr(func_or_class, 'name') and func_or_class.name:
            return func_or_class.name

        # Use __name__
        return func_or_class.__name__

    def _extract_kernel_name(self, cls: Type) -> str:
        """Extract kernel name from class.

        Args:
            cls: Kernel class

        Returns:
            Kernel name (prefers class.op_type over __name__)
        """
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
