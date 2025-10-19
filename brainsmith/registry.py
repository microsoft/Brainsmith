# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Component registration system.

This module provides the central registry for all Brainsmith components:
steps, kernels, and backends. Components self-register using the global
registry instance.

The registry supports:
- Explicit registration with source/name
- Auto-detection of source from context
- Type-safe metadata extraction from base classes
- Decorator-based registration
"""

import inspect
import logging
from typing import Any, Callable, Dict, Optional, Type, Union
from pathlib import Path

logger = logging.getLogger(__name__)

# Current source context (set during plugin discovery)
_current_source: Optional[str] = None


class Registry:
    """Central registry for Brainsmith components.

    Stores steps, kernels, and backends with source-prefixed names.
    Provides registration methods that auto-detect source and extract
    metadata from component classes.

    Examples:
        >>> from brainsmith import registry
        >>> registry.step(my_step_function)
        >>> registry.kernel(MyKernelClass)
        >>> registry.backend(MyBackendClass)
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
        """Register a step.

        Args:
            func_or_class: Step function or Step class
            source: Source name (auto-detected if None)
            name: Step name (uses func/class name if None)

        Returns:
            The original func_or_class (for use as decorator)

        Examples:
            >>> @registry.step
            ... def my_step(model, **kwargs):
            ...     return model

            >>> registry.step(my_step, source='user', name='custom_name')
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
        op_type: Optional[str] = None,
        infer_transform: Optional[Type] = None,
        domain: Optional[str] = None
    ) -> Type:
        """Register a kernel (custom operation).

        Metadata can come from class attributes or parameters.
        Parameters take precedence over class attributes.

        Args:
            cls: CustomOp subclass (or any class with kernel metadata)
            source: Source name (auto-detected if None)
            name: Kernel name (uses cls.op_type or cls.__name__ if None)
            op_type: ONNX op type (overrides cls.op_type)
            infer_transform: InferTransform class (overrides cls.infer_transform)
            domain: ONNX domain (overrides cls.domain, defaults to 'finn.custom')

        Returns:
            The original class (for use as decorator)

        Examples:
            >>> # Approach 1: Class has metadata attributes
            >>> @registry.kernel
            ... class LayerNorm(CustomOp):
            ...     op_type = "LayerNorm"
            ...     infer_transform = InferLayerNorm

            >>> # Approach 2: Pass metadata as parameters
            >>> registry.kernel(
            ...     OldKernel,
            ...     source='brainsmith',
            ...     op_type='Softmax',
            ...     infer_transform=InferSoftmax
            ... )
        """
        source = source or self._detect_source(cls)
        name = name or self._extract_kernel_name(cls)

        full_name = f"{source}:{name}"

        if full_name in self._kernels:
            logger.warning(f"Overriding existing kernel: {full_name}")

        # Extract metadata - parameters override class attributes
        metadata = {
            'class': cls,
            'infer': infer_transform or getattr(cls, 'infer_transform', None),
            'op_type': op_type or getattr(cls, 'op_type', None),
            'domain': domain or getattr(cls, 'domain', 'finn.custom')
        }

        self._kernels[full_name] = metadata
        logger.debug(f"Registered kernel: {full_name} (op_type={metadata['op_type']})")

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
        """Register a backend.

        Metadata can come from class attributes or parameters.
        Parameters take precedence over class attributes.

        Args:
            cls: Backend subclass (or any class with backend metadata)
            source: Source name (auto-detected if None)
            name: Backend name (uses cls.__name__ if None)
            target_kernel: Full kernel name this backend targets (overrides cls.target_kernel)
            language: Backend language 'hls' or 'rtl' (overrides cls.language)
            variant: Optional variant name (overrides cls.variant)

        Returns:
            The original class (for use as decorator)

        Examples:
            >>> # Approach 1: Class has metadata attributes
            >>> @registry.backend
            ... class LayerNormHLS(Backend):
            ...     target_kernel = 'brainsmith:LayerNorm'
            ...     language = 'hls'

            >>> # Approach 2: Pass metadata as parameters
            >>> registry.backend(
            ...     OldBackend,
            ...     source='brainsmith',
            ...     target_kernel='brainsmith:Softmax',
            ...     language='hls'
            ... )
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

        # Priority 3: Config default
        try:
            from brainsmith.settings import get_config
            return get_config().default_source
        except Exception:
            return 'brainsmith'  # Ultimate fallback

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
            Kernel name (prefers op_type over __name__)
        """
        # Prefer op_type (e.g., "LayerNorm" for ONNX compatibility)
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


# === Legacy Decorator Stubs ===
# These are no-op decorators that allow old kernel/backend files to import
# without error. Registration for legacy components happens via the legacy
# bridge in plugins.py instead.

def kernel(*args, **kwargs):
    """Legacy decorator stub - does nothing.

    Old kernel files still import this, but registration happens via
    the legacy bridge in plugins.py. This stub prevents ImportError.
    """
    def decorator(cls):
        return cls  # Just return the class unchanged

    # Support both @kernel and @kernel(...) syntax
    if args and callable(args[0]):
        return args[0]
    return decorator


def backend(*args, **kwargs):
    """Legacy decorator stub - does nothing.

    Old backend files still import this, but registration happens via
    the legacy bridge in plugins.py. This stub prevents ImportError.
    """
    def decorator(cls):
        return cls  # Just return the class unchanged

    # Support both @backend and @backend(...) syntax
    if args and callable(args[0]):
        return args[0]
    return decorator


def step(*args, **kwargs):
    """Legacy decorator stub - does nothing.

    Old step files still import this, but registration happens via
    the legacy bridge in plugins.py. This stub prevents ImportError.
    """
    def decorator(func):
        return func  # Just return the function unchanged

    # Support both @step and @step(...) syntax
    if args and callable(args[0]):
        return args[0]
    return decorator


def transform(*args, **kwargs):
    """Legacy decorator stub - does nothing.

    Old transform files still import this, but registration happens via
    the legacy bridge in plugins.py. This stub prevents ImportError.
    """
    def decorator(cls):
        return cls  # Just return the class unchanged

    # Support both @transform and @transform(...) syntax
    if args and callable(args[0]):
        return args[0]
    return decorator
