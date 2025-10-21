# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Component registration system.

This module provides the central registry for all Brainsmith components:
steps, kernels, and backends.

Registration Patterns:
1. **Decorators** (recommended): @step, @kernel, @backend
   - User-facing, ergonomic API
   - Deferred registration (lazy processing on first lookup)
   - Example: @kernel(name='LayerNorm', op_type='LayerNorm')

2. **Manifests** (for FINN integration):
   - String-based lazy loading for performance
   - Methods: kernel_lazy(), backend_lazy(), step_lazy()
   - Imports happen on first access

Internal Implementation:
- Registry.step(), Registry.kernel(), Registry.backend() methods are internal
- Called by deferred registration processor
- Users should NOT call these methods directly
"""

import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Type, Union
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

# Current source context (set during plugin discovery)
_current_source: Optional[str] = None

# === Deferred Registration Infrastructure ===
# Components marked with decorators but not yet registered

# Deferred storage - components waiting for registration
_deferred_steps: List[Callable] = []
_deferred_kernels: List[Type] = []
_deferred_backends: List[Type] = []
_registration_processed = False
_registration_lock = Lock()  # Thread safety for deferred processing


# === Helper Functions ===

def _get_class_attribute(cls: Type, attr_name: str, default: Any = None) -> Any:
    """Safely get class attribute, handling properties correctly.

    When a class has a property, getattr() returns the property descriptor,
    not the actual value. This function detects properties and evaluates them.

    Args:
        cls: Class to get attribute from
        attr_name: Attribute name
        default: Default value if attribute doesn't exist

    Returns:
        Attribute value (property values are evaluated)
    """
    try:
        # Use getattr_static to check if it's a property
        attr = inspect.getattr_static(cls, attr_name)

        if isinstance(attr, property):
            # It's a property - call the getter with the class
            return attr.fget(cls)
        else:
            # Regular attribute
            return attr
    except AttributeError:
        return default


# === Decorator Functions (User API) ===

def step(
    _func_or_path: Optional[Union[Callable, str]] = None,
    *,
    name: Optional[str] = None,
    callable_path: Optional[str] = None,
    **metadata
) -> Optional[Callable]:
    """Universal step registration: decorator, explicit, or lazy (manifest).

    Detects argument type and dispatches to appropriate registration method:
    - String path → lazy manifest registration (stores path, imports on first use)
    - Callable → deferred registration (marks for processing on first lookup)

    Supports all syntax patterns:
    - @step (decorator without params)
    - @step(...) (decorator with params)
    - step(my_function) (explicit function registration)
    - step('path.to.my_function', ...) (manifest lazy loading)

    Args:
        _func_or_path: Function to register OR string import path
        name: Step name (defaults to function name)
        callable_path: Deprecated alias for _func_or_path (string path)
        **metadata: Additional metadata (reserved for future use)

    Returns:
        Original function (for decorator/explicit), None (for lazy string)

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

        >>> # Manifest lazy loading (string path)
        >>> step('finn.transforms.streamline.streamline', name='streamline')
    """
    # Handle deprecated parameter names
    if callable_path is not None and _func_or_path is None:
        _func_or_path = callable_path

    def register_step(func_or_path: Union[Callable, str]) -> Optional[Callable]:
        # Case 1: String path → lazy manifest registration
        if isinstance(func_or_path, str):
            if not name:
                raise ValueError("name parameter required for string path registration")

            # Call Registry.step_lazy() to register string path
            registry.step_lazy(
                name=name,
                callable_path=func_or_path,
                **metadata
            )
            logger.debug(f"Registered step (lazy): {_current_source or 'brainsmith'}:{name}")
            return None  # Nothing to return for string registration

        # Case 2: Callable → deferred registration
        elif callable(func_or_path):
            # Store metadata on function
            func_or_path.__brainsmith_step__ = {
                'name': name or func_or_path.__name__,
                'metadata': metadata
            }

            # Add to deferred queue
            _deferred_steps.append(func_or_path)

            logger.debug(
                f"Marked step for deferred registration: {func_or_path.__module__}.{func_or_path.__name__}"
            )

            return func_or_path

        else:
            raise TypeError(
                f"step() expects a callable or string path, got {type(func_or_path)}"
            )

    # Support decorator syntax: @step, @step(), @step(name='...')
    if _func_or_path is not None:
        # Called as: @step, step(my_func), or step('path.to.func', ...)
        return register_step(_func_or_path)
    else:
        # Called as: @step(...) with parameters only
        return register_step


def kernel(
    _cls_or_path: Optional[Union[Type, str]] = None,
    *,
    name: Optional[str] = None,
    op_type: Optional[str] = None,
    infer_transform: Optional[Union[Type, str]] = None,
    domain: Optional[str] = None,
    infer_path: Optional[str] = None,
    class_path: Optional[str] = None,
    **metadata
) -> Optional[Type]:
    """Universal kernel registration: decorator, explicit, or lazy (manifest).

    Detects argument type and dispatches to appropriate registration method:
    - String path → lazy manifest registration (stores path, imports on first use)
    - Class → deferred registration (marks for processing on first lookup)

    Supports all syntax patterns:
    - @kernel (decorator without params)
    - @kernel(...) (decorator with params)
    - kernel(MyClass) (explicit class registration)
    - kernel('path.to.MyClass', ...) (manifest lazy loading)

    Args:
        _cls_or_path: Class to register OR string import path
        name: Kernel name (defaults to op_type or class name)
        op_type: ONNX op_type (defaults to class.op_type)
        infer_transform: InferTransform class (for class) or path (for string)
        domain: ONNX domain (defaults to class.domain or 'finn.custom')
        infer_path: Deprecated alias for infer_transform (string path)
        class_path: Deprecated alias for _cls_or_path (string path)
        **metadata: Additional metadata

    Returns:
        Original class (for decorator/explicit), None (for lazy string)

    Examples:
        >>> # Decorator without params
        >>> @kernel
        ... class LayerNorm(CustomOp):
        ...     op_type = "LayerNorm"

        >>> # Decorator with params
        >>> @kernel(name='CustomName', op_type='Softmax')
        ... class MyKernel(CustomOp):
        ...     pass

        >>> # Explicit class registration
        >>> kernel(MyKernelClass, name='MyKernel')

        >>> # Manifest lazy loading (string path)
        >>> kernel('finn.kernels.mvau.MVAU', name='MVAU', op_type='MVAU')
    """
    # Handle deprecated parameter names
    if class_path is not None and _cls_or_path is None:
        _cls_or_path = class_path
    if infer_path is not None and infer_transform is None:
        infer_transform = infer_path

    def register_kernel(cls_or_path: Union[Type, str]) -> Optional[Type]:
        # Case 1: String path → lazy manifest registration
        if isinstance(cls_or_path, str):
            if not name:
                raise ValueError("name parameter required for string path registration")

            # Call Registry.kernel_lazy() to register string path
            registry.kernel_lazy(
                name=name,
                class_path=cls_or_path,
                infer_path=infer_transform if isinstance(infer_transform, str) else None,
                op_type=op_type,
                domain=domain
            )
            logger.debug(f"Registered kernel (lazy): {_current_source or 'brainsmith'}:{name}")
            return None  # Nothing to return for string registration

        # Case 2: Class → deferred registration
        elif isinstance(cls_or_path, type):
            # Store metadata on class
            # Parameters override class attributes
            # Use _get_class_attribute to properly handle properties
            cls_or_path.__brainsmith_kernel__ = {
                'name': name or _get_class_attribute(cls_or_path, 'op_type', cls_or_path.__name__),
                'op_type': op_type or _get_class_attribute(cls_or_path, 'op_type', None),
                'infer_transform': infer_transform if not isinstance(infer_transform, str) else None or _get_class_attribute(cls_or_path, 'infer_transform', None),
                'domain': domain or _get_class_attribute(cls_or_path, 'domain', None),
                'metadata': metadata
            }

            # Add to deferred queue
            _deferred_kernels.append(cls_or_path)

            logger.debug(
                f"Marked kernel for deferred registration: {cls_or_path.__module__}.{cls_or_path.__name__}"
            )

            return cls_or_path

        else:
            raise TypeError(
                f"kernel() expects a class or string path, got {type(cls_or_path)}"
            )

    # Support decorator syntax: @kernel, @kernel(), @kernel(name='...')
    if _cls_or_path is not None:
        # Called as: @kernel, kernel(MyClass), or kernel('path.to.Class', ...)
        return register_kernel(_cls_or_path)
    else:
        # Called as: @kernel(...) with parameters only
        return register_kernel


def backend(
    _cls_or_path: Optional[Union[Type, str]] = None,
    *,
    name: Optional[str] = None,
    target_kernel: Optional[str] = None,
    language: Optional[str] = None,
    variant: Optional[str] = None,
    infer_path: Optional[str] = None,
    class_path: Optional[str] = None,
    **metadata
) -> Optional[Type]:
    """Universal backend registration: decorator, explicit, or lazy (manifest).

    Detects argument type and dispatches to appropriate registration method:
    - String path → lazy manifest registration (stores path, imports on first use)
    - Class → deferred registration (marks for processing on first lookup)

    Supports all syntax patterns:
    - @backend (decorator without params)
    - @backend(...) (decorator with params)
    - backend(MyClass) (explicit class registration)
    - backend('path.to.MyClass', ...) (manifest lazy loading)

    Args:
        _cls_or_path: Class to register OR string import path
        name: Backend name (defaults to class name)
        target_kernel: Target kernel full name (e.g., 'brainsmith:LayerNorm')
        language: Backend language ('hls' or 'rtl')
        variant: Optional variant name
        infer_path: Optional infer transform path (for lazy loading)
        class_path: Deprecated alias for _cls_or_path (string path)
        **metadata: Additional metadata

    Returns:
        Original class (for decorator/explicit), None (for lazy string)

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

        >>> # Manifest lazy loading (string path)
        >>> backend('finn.backends.mvau.MVAU_HLS',
        ...         name='MVAU_HLS', target_kernel='finn:MVAU', language='hls')
    """
    # Handle deprecated parameter names
    if class_path is not None and _cls_or_path is None:
        _cls_or_path = class_path

    def register_backend(cls_or_path: Union[Type, str]) -> Optional[Type]:
        # Case 1: String path → lazy manifest registration
        if isinstance(cls_or_path, str):
            if not name:
                raise ValueError("name parameter required for string path registration")

            # Call Registry.backend_lazy() to register string path
            registry.backend_lazy(
                name=name,
                class_path=cls_or_path,
                infer_path=infer_path,
                target_kernel=target_kernel,
                language=language,
                variant=variant
            )
            logger.debug(f"Registered backend (lazy): {_current_source or 'brainsmith'}:{name}")
            return None  # Nothing to return for string registration

        # Case 2: Class → deferred registration
        elif isinstance(cls_or_path, type):
            # Store metadata on class
            # Parameters override class attributes
            cls_or_path.__brainsmith_backend__ = {
                'name': name or cls_or_path.__name__,
                'target_kernel': target_kernel or getattr(cls_or_path, 'target_kernel', None),
                'language': language or getattr(cls_or_path, 'language', None),
                'variant': variant or getattr(cls_or_path, 'variant', None),
                'metadata': metadata
            }

            # Add to deferred queue
            _deferred_backends.append(cls_or_path)

            logger.debug(
                f"Marked backend for deferred registration: {cls_or_path.__module__}.{cls_or_path.__name__}"
            )

            return cls_or_path

        else:
            raise TypeError(
                f"backend() expects a class or string path, got {type(cls_or_path)}"
            )

    # Support decorator syntax: @backend, @backend(), @backend(name='...')
    if _cls_or_path is not None:
        # Called as: @backend, backend(MyClass), or backend('path.to.Class', ...)
        return register_backend(_cls_or_path)
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
        are called internally by the deferred registration processor.
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

        INTERNAL: Called by deferred registration processor.
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
        op_type: Optional[str] = None,
        infer_transform: Optional[Type] = None,
        domain: Optional[str] = None
    ) -> Type:
        """Register a kernel (internal method).

        INTERNAL: Called by deferred registration processor.
        Users should use @kernel decorator instead.

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
            The original class
        """
        source = source or self._detect_source(cls)
        name = name or self._extract_kernel_name(cls)

        full_name = f"{source}:{name}"

        if full_name in self._kernels:
            logger.warning(f"Overriding existing kernel: {full_name}")

        # Extract metadata - parameters override class attributes
        # Use _get_class_attribute to properly handle properties
        metadata = {
            'class': cls,
            'infer': infer_transform or _get_class_attribute(cls, 'infer_transform', None),
            'op_type': op_type or _get_class_attribute(cls, 'op_type', None),
            'domain': domain or _get_class_attribute(cls, 'domain', 'finn.custom')
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
        """Register a backend (internal method).

        INTERNAL: Called by deferred registration processor.
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

    def kernel_lazy(
        self,
        *,
        name: str,
        class_path: str,
        infer_path: Optional[str] = None,
        op_type: Optional[str] = None,
        domain: Optional[str] = None
    ):
        """Register a kernel with lazy loading (from manifest).

        Instead of importing the class, stores import path as string.
        The class is imported on first access.

        Args:
            name: Kernel name (without source prefix, uses current context)
            class_path: Full import path to kernel class
            infer_path: Optional full import path to infer transform
            op_type: Optional ONNX op type
            domain: Optional ONNX domain

        Examples:
            >>> with source_context('finn'):
            ...     registry.kernel_lazy(
            ...         name='MVAU',
            ...         class_path='finn_xsi.kernels.mvau.MVAU',
            ...         infer_path='finn_xsi.infer.mvau.InferMVAU'
            ...     )
        """
        source = _current_source or 'brainsmith'
        full_name = f"{source}:{name}"

        if full_name in self._kernels:
            logger.warning(f"Overriding existing kernel: {full_name}")

        # Store import paths as strings (lazy)
        metadata = {
            'class': class_path,  # String, not imported yet
            'infer': infer_path,  # String or None
            'op_type': op_type or name,
            'domain': domain or 'finn.custom'
        }

        self._kernels[full_name] = metadata
        logger.debug(f"Registered kernel (lazy): {full_name}")

    def backend_lazy(
        self,
        *,
        name: str,
        class_path: str,
        infer_path: Optional[str] = None,
        target_kernel: Optional[str] = None,
        language: Optional[str] = None,
        variant: Optional[str] = None
    ):
        """Register a backend with lazy loading (from manifest).

        Instead of importing the class, stores import path as string.
        The class is imported on first access.

        Args:
            name: Backend name (without source prefix, uses current context)
            class_path: Full import path to backend class
            infer_path: Optional full import path to infer transform
            target_kernel: Kernel this backend targets
            language: Backend language ('hls' or 'rtl')
            variant: Optional variant name

        Examples:
            >>> with source_context('finn'):
            ...     registry.backend_lazy(
            ...         name='MVAU_HLS',
            ...         class_path='finn_xsi.backends.mvau.MVAU_HLS',
            ...         target_kernel='finn:MVAU',
            ...         language='hls'
            ...     )
        """
        source = _current_source or 'brainsmith'
        full_name = f"{source}:{name}"

        if full_name in self._backends:
            logger.warning(f"Overriding existing backend: {full_name}")

        # Store import paths as strings (lazy)
        metadata = {
            'class': class_path,  # String, not imported yet
            'target_kernel': target_kernel or name,
            'language': language,
            'variant': variant
        }

        self._backends[full_name] = metadata
        logger.debug(f"Registered backend (lazy): {full_name}")

    def step_lazy(
        self,
        *,
        name: str,
        callable_path: str,
        **params
    ):
        """Register a step with lazy loading (from manifest).

        Instead of importing the callable, stores import path as string.
        The callable is imported on first access.

        Args:
            name: Step name (without source prefix, uses current context)
            callable_path: Full import path to step callable
            **params: Additional step parameters

        Examples:
            >>> with source_context('finn'):
            ...     registry.step_lazy(
            ...         name='streamline',
            ...         callable_path='finn.transforms.streamline.streamline'
            ...     )
        """
        source = _current_source or 'brainsmith'
        full_name = f"{source}:{name}"

        if full_name in self._steps:
            logger.warning(f"Overriding existing step: {full_name}")

        # For steps, we store the path directly as the value
        # The loader will handle lazy import
        self._steps[full_name] = callable_path  # String, not imported yet
        logger.debug(f"Registered step (lazy): {full_name}")

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


# Export decorators for public API
__all__ = ['Registry', 'registry', 'source_context', 'step', 'kernel', 'backend']
