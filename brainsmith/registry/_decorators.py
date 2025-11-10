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

from ._state import _component_index, _discovered_sources
from ._metadata import ComponentMetadata, ComponentType, ImportSpec

logger = logging.getLogger(__name__)

# Current source context (set during plugin discovery)
# Using ContextVar for thread-safe source tracking
_current_source: ContextVar[Optional[str]] = ContextVar('current_source', default=None)


# ============================================================================
# Source Detection
# ============================================================================

def _detect_source(obj: Any) -> str:
    """Detect component source with priority: context > discovered sources > custom.

    Priority order:
    1. Active source_context (import location - highest priority)
       - Used during discovery to classify components by where they're imported
       - Example: importing third-party kernel into brainsmith/kernels/__init__.py

    2. Discovered sources matching (definition location - fallback)
       - Checks all sources registered during discovery (entrypoints + component_sources)
       - Supports both standard (source.category) and unique (source__category) patterns
       - Unified handling for all sources (brainsmith, finn, custom, etc.)

    3. 'custom' (unknown/unregistered - lowest priority)
       - Components not matching any registered source
       - Not cached to manifest (ephemeral, must be reimported each run)

    Args:
        obj: Component being registered (class or function)

    Returns:
        Source name ('brainsmith', 'finn', 'project', 'custom', or custom source name)

    Examples:
        >>> # During discovery with source_context
        >>> with source_context('brainsmith'):
        ...     _detect_source(SomeKernel)  # Returns 'brainsmith'

        >>> # Discovered sources (brainsmith, finn, custom)
        >>> from brainsmith.kernels.mvau import MVAU
        >>> _detect_source(MVAU)  # Returns 'brainsmith'

        >>> from finn.custom_op.fpgadataflow.mvau import MVAU
        >>> _detect_source(MVAU)  # Returns 'finn'

        >>> # Custom source (via component_sources or entrypoint)
        >>> from acme.kernels.custom_op import CustomOp
        >>> _detect_source(CustomOp)  # Returns 'acme' (if discovered)

        >>> # Unknown module
        >>> _detect_source(UnknownKernel)  # Returns 'custom'
    """
    # 1. Active source_context (import location)
    if ctx_source := _current_source.get():
        return ctx_source

    # 2. Module prefix matching against discovered sources
    module_name = obj.__module__

    # Check all discovered sources (from entrypoints and component_sources)
    for source in _discovered_sources:
        # Match both standard (source.category) and unique (source__category) patterns
        if module_name.startswith(f'{source}.') or module_name.startswith(f'{source}__'):
            logger.debug(f"Detected source '{source}' for {obj.__name__} via discovered sources")
            return source

    # 3. Unknown/custom (ephemeral)
    logger.debug(f"No source match for {obj.__name__} (module={module_name}), defaulting to 'custom'")
    return 'custom'


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
    name: Optional[str] = None,
    **kwargs  # Accept and ignore extra parameters (category, description, etc.) for backwards compat
) -> Union[Callable, Type]:
    """Register step in global component index."""
    if 'source' in kwargs:
        raise ValueError(
            "'source' parameter not supported. Use source_context() or configure component_sources in brainsmith.yaml"
        )

    source = _detect_source(func_or_class)
    name = name or getattr(func_or_class, 'name', None) or func_or_class.__name__

    full_name = f"{source}:{name}"

    # Skip if already registered with loaded object (avoid redundant work)
    if full_name in _component_index:
        existing = _component_index[full_name]
        if existing.loaded_obj is not None:
            # Already registered and loaded - skip re-registration
            logger.debug(f"Step {full_name} already registered, skipping")
            return func_or_class

        # Entry exists from manifest - update it instead of replacing
        # This preserves manifest metadata
        logger.debug(f"Updating existing step registration: {full_name}")
        existing.loaded_obj = func_or_class

        # Attach registry name to function/class for O(1) reverse lookup
        func_or_class.__registry_name__ = full_name
        return func_or_class

    # New registration (no manifest entry exists)
    # Create index entry for standalone decorator usage
    logger.debug(f"Registering step: {full_name}")
    import_spec = ImportSpec(
        module=func_or_class.__module__,
        attr=func_or_class.__name__
    )
    _component_index[full_name] = ComponentMetadata(
        name=name,
        source=source,
        component_type=ComponentType.STEP,
        import_spec=import_spec,
        loaded_obj=func_or_class  # Already loaded
    )

    # Attach registry name to function/class for O(1) reverse lookup
    func_or_class.__registry_name__ = full_name

    return func_or_class


def _register_kernel(
    cls: Type,
    *,
    name: Optional[str] = None,
    infer_transform: Optional[Type] = None,
    is_infrastructure: bool = False,  # True for topology-based kernels (DuplicateStreams, FIFO)
    **kwargs  # Accept and ignore extra parameters (e.g., op_type, domain) for backwards compat
) -> Type:
    """Register kernel in global component index."""
    if 'source' in kwargs:
        raise ValueError(
            "'source' parameter not supported. Use source_context() or configure component_sources in brainsmith.yaml"
        )

    source = _detect_source(cls)
    name = name or getattr(cls, 'op_type', None) or cls.__name__

    full_name = f"{source}:{name}"

    # Skip if already registered with loaded object (avoid redundant work)
    if full_name in _component_index:
        existing = _component_index[full_name]
        if existing.loaded_obj is not None:
            # Already registered and loaded - skip re-registration
            logger.debug(f"Kernel {full_name} already registered, skipping")
            return cls

        # Entry exists from manifest - update it instead of replacing
        # This preserves manifest metadata (is_infrastructure, backends, etc.)
        logger.debug(f"Updating existing kernel registration: {full_name}")

        # Extract infer_transform from decorator parameters or class attribute
        if infer_transform is None:
            infer_transform = getattr(cls, 'infer_transform', None)
        infer_transform = _convert_lazy_import_spec(infer_transform)

        # Update the existing metadata
        existing.loaded_obj = cls
        if infer_transform is not None:
            existing.kernel_infer = infer_transform

        # Attach registry name to class for O(1) reverse lookup
        cls.__registry_name__ = full_name
        return cls

    # New registration (no manifest entry exists)
    # Extract infer_transform (class attribute or parameter)
    # Supports both class references and string-based lazy import specs
    # Note: infer_transform is optional - not all kernels need it (e.g., test kernels)
    if infer_transform is None:
        infer_transform = getattr(cls, 'infer_transform', None)

    # Handle string-based lazy import specs (format: 'module.path:ClassName')
    infer_transform = _convert_lazy_import_spec(infer_transform)

    # Create index entry for standalone decorator usage
    logger.debug(f"Registering kernel: {full_name}")
    import_spec = ImportSpec(
        module=cls.__module__,
        attr=cls.__name__
    )
    _component_index[full_name] = ComponentMetadata(
        name=name,
        source=source,
        component_type=ComponentType.KERNEL,
        import_spec=import_spec,
        loaded_obj=cls,  # Already loaded
        is_infrastructure=is_infrastructure,
    )

    # Populate type-specific metadata (inline)
    meta = _component_index[full_name]
    # Kernel-specific metadata (infer_transform already converted above)
    meta.kernel_infer = infer_transform

    # Attach registry name to class for O(1) reverse lookup
    cls.__registry_name__ = full_name

    return cls


def _register_backend(
    cls: Type,
    *,
    name: Optional[str] = None,
    target_kernel: Optional[str] = None,
    language: Optional[str] = None,
    variant: Optional[str] = None,
    **kwargs  # Accept and ignore extra parameters (e.g., author, description) for backwards compat
) -> Type:
    """Register backend in global component index."""
    if 'source' in kwargs:
        raise ValueError(
            "'source' parameter not supported. Use source_context() or configure component_sources in brainsmith.yaml"
        )

    source = _detect_source(cls)
    name = name or cls.__name__

    full_name = f"{source}:{name}"

    # Skip if already registered with loaded object (avoid redundant work)
    if full_name in _component_index:
        existing = _component_index[full_name]
        if existing.loaded_obj is not None:
            # Already registered and loaded - skip re-registration
            logger.debug(f"Backend {full_name} already registered, skipping")
            return cls

        # Entry exists from manifest - update it instead of replacing
        # This preserves manifest metadata
        logger.debug(f"Updating existing backend registration: {full_name}")

        # Extract metadata - parameters override class attributes
        target = target_kernel or getattr(cls, 'target_kernel', None)
        lang = language or getattr(cls, 'language', None)

        # Validate required fields
        if not target:
            raise ValueError(f"Backend {full_name} missing 'target_kernel' attribute")
        if not lang:
            raise ValueError(f"Backend {full_name} missing 'language' attribute")

        # Normalize unqualified target to backend's source
        if ':' not in target:
            target = f"{source}:{target}"

        # Update the existing metadata
        existing.loaded_obj = cls
        existing.backend_target = target
        existing.backend_language = lang

        # Link backend to its target kernel (if kernel exists)
        kernel_meta = _component_index.get(target)
        if kernel_meta:
            if kernel_meta.kernel_backends is None:
                kernel_meta.kernel_backends = []
            if full_name not in kernel_meta.kernel_backends:
                kernel_meta.kernel_backends.append(full_name)

        # Attach registry name to class for O(1) reverse lookup
        cls.__registry_name__ = full_name
        return cls

    # New registration (no manifest entry exists)
    # Extract metadata - parameters override class attributes
    target = target_kernel or getattr(cls, 'target_kernel', None)
    lang = language or getattr(cls, 'language', None)

    # Validate required fields
    if not target:
        raise ValueError(f"Backend {full_name} missing 'target_kernel' attribute")
    if not lang:
        raise ValueError(f"Backend {full_name} missing 'language' attribute")

    # Normalize unqualified target to backend's source
    if ':' not in target:
        target = f"{source}:{target}"

    # Create index entry for standalone decorator usage
    logger.debug(f"Registering backend: {full_name} (target={target}, lang={lang})")
    import_spec = ImportSpec(
        module=cls.__module__,
        attr=cls.__name__
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

    # Attach registry name to class for O(1) reverse lookup
    cls.__registry_name__ = full_name

    return cls


# ============================================================================
# Public Decorator API ===
# Direct decorator implementations (no factory needed)

def step(obj=None, **kwargs):
    """Register a pipeline transformation step.

    Decorator for model transformation functions. Steps are discovered automatically
    and can be used in pipeline workflows.

    Args:
        obj: Function or class being decorated (automatic when using @step without parens)
        **kwargs: Optional configuration:
            name: Step name (default: function/class __name__)

    Returns:
        The decorated callable, unchanged but registered in component index

    Examples:
        >>> from brainsmith.registry import step
        >>>
        >>> @step
        ... def my_optimization(model, config):
        ...     # Transform model
        ...     return model
        ...
        >>> @step(name="CustomStepName")
        ... def complex_transform(model, config):
        ...     return model
    """
    def register(func_or_class):
        kwargs.setdefault('name', func_or_class.__name__)
        _register_step(func_or_class, **kwargs)
        return func_or_class
    return register(obj) if obj is not None else register


def kernel(obj=None, **kwargs):
    """Register a kernel class.

    Decorator for hardware kernel implementations. Supports both `@kernel` and
    `@kernel(name='CustomName')` syntax.

    Args:
        obj: Class being decorated (automatic when using @kernel without parens)
        **kwargs: Optional configuration:
            name: Kernel name (default: cls.op_type or cls.__name__)
            infer_transform: Transform class for ONNX → kernel inference
            is_infrastructure: Mark as topology kernel like FIFO (default: False)

    Returns:
        The decorated class, unchanged but registered in component index

    Examples:
        >>> from brainsmith.registry import kernel
        >>> from brainsmith.dataflow import KernelOp
        >>>
        >>> @kernel
        ... class MyKernel(KernelOp):
        ...     op_type = "MyKernel"
        ...
        >>> @kernel(name="CustomName", is_infrastructure=True)
        ... class FIFO(HWCustomOp):
        ...     pass
    """
    def register(cls):
        kwargs.setdefault('name', getattr(cls, 'op_type', cls.__name__))
        _register_kernel(cls, **kwargs)
        return cls
    return register(obj) if obj is not None else register


def backend(obj=None, **kwargs):
    """Register a backend implementation.

    Decorator for HLS or RTL backend implementations. Backends must specify their
    target kernel and implementation language.

    Args:
        obj: Class being decorated (automatic when using @backend without parens)
        **kwargs: Backend configuration:
            target_kernel (required): Kernel name this backend implements
            language (required): Implementation language ('hls' or 'rtl')
            name: Backend class name (default: cls.__name__)
            variant: Optional variant identifier

    Returns:
        The decorated class, unchanged but registered in component index

    Raises:
        ValueError: If target_kernel or language missing

    Examples:
        >>> from brainsmith.registry import backend
        >>>
        >>> @backend(target_kernel='MVAU', language='hls')
        ... class MVAU_hls:
        ...     @staticmethod
        ...     def get_nodeattr_types():
        ...         return {...}
        ...
        >>> @backend(target_kernel='MVAU', language='rtl', variant='fast')
        ... class MVAU_rtl_fast:
        ...     pass
    """
    def register(cls):
        kwargs.setdefault('name', cls.__name__)
        _register_backend(cls, **kwargs)
        return cls
    return register(obj) if obj is not None else register


# Context manager for source detection
class source_context:
    """Context manager for setting component source during discovery.

    Args:
        source_name: Source identifier for components loaded in this context

    Examples:
        >>> with source_context('project'):
        ...     import project_kernels  # Registered with source='project'
    """

    def __init__(self, source_name: str):
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
