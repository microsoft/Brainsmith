# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Component metadata structures for the registry system.

Defines data structures used throughout the component registry:
- ComponentType: Enum for component types (kernel, backend, step)
- ImportSpec: Lazy import specification
- ComponentMetadata: Metadata for registered components
- Helper functions for metadata manipulation
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional, Type, Union


class ComponentType(Enum):
    """Component type enumeration.

    Provides type-safe component type identifiers with string conversion
    support for JSON serialization.
    """
    KERNEL = auto()
    BACKEND = auto()
    STEP = auto()

    def __str__(self) -> str:
        """String representation for display and serialization."""
        return self.name.lower()

    @classmethod
    def from_string(cls, s: str) -> 'ComponentType':
        """Parse component type from string.

        Args:
            s: Component type string ('kernel', 'backend', or 'step')

        Returns:
            Corresponding ComponentType enum value

        Raises:
            ValueError: If string doesn't match any component type

        Example:
            >>> ComponentType.from_string('kernel')
            <ComponentType.KERNEL: 1>
            >>> ComponentType.from_string('backend')
            <ComponentType.BACKEND: 2>
        """
        try:
            return cls[s.upper()]
        except KeyError:
            valid = ', '.join([t.name.lower() for t in cls])
            raise ValueError(
                f"Invalid component type: '{s}'. "
                f"Must be one of: {valid}"
            )


@dataclass
class ImportSpec:
    """Lazy import specification: module + attr.

    Stores the information needed to lazy-load a component:
    - module: Python module path (e.g., 'brainsmith.kernels.layernorm')
    - attr: Attribute name in module (e.g., 'LayerNorm')
    """
    module: str
    attr: str


@dataclass
class ComponentMetadata:
    """Component metadata for unified lazy loading.

    Represents a discovered component with all information needed to:
    - Identify it (name, source, type)
    - Load it on demand (import_spec)
    - Track its state (loaded_obj)
    - Store type-specific metadata (kernel/backend fields)
    """
    name: str
    source: str
    component_type: ComponentType
    import_spec: ImportSpec
    loaded_obj: Optional[Any] = None

    # Type-specific metadata fields (only populated for relevant types)
    # Kernel metadata
    kernel_infer: Optional[Any] = None  # InferTransform class or lazy import spec
    kernel_backends: Optional[list[str]] = None  # List of backend names targeting this kernel

    # Backend metadata
    backend_target: Optional[str] = None  # Target kernel name (source:name format)
    backend_language: Optional[str] = None  # 'hls' or 'rtl'

    @property
    def full_name(self) -> str:
        """Get source-prefixed name (e.g., 'brainsmith:LayerNorm')."""
        return f"{self.source}:{self.name}"


# ============================================================================
# Metadata Helper Functions
# ============================================================================

def resolve_lazy_class(spec: Union[Type, Dict[str, str], None]) -> Optional[Type]:
    """Resolve lazy class import spec to actual class.

    Supports two formats:
    - Direct class reference (already loaded)
    - Lazy spec: {'module': 'path.to.module', 'class_name': 'ClassName'}

    Args:
        spec: Class reference, lazy import dict, or None

    Returns:
        Resolved class object, or None if spec was None

    Examples:
        >>> resolve_lazy_class(MyClass)  # Direct reference
        <class 'MyClass'>
        >>> resolve_lazy_class({'module': 'foo.bar', 'class_name': 'Baz'})
        <class 'foo.bar.Baz'>
        >>> resolve_lazy_class(None)
        None
    """
    if spec is None:
        return None

    if isinstance(spec, dict) and 'module' in spec:
        import importlib
        module = importlib.import_module(spec['module'])
        return getattr(module, spec['class_name'])

    return spec
