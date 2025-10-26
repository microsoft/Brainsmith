# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Component metadata structures for the registry system.

Defines data structures used throughout the component registry:
- ImportSpec: Lazy import specification
- ComponentMetadata: Metadata for registered components
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional


@dataclass
class ImportSpec:
    """Lazy import specification: module + attr + metadata.

    Stores the information needed to lazy-load a component:
    - module: Python module path (e.g., 'brainsmith.kernels.layernorm')
    - attr: Attribute name in module (e.g., 'LayerNorm')
    - extra: Type-specific metadata (e.g., infer_transform, language)
    """
    module: str
    attr: str
    extra: Dict[str, Any] = field(default_factory=dict)


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
    component_type: Literal['kernel', 'backend', 'step']
    import_spec: ImportSpec
    loaded_obj: Optional[Any] = None

    # Type-specific metadata fields (only populated for relevant types)
    # Kernel metadata
    kernel_infer: Optional[Any] = None  # InferTransform class or lazy import spec
    kernel_domain: Optional[str] = None

    # Backend metadata
    backend_target: Optional[str] = None  # Target kernel name (source:name format)
    backend_language: Optional[str] = None  # 'hls' or 'rtl'

    @property
    def full_name(self) -> str:
        """Get source-prefixed name (e.g., 'brainsmith:LayerNorm')."""
        return f"{self.source}:{self.name}"

    @property
    def is_loaded(self) -> bool:
        """Check if component has been imported."""
        return self.loaded_obj is not None
