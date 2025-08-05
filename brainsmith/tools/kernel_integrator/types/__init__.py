"""
Kernel Integrator Type System.

This package contains the refactored type system for the kernel integrator,
organized into logical layers to eliminate circular dependencies.

Modules:
- core: Base types with no dependencies (enums, basic specs)
- rtl: RTL parsing types (ports, parameters, modules)
- metadata: Higher-level metadata types
- generation: Code generation types
- binding: Code generation binding types
- config: Configuration types
"""

# Core types
from .core import (
    PortDirection,
    DatatypeSpec,
    DimensionSpec,
)

# RTL types
from .rtl import (
    Port,
    Parameter,
    ParsedModule,
    ValidationError,
    ValidationResult,
)

# Metadata types
from .metadata import (
    InterfaceMetadata,
    KernelMetadata,
)

# Generation types
from .generation import (
    GeneratedFile,
    GenerationContext,
    GenerationResult,
)

# Binding types
from .binding import (
    IOSpec,
    AttributeBinding,
    CodegenBinding,
)

# Config types
from .config import Config

__all__ = [
    # Core
    "PortDirection",
    "DatatypeSpec", 
    "DimensionSpec",
    # RTL
    "Port",
    "Parameter",
    "ParsedModule",
    "ValidationError",
    "ValidationResult",
    # Metadata
    "InterfaceMetadata",
    "KernelMetadata",
    # Generation
    "GeneratedFile",
    "GenerationContext",
    "GenerationResult",
    # Binding
    "IOSpec",
    "AttributeBinding",
    "CodegenBinding",
    # Config
    "Config",
]