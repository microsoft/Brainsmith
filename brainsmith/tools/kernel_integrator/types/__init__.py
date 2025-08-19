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

# RTL types
from .rtl import (
    Direction,
    Port,
    Parameter,
    ParsedModule,
    ValidationError,
    ValidationResult,
    ProtocolValidationResult,
    PortGroup,
    PragmaType,
)

# Metadata types
from .metadata import (
    Interface,
    AXIStreamInterface,
    AXILiteInterface,
    ControlInterface,
    KernelMetadata
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
    # RTL
    "Direction",
    "Port",
    "Parameter",
    "ParsedModule",
    "ValidationError",
    "ValidationResult",
    "ProtocolValidationResult",
    "PortGroup",
    "PragmaType",
    # Metadata
    "Interface",
    "AXIStreamInterface",
    "AXILiteInterface",
    "ControlInterface",
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